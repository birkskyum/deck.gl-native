//! Port of `@math.gl/web-mercator`'s `flyToViewport` and `getFlyToDuration`: the "Smooth and
//! efficient zooming and panning" flight path of van Wijk and Nuij, as used by mapbox-gl's
//! and maplibre's `flyTo`.

use crate::web_mercator::{lng_lat_to_world, scale_to_zoom, world_to_lng_lat, zoom_to_scale};

const EPSILON: f64 = 0.01;

/// Tuning of the flight path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyToOptions {
    /// The zooming "curve" along the path (rho). Default 1.414
    pub curve: f64,
    /// Average speed in relation to `curve`; higher is shorter. Default 1.2
    pub speed: f64,
    /// Average speed in screenfuls per second; `speed` is ignored when set
    pub screen_speed: Option<f64>,
    /// Longest allowed duration in ms; a longer flight gets duration 0 (a jump)
    pub max_duration: Option<f64>,
}

impl Default for FlyToOptions {
    fn default() -> Self {
        Self {
            curve: 1.414,
            speed: 1.2,
            screen_speed: None,
            max_duration: None,
        }
    }
}

/// The parts of a viewport the flight path depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyToProps {
    pub width: f64,
    pub height: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
}

struct Params {
    start_zoom: f64,
    start_center: [f64; 2],
    u_delta: [f64; 2],
    w0: f64,
    u1: f64,
    s: f64,
    rho: f64,
    rho2: f64,
    r0: f64,
}

fn params(start: &FlyToProps, end: &FlyToProps, opts: &FlyToOptions) -> Params {
    let rho = opts.curve;
    let start_zoom = start.zoom;
    let start_scale = zoom_to_scale(start_zoom);
    let scale = zoom_to_scale(end.zoom - start_zoom);
    let start_center = lng_lat_to_world([start.longitude, start.latitude]);
    let end_center = lng_lat_to_world([end.longitude, end.latitude]);
    let u_delta = [end_center[0] - start_center[0], end_center[1] - start_center[1]];
    let w0 = start.width.max(start.height);
    let w1 = w0 / scale;
    let u1 = (u_delta[0] * u_delta[0] + u_delta[1] * u_delta[1]).sqrt() * start_scale;
    // u1 can be 0 if the end center is the same as the start center
    let u1_safe = u1.max(EPSILON);
    let rho2 = rho * rho;
    let b0 = (w1 * w1 - w0 * w0 + rho2 * rho2 * u1_safe * u1_safe) / (2.0 * w0 * rho2 * u1_safe);
    let b1 = (w1 * w1 - w0 * w0 - rho2 * rho2 * u1_safe * u1_safe) / (2.0 * w1 * rho2 * u1_safe);
    let r0 = ((b0 * b0 + 1.0).sqrt() - b0).ln();
    let r1 = ((b1 * b1 + 1.0).sqrt() - b1).ln();
    Params {
        start_zoom,
        start_center,
        u_delta,
        w0,
        u1,
        s: (r1 - r0) / rho,
        rho,
        rho2,
        r0,
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Longitude, latitude and zoom at fraction `t` (0 to 1) of the flight from `start` to `end`.
pub fn fly_to_viewport(start: &FlyToProps, end: &FlyToProps, t: f64, opts: &FlyToOptions) -> [f64; 3] {
    let p = params(start, end, opts);
    // If the change in center is too small, interpolate linearly
    if p.u1 < EPSILON {
        return [
            lerp(start.longitude, end.longitude, t),
            lerp(start.latitude, end.latitude, t),
            lerp(start.zoom, end.zoom, t),
        ];
    }
    let s = t * p.s;
    let w = p.r0.cosh() / (p.r0 + p.rho * s).cosh();
    let u = (p.w0 * ((p.r0.cosh() * (p.r0 + p.rho * s).tanh() - p.r0.sinh()) / p.rho2)) / p.u1;
    let scale_increment = 1.0 / w;
    let zoom = p.start_zoom + scale_to_zoom(scale_increment);
    let center = world_to_lng_lat([
        p.start_center[0] + p.u_delta[0] * u,
        p.start_center[1] + p.u_delta[1] * u,
    ]);
    [center[0], center[1], zoom]
}

/// Duration in milliseconds of the flight from `start` to `end`.
pub fn get_fly_to_duration(start: &FlyToProps, end: &FlyToProps, opts: &FlyToOptions) -> f64 {
    let p = params(start, end, opts);
    let length = 1000.0 * p.s;
    let duration = match opts.screen_speed {
        Some(screen_speed) if screen_speed.is_finite() => length / (screen_speed / p.rho),
        _ => length / opts.speed,
    };
    match opts.max_duration {
        Some(max) if max.is_finite() && duration > max => 0.0,
        _ => duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(longitude: f64, latitude: f64, zoom: f64) -> FlyToProps {
        FlyToProps {
            width: 800.0,
            height: 600.0,
            longitude,
            latitude,
            zoom,
        }
    }

    #[test]
    fn flight_starts_and_ends_at_the_given_views() {
        let start = props(-122.4, 37.8, 12.0);
        let end = props(-74.0, 40.7, 10.0);
        let opts = FlyToOptions::default();
        let a = fly_to_viewport(&start, &end, 0.0, &opts);
        assert!((a[0] - start.longitude).abs() < 1e-9 && (a[1] - start.latitude).abs() < 1e-9);
        assert!((a[2] - start.zoom).abs() < 1e-9);
        let b = fly_to_viewport(&start, &end, 1.0, &opts);
        assert!((b[0] - end.longitude).abs() < 1e-6, "{b:?}");
        assert!((b[1] - end.latitude).abs() < 1e-6, "{b:?}");
        assert!((b[2] - end.zoom).abs() < 1e-6, "{b:?}");
        // A long flight zooms out in the middle
        let mid = fly_to_viewport(&start, &end, 0.5, &opts);
        assert!(mid[2] < start.zoom.min(end.zoom), "{mid:?}");
        assert!(mid[0] > start.longitude && mid[0] < end.longitude);
        let duration = get_fly_to_duration(&start, &end, &opts);
        assert!(duration > 1000.0 && duration < 20000.0, "{duration}");
        assert_eq!(
            get_fly_to_duration(
                &start,
                &end,
                &FlyToOptions {
                    max_duration: Some(1000.0),
                    ..Default::default()
                }
            ),
            0.0
        );
    }

    #[test]
    fn same_center_interpolates_zoom_linearly() {
        let start = props(10.0, 50.0, 4.0);
        let end = props(10.0, 50.0, 8.0);
        let mid = fly_to_viewport(&start, &end, 0.5, &FlyToOptions::default());
        assert!((mid[2] - 6.0).abs() < 1e-9 && (mid[0] - 10.0).abs() < 1e-9);
    }
}

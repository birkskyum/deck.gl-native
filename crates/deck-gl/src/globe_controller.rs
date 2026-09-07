//! Interactive control of a globe view: drag rotates the globe under the pointer at a constant
//! screen speed (deck.gl's `GlobeViewport.panByPosition`), right drag changes bearing and
//! pitch, scroll zooms, keyboard moves. Zoom follows latitude so the globe and the map agree.

use crate::deck::ViewState;
use crate::viewport::{globe_zoom_adjust, Viewport};
use crate::views::{AnyViewState, GlobeViewProps, View};

/// Limits of the globe camera, deck.gl's defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeConstraints {
    pub min_zoom: f64,
    pub max_zoom: f64,
    pub min_pitch: f64,
    pub max_pitch: f64,
}

impl Default for GlobeConstraints {
    fn default() -> Self {
        Self {
            min_zoom: 0.0,
            max_zoom: 20.0,
            min_pitch: 0.0,
            max_pitch: 60.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlobeController {
    props: GlobeViewProps,
    view_state: ViewState,
    width: f64,
    height: f64,
    pub constraints: GlobeConstraints,
    /// Longitude, latitude, zoom and pixel when the drag started
    start_pan: Option<([f64; 3], [f64; 2])>,
    start_rotate: Option<([f64; 2], f64, f64)>,
    start_zoom: Option<f64>,
    interacted: bool,
}

impl GlobeController {
    pub fn new(props: GlobeViewProps, view_state: ViewState, width: f64, height: f64) -> Self {
        let mut controller = Self {
            props,
            view_state,
            width: width.max(1.0),
            height: height.max(1.0),
            constraints: GlobeConstraints::default(),
            start_pan: None,
            start_rotate: None,
            start_zoom: None,
            interacted: false,
        };
        controller.view_state = controller.constrain(view_state);
        controller
    }

    pub fn view_state(&self) -> ViewState {
        self.view_state
    }

    pub fn any_view_state(&self) -> AnyViewState {
        AnyViewState::Globe(self.view_state)
    }

    pub fn set_view_state(&mut self, view_state: ViewState) {
        self.view_state = self.constrain(view_state);
    }

    pub fn set_size(&mut self, width: f64, height: f64) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
    }

    pub fn interacted(&self) -> bool {
        self.interacted
    }

    pub fn is_active(&self) -> bool {
        self.start_pan.is_some() || self.start_rotate.is_some() || self.start_zoom.is_some()
    }

    pub fn viewport(&self) -> Viewport {
        View::Globe(self.props).make_viewport(&AnyViewState::Globe(self.view_state), self.width, self.height)
    }

    fn constrain(&self, mut v: ViewState) -> ViewState {
        let c = &self.constraints;
        v.latitude = v.latitude.clamp(-90.0, 90.0);
        if !(-180.0..=180.0).contains(&v.longitude) {
            v.longitude = (v.longitude + 180.0).rem_euclid(360.0) - 180.0;
        }
        if !(-180.0..=180.0).contains(&v.bearing) {
            v.bearing = (v.bearing + 180.0).rem_euclid(360.0) - 180.0;
        }
        v.pitch = v.pitch.clamp(c.min_pitch, c.max_pitch);
        // Zoom limits follow the latitude adjustment so they mean the same at every latitude
        let adjustment = globe_zoom_adjust(v.latitude, true) - globe_zoom_adjust(0.0, true);
        v.zoom = v.zoom.clamp(c.min_zoom + adjustment, c.max_zoom + adjustment);
        v
    }

    pub fn pan_start(&mut self, pixel: [f64; 2]) {
        let v = self.view_state;
        self.start_pan = Some(([v.longitude, v.latitude, v.zoom], pixel));
    }

    pub fn pan(&mut self, pixel: [f64; 2]) {
        let Some((start, start_pixel)) = self.start_pan else {
            return;
        };
        let [longitude, latitude, zoom] = self.viewport().globe_pan_by_position(start, pixel, start_pixel);
        self.view_state = self.constrain(ViewState {
            longitude,
            latitude,
            zoom,
            ..self.view_state
        });
        self.interacted = true;
    }

    pub fn pan_end(&mut self) {
        self.start_pan = None;
    }

    pub fn rotate_start(&mut self, pixel: [f64; 2]) {
        self.start_rotate = Some((pixel, self.view_state.bearing, self.view_state.pitch));
    }

    /// Right drag: a full drag across the width turns the bearing by 180 degrees, vertical
    /// drags move the pitch towards its limits, as in deck.gl's MapState.
    pub fn rotate(&mut self, pixel: [f64; 2]) {
        let Some((start, start_bearing, start_pitch)) = self.start_rotate else {
            return;
        };
        let delta_scale_x = (pixel[0] - start[0]) / self.width;
        let delta_scale_y = ((pixel[1] - start[1]) / self.height).clamp(-1.0, 1.0);
        let bearing = start_bearing + 180.0 * delta_scale_x;
        let c = &self.constraints;
        let pitch = if delta_scale_y > 0.0 {
            start_pitch + delta_scale_y * (c.max_pitch - start_pitch)
        } else {
            start_pitch - delta_scale_y * (c.min_pitch - start_pitch)
        };
        self.view_state = self.constrain(ViewState {
            bearing,
            pitch,
            ..self.view_state
        });
        self.interacted = true;
    }

    pub fn rotate_end(&mut self) {
        self.start_rotate = None;
    }

    pub fn rotate_by(&mut self, bearing: f64, pitch: f64) {
        self.view_state = self.constrain(ViewState {
            bearing: self.view_state.bearing + bearing,
            pitch: self.view_state.pitch + pitch,
            ..self.view_state
        });
        self.interacted = true;
    }

    /// Zoom by `delta` levels around the centre (the globe keeps its target).
    pub fn zoom_by(&mut self, delta: f64) {
        self.view_state = self.constrain(ViewState {
            zoom: self.view_state.zoom + delta,
            ..self.view_state
        });
        self.interacted = true;
    }

    pub fn zoom_start(&mut self) {
        self.start_zoom = Some(self.view_state.zoom);
    }

    pub fn zoom(&mut self, scale: f64) {
        let Some(start) = self.start_zoom else { return };
        self.view_state = self.constrain(ViewState {
            zoom: start + scale.max(f64::EPSILON).log2(),
            ..self.view_state
        });
        self.interacted = true;
    }

    pub fn zoom_end(&mut self) {
        self.start_zoom = None;
    }

    pub fn zoom_in(&mut self) {
        self.zoom_by(1.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_by(-1.0);
    }

    /// Move by screen pixels (keyboard).
    pub fn move_by(&mut self, pixels: [f64; 2]) {
        let centre = [self.width / 2.0, self.height / 2.0];
        self.pan_start(centre);
        self.pan([centre[0] + pixels[0], centre[1] + pixels[1]]);
        self.pan_end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_rotates_the_globe_and_keeps_limits() {
        let start = ViewState {
            longitude: 10.0,
            latitude: 20.0,
            zoom: 2.0,
            pitch: 0.0,
            bearing: 0.0,
        };
        let mut c = GlobeController::new(GlobeViewProps::default(), start, 400.0, 300.0);
        c.pan_start([200.0, 150.0]);
        c.pan([250.0, 150.0]);
        c.pan_end();
        assert!(
            c.view_state().longitude < 10.0,
            "dragging right moves the globe east under the pointer"
        );
        assert!((c.view_state().latitude - 20.0).abs() < 1e-9);
        c.pan_start([200.0, 150.0]);
        c.pan([200.0, 100.0]);
        c.pan_end();
        assert!(c.view_state().latitude < 20.0, "dragging up shows the south");
        assert!(c.interacted());
        c.rotate_start([0.0, 0.0]);
        c.rotate([200.0, 300.0]);
        c.rotate_end();
        assert!((c.view_state().bearing - 90.0).abs() < 1e-9);
        assert!(
            (c.view_state().pitch - 60.0).abs() < 1e-9,
            "pitch reaches its limit"
        );
        c.zoom_by(30.0);
        let adjustment = globe_zoom_adjust(c.view_state().latitude, true) - globe_zoom_adjust(0.0, true);
        assert!(
            (c.view_state().zoom - (20.0 + adjustment)).abs() < 1e-9,
            "max zoom follows latitude"
        );
        c.zoom_start();
        c.zoom(0.25);
        c.zoom_end();
        assert!((c.view_state().zoom - (18.0 + adjustment)).abs() < 1e-9);
    }
}

//! Port of deck.gl's `MapController` and `MapState`: turns pointer, wheel and key input into
//! view state changes with deck.gl's gestures (drag to pan, right drag or a modifier to rotate
//! and pitch, wheel to zoom around the cursor) and inertia after a pan.
//!
//! The controller is independent of any windowing library: feed it pixel positions in logical
//! pixels from the top left, then read [`MapController::view_state`] each frame after calling
//! [`MapController::tick`].

use crate::deck::ViewState;
use crate::transition::{TransitionProps, ViewStateTransition};
use crate::viewport::{Viewport, WebMercatorViewportOptions};

const PITCH_MOUSE_THRESHOLD: f64 = 5.0;
const PITCH_ACCEL: f64 = 1.2;

/// Limits applied to every view state change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraints {
    pub min_zoom: f64,
    pub max_zoom: f64,
    pub min_pitch: f64,
    pub max_pitch: f64,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            min_zoom: 0.0,
            max_zoom: 20.0,
            min_pitch: 0.0,
            max_pitch: 60.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PanInertia {
    start: ViewState,
    start_lng_lat: [f64; 2],
    from_pixel: [f64; 2],
    to_pixel: [f64; 2],
    start_time: f64,
    duration: f64,
}

/// Interactive control of a Web Mercator view.
#[derive(Clone, Debug)]
pub struct MapController {
    view_state: ViewState,
    width: f64,
    height: f64,
    pub constraints: Constraints,
    /// Duration of the pan inertia in milliseconds; 0 disables it
    pub inertia: f64,
    start_pan_lng_lat: Option<[f64; 2]>,
    start_rotate: Option<([f64; 2], f64, f64)>,
    start_zoom: Option<([f64; 2], f64)>,
    /// Recent pan samples as (pixel, time in ms) for the inertia velocity
    samples: Vec<([f64; 2], f64)>,
    animation: Option<PanInertia>,
    transition: Option<ViewStateTransition>,
    interacted: bool,
}

impl MapController {
    pub fn new(view_state: ViewState, width: f64, height: f64) -> Self {
        Self {
            view_state,
            width,
            height,
            constraints: Constraints::default(),
            inertia: 300.0,
            start_pan_lng_lat: None,
            start_rotate: None,
            start_zoom: None,
            samples: Vec::new(),
            animation: None,
            transition: None,
            interacted: false,
        }
    }

    /// Animate to `end` with deck.gl's transition props, starting at `now` (milliseconds,
    /// the same clock as [`MapController::tick`]). Interrupts a running transition according to
    /// the new props' `interruption`. A zero duration jumps straight to `end`.
    pub fn transition_to(&mut self, end: ViewState, props: TransitionProps, now: f64) {
        let (current, proceed) =
            crate::transition::interrupt(self.transition.as_ref(), self.view_state, props.interruption);
        if !proceed {
            return;
        }
        self.animation = None;
        self.interacted = true;
        self.view_state = self.constrain(current);
        let end = self.constrain(end);
        self.transition = ViewStateTransition::new(self.view_state, end, self.width, self.height, props, now);
        if self.transition.is_none() {
            self.view_state = end;
        }
    }

    /// Fly to `end` along the van Wijk and Nuij path with an automatic duration.
    pub fn fly_to(&mut self, end: ViewState, now: f64) {
        self.transition_to(end, TransitionProps::fly_to(), now);
    }

    /// The transition in flight, if any.
    pub fn transition(&self) -> Option<&ViewStateTransition> {
        self.transition.as_ref()
    }

    pub fn view_state(&self) -> ViewState {
        self.view_state
    }

    /// Replace the view state, for instance from an external animation.
    /// Jump to a view state, ending any transition.
    pub fn set_view_state(&mut self, view_state: ViewState) {
        self.transition = None;
        self.view_state = self.constrain(view_state);
    }

    pub fn set_size(&mut self, width: f64, height: f64) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
    }

    /// True once the user has panned, rotated or zoomed.
    pub fn interacted(&self) -> bool {
        self.interacted
    }

    /// Whether a gesture or inertia is in progress.
    pub fn is_active(&self) -> bool {
        self.start_pan_lng_lat.is_some()
            || self.start_rotate.is_some()
            || self.start_zoom.is_some()
            || self.animation.is_some()
            || self.transition.is_some()
    }

    fn viewport(&self, view_state: &ViewState) -> Viewport {
        Viewport::web_mercator(&WebMercatorViewportOptions {
            width: self.width,
            height: self.height,
            longitude: view_state.longitude,
            latitude: view_state.latitude,
            zoom: view_state.zoom,
            pitch: view_state.pitch,
            bearing: view_state.bearing,
            ..Default::default()
        })
    }

    fn unproject(&self, pixel: [f64; 2]) -> [f64; 2] {
        let p = self.viewport(&self.view_state).unproject(
            crate::glam::DVec2::new(pixel[0], pixel[1]),
            None,
            true,
            None,
        );
        [p.x, p.y]
    }

    fn constrain(&self, mut state: ViewState) -> ViewState {
        let c = &self.constraints;
        if !(-180.0..=180.0).contains(&state.longitude) {
            state.longitude = (state.longitude + 180.0).rem_euclid(360.0) - 180.0;
        }
        if !(-180.0..=180.0).contains(&state.bearing) {
            state.bearing = (state.bearing + 180.0).rem_euclid(360.0) - 180.0;
        }
        state.latitude = state.latitude.clamp(-85.05, 85.05);
        state.pitch = state.pitch.clamp(c.min_pitch, c.max_pitch);
        state.zoom = state.zoom.clamp(c.min_zoom, c.max_zoom);
        state
    }

    fn set(&mut self, state: ViewState) {
        self.view_state = self.constrain(state);
        self.interacted = true;
    }

    /// Start dragging at `pixel` (`now` in milliseconds, for inertia).
    pub fn pan_start(&mut self, pixel: [f64; 2], now: f64) {
        self.animation = None;
        self.transition = None;
        self.start_pan_lng_lat = Some(self.unproject(pixel));
        self.samples.clear();
        self.samples.push((pixel, now));
    }

    /// Drag to `pixel`: the point grabbed at `pan_start` follows the cursor.
    pub fn pan(&mut self, pixel: [f64; 2], now: f64) {
        let Some(anchor) = self.start_pan_lng_lat else {
            return;
        };
        let [longitude, latitude] = self.viewport(&self.view_state).pan_by_position(anchor, pixel);
        self.set(ViewState {
            longitude,
            latitude,
            ..self.view_state
        });
        self.samples.push((pixel, now));
        if self.samples.len() > 8 {
            self.samples.remove(0);
        }
    }

    /// Release the drag; a fast release keeps moving with inertia.
    pub fn pan_end(&mut self, now: f64) {
        let anchor = self.start_pan_lng_lat.take();
        let velocity = self.release_velocity(now);
        if let (Some(anchor), Some((pixel, velocity))) = (anchor, velocity) {
            if self.inertia > 0.0 && (velocity[0].abs() + velocity[1].abs()) > 0.05 {
                let to = [
                    pixel[0] + velocity[0] * self.inertia / 2.0,
                    pixel[1] + velocity[1] * self.inertia / 2.0,
                ];
                self.animation = Some(PanInertia {
                    start: self.view_state,
                    start_lng_lat: anchor,
                    from_pixel: pixel,
                    to_pixel: to,
                    start_time: now,
                    duration: self.inertia,
                });
            }
        }
        self.samples.clear();
    }

    /// Pixel velocity (per millisecond) over the last samples, and the last pixel, when the
    /// pointer was still moving at release.
    fn release_velocity(&self, now: f64) -> Option<([f64; 2], [f64; 2])> {
        let (last_pixel, last_time) = *self.samples.last()?;
        if now - last_time > 50.0 {
            return None;
        }
        let (first_pixel, first_time) = self.samples.first().copied()?;
        let dt = last_time - first_time;
        if dt <= 0.0 {
            return None;
        }
        Some((
            last_pixel,
            [
                (last_pixel[0] - first_pixel[0]) / dt,
                (last_pixel[1] - first_pixel[1]) / dt,
            ],
        ))
    }

    /// Advance a transition or inertia; returns true while the view is still changing.
    pub fn tick(&mut self, now: f64) -> bool {
        if let Some(transition) = self.transition {
            self.view_state = self.constrain(transition.at(now));
            if transition.is_done(now) {
                self.transition = None;
            }
            return true;
        }
        let Some(animation) = self.animation else {
            return false;
        };
        let t = ((now - animation.start_time) / animation.duration).clamp(0.0, 1.0);
        // ease out
        let eased = t * (2.0 - t);
        let pixel = [
            animation.from_pixel[0] + (animation.to_pixel[0] - animation.from_pixel[0]) * eased,
            animation.from_pixel[1] + (animation.to_pixel[1] - animation.from_pixel[1]) * eased,
        ];
        let [longitude, latitude] = self
            .viewport(&animation.start)
            .pan_by_position(animation.start_lng_lat, pixel);
        self.view_state = self.constrain(ViewState {
            longitude,
            latitude,
            ..animation.start
        });
        if t >= 1.0 {
            self.animation = None;
            self.transition = None;
        }
        true
    }

    pub fn rotate_start(&mut self, pixel: [f64; 2]) {
        self.animation = None;
        self.transition = None;
        self.start_rotate = Some((pixel, self.view_state.bearing, self.view_state.pitch));
    }

    /// Drag to rotate: horizontal motion turns the bearing, vertical motion pitches, like
    /// deck.gl's `_getNewRotation`.
    pub fn rotate(&mut self, pixel: [f64; 2]) {
        let Some((start, start_bearing, start_pitch)) = self.start_rotate else {
            return;
        };
        let delta_x = pixel[0] - start[0];
        let delta_y = pixel[1] - start[1];
        let delta_scale_x = delta_x / self.width;
        let mut delta_scale_y = 0.0;
        if delta_y > 0.0 {
            if (self.height - start[1]).abs() > PITCH_MOUSE_THRESHOLD {
                delta_scale_y = delta_y / (start[1] - self.height) * PITCH_ACCEL;
            }
        } else if delta_y < 0.0 && start[1] > PITCH_MOUSE_THRESHOLD {
            delta_scale_y = 1.0 - pixel[1] / start[1];
        }
        delta_scale_y = delta_scale_y.clamp(-1.0, 1.0);
        let bearing = start_bearing + 180.0 * delta_scale_x;
        let pitch = if delta_scale_y > 0.0 {
            start_pitch + delta_scale_y * (self.constraints.max_pitch - start_pitch)
        } else if delta_scale_y < 0.0 {
            start_pitch - delta_scale_y * (self.constraints.min_pitch - start_pitch)
        } else {
            start_pitch
        };
        self.set(ViewState {
            bearing,
            pitch,
            ..self.view_state
        });
    }

    pub fn rotate_end(&mut self) {
        self.start_rotate = None;
    }

    /// Zoom by `delta` levels keeping the ground under `pixel` fixed (wheel input).
    pub fn zoom_by(&mut self, pixel: [f64; 2], delta: f64) {
        self.animation = None;
        self.transition = None;
        let anchor = self.unproject(pixel);
        let zoom = (self.view_state.zoom + delta).clamp(self.constraints.min_zoom, self.constraints.max_zoom);
        let zoomed = self.viewport(&ViewState {
            zoom,
            ..self.view_state
        });
        let [longitude, latitude] = zoomed.pan_by_position(anchor, pixel);
        self.set(ViewState {
            zoom,
            longitude,
            latitude,
            ..self.view_state
        });
    }

    /// Start a pinch or drag zoom at `pixel`.
    pub fn zoom_start(&mut self, pixel: [f64; 2]) {
        self.animation = None;
        self.transition = None;
        self.start_zoom = Some((self.unproject(pixel), self.view_state.zoom));
    }

    /// Scale the view relative to the zoom at `zoom_start`, anchored at `pixel`.
    pub fn zoom(&mut self, pixel: [f64; 2], scale: f64) {
        let Some((anchor, start_zoom)) = self.start_zoom else {
            return;
        };
        let zoom = (start_zoom + scale.log2()).clamp(self.constraints.min_zoom, self.constraints.max_zoom);
        let zoomed = self.viewport(&ViewState {
            zoom,
            ..self.view_state
        });
        let [longitude, latitude] = zoomed.pan_by_position(anchor, pixel);
        self.set(ViewState {
            zoom,
            longitude,
            latitude,
            ..self.view_state
        });
    }

    pub fn zoom_end(&mut self) {
        self.start_zoom = None;
    }

    /// Keyboard style zoom around the centre; deck.gl's default speed doubles the scale.
    pub fn zoom_in(&mut self) {
        self.zoom_by([self.width / 2.0, self.height / 2.0], 1.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_by([self.width / 2.0, self.height / 2.0], -1.0);
    }

    /// Move the view by a pixel offset (positive x moves the map right).
    pub fn move_by(&mut self, pixels: [f64; 2]) {
        self.animation = None;
        self.transition = None;
        let center = [self.width / 2.0, self.height / 2.0];
        let anchor = self.unproject(center);
        let [longitude, latitude] = self
            .viewport(&self.view_state)
            .pan_by_position(anchor, [center[0] + pixels[0], center[1] + pixels[1]]);
        self.set(ViewState {
            longitude,
            latitude,
            ..self.view_state
        });
    }

    pub fn rotate_by(&mut self, bearing: f64, pitch: f64) {
        self.animation = None;
        self.transition = None;
        self.set(ViewState {
            bearing: self.view_state.bearing + bearing,
            pitch: self.view_state.pitch + pitch,
            ..self.view_state
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glam::DVec3;

    fn controller() -> MapController {
        MapController::new(
            ViewState {
                longitude: -122.4,
                latitude: 37.8,
                zoom: 12.0,
                pitch: 30.0,
                bearing: 20.0,
            },
            800.0,
            600.0,
        )
    }

    fn pixel_of(c: &MapController, lng_lat: [f64; 2]) -> [f64; 2] {
        let p = c
            .viewport(&c.view_state())
            .project(DVec3::new(lng_lat[0], lng_lat[1], 0.0), true);
        [p.x, p.y]
    }

    #[test]
    fn panning_keeps_the_grabbed_point_under_the_cursor() {
        let mut c = controller();
        c.inertia = 0.0;
        let grabbed = c.unproject([300.0, 400.0]);
        c.pan_start([300.0, 400.0], 0.0);
        c.pan([420.0, 350.0], 16.0);
        c.pan_end(32.0);
        let p = pixel_of(&c, grabbed);
        assert!(
            (p[0] - 420.0).abs() < 1e-6 && (p[1] - 350.0).abs() < 1e-6,
            "{p:?}"
        );
        assert!(c.interacted());
    }

    #[test]
    fn inertia_keeps_moving_after_a_fast_release() {
        let mut c = controller();
        c.pan_start([400.0, 300.0], 0.0);
        c.pan([420.0, 300.0], 16.0);
        c.pan([440.0, 300.0], 32.0);
        let before = c.view_state();
        c.pan_end(40.0);
        assert!(c.is_active());
        assert!(c.tick(190.0));
        let mid = c.view_state();
        assert_ne!(mid.longitude, before.longitude, "still moving");
        assert!(c.tick(400.0), "the last step applies the end position");
        assert!(!c.is_active());
        assert!(!c.tick(500.0), "nothing left to animate");
        let end = c.view_state();
        assert!((end.longitude - mid.longitude).abs() < (mid.longitude - before.longitude).abs());
    }

    #[test]
    fn wheel_zoom_is_anchored_at_the_cursor() {
        let mut c = controller();
        let anchor = c.unproject([100.0, 500.0]);
        c.zoom_by([100.0, 500.0], 1.5);
        assert!((c.view_state().zoom - 13.5).abs() < 1e-9);
        let p = pixel_of(&c, anchor);
        assert!(
            (p[0] - 100.0).abs() < 1e-6 && (p[1] - 500.0).abs() < 1e-6,
            "{p:?}"
        );
        c.zoom_by([100.0, 500.0], 100.0);
        assert_eq!(c.view_state().zoom, 20.0, "clamped to max zoom");
    }

    #[test]
    fn rotating_changes_bearing_and_pitch_within_limits() {
        let mut c = controller();
        c.rotate_start([400.0, 300.0]);
        c.rotate([600.0, 300.0]);
        assert!((c.view_state().bearing - (20.0 + 180.0 * 0.25)).abs() < 1e-9);
        c.rotate([400.0, 0.0]);
        assert!(
            (c.view_state().pitch - 60.0).abs() < 1e-9,
            "dragging up to the top pitches fully"
        );
        c.rotate([400.0, 600.0]);
        assert!(
            (c.view_state().pitch - 0.0).abs() < 1e-9,
            "dragging to the bottom flattens"
        );
        c.rotate_end();
        c.rotate_by(200.0, 0.0);
        assert!(
            c.view_state().bearing <= 180.0,
            "bearing wraps: {}",
            c.view_state().bearing
        );
    }

    #[test]
    fn transitions_advance_and_gestures_interrupt_them() {
        use crate::transition::{TransitionInterruption, TransitionProps};
        let start = ViewState {
            longitude: 0.0,
            latitude: 0.0,
            zoom: 4.0,
            pitch: 0.0,
            bearing: 0.0,
        };
        let end = ViewState {
            longitude: 10.0,
            zoom: 6.0,
            ..start
        };
        let mut c = MapController::new(start, 400.0, 300.0);
        c.transition_to(end, TransitionProps::linear(1000.0), 0.0);
        assert!(c.is_active() && c.interacted());
        assert!(c.tick(500.0));
        assert!((c.view_state().longitude - 5.0).abs() < 1e-9);
        assert!((c.view_state().zoom - 5.0).abs() < 1e-9);
        // Break (the default) restarts from the current view
        let elsewhere = ViewState {
            longitude: -20.0,
            ..start
        };
        c.transition_to(elsewhere, TransitionProps::linear(1000.0), 500.0);
        c.tick(1000.0);
        assert!(
            (c.view_state().longitude - (-7.5)).abs() < 1e-9,
            "{:?}",
            c.view_state()
        );
        // Ignore keeps the running transition
        c.transition_to(
            end,
            TransitionProps::linear(1000.0).with_interruption(TransitionInterruption::Ignore),
            1000.0,
        );
        c.tick(1500.0);
        assert!((c.view_state().longitude - (-20.0)).abs() < 1e-9);
        assert!(!c.is_active(), "finished at 1500");
        // Snap to end jumps to the running transition's end before starting the new one
        c.transition_to(end, TransitionProps::linear(1000.0), 2000.0);
        c.transition_to(
            start,
            TransitionProps::linear(1000.0).with_interruption(TransitionInterruption::SnapToEnd),
            2500.0,
        );
        assert_eq!(c.view_state(), end);
        // A gesture ends a transition
        c.tick(2600.0);
        c.pan_start([10.0, 10.0], 2600.0);
        assert!(c.transition().is_none());
        // Fly to has an automatic duration and lands exactly
        let mut c = MapController::new(start, 400.0, 300.0);
        c.fly_to(end, 0.0);
        let duration = c.transition().unwrap().duration_ms();
        assert!(duration > 500.0);
        c.tick(duration / 2.0);
        let mid = c.view_state();
        assert!(
            mid.longitude > 0.0 && mid.longitude < 10.0 && mid.zoom < 6.0,
            "midway {mid:?}"
        );
        c.tick(duration);
        assert!((c.view_state().longitude - 10.0).abs() < 1e-6 && (c.view_state().zoom - 6.0).abs() < 1e-6);
        assert!(!c.is_active());
        // A zero duration jumps
        c.transition_to(start, TransitionProps::linear(0.0), 9000.0);
        assert_eq!(c.view_state(), start);
    }
}

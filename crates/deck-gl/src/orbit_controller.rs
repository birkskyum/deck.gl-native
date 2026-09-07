//! Port of deck.gl's `OrbitController` and `OrthographicController` (which extends it): drag to
//! pan the target, drag with the right button to rotate (orbit view only), scroll to zoom
//! around the cursor, keyboard moves. Like [`crate::MapController`] it is independent of the
//! windowing library; feed it pointer positions in logical pixels.

use glam::{DVec2, DVec3};

use crate::viewport::Viewport;
use crate::views::{AnyViewState, OrbitViewState, OrthographicViewState, View};

/// Limits of an orbit or orthographic camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitConstraints {
    pub min_zoom: f64,
    pub max_zoom: f64,
    /// Rotation around X in degrees, orbit view only
    pub min_rotation_x: f64,
    pub max_rotation_x: f64,
}

impl Default for OrbitConstraints {
    fn default() -> Self {
        Self {
            min_zoom: f64::NEG_INFINITY,
            max_zoom: f64::INFINITY,
            min_rotation_x: -90.0,
            max_rotation_x: 90.0,
        }
    }
}

/// Interactive control of an orbit or orthographic view.
#[derive(Clone, Debug)]
pub struct OrbitController {
    view: View,
    state: AnyViewState,
    width: f64,
    height: f64,
    pub constraints: OrbitConstraints,
    /// World position grabbed at the start of a pan
    start_pan_position: Option<DVec3>,
    start_rotate: Option<(DVec2, f64, f64)>,
    /// World position under the pointer and the zoom when a pinch started
    start_zoom: Option<(DVec3, f64)>,
    interacted: bool,
}

impl OrbitController {
    /// `view` must be an orbit or orthographic view; other views fall back to an orbit view.
    pub fn new(view: View, state: AnyViewState, width: f64, height: f64) -> Self {
        let view = match view {
            View::Orbit(_) | View::Orthographic(_) => view,
            _ => View::Orbit(Default::default()),
        };
        let mut controller = Self {
            view,
            state: view.default_view_state(),
            width: width.max(1.0),
            height: height.max(1.0),
            constraints: OrbitConstraints::default(),
            start_pan_position: None,
            start_rotate: None,
            start_zoom: None,
            interacted: false,
        };
        controller.set_view_state(state);
        controller
    }

    pub fn view(&self) -> View {
        self.view
    }

    pub fn view_state(&self) -> AnyViewState {
        self.state
    }

    /// Jump to a view state; states of another kind are ignored.
    pub fn set_view_state(&mut self, state: AnyViewState) {
        let matches = matches!(
            (&self.view, &state),
            (View::Orbit(_), AnyViewState::Orbit(_)) | (View::Orthographic(_), AnyViewState::Orthographic(_))
        );
        if matches {
            self.state = self.constrain(state);
        }
    }

    pub fn set_size(&mut self, width: f64, height: f64) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
    }

    /// Whether the user has moved the camera since creation.
    pub fn interacted(&self) -> bool {
        self.interacted
    }

    /// Whether a gesture is in progress.
    pub fn is_active(&self) -> bool {
        self.start_pan_position.is_some() || self.start_rotate.is_some() || self.start_zoom.is_some()
    }

    pub fn viewport(&self) -> Viewport {
        self.view.make_viewport(&self.state, self.width, self.height)
    }

    fn current_zoom(&self) -> f64 {
        match self.state {
            AnyViewState::Orbit(s) => s.zoom,
            AnyViewState::Orthographic(s) => s.zoom,
            _ => 0.0,
        }
    }

    fn target(&self) -> DVec3 {
        match self.state {
            AnyViewState::Orbit(s) => DVec3::from(s.target),
            AnyViewState::Orthographic(s) => DVec3::from(s.target),
            _ => DVec3::ZERO,
        }
    }

    fn with_zoom_and_target(&self, zoom: f64, target: DVec3) -> AnyViewState {
        match self.state {
            AnyViewState::Orbit(s) => AnyViewState::Orbit(OrbitViewState {
                zoom,
                target: target.to_array(),
                ..s
            }),
            AnyViewState::Orthographic(_) => AnyViewState::Orthographic(OrthographicViewState {
                zoom,
                target: target.to_array(),
                zoom_x: None,
                zoom_y: None,
            }),
            other => other,
        }
    }

    fn constrain(&self, state: AnyViewState) -> AnyViewState {
        let c = &self.constraints;
        match state {
            AnyViewState::Orbit(s) => AnyViewState::Orbit(OrbitViewState {
                zoom: s.zoom.clamp(c.min_zoom, c.max_zoom),
                rotation_x: s.rotation_x.clamp(c.min_rotation_x, c.max_rotation_x),
                ..s
            }),
            AnyViewState::Orthographic(s) => AnyViewState::Orthographic(OrthographicViewState {
                zoom: s.zoom.clamp(c.min_zoom, c.max_zoom),
                zoom_x: s.zoom_x.map(|z| z.clamp(c.min_zoom, c.max_zoom)),
                zoom_y: s.zoom_y.map(|z| z.clamp(c.min_zoom, c.max_zoom)),
                ..s
            }),
            other => other,
        }
    }

    fn unproject(&self, pixel: [f64; 2]) -> DVec3 {
        self.viewport()
            .unproject(DVec2::new(pixel[0], pixel[1]), None, true, None)
    }

    /// Start dragging at `pixel`.
    pub fn pan_start(&mut self, pixel: [f64; 2]) {
        self.start_pan_position = Some(self.unproject(pixel));
    }

    /// Drag: the grabbed world position follows the pointer.
    pub fn pan(&mut self, pixel: [f64; 2]) {
        let Some(start) = self.start_pan_position else {
            return;
        };
        let target = self
            .viewport()
            .pan_target_by_position(start, DVec2::new(pixel[0], pixel[1]));
        self.state = self.constrain(self.with_zoom_and_target(self.current_zoom(), target));
        self.interacted = true;
    }

    pub fn pan_end(&mut self) {
        self.start_pan_position = None;
    }

    /// Start rotating at `pixel` (orbit view only).
    pub fn rotate_start(&mut self, pixel: [f64; 2]) {
        if let AnyViewState::Orbit(s) = self.state {
            self.start_rotate = Some((DVec2::new(pixel[0], pixel[1]), s.rotation_x, s.rotation_orbit));
        }
    }

    /// Rotate: a full drag across the width or height turns 180 degrees.
    pub fn rotate(&mut self, pixel: [f64; 2]) {
        let Some((start, start_x, start_orbit)) = self.start_rotate else {
            return;
        };
        let mut delta_scale_x = (pixel[0] - start.x) / self.width;
        let delta_scale_y = (pixel[1] - start.y) / self.height;
        if !(-90.0..=90.0).contains(&start_x) {
            // Looking at the back of the scene: invert the horizontal drag
            delta_scale_x = -delta_scale_x;
        }
        self.rotate_to(
            start_x + delta_scale_y * 180.0,
            start_orbit + delta_scale_x * 180.0,
        );
    }

    pub fn rotate_end(&mut self) {
        self.start_rotate = None;
    }

    fn rotate_to(&mut self, rotation_x: f64, rotation_orbit: f64) {
        if let AnyViewState::Orbit(s) = self.state {
            self.state = self.constrain(AnyViewState::Orbit(OrbitViewState {
                rotation_x,
                rotation_orbit,
                ..s
            }));
            self.interacted = true;
        }
    }

    /// Rotate by degrees around the orbit axis and X (keyboard).
    pub fn rotate_by(&mut self, orbit: f64, x: f64) {
        if let AnyViewState::Orbit(s) = self.state {
            self.rotate_to(s.rotation_x + x, s.rotation_orbit + orbit);
        }
    }

    /// Zoom by `delta` levels keeping the world position under `pixel` in place (scroll).
    pub fn zoom_by(&mut self, pixel: [f64; 2], delta: f64) {
        let anchor = self.unproject(pixel);
        self.zoom_to(anchor, self.current_zoom() + delta, pixel);
    }

    /// Start a pinch zoom at `pixel`.
    pub fn zoom_start(&mut self, pixel: [f64; 2]) {
        self.start_zoom = Some((self.unproject(pixel), self.current_zoom()));
    }

    /// Pinch: `scale` is the accumulated relative scale since `zoom_start`.
    pub fn zoom(&mut self, pixel: [f64; 2], scale: f64) {
        let Some((anchor, start_zoom)) = self.start_zoom else {
            return;
        };
        self.zoom_to(anchor, start_zoom + scale.max(f64::EPSILON).log2(), pixel);
    }

    pub fn zoom_end(&mut self) {
        self.start_zoom = None;
    }

    fn zoom_to(&mut self, anchor: DVec3, zoom: f64, pixel: [f64; 2]) {
        let zoom = zoom.clamp(self.constraints.min_zoom, self.constraints.max_zoom);
        let zoomed = self.with_zoom_and_target(zoom, self.target());
        let viewport = self.view.make_viewport(&zoomed, self.width, self.height);
        let target = viewport.pan_target_by_position(anchor, DVec2::new(pixel[0], pixel[1]));
        self.state = self.constrain(self.with_zoom_and_target(zoom, target));
        self.interacted = true;
    }

    pub fn zoom_in(&mut self) {
        self.zoom_by([self.width / 2.0, self.height / 2.0], 1.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_by([self.width / 2.0, self.height / 2.0], -1.0);
    }

    /// Move the view by screen pixels (keyboard).
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
    use crate::views::{OrbitViewProps, OrthographicViewProps};

    fn ortho(zoom: f64) -> OrbitController {
        OrbitController::new(
            View::Orthographic(OrthographicViewProps::default()),
            AnyViewState::Orthographic(OrthographicViewState {
                target: [10.0, 20.0, 0.0],
                zoom,
                ..Default::default()
            }),
            200.0,
            100.0,
        )
    }

    fn target(c: &OrbitController) -> [f64; 3] {
        c.target().to_array()
    }

    #[test]
    fn orthographic_pan_moves_the_target_with_the_pointer() {
        let mut c = ortho(1.0);
        c.pan_start([100.0, 50.0]);
        c.pan([110.0, 56.0]);
        c.pan_end();
        // Zoom 1: two pixels per unit, and +y is down on screen
        let t = target(&c);
        assert!((t[0] - 5.0).abs() < 1e-9 && (t[1] - 17.0).abs() < 1e-9, "{t:?}");
        assert!(c.interacted());
    }

    #[test]
    fn zoom_keeps_the_world_position_under_the_cursor() {
        let mut c = ortho(0.0);
        let before = c.unproject([150.0, 25.0]);
        c.zoom_by([150.0, 25.0], 2.0);
        assert!((c.current_zoom() - 2.0).abs() < 1e-9);
        let after = c.unproject([150.0, 25.0]);
        assert!((before - after).length() < 1e-9, "{before:?} vs {after:?}");
        c.constraints.max_zoom = 2.5;
        c.zoom_in();
        assert!((c.current_zoom() - 2.5).abs() < 1e-9, "clamped");
        // Pinch from the start zoom
        c.zoom_start([100.0, 50.0]);
        c.zoom([100.0, 50.0], 0.5);
        c.zoom_end();
        assert!((c.current_zoom() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn orbit_rotates_and_pans_in_three_dimensions() {
        let mut c = OrbitController::new(
            View::Orbit(OrbitViewProps::default()),
            AnyViewState::Orbit(OrbitViewState {
                target: [0.0, 0.0, 0.0],
                zoom: 2.0,
                rotation_x: 30.0,
                rotation_orbit: 0.0,
            }),
            400.0,
            200.0,
        );
        c.rotate_start([100.0, 100.0]);
        c.rotate([200.0, 150.0]);
        c.rotate_end();
        match c.view_state() {
            AnyViewState::Orbit(s) => {
                assert!(
                    (s.rotation_orbit - 45.0).abs() < 1e-9 && (s.rotation_x - 75.0).abs() < 1e-9,
                    "{s:?}"
                );
            }
            other => panic!("{other:?}"),
        }
        c.rotate_by(0.0, 40.0);
        match c.view_state() {
            AnyViewState::Orbit(s) => assert!((s.rotation_x - 90.0).abs() < 1e-9, "clamped {s:?}"),
            other => panic!("{other:?}"),
        }
        // Panning keeps the grabbed point under the pointer
        let grabbed = c.unproject([150.0, 80.0]);
        c.pan_start([150.0, 80.0]);
        c.pan([190.0, 110.0]);
        c.pan_end();
        let now = c.unproject([190.0, 110.0]);
        assert!((grabbed - now).length() < 1e-6, "{grabbed:?} vs {now:?}");
        // The orthographic controller ignores rotation
        let mut o = ortho(0.0);
        o.rotate_start([0.0, 0.0]);
        o.rotate([50.0, 50.0]);
        assert_eq!(target(&o), [10.0, 20.0, 0.0]);
        assert!(!o.interacted());
    }
}

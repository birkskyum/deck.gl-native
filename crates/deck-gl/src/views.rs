//! deck.gl's views other than the map: `OrthographicView`, `OrbitView` and `FirstPersonView`,
//! with their view states and the viewports they produce.

use glam::DVec3;

use crate::deck::ViewState;
use crate::viewport::{
    FirstPersonViewportOptions, OrbitViewportOptions, OrthographicViewportOptions, Viewport,
};

/// Axis an `OrbitView` rotates around freely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrbitAxis {
    Y,
    #[default]
    Z,
}

/// `OrthographicView` props: a 2D view of cartesian coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthographicViewProps {
    pub near: f64,
    pub far: f64,
    /// Top left screen coordinates (`true`, the default) or bottom left
    pub flip_y: bool,
}

impl Default for OrthographicViewProps {
    fn default() -> Self {
        Self {
            near: 0.1,
            far: 1000.0,
            flip_y: true,
        }
    }
}

/// `OrbitView` props: a 3D view orbiting a target in cartesian coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitViewProps {
    pub orbit_axis: OrbitAxis,
    /// Field of view in degrees
    pub fovy: f64,
    pub near: f64,
    pub far: f64,
    pub orthographic: bool,
}

impl Default for OrbitViewProps {
    fn default() -> Self {
        Self {
            orbit_axis: OrbitAxis::Z,
            fovy: 50.0,
            near: 0.1,
            far: 1000.0,
            orthographic: false,
        }
    }
}

/// `FirstPersonView` props: the camera sits at the view state's position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstPersonViewProps {
    pub fovy: f64,
    pub near: f64,
    pub far: f64,
    /// Pixels per meter
    pub focal_distance: f64,
}

impl Default for FirstPersonViewProps {
    fn default() -> Self {
        Self {
            fovy: 75.0,
            near: 0.1,
            far: 1000.0,
            focal_distance: 1.0,
        }
    }
}

/// Which kind of camera a deck uses, deck.gl's `views` prop (one view).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum View {
    /// Web Mercator map, driven by a [`ViewState`]
    #[default]
    Map,
    Orthographic(OrthographicViewProps),
    Orbit(OrbitViewProps),
    FirstPerson(FirstPersonViewProps),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthographicViewState {
    /// World position at the centre of the viewport
    pub target: [f64; 3],
    /// `zoom: 0` maps one world unit to one pixel; each level doubles the size
    pub zoom: f64,
    /// Independent zoom along X, overriding `zoom`
    pub zoom_x: Option<f64>,
    pub zoom_y: Option<f64>,
}

impl Default for OrthographicViewState {
    fn default() -> Self {
        Self {
            target: [0.0; 3],
            zoom: 0.0,
            zoom_x: None,
            zoom_y: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitViewState {
    pub target: [f64; 3],
    pub zoom: f64,
    /// Rotation around the orbit axis in degrees
    pub rotation_orbit: f64,
    /// Rotation around the X axis in degrees
    pub rotation_x: f64,
}

impl Default for OrbitViewState {
    fn default() -> Self {
        Self {
            target: [0.0; 3],
            zoom: 0.0,
            rotation_orbit: 0.0,
            rotation_x: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstPersonViewState {
    /// Anchor of the camera in the geospatial case
    pub longitude: Option<f64>,
    pub latitude: Option<f64>,
    /// Meter offsets of the camera from the anchor (or world position when not geospatial)
    pub position: [f64; 3],
    /// Heading in degrees, 0 is north
    pub bearing: f64,
    /// Tilt in degrees, 0 is horizontal, positive looks down
    pub pitch: f64,
}

impl Default for FirstPersonViewState {
    fn default() -> Self {
        Self {
            longitude: None,
            latitude: None,
            position: [0.0; 3],
            bearing: 0.0,
            pitch: 0.0,
        }
    }
}

/// The view state of any [`View`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnyViewState {
    Map(ViewState),
    Orthographic(OrthographicViewState),
    Orbit(OrbitViewState),
    FirstPerson(FirstPersonViewState),
}

impl Default for AnyViewState {
    fn default() -> Self {
        Self::Map(ViewState::default())
    }
}

impl From<ViewState> for AnyViewState {
    fn from(state: ViewState) -> Self {
        Self::Map(state)
    }
}

impl AnyViewState {
    /// The map view state, when this is one.
    pub fn map(&self) -> Option<ViewState> {
        match self {
            Self::Map(state) => Some(*state),
            _ => None,
        }
    }
}

impl View {
    /// The default view state of this view kind.
    pub fn default_view_state(&self) -> AnyViewState {
        match self {
            View::Map => AnyViewState::Map(ViewState::default()),
            View::Orthographic(_) => AnyViewState::Orthographic(OrthographicViewState::default()),
            View::Orbit(_) => AnyViewState::Orbit(OrbitViewState::default()),
            View::FirstPerson(_) => AnyViewState::FirstPerson(FirstPersonViewState::default()),
        }
    }

    /// Build the viewport for `state`; a state of another kind falls back to this view's
    /// default state.
    pub fn make_viewport(&self, state: &AnyViewState, width: f64, height: f64) -> Viewport {
        match (self, state) {
            (View::Map, AnyViewState::Map(vs)) => map_viewport(vs, width, height),
            (View::Orthographic(props), AnyViewState::Orthographic(vs)) => {
                Viewport::orthographic(&OrthographicViewportOptions {
                    width,
                    height,
                    target: DVec3::from(vs.target),
                    zoom: vs.zoom,
                    zoom_x: vs.zoom_x,
                    zoom_y: vs.zoom_y,
                    near: props.near,
                    far: props.far,
                    flip_y: props.flip_y,
                    ..Default::default()
                })
            }
            (View::Orbit(props), AnyViewState::Orbit(vs)) => Viewport::orbit(&OrbitViewportOptions {
                width,
                height,
                orbit_axis: props.orbit_axis,
                target: DVec3::from(vs.target),
                zoom: vs.zoom,
                rotation_orbit: vs.rotation_orbit,
                rotation_x: vs.rotation_x,
                fovy: props.fovy,
                near: props.near,
                far: props.far,
                orthographic: props.orthographic,
                ..Default::default()
            }),
            (View::FirstPerson(props), AnyViewState::FirstPerson(vs)) => {
                Viewport::first_person(&FirstPersonViewportOptions {
                    width,
                    height,
                    longitude: vs.longitude,
                    latitude: vs.latitude,
                    position: DVec3::from(vs.position),
                    bearing: vs.bearing,
                    pitch: vs.pitch,
                    fovy: props.fovy,
                    near: props.near,
                    far: props.far,
                    focal_distance: props.focal_distance,
                    ..Default::default()
                })
            }
            _ => self.make_viewport(&self.default_view_state(), width, height),
        }
    }
}

/// The Web Mercator viewport of a map view state.
pub fn map_viewport(view_state: &ViewState, width: f64, height: f64) -> Viewport {
    Viewport::web_mercator(&crate::viewport::WebMercatorViewportOptions {
        width,
        height,
        longitude: view_state.longitude,
        latitude: view_state.latitude,
        zoom: view_state.zoom,
        pitch: view_state.pitch,
        bearing: view_state.bearing,
        ..Default::default()
    })
}

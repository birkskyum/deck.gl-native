//! deck.gl's views other than the map: `OrthographicView`, `OrbitView` and `FirstPersonView`,
//! with their view states and the viewports they produce.

use glam::DVec3;

use std::sync::Arc;

use crate::deck::{PickingInfo, ViewState};
use crate::viewport::{
    FirstPersonViewportOptions, GlobeViewportOptions, OrbitViewportOptions, OrthographicViewportOptions,
    Padding, Viewport,
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

/// `GlobeView` props: the earth as a sphere, driven by a map [`ViewState`]. Above zoom 12
/// the view switches to the Web Mercator map, as deck.gl does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeViewProps {
    /// Degrees per mesh segment when flat geometry is turned into 3D
    pub resolution: f64,
    pub near_z_multiplier: f64,
    pub far_z_multiplier: f64,
    /// Camera altitude relative to the viewport height
    pub altitude: f64,
}

impl Default for GlobeViewProps {
    fn default() -> Self {
        Self {
            resolution: 10.0,
            near_z_multiplier: 0.5,
            far_z_multiplier: 1.0,
            altitude: 1.5,
        }
    }
}

/// Which kind of camera a deck uses, deck.gl's `views` prop (one view).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum View {
    /// Web Mercator map, driven by a [`ViewState`]
    #[default]
    Map,
    /// The earth as a sphere, driven by a [`ViewState`]
    Globe(GlobeViewProps),
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
    Globe(ViewState),
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
    /// The map view state, when this is one (globe states use the same type).
    pub fn map(&self) -> Option<ViewState> {
        match self {
            Self::Map(state) | Self::Globe(state) => Some(*state),
            _ => None,
        }
    }
}

impl View {
    /// The default view state of this view kind.
    pub fn default_view_state(&self) -> AnyViewState {
        match self {
            View::Map => AnyViewState::Map(ViewState::default()),
            View::Globe(_) => AnyViewState::Globe(ViewState::default()),
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
            (View::Globe(props), AnyViewState::Globe(vs)) => {
                if vs.zoom > 12.0 {
                    return map_viewport(vs, width, height);
                }
                Viewport::globe(&GlobeViewportOptions {
                    width,
                    height,
                    longitude: vs.longitude,
                    latitude: vs.latitude,
                    zoom: vs.zoom,
                    bearing: vs.bearing,
                    pitch: vs.pitch,
                    altitude: props.altitude,
                    near_z_multiplier: props.near_z_multiplier,
                    far_z_multiplier: props.far_z_multiplier,
                    resolution: props.resolution,
                    ..Default::default()
                })
            }
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

/// A position or size of a view: absolute pixels or a percentage of the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Extent {
    Pixels(f64),
    Percent(f64),
}

impl Extent {
    /// `"50%"` or a number of pixels.
    pub fn parse(text: &str) -> Option<Extent> {
        let text = text.trim();
        if let Some(percent) = text.strip_suffix('%') {
            percent.trim().parse().ok().map(Extent::Percent)
        } else {
            text.parse().ok().map(Extent::Pixels)
        }
    }

    pub fn resolve(&self, total: f64) -> f64 {
        match self {
            Extent::Pixels(px) => *px,
            Extent::Percent(pct) => total * pct / 100.0,
        }
    }
}

/// Padding of a view, each side as pixels or a percentage of the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewPadding {
    pub left: Extent,
    pub right: Extent,
    pub top: Extent,
    pub bottom: Extent,
}

impl ViewPadding {
    pub fn pixels(left: f64, right: f64, top: f64, bottom: f64) -> Self {
        Self {
            left: Extent::Pixels(left),
            right: Extent::Pixels(right),
            top: Extent::Pixels(top),
            bottom: Extent::Pixels(bottom),
        }
    }
}

/// The rectangle of a view on the canvas, in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub padding: Option<Padding>,
}

impl ViewRect {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

/// One of a deck's views: a camera kind placed in a rectangle of the canvas, deck.gl's `View`
/// props `id`, `x`, `y`, `width`, `height` and `padding`.
#[derive(Clone, Debug, PartialEq)]
pub struct DeckView {
    pub id: String,
    pub view: View,
    pub x: Extent,
    pub y: Extent,
    pub width: Extent,
    pub height: Extent,
    pub padding: Option<ViewPadding>,
}

impl DeckView {
    /// A view filling the canvas.
    pub fn new(id: impl Into<String>, view: View) -> Self {
        Self {
            id: id.into(),
            view,
            x: Extent::Pixels(0.0),
            y: Extent::Pixels(0.0),
            width: Extent::Percent(100.0),
            height: Extent::Percent(100.0),
            padding: None,
        }
    }

    pub fn with_rect(mut self, x: Extent, y: Extent, width: Extent, height: Extent) -> Self {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_padding(mut self, padding: ViewPadding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// Resolve the rectangle on a canvas of `width` x `height` logical pixels.
    pub fn rect(&self, width: f64, height: f64) -> ViewRect {
        ViewRect {
            x: self.x.resolve(width),
            y: self.y.resolve(height),
            width: self.width.resolve(width),
            height: self.height.resolve(height),
            padding: self.padding.map(|p| Padding {
                left: p.left.resolve(width),
                right: p.right.resolve(width),
                top: p.top.resolve(height),
                bottom: p.bottom.resolve(height),
            }),
        }
    }

    /// The viewport of this view for `state` on a canvas of `width` x `height`; `None` when
    /// the rectangle has no area.
    pub fn make_viewport(&self, state: &AnyViewState, width: f64, height: f64) -> Option<Viewport> {
        let rect = self.rect(width, height);
        if rect.width < 1.0 || rect.height < 1.0 {
            return None;
        }
        let mut viewport = self.view.make_viewport(state, rect.width, rect.height);
        viewport.id = self.id.clone();
        viewport.x = rect.x;
        viewport.y = rect.y;
        if let Some(padding) = rect.padding {
            viewport = viewport.with_padding(padding);
        }
        Some(viewport)
    }
}

/// The function behind a [`LayerFilter`]: layer id and view id in, whether to draw.
pub type LayerFilterFn = dyn Fn(&str, &str) -> bool + Send + Sync;

/// Decides which layers a view draws, deck.gl's `layerFilter`: called with the layer's id
/// and the view's id.
#[derive(Clone)]
pub struct LayerFilter(pub Arc<LayerFilterFn>);

impl LayerFilter {
    pub fn new(f: impl Fn(&str, &str) -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn allows(&self, layer_id: &str, view_id: &str) -> bool {
        (self.0)(layer_id, view_id)
    }
}

impl std::fmt::Debug for LayerFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LayerFilter")
    }
}

/// Silence an unused import when picking info is only referenced in docs.
#[allow(dead_code)]
fn _picking_info_link(_: &PickingInfo) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extents_resolve_against_the_canvas() {
        assert_eq!(Extent::parse("50%"), Some(Extent::Percent(50.0)));
        assert_eq!(Extent::parse(" 120 "), Some(Extent::Pixels(120.0)));
        assert_eq!(Extent::parse("wide"), None);
        let view = DeckView::new("mini", View::Map)
            .with_rect(
                Extent::Percent(70.0),
                Extent::Pixels(10.0),
                Extent::Percent(30.0),
                Extent::Pixels(150.0),
            )
            .with_padding(ViewPadding::pixels(5.0, 0.0, 0.0, 10.0));
        let rect = view.rect(1000.0, 500.0);
        assert_eq!(
            (rect.x, rect.y, rect.width, rect.height),
            (700.0, 10.0, 300.0, 150.0)
        );
        assert_eq!(rect.padding.unwrap().left, 5.0);
        assert!(rect.contains(701.0, 20.0) && !rect.contains(699.0, 20.0));
        let viewport = view
            .make_viewport(&AnyViewState::Map(ViewState::default()), 1000.0, 500.0)
            .unwrap();
        assert_eq!(
            (viewport.x, viewport.y, viewport.width, viewport.height),
            (700.0, 10.0, 300.0, 150.0)
        );
        assert!(DeckView::new("none", View::Map)
            .with_rect(
                Extent::Pixels(0.0),
                Extent::Pixels(0.0),
                Extent::Pixels(0.0),
                Extent::Percent(100.0)
            )
            .make_viewport(&AnyViewState::Map(ViewState::default()), 100.0, 100.0)
            .is_none());
    }
}

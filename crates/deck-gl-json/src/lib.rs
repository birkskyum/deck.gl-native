//! JSON layer descriptions for deck.gl-native.
//!
//! The format is the one used by `@deck.gl/json` and pydeck: a description object with
//! `initialViewState` and `layers`, where each layer carries `@@type` and camelCase deck.gl props.
//! Accessor strings prefixed with `@@=` are expressions evaluated per row of `data` (see
//! [`expression`]), and `data`, `image` and `iconAtlas` may be inline, local files or URLs.
//!
//! ```
//! let spec = r#"{
//!   "initialViewState": {"longitude": -122.4, "latitude": 37.8, "zoom": 12},
//!   "layers": [{
//!     "@@type": "ScatterplotLayer",
//!     "data": [{"position": [-122.4, 37.8], "size": 100}],
//!     "getPosition": "@@=position",
//!     "getRadius": "@@=size * 2",
//!     "getFillColor": [255, 140, 0]
//!   }]
//! }"#;
//! let deck = deck_gl_json::JsonConverter::new().parse(spec).unwrap();
//! assert_eq!(deck.layers.len(), 1);
//! assert_eq!(deck.view_state.unwrap().zoom, 12.0);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use deck_gl::{
    AnyViewState, DeckError, DeckView, Extent, FirstPersonViewProps, FirstPersonViewState, GlobeViewProps,
    Layer, LightingEffect, OrbitAxis, OrbitViewProps, OrbitViewState, OrthographicViewProps,
    OrthographicViewState, View, ViewPadding, ViewState,
};
use serde_json::Value;

pub mod data;
pub mod expression;
mod layers;
pub mod props;

pub use expression::{Expr, ExpressionError};
pub use props::{Props, Rows};

#[derive(Debug, thiserror::Error)]
pub enum JsonError {
    #[error("invalid JSON description: {0}")]
    Parse(String),
    #[error("layer `{layer}` prop `{prop}`: {message}")]
    Prop {
        layer: String,
        prop: String,
        message: String,
    },
    #[error("layer `{layer}`: {message}")]
    Layer { layer: String, message: String },
    #[error("could not load `{url}`: {message}")]
    Load { url: String, message: String },
    #[error(transparent)]
    Deck(#[from] DeckError),
}

pub type Result<T> = std::result::Result<T, JsonError>;

/// How external references in a description are resolved.
#[derive(Clone, Debug, Default)]
pub struct ConvertOptions {
    /// Directory that relative `data`, `image` and `iconAtlas` paths resolve against.
    pub base_dir: Option<PathBuf>,
    /// Arrow tables a layer can use as `"data": "@@table:name"`, with `@@column:name`
    /// accessors (or `@@=name` for a plain column name). Columns are read directly, so large
    /// data never goes through JSON.
    pub tables: HashMap<String, RecordBatch>,
}

/// A converted description.
pub struct JsonDeck {
    /// `initialViewState` (or `viewState`), when present.
    pub view_state: Option<ViewState>,
    pub layers: Vec<Box<dyn Layer>>,
    /// The `LightingEffect` among `effects`, when present.
    pub lighting: Option<LightingEffect>,
    /// `repeat` of the `MapView` among `views`: draw world copies across the antimeridian.
    pub repeat: bool,
    /// The view among `views` (a map unless an `OrthographicView`, `OrbitView` or
    /// `FirstPersonView` is given).
    pub view: View,
    /// `initialViewState` read for `view`, of any kind; `view_state` is its map form.
    pub camera: Option<AnyViewState>,
    /// All views of `views` with their rectangles, for decks with several views (empty when
    /// the description has at most one full size view).
    pub views: Vec<DeckView>,
    /// `initialViewState` per view id when it is keyed by view ids, else the shared state.
    pub cameras: HashMap<String, AnyViewState>,
    /// Layer types and props that were ignored, mirroring deck.gl's console warnings.
    pub warnings: Vec<String>,
}

impl std::fmt::Debug for JsonDeck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonDeck")
            .field("view_state", &self.view_state)
            .field(
                "layers",
                &self.layers.iter().map(|layer| layer.id()).collect::<Vec<_>>(),
            )
            .field("warnings", &self.warnings)
            .finish()
    }
}

/// Converts JSON descriptions into layers.
#[derive(Clone, Debug, Default)]
pub struct JsonConverter {
    pub options: ConvertOptions,
}

impl JsonConverter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve relative paths against `dir`.
    pub fn with_base_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            options: ConvertOptions {
                base_dir: Some(dir.into()),
                ..Default::default()
            },
        }
    }

    /// Make an Arrow table available as `"data": "@@table:<name>"`.
    pub fn with_table(mut self, name: impl Into<String>, batch: RecordBatch) -> Self {
        self.options.tables.insert(name.into(), batch);
        self
    }

    /// Parse and convert a description object or a layer array.
    pub fn parse(&self, text: &str) -> Result<JsonDeck> {
        let value: Value = serde_json::from_str(text).map_err(|e| JsonError::Parse(e.to_string()))?;
        self.convert(&value)
    }

    /// Load a description from a file. Relative paths inside it resolve against its directory.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<JsonDeck> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| JsonError::Load {
            url: path.display().to_string(),
            message: e.to_string(),
        })?;
        let converter = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => Self::with_base_dir(dir),
            _ => Self::new(),
        };
        converter.parse(&text)
    }

    /// Convert a parsed description object or layer array.
    pub fn convert(&self, value: &Value) -> Result<JsonDeck> {
        let mut warnings = Vec::new();
        let mut lighting = None;
        let mut repeat = false;
        let mut view = View::Map;
        let mut camera = None;
        let mut deck_views: Vec<DeckView> = Vec::new();
        let mut cameras = HashMap::new();
        let (view_state, layers) = match value {
            Value::Array(_) => (None, self.convert_layers(value, &mut warnings)?),
            Value::Object(map) => {
                if let Some(Value::Array(views)) = map.get("views") {
                    for item in views {
                        if item.get(props::TYPE_KEY).and_then(Value::as_str) == Some("MapView") {
                            repeat |= item.get("repeat").and_then(Value::as_bool).unwrap_or(false);
                        }
                        match view_from_value(item) {
                            Ok(kind) => match deck_view_from_value(item, kind.unwrap_or(View::Map)) {
                                Ok(deck_view) => deck_views.push(deck_view),
                                Err(warning) => warnings.push(warning),
                            },
                            Err(warning) => warnings.push(warning),
                        }
                    }
                }
                let state_value = map
                    .get("initialViewState")
                    .or_else(|| map.get("viewState"))
                    .filter(|v| !v.is_null());
                if let Some(first) = deck_views.first() {
                    view = first.view;
                }
                if let Some(state) = state_value {
                    let keyed = !deck_views.is_empty()
                        && state.as_object().is_some_and(|map| {
                            !map.is_empty()
                                && map
                                    .iter()
                                    .all(|(k, v)| v.is_object() && deck_views.iter().any(|d| &d.id == k))
                        });
                    if keyed {
                        for deck_view in &deck_views {
                            if let Some(v) = state.get(&deck_view.id) {
                                cameras.insert(
                                    deck_view.id.clone(),
                                    any_view_state_from_value(v, &deck_view.view)?,
                                );
                            }
                        }
                        camera = deck_views.first().and_then(|d| cameras.get(&d.id).copied());
                    } else {
                        let shared = any_view_state_from_value(state, &view)?;
                        camera = Some(shared);
                        for deck_view in &deck_views {
                            if same_view_kind(&deck_view.view, &view) {
                                cameras.insert(deck_view.id.clone(), shared);
                            }
                        }
                    }
                }
                let view_state = camera.and_then(|c| c.map());
                let layers = match map.get("layers") {
                    Some(layers) => self.convert_layers(layers, &mut warnings)?,
                    None => Vec::new(),
                };
                if let Some(Value::Array(effects)) = map.get("effects") {
                    for effect in effects {
                        match effect.get(props::TYPE_KEY).and_then(Value::as_str) {
                            Some("LightingEffect") => lighting = Some(lighting_from_value(effect)?),
                            Some(other) => warnings
                                .push(format!("effect `{other}` is not available yet and was skipped")),
                            None => warnings.push("effect without @@type was skipped".to_string()),
                        }
                    }
                }
                (view_state, layers)
            }
            other => {
                return Err(JsonError::Parse(format!(
                    "expected a description object or an array of layers, got {}",
                    props::describe(other)
                )))
            }
        };
        Ok(JsonDeck {
            view_state,
            layers,
            lighting,
            repeat,
            view,
            camera,
            views: if deck_views.len() > 1 || deck_views.first().is_some_and(|d| !is_full_size(d)) {
                deck_views
            } else {
                Vec::new()
            },
            cameras,
            warnings,
        })
    }

    /// Convert a layer array (nested arrays and null entries are allowed, as in deck.gl).
    pub fn convert_layers(&self, value: &Value, warnings: &mut Vec<String>) -> Result<Vec<Box<dyn Layer>>> {
        let mut layers = Vec::new();
        self.collect_layers(value, &mut layers, warnings)?;
        Ok(layers)
    }

    fn collect_layers(
        &self,
        value: &Value,
        layers: &mut Vec<Box<dyn Layer>>,
        warnings: &mut Vec<String>,
    ) -> Result<()> {
        match value {
            Value::Null | Value::Bool(false) => Ok(()),
            Value::Array(items) => {
                for item in items {
                    self.collect_layers(item, layers, warnings)?;
                }
                Ok(())
            }
            Value::Object(_) => {
                if let Some(layer) = layers::convert_layer(self, value, warnings)? {
                    layers.push(layer);
                }
                Ok(())
            }
            other => Err(JsonError::Parse(format!(
                "expected a layer object, got {}",
                props::describe(other)
            ))),
        }
    }
}

/// A `LightingEffect` object: every other field is a light with its own `@@type`
/// (`AmbientLight`, `DirectionalLight` or `PointLight`), as in deck.gl JSON.
pub fn lighting_from_value(value: &Value) -> Result<LightingEffect> {
    let map = value
        .as_object()
        .ok_or_else(|| JsonError::Parse("LightingEffect must be an object".into()))?;
    let mut effect = LightingEffect::default();
    let mut has_ambient = false;
    let mut directional = Vec::new();
    let mut point = Vec::new();
    for (name, light) in map {
        if name == props::TYPE_KEY || name == "id" {
            continue;
        }
        let object = light
            .as_object()
            .ok_or_else(|| JsonError::Parse(format!("light `{name}` must be an object")))?;
        let number = |key: &str, default: f32| -> Result<f32> {
            match object.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(v) => {
                    props::convert::f32(v).map_err(|m| JsonError::Parse(format!("light `{name}` {key}: {m}")))
                }
            }
        };
        let vec3 = |key: &str, default: [f32; 3]| -> Result<[f32; 3]> {
            match object.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(v) => props::convert::vec3(v)
                    .map_err(|m| JsonError::Parse(format!("light `{name}` {key}: {m}"))),
            }
        };
        match object.get(props::TYPE_KEY).and_then(Value::as_str) {
            Some("AmbientLight") => {
                effect.ambient = deck_gl::AmbientLight {
                    color: vec3("color", [255.0, 255.0, 255.0])?,
                    intensity: number("intensity", 1.0)?,
                };
                has_ambient = true;
            }
            Some("DirectionalLight") => directional.push(deck_gl::DirectionalLight {
                color: vec3("color", [255.0, 255.0, 255.0])?,
                intensity: number("intensity", 1.0)?,
                direction: vec3("direction", [0.0, 0.0, -1.0])?,
            }),
            Some("PointLight") => point.push(deck_gl::PointLight {
                color: vec3("color", [255.0, 255.0, 255.0])?,
                intensity: number("intensity", 1.0)?,
                position: vec3("position", [0.0, 0.0, 0.0])?,
                attenuation: vec3("attenuation", [1.0, 0.0, 0.0])?,
            }),
            Some(other) => {
                return Err(JsonError::Parse(format!(
                    "light `{name}`: unknown type `{other}`"
                )))
            }
            None => return Err(JsonError::Parse(format!("light `{name}` is missing `@@type`"))),
        }
    }
    if !has_ambient {
        effect.ambient.intensity = 0.0;
    }
    effect.directional = directional;
    effect.point = point;
    Ok(effect)
}

/// One entry of `views`: `MapView`, `OrthographicView`, `OrbitView` or `FirstPersonView` with
/// deck.gl's props. `Ok(None)` for a `MapView` (the default); `Err` carries a warning for
/// unknown view types.
pub fn view_from_value(value: &Value) -> std::result::Result<Option<View>, String> {
    let map = value
        .as_object()
        .ok_or_else(|| "view must be an object".to_string())?;
    let number = |key: &str, default: f64| -> std::result::Result<f64, String> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => props::convert::number(v).map_err(|m| format!("view `{key}`: {m}")),
        }
    };
    let boolean = |key: &str, default: bool| map.get(key).and_then(Value::as_bool).unwrap_or(default);
    match map.get(props::TYPE_KEY).and_then(Value::as_str) {
        Some("MapView") => Ok(None),
        Some("GlobeView") => {
            let d = GlobeViewProps::default();
            Ok(Some(View::Globe(GlobeViewProps {
                resolution: number("resolution", d.resolution)?,
                near_z_multiplier: number("nearZMultiplier", d.near_z_multiplier)?,
                far_z_multiplier: number("farZMultiplier", d.far_z_multiplier)?,
                altitude: number("altitude", d.altitude)?,
            })))
        }
        Some("OrthographicView") => {
            let d = OrthographicViewProps::default();
            Ok(Some(View::Orthographic(OrthographicViewProps {
                near: number("near", d.near)?,
                far: number("far", d.far)?,
                flip_y: boolean("flipY", d.flip_y),
            })))
        }
        Some("OrbitView") => {
            let d = OrbitViewProps::default();
            let orbit_axis = match map.get("orbitAxis").and_then(Value::as_str) {
                None | Some("Z") => OrbitAxis::Z,
                Some("Y") => OrbitAxis::Y,
                Some(other) => return Err(format!("view `orbitAxis`: expected Y or Z, got `{other}`")),
            };
            Ok(Some(View::Orbit(OrbitViewProps {
                orbit_axis,
                fovy: number("fovy", d.fovy)?,
                near: number("near", d.near)?,
                far: number("far", d.far)?,
                orthographic: boolean("orthographic", d.orthographic),
            })))
        }
        Some("FirstPersonView") => {
            let d = FirstPersonViewProps::default();
            Ok(Some(View::FirstPerson(FirstPersonViewProps {
                fovy: number("fovy", d.fovy)?,
                near: number("near", d.near)?,
                far: number("far", d.far)?,
                focal_distance: number("focalDistance", d.focal_distance)?,
            })))
        }
        Some(other) => Err(format!("view `{other}` is not available yet and was skipped")),
        None => Err("view without @@type was skipped".to_string()),
    }
}

/// The placement props of a view: `id`, `x`, `y`, `width`, `height` (pixels or `"NN%"`) and
/// `padding`.
pub fn deck_view_from_value(value: &Value, kind: View) -> std::result::Result<DeckView, String> {
    let map = value
        .as_object()
        .ok_or_else(|| "view must be an object".to_string())?;
    let extent = |key: &str, default: Extent| -> std::result::Result<Extent, String> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(Value::String(text)) => Extent::parse(text)
                .ok_or_else(|| format!("view `{key}`: expected pixels or a percentage, got `{text}`")),
            Some(v) => props::convert::number(v)
                .map(Extent::Pixels)
                .map_err(|m| format!("view `{key}`: {m}")),
        }
    };
    let id = map
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            map.get(props::TYPE_KEY)
                .and_then(Value::as_str)
                .unwrap_or("view")
                .to_string()
        });
    let mut view = DeckView::new(id, kind).with_rect(
        extent("x", Extent::Pixels(0.0))?,
        extent("y", Extent::Pixels(0.0))?,
        extent("width", Extent::Percent(100.0))?,
        extent("height", Extent::Percent(100.0))?,
    );
    if let Some(Value::Object(padding)) = map.get("padding") {
        let side = |key: &str| -> std::result::Result<Extent, String> {
            match padding.get(key) {
                None | Some(Value::Null) => Ok(Extent::Pixels(0.0)),
                Some(Value::String(text)) => {
                    Extent::parse(text).ok_or_else(|| format!("view padding `{key}`: got `{text}`"))
                }
                Some(v) => props::convert::number(v)
                    .map(Extent::Pixels)
                    .map_err(|m| format!("view padding `{key}`: {m}")),
            }
        };
        view = view.with_padding(ViewPadding {
            left: side("left")?,
            right: side("right")?,
            top: side("top")?,
            bottom: side("bottom")?,
        });
    }
    Ok(view)
}

fn is_full_size(view: &DeckView) -> bool {
    view.x == Extent::Pixels(0.0)
        && view.y == Extent::Pixels(0.0)
        && view.width == Extent::Percent(100.0)
        && view.height == Extent::Percent(100.0)
        && view.padding.is_none()
}

fn same_view_kind(a: &View, b: &View) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

/// Read the `initialViewState` of a view of any kind.
pub fn any_view_state_from_value(value: &Value, view: &View) -> Result<AnyViewState> {
    let map = value
        .as_object()
        .ok_or_else(|| JsonError::Parse("view state must be an object".into()))?;
    let number = |key: &str, default: f64| -> Result<f64> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => {
                props::convert::number(v).map_err(|m| JsonError::Parse(format!("view state `{key}`: {m}")))
            }
        }
    };
    let optional = |key: &str| -> Result<Option<f64>> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => props::convert::number(v)
                .map(Some)
                .map_err(|m| JsonError::Parse(format!("view state `{key}`: {m}"))),
        }
    };
    let target = |default: [f64; 3]| -> Result<[f64; 3]> {
        match map.get("target") {
            None | Some(Value::Null) => Ok(default),
            Some(v) => {
                let n = props::convert::numbers(v, 2, 3)
                    .map_err(|m| JsonError::Parse(format!("view state `target`: {m}")))?;
                Ok([n[0], n[1], n.get(2).copied().unwrap_or(0.0)])
            }
        }
    };
    Ok(match view {
        View::Map => AnyViewState::Map(view_state_from_value(value)?),
        View::Globe(_) => AnyViewState::Globe(view_state_from_value(value)?),
        View::Orthographic(_) => {
            let d = OrthographicViewState::default();
            AnyViewState::Orthographic(OrthographicViewState {
                target: target(d.target)?,
                zoom: number("zoom", d.zoom)?,
                zoom_x: optional("zoomX")?,
                zoom_y: optional("zoomY")?,
            })
        }
        View::Orbit(_) => {
            let d = OrbitViewState::default();
            AnyViewState::Orbit(OrbitViewState {
                target: target(d.target)?,
                zoom: number("zoom", d.zoom)?,
                rotation_orbit: number("rotationOrbit", d.rotation_orbit)?,
                rotation_x: number("rotationX", d.rotation_x)?,
            })
        }
        View::FirstPerson(_) => {
            let d = FirstPersonViewState::default();
            let position = match map.get("position") {
                None | Some(Value::Null) => d.position,
                Some(v) => {
                    let n = props::convert::numbers(v, 3, 3)
                        .map_err(|m| JsonError::Parse(format!("view state `position`: {m}")))?;
                    [n[0], n[1], n[2]]
                }
            };
            AnyViewState::FirstPerson(FirstPersonViewState {
                longitude: optional("longitude")?,
                latitude: optional("latitude")?,
                position,
                bearing: number("bearing", d.bearing)?,
                pitch: number("pitch", d.pitch)?,
            })
        }
    })
}

/// Read `longitude`, `latitude`, `zoom`, `pitch` and `bearing`; missing fields are zero.
pub fn view_state_from_value(value: &Value) -> Result<ViewState> {
    let map = value
        .as_object()
        .ok_or_else(|| JsonError::Parse("view state must be an object".into()))?;
    let number = |key: &str| -> Result<f64> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(0.0),
            Some(v) => {
                props::convert::number(v).map_err(|m| JsonError::Parse(format!("view state `{key}`: {m}")))
            }
        }
    };
    Ok(ViewState {
        longitude: number("longitude")?,
        latitude: number("latitude")?,
        zoom: number("zoom")?,
        pitch: number("pitch")?,
        bearing: number("bearing")?,
    })
}

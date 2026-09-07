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
use deck_gl::{DeckError, Layer, LightingEffect, ViewState};
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
        let (view_state, layers) = match value {
            Value::Array(_) => (None, self.convert_layers(value, &mut warnings)?),
            Value::Object(map) => {
                let view_state = map
                    .get("initialViewState")
                    .or_else(|| map.get("viewState"))
                    .filter(|v| !v.is_null())
                    .map(view_state_from_value)
                    .transpose()?;
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

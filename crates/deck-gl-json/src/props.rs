//! Typed readers for the camelCase props of a JSON layer object, including `@@=` accessors.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use deck_gl::glam::DMat4;
use deck_gl::wgpu;
use deck_gl::{
    Accessor, Color, CoordinateSystem, CullMode, EasingKind, Extensions, LayerExtension, LayerProps,
    Material, Operation, PropTransition, PropTransitions, RenderParameters, Unit,
};
use deck_gl_layers::{
    BrushingExtension, BrushingTarget, ClipExtension, CollisionFilterExtension, DataFilterExtension,
    FillPattern, FillPatternAtlas, FillStyleExtension, FilterCategories, FilterValues, MaskExtension,
    PathStyleExtension, PathStyleTarget,
};
use serde_json::{Map, Value};

use crate::expression::Expr;
use crate::{data, ConvertOptions};
use crate::{JsonError, Result};

/// The rows of a layer's `data`, one JSON value per object.
pub type Rows = Arc<Vec<Value>>;

/// Prefix of accessor expression strings.
pub const FUNCTION_IDENTIFIER: &str = "@@=";
/// Prefix of a column accessor on Arrow table data.
pub const COLUMN_IDENTIFIER: &str = "@@column:";
/// Prefix of a `data` string naming an Arrow table.
pub const TABLE_IDENTIFIER: &str = "@@table:";
/// Prefix of constant and enumeration references.
pub const CONSTANT_IDENTIFIER: &str = "@@#";
/// Key holding the layer class name.
pub const TYPE_KEY: &str = "@@type";

/// Props that have no meaning in a static description and are ignored without a warning.
const SILENTLY_IGNORED: &[&str] = &[
    "onHover",
    "onClick",
    "onDragStart",
    "onDrag",
    "onDragEnd",
    "onDataLoad",
    "onError",
    "updateTriggers",
    "dataComparator",
    "_dataDiff",
    "fetch",
    "loaders",
    "loadOptions",
    "autoHighlight",
];

/// The default for an accessor prop that is not in the JSON.
pub enum AccessorDefault<T> {
    /// An expression evaluated against each row, such as `position`.
    Expr(String),
    /// A constant.
    Value(T),
}

impl<T: Clone> From<&Accessor<T>> for AccessorDefault<T> {
    fn from(accessor: &Accessor<T>) -> Self {
        match accessor {
            Accessor::Constant(value) => AccessorDefault::Value(value.clone()),
            Accessor::Column(name) => AccessorDefault::Expr(name.clone()),
            // Function defaults have no JSON form: use their value for the first row
            Accessor::Func(f) => AccessorDefault::Value(f(0)),
        }
    }
}

impl<T> From<&str> for AccessorDefault<T> {
    fn from(expression: &str) -> Self {
        AccessorDefault::Expr(expression.to_string())
    }
}

/// A layer object being converted. Tracks which keys were read so unknown props can be reported.
pub struct Props<'a> {
    pub layer_type: String,
    pub id: String,
    object: &'a Map<String, Value>,
    used: RefCell<HashSet<String>>,
    /// Warnings raised while reading props, reported by [`Props::finish`]
    notes: RefCell<Vec<String>>,
    rows: Option<Rows>,
    /// True when `data` is an Arrow table, so accessors are columns rather than expressions.
    table: bool,
    /// How external files are resolved, for props that load images
    options: Option<&'a ConvertOptions>,
}

impl<'a> Props<'a> {
    pub fn new(layer_type: &str, object: &'a Map<String, Value>) -> Self {
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or(layer_type)
            .to_string();
        let props = Self {
            layer_type: layer_type.to_string(),
            id,
            object,
            used: RefCell::new(HashSet::new()),
            notes: RefCell::new(Vec::new()),
            rows: None,
            table: false,
            options: None,
        };
        props.mark("id");
        props.mark(TYPE_KEY);
        props
    }

    /// Attach the rows that accessor expressions evaluate against.
    pub fn set_rows(&mut self, rows: Rows) {
        self.rows = Some(rows);
    }

    /// Attach the options external files (a pattern atlas) are loaded with.
    pub fn set_options(&mut self, options: &'a ConvertOptions) {
        self.options = Some(options);
    }

    pub fn rows(&self) -> Option<&Rows> {
        self.rows.as_ref()
    }

    /// Mark the data as an Arrow table: accessors name columns.
    pub fn set_table(&mut self) {
        self.table = true;
        self.rows = None;
    }

    pub fn is_table(&self) -> bool {
        self.table
    }

    fn mark(&self, key: &str) {
        self.used.borrow_mut().insert(key.to_string());
    }

    /// Raw access to a prop. Marks it as consumed.
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        self.mark(key);
        self.object.get(key)
    }

    pub fn error(&self, key: &str, message: impl Into<String>) -> JsonError {
        JsonError::Prop {
            layer: self.id.clone(),
            prop: key.to_string(),
            message: message.into(),
        }
    }

    pub fn bool(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or(0.0) != 0.0),
            Some(other) => Err(self.error(key, format!("expected a boolean, got {}", describe(other)))),
        }
    }

    pub fn f64(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(value) => convert::number(value).map_err(|m| self.error(key, m)),
        }
    }

    pub fn f32(&self, key: &str, default: f32) -> Result<f32> {
        self.f64(key, default as f64).map(|v| v as f32)
    }

    pub fn u32(&self, key: &str, default: u32) -> Result<u32> {
        let value = self.f64(key, default as f64)?;
        if value < 0.0 || value.fract() != 0.0 {
            return Err(self.error(key, format!("expected a non negative integer, got {value}")));
        }
        Ok(value as u32)
    }

    pub fn string(&self, key: &str) -> Result<Option<String>> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(self.error(key, format!("expected a string, got {}", describe(other)))),
        }
    }

    pub fn unit(&self, key: &str, default: Unit) -> Result<Unit> {
        match self
            .string(key)?
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None => Ok(default),
            Some("meters") => Ok(Unit::Meters),
            Some("pixels") => Ok(Unit::Pixels),
            Some("common") => Ok(Unit::Common),
            Some(other) => Err(self.error(key, format!("expected meters, pixels or common, got `{other}`"))),
        }
    }

    pub fn color(&self, key: &str, default: Color) -> Result<Color> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(value) => convert::color(value).map_err(|m| self.error(key, m)),
        }
    }

    pub fn vec2(&self, key: &str, default: [f32; 2]) -> Result<[f32; 2]> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(value) => convert::vec2(value).map_err(|m| self.error(key, m)),
        }
    }

    pub fn vec3_f64(&self, key: &str, default: [f64; 3]) -> Result<[f64; 3]> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(value) => convert::position(value).map_err(|m| self.error(key, m)),
        }
    }

    /// A list of numbers, such as `bounds` or `modelMatrix`.
    pub fn numbers(&self, key: &str) -> Result<Option<Vec<f64>>> {
        match self.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => convert::numbers(value, 0, usize::MAX)
                .map(Some)
                .map_err(|m| self.error(key, m)),
        }
    }

    /// An accessor prop: a constant JSON value or a `@@=` expression over the rows.
    pub fn accessor<T>(
        &self,
        key: &str,
        default: impl Into<AccessorDefault<T>>,
        convert: fn(&Value) -> std::result::Result<T, String>,
    ) -> Result<Accessor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        match self.get(key) {
            None | Some(Value::Null) => match default.into() {
                AccessorDefault::Value(value) => Ok(Accessor::Constant(value)),
                AccessorDefault::Expr(source) => self.accessor_from_expression(key, &source, convert),
            },
            Some(Value::String(s)) if s.starts_with(COLUMN_IDENTIFIER) => {
                if !self.table {
                    return Err(self.error(key, "column accessors need `data` to be an Arrow table"));
                }
                Ok(Accessor::column(&s[COLUMN_IDENTIFIER.len()..]))
            }
            Some(Value::String(s)) if s.starts_with(FUNCTION_IDENTIFIER) => {
                self.accessor_from_expression(key, &s[FUNCTION_IDENTIFIER.len()..], convert)
            }
            Some(value) => convert(value)
                .map(Accessor::Constant)
                .map_err(|m| self.error(key, m)),
        }
    }

    fn accessor_from_expression<T>(
        &self,
        key: &str,
        source: &str,
        convert: fn(&Value) -> std::result::Result<T, String>,
    ) -> Result<Accessor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let expr = Expr::parse(source).map_err(|e| self.error(key, e.0))?;
        if expr.is_constant() {
            return convert(&expr.eval(&Value::Null))
                .map(Accessor::Constant)
                .map_err(|m| self.error(key, format!("{m} (expression `{source}`)")));
        }
        if self.table {
            // On a table only a plain column name can be evaluated; use @@column: for clarity.
            return match expr {
                Expr::Identifier(name) => Ok(Accessor::column(name)),
                _ => Err(self.error(
                    key,
                    format!(
                        "expressions are not evaluated over Arrow tables; use a column name (`{source}`)"
                    ),
                )),
            };
        }
        let rows = self
            .rows
            .clone()
            .ok_or_else(|| self.error(key, "an accessor expression needs `data` rows"))?;
        let mut values = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let value = convert(&expr.eval(row))
                .map_err(|m| self.error(key, format!("row {index}: {m} (expression `{source}`)")))?;
            values.push(value);
        }
        let values = Arc::new(values);
        Ok(Accessor::func(move |i| values[i.min(values.len() - 1)].clone()))
    }

    /// The props shared by every layer.
    pub fn base(&self) -> Result<LayerProps> {
        let defaults = LayerProps::default();
        let mut base = LayerProps::new(&self.id);
        base.visible = self.bool("visible", defaults.visible)?;
        base.opacity = self.f64("opacity", defaults.opacity)?;
        base.pickable = self.bool("pickable", defaults.pickable)?;
        base.auto_highlight = self.bool("autoHighlight", defaults.auto_highlight)?;
        base.coordinate_system = self.coordinate_system()?;
        base.coordinate_origin = self.vec3_f64("coordinateOrigin", defaults.coordinate_origin)?;
        base.model_matrix = match self.numbers("modelMatrix")? {
            None => None,
            Some(values) if values.len() == 16 => {
                let mut cols = [0.0; 16];
                cols.copy_from_slice(&values);
                Some(DMat4::from_cols_array(&cols))
            }
            Some(values) => {
                return Err(self.error(
                    "modelMatrix",
                    format!("expected 16 numbers, got {}", values.len()),
                ))
            }
        };
        base.wrap_longitude = self.bool("wrapLongitude", defaults.wrap_longitude)?;
        base.shadow_enabled = self.bool("shadowEnabled", defaults.shadow_enabled)?;
        base.highlight_color = self.color("highlightColor", defaults.highlight_color)?;
        base.material = match self.get("material") {
            None | Some(Value::Null) | Some(Value::Bool(true)) => Material::default(),
            Some(Value::Bool(false)) => Material::unlit(),
            Some(Value::Object(map)) => {
                let d = Material::default();
                let number = |key: &str, default: f32| -> Result<f32> {
                    match map.get(key) {
                        None | Some(Value::Null) => Ok(default),
                        Some(v) => convert::f32(v).map_err(|m| self.error("material", format!("{key}: {m}"))),
                    }
                };
                Material {
                    unlit: false,
                    ambient: number("ambient", d.ambient)?,
                    diffuse: number("diffuse", d.diffuse)?,
                    shininess: number("shininess", d.shininess)?,
                    specular_color: match map.get("specularColor") {
                        None | Some(Value::Null) => d.specular_color,
                        Some(v) => {
                            let c = convert::numbers(v, 3, 3).map_err(|m| self.error("material", m))?;
                            [c[0] as f32, c[1] as f32, c[2] as f32]
                        }
                    },
                }
            }
            Some(other) => {
                return Err(self.error(
                    "material",
                    format!("expected true, false or an object, got {}", describe(other)),
                ))
            }
        };
        base.parameters = match self.get("parameters") {
            None | Some(Value::Null) => RenderParameters::default(),
            Some(Value::Object(map)) => render_parameters(map, |key| {
                self.warn(format!("parameters.{key} is not supported yet and was ignored"))
            })
            .map_err(|m| self.error("parameters", m))?,
            Some(other) => {
                return Err(self.error(
                    "parameters",
                    format!("expected an object, got {}", describe(other)),
                ))
            }
        };
        base.highlighted_object_index = match self.get("highlightedObjectIndex") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let index = convert::number(value).map_err(|m| self.error("highlightedObjectIndex", m))?;
                (index >= 0.0).then_some(index as u32)
            }
        };
        base.extensions = self.extensions()?;
        base.transitions = self.transitions()?;
        base.operation = match self.string("operation")?.as_deref() {
            None | Some("draw") => Operation::DRAW,
            Some("mask") => Operation::MASK,
            Some("mask+draw") | Some("draw+mask") => {
                self.warn(
                    "operation `mask+draw` only renders the mask for now; add a layer to draw the geometry",
                );
                Operation::MASK
            }
            Some(other) => {
                return Err(self.error("operation", format!("expected draw or mask, got `{other}`")))
            }
        };
        Ok(base)
    }

    /// deck.gl's `transitions` prop: `{"getRadius": 300, "opacity": {"duration": 500,
    /// "easing": "easeInOut"}, "getPosition": {"type": "spring", "stiffness": 0.05,
    /// "damping": 0.5}}`.
    fn transitions(&self) -> Result<PropTransitions> {
        let key = "transitions";
        let map = match self.get(key) {
            None | Some(Value::Null) => return Ok(PropTransitions::default()),
            Some(Value::Object(map)) => map,
            Some(other) => {
                return Err(self.error(key, format!("expected an object, got {}", describe(other))));
            }
        };
        let mut transitions = PropTransitions::default();
        for (prop, value) in map {
            let transition = match value {
                Value::Number(n) => PropTransition::interpolation(n.as_f64().unwrap_or(0.0)),
                Value::Object(settings) => {
                    let number = |name: &str, default: f64| -> Result<f64> {
                        match settings.get(name) {
                            None | Some(Value::Null) => Ok(default),
                            Some(v) => {
                                convert::number(v).map_err(|m| self.error(key, format!("{prop}.{name}: {m}")))
                            }
                        }
                    };
                    let kind = settings
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("interpolation");
                    match kind {
                        "spring" => PropTransition::Spring {
                            stiffness: number("stiffness", 0.05)?,
                            damping: number("damping", 0.5)?,
                        },
                        "interpolation" => {
                            let easing = match settings.get("easing").and_then(Value::as_str) {
                                None | Some("linear") => EasingKind::Linear,
                                Some("easeIn") | Some("ease-in") => EasingKind::EaseIn,
                                Some("easeOut") | Some("ease-out") => EasingKind::EaseOut,
                                Some("easeInOut") | Some("ease-in-out") => EasingKind::EaseInOut,
                                Some(other) => {
                                    self.warn(format!(
                                        "transitions.{prop}: easing `{other}` is unknown, using linear"
                                    ));
                                    EasingKind::Linear
                                }
                            };
                            PropTransition::Interpolation {
                                duration_ms: number("duration", 0.0)?,
                                easing,
                            }
                        }
                        other => {
                            return Err(self.error(
                                key,
                                format!("{prop}: transition type `{other}` is not interpolation or spring"),
                            ));
                        }
                    }
                }
                other => {
                    return Err(self.error(
                        key,
                        format!(
                            "{prop}: expected a duration or an object, got {}",
                            describe(other)
                        ),
                    ))
                }
            };
            transitions.0.insert(prop.clone(), transition);
        }
        Ok(transitions)
    }

    /// deck.gl decides clipping and masking by instance for layers with an `instancePositions`
    /// attribute: everything but the path, polygon, bitmap and tile layers.
    fn instanced_layer(&self) -> bool {
        !matches!(
            self.layer_type.as_str(),
            "PathLayer"
                | "TripsLayer"
                | "SolidPolygonLayer"
                | "PolygonLayer"
                | "GeoJsonLayer"
                | "BitmapLayer"
                | "H3HexagonLayer"
                | "S2Layer"
                | "GeohashLayer"
                | "QuadkeyLayer"
                | "MVTLayer"
                | "TileLayer"
                | "WMSLayer"
                | "ContourLayer"
        )
    }

    fn fill_style_extension(&self, options: &Map<String, Value>) -> Result<FillStyleExtension> {
        let defaults = FillStyleExtension::default();
        let pattern = options.get("pattern").and_then(Value::as_bool).unwrap_or(false);
        let atlas = match (self.string("fillPatternAtlas")?, self.get("fillPatternMapping")) {
            (Some(source), Some(mapping)) => {
                let load = self.options.ok_or_else(|| {
                    self.error("fillPatternAtlas", "no converter options to load the atlas with")
                })?;
                let image = data::load_image(&source, load)
                    .map_err(|e| self.error("fillPatternAtlas", e.to_string()))?;
                let mapping = data::load_json(mapping, load)
                    .map_err(|e| self.error("fillPatternMapping", e.to_string()))?;
                let Some(object) = mapping.as_object() else {
                    return Err(self.error("fillPatternMapping", "expected an object of patterns"));
                };
                let mut patterns = HashMap::new();
                for (name, entry) in object {
                    let number = |key: &str| -> Result<u32> {
                        entry
                            .get(key)
                            .and_then(Value::as_f64)
                            .map(|v| v as u32)
                            .ok_or_else(|| {
                                self.error(
                                    "fillPatternMapping",
                                    format!("pattern `{name}` is missing `{key}`"),
                                )
                            })
                    };
                    patterns.insert(
                        name.clone(),
                        FillPattern {
                            x: number("x")?,
                            y: number("y")?,
                            width: number("width")?,
                            height: number("height")?,
                        },
                    );
                }
                Some(Arc::new(FillPatternAtlas {
                    image,
                    mapping: patterns,
                }))
            }
            (None, None) => None,
            _ => {
                return Err(self.error(
                    "fillPatternAtlas",
                    "fillPatternAtlas and fillPatternMapping must be given together",
                ))
            }
        };
        Ok(FillStyleExtension {
            pattern,
            fill_pattern_enabled: self.bool("fillPatternEnabled", defaults.fill_pattern_enabled)?,
            fill_pattern_atlas: atlas,
            fill_pattern_mask: self.bool("fillPatternMask", defaults.fill_pattern_mask)?,
            get_fill_pattern: self.accessor("getFillPattern", &defaults.get_fill_pattern, convert::string)?,
            get_fill_pattern_scale: self.accessor(
                "getFillPatternScale",
                &defaults.get_fill_pattern_scale,
                convert::f32,
            )?,
            get_fill_pattern_offset: self.accessor(
                "getFillPatternOffset",
                &defaults.get_fill_pattern_offset,
                convert::vec2,
            )?,
            ..FillStyleExtension::default()
        })
    }

    fn path_style_extension(&self, options: &Map<String, Value>) -> Result<PathStyleExtension> {
        let defaults = PathStyleExtension::default();
        let flag = |key: &str| options.get(key).and_then(Value::as_bool).unwrap_or(false);
        if flag("highPrecisionDash") {
            self.warn("PathStyleExtension highPrecisionDash is not supported yet and was ignored");
        }
        Ok(PathStyleExtension {
            dash: flag("dash") || flag("highPrecisionDash"),
            offset: flag("offset"),
            target: if self.layer_type == "ScatterplotLayer" {
                PathStyleTarget::Scatterplot
            } else {
                PathStyleTarget::Path
            },
            get_dash_array: self.accessor("getDashArray", &defaults.get_dash_array, convert::vec2)?,
            get_offset: self.accessor("getOffset", &defaults.get_offset, convert::f32)?,
            dash_justified: self.bool("dashJustified", defaults.dash_justified)?,
            dash_gap_pickable: self.bool("dashGapPickable", defaults.dash_gap_pickable)?,
        })
    }

    fn collision_filter_extension(&self) -> Result<CollisionFilterExtension> {
        let defaults = CollisionFilterExtension::default();
        if self.get("collisionTestProps").is_some() {
            self.warn("collisionTestProps is not supported yet and was ignored");
        }
        Ok(CollisionFilterExtension {
            get_collision_priority: self.accessor(
                "getCollisionPriority",
                &defaults.get_collision_priority,
                convert::f32,
            )?,
            collision_enabled: self.bool("collisionEnabled", defaults.collision_enabled)?,
            collision_group: self.string("collisionGroup")?.unwrap_or(defaults.collision_group),
        })
    }

    fn mask_extension(&self) -> Result<MaskExtension> {
        let defaults = MaskExtension::default();
        Ok(MaskExtension {
            mask_id: self.string("maskId")?.unwrap_or_default(),
            mask_by_instance: self.bool("maskByInstance", self.instanced_layer())?,
            mask_inverted: self.bool("maskInverted", defaults.mask_inverted)?,
        })
    }

    /// deck.gl's `extensions` prop: `{"@@type": "DataFilterExtension", ...options}` objects,
    /// with the extension's props (`getFilterValue`, `filterRange`, ...) on the layer itself.
    fn extensions(&self) -> Result<Extensions> {
        let list = match self.get("extensions") {
            None | Some(Value::Null) => return Ok(Extensions::default()),
            Some(Value::Array(list)) => list,
            Some(other) => {
                return Err(self.error(
                    "extensions",
                    format!("expected an array, got {}", describe(other)),
                ))
            }
        };
        let mut extensions: Vec<Arc<dyn LayerExtension>> = Vec::new();
        for entry in list {
            let (kind, options) = match entry {
                Value::String(name) => (
                    name.trim_start_matches(CONSTANT_IDENTIFIER).to_string(),
                    Map::new(),
                ),
                Value::Object(map) => (
                    map.get(TYPE_KEY)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    map.clone(),
                ),
                other => {
                    return Err(self.error(
                        "extensions",
                        format!("expected an object with {TYPE_KEY}, got {}", describe(other)),
                    ))
                }
            };
            match kind.as_str() {
                "DataFilterExtension" => extensions.push(Arc::new(self.data_filter_extension(&options)?)),
                "BrushingExtension" => extensions.push(Arc::new(self.brushing_extension()?)),
                "ClipExtension" => extensions.push(Arc::new(self.clip_extension()?)),
                "MaskExtension" => extensions.push(Arc::new(self.mask_extension()?)),
                "CollisionFilterExtension" => extensions.push(Arc::new(self.collision_filter_extension()?)),
                "FillStyleExtension" => extensions.push(Arc::new(self.fill_style_extension(&options)?)),
                "PathStyleExtension" => extensions.push(Arc::new(self.path_style_extension(&options)?)),
                "" => return Err(self.error("extensions", format!("each extension needs a {TYPE_KEY}"))),
                other => self.warn(format!(
                    "extension `{other}` is not supported yet and was ignored"
                )),
            }
        }
        Ok(Extensions::new(extensions))
    }

    fn brushing_extension(&self) -> Result<BrushingExtension> {
        let defaults = BrushingExtension::default();
        let target = match self.string("brushingTarget")?.as_deref() {
            None | Some("source") => BrushingTarget::Source,
            Some("target") => BrushingTarget::Target,
            Some("source_target") => BrushingTarget::SourceTarget,
            Some("custom") => BrushingTarget::Custom,
            Some(other) => {
                return Err(self.error(
                    "brushingTarget",
                    format!("expected source, target, source_target or custom, got `{other}`"),
                ))
            }
        };
        Ok(BrushingExtension {
            get_brushing_target: self.accessor(
                "getBrushingTarget",
                &defaults.get_brushing_target,
                convert::vec2,
            )?,
            brushing_target: target,
            brushing_enabled: self.bool("brushingEnabled", defaults.brushing_enabled)?,
            brushing_radius: self.f32("brushingRadius", defaults.brushing_radius)?,
        })
    }

    fn clip_extension(&self) -> Result<ClipExtension> {
        let bounds = match self.get("clipBounds") {
            None | Some(Value::Null) => ClipExtension::default().clip_bounds,
            Some(value) => {
                let n = convert::numbers(value, 4, 4).map_err(|m| self.error("clipBounds", m))?;
                [n[0], n[1], n[2], n[3]]
            }
        };
        Ok(ClipExtension {
            clip_bounds: bounds,
            clip_by_instance: self.bool("clipByInstance", self.instanced_layer())?,
        })
    }

    fn data_filter_extension(&self, options: &Map<String, Value>) -> Result<DataFilterExtension> {
        let option = |key: &str, default: u32| -> Result<u32> {
            match options.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(value) => convert::number(value)
                    .map(|n| n as u32)
                    .map_err(|m| self.error("extensions", format!("DataFilterExtension {key}: {m}"))),
            }
        };
        for key in options.keys() {
            if !["filterSize", "categorySize", "fp64", "countItems", TYPE_KEY].contains(&key.as_str()) {
                self.warn(format!(
                    "DataFilterExtension option `{key}` is unknown and was ignored"
                ));
            }
        }
        for key in ["fp64", "countItems"] {
            if options.get(key).and_then(Value::as_bool) == Some(true) {
                self.warn(format!(
                    "DataFilterExtension `{key}` is not supported yet and was ignored"
                ));
            }
        }
        let filter_size = option("filterSize", 1)?;
        let category_size = option("categorySize", 0)?;

        let get_filter_value = match filter_size {
            0 | 1 => {
                FilterValues::One(self.accessor("getFilterValue", &Accessor::Constant(0.0), convert::f32)?)
            }
            2 => FilterValues::Two(self.accessor(
                "getFilterValue",
                &Accessor::Constant([0.0; 2]),
                convert::vec2,
            )?),
            3 => FilterValues::Three(self.accessor(
                "getFilterValue",
                &Accessor::Constant([0.0; 3]),
                convert::vec3,
            )?),
            4 => FilterValues::Four(self.accessor(
                "getFilterValue",
                &Accessor::Constant([0.0; 4]),
                convert::vec4,
            )?),
            other => {
                return Err(self.error(
                    "extensions",
                    format!("DataFilterExtension filterSize {other} is not 0 to 4"),
                ))
            }
        };
        let ranges = |key: &str| -> Result<Option<Vec<[f32; 2]>>> {
            let pair = |value: &Value| convert::numbers(value, 2, 2).map(|n| [n[0] as f32, n[1] as f32]);
            match self.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Array(items)) if items.first().is_some_and(Value::is_array) => items
                    .iter()
                    .map(pair)
                    .collect::<std::result::Result<Vec<_>, String>>()
                    .map(Some)
                    .map_err(|m| self.error(key, m)),
                Some(value) => pair(value).map(|p| Some(vec![p])).map_err(|m| self.error(key, m)),
            }
        };
        let filter_range = ranges("filterRange")?.unwrap_or_else(|| vec![[-1.0, 1.0]]);
        let filter_soft_range = ranges("filterSoftRange")?;
        let (get_filter_category, filter_categories) = match category_size {
            0 => (None, vec![vec![0]]),
            1..=4 => {
                let (categories, shown) = self.filter_categories(category_size as usize)?;
                (Some(categories), shown)
            }
            other => {
                return Err(self.error(
                    "extensions",
                    format!("DataFilterExtension categorySize {other} is not 0 to 4"),
                ))
            }
        };
        let defaults = DataFilterExtension::default();
        Ok(DataFilterExtension {
            get_filter_value,
            filter_range,
            filter_soft_range,
            filter_enabled: self.bool("filterEnabled", defaults.filter_enabled)?,
            filter_transform_size: self.bool("filterTransformSize", defaults.filter_transform_size)?,
            filter_transform_color: self.bool("filterTransformColor", defaults.filter_transform_color)?,
            get_filter_category,
            filter_categories,
        })
    }

    /// `getFilterCategory` and `filterCategories` with names or numbers as categories, mapped
    /// to keys in order of appearance per channel, as deck.gl does.
    fn filter_categories(&self, size: usize) -> Result<(FilterCategories, Vec<Vec<u32>>)> {
        fn name(value: &Value) -> std::result::Result<String, String> {
            match value {
                Value::String(s) => Ok(s.clone()),
                Value::Number(n) => Ok(n.to_string()),
                Value::Bool(b) => Ok(b.to_string()),
                other => Err(format!(
                    "expected a category name or number, got {}",
                    describe(other)
                )),
            }
        }
        fn names(value: &Value) -> std::result::Result<Vec<String>, String> {
            match value {
                Value::Array(items) => items.iter().map(name).collect(),
                other => Ok(vec![name(other)?]),
            }
        }
        let accessor = self.accessor(
            "getFilterCategory",
            &Accessor::Constant(vec!["0".to_string()]),
            names,
        )?;
        let mut maps: Vec<HashMap<String, u32>> = vec![HashMap::new(); size];
        let mut key_of = |channel: usize, name: &str| -> u32 {
            let map = &mut maps[channel];
            let next = map.len() as u32;
            *map.entry(name.to_string()).or_insert(next)
        };
        let keys_of = |key_of: &mut dyn FnMut(usize, &str) -> u32, names: &[String]| -> [u32; 4] {
            let mut keys = [0u32; 4];
            for (channel, key) in keys.iter_mut().enumerate().take(size) {
                *key = key_of(channel, names.get(channel).map(String::as_str).unwrap_or(""));
            }
            keys
        };
        let per_row: Vec<[u32; 4]> = match &accessor {
            Accessor::Column(_) => {
                return Err(self.error(
                    "getFilterCategory",
                    "category filters need `data` rows, not an Arrow table",
                ))
            }
            Accessor::Constant(constant) => vec![keys_of(&mut key_of, constant)],
            Accessor::Func(f) => (0..self.rows().map_or(0, |rows| rows.len()))
                .map(|i| keys_of(&mut key_of, &f(i)))
                .collect(),
        };
        let shown_key = "filterCategories";
        let mut shown: Vec<Vec<u32>> = vec![Vec::new(); size];
        match self.get(shown_key) {
            None | Some(Value::Null) => shown[0].push(key_of(0, "0")),
            Some(value) => {
                let lists: Vec<Vec<String>> = match value {
                    Value::Array(items) if size > 1 && items.iter().all(Value::is_array) => items
                        .iter()
                        .map(names)
                        .collect::<std::result::Result<_, String>>()
                        .map_err(|m| self.error(shown_key, m))?,
                    other => vec![names(other).map_err(|m| self.error(shown_key, m))?],
                };
                for (channel, list) in lists.iter().enumerate().take(size) {
                    shown[channel] = list.iter().map(|n| key_of(channel, n)).collect();
                }
            }
        }
        let constant = matches!(accessor, Accessor::Constant(_)) || per_row.is_empty();
        let first = per_row.first().copied().unwrap_or_default();
        let per_row = Arc::new(per_row);
        let last = per_row.len().saturating_sub(1);
        let categories = match size {
            1 if constant => FilterCategories::One(Accessor::Constant(first[0])),
            1 => FilterCategories::One(Accessor::func(move |i| per_row[i.min(last)][0])),
            2 if constant => FilterCategories::Two(Accessor::Constant([first[0], first[1]])),
            2 => FilterCategories::Two(Accessor::func(move |i| {
                let k = per_row[i.min(last)];
                [k[0], k[1]]
            })),
            3 if constant => FilterCategories::Three(Accessor::Constant([first[0], first[1], first[2]])),
            3 => FilterCategories::Three(Accessor::func(move |i| {
                let k = per_row[i.min(last)];
                [k[0], k[1], k[2]]
            })),
            _ if constant => FilterCategories::Four(Accessor::Constant(first)),
            _ => FilterCategories::Four(Accessor::func(move |i| per_row[i.min(last)])),
        };
        Ok((categories, shown))
    }

    fn coordinate_system(&self) -> Result<CoordinateSystem> {
        let key = "coordinateSystem";
        let value = match self.get(key) {
            None | Some(Value::Null) => return Ok(CoordinateSystem::Default),
            Some(value) => value,
        };
        let name = match value {
            Value::Number(n) => {
                return match n.as_f64().unwrap_or(f64::NAN) as i64 {
                    -1 => Ok(CoordinateSystem::Default),
                    0 => Ok(CoordinateSystem::Cartesian),
                    1 => Ok(CoordinateSystem::LngLat),
                    2 => Ok(CoordinateSystem::MeterOffsets),
                    3 => Ok(CoordinateSystem::LngLatOffsets),
                    other => Err(self.error(key, format!("unknown coordinate system {other}"))),
                };
            }
            Value::String(s) => s
                .trim_start_matches(CONSTANT_IDENTIFIER)
                .trim_start_matches("COORDINATE_SYSTEM.")
                .to_ascii_uppercase(),
            other => {
                return Err(self.error(key, format!("expected a number or name, got {}", describe(other))))
            }
        };
        match name.as_str() {
            "DEFAULT" => Ok(CoordinateSystem::Default),
            "CARTESIAN" => Ok(CoordinateSystem::Cartesian),
            "LNGLAT" => Ok(CoordinateSystem::LngLat),
            "METER_OFFSETS" => Ok(CoordinateSystem::MeterOffsets),
            "LNGLAT_OFFSETS" => Ok(CoordinateSystem::LngLatOffsets),
            other => Err(self.error(key, format!("unknown coordinate system `{other}`"))),
        }
    }

    /// Record a warning about this layer, reported by [`Props::finish`].
    pub fn warn(&self, message: impl std::fmt::Display) {
        self.notes
            .borrow_mut()
            .push(format!("layer `{}` ({}): {message}", self.id, self.layer_type));
    }

    /// Report props that were never read, mirroring deck.gl's warnings about unknown props.
    pub fn finish(&self, warnings: &mut Vec<String>) {
        warnings.append(&mut self.notes.borrow_mut());
        let used = self.used.borrow();
        for key in self.object.keys() {
            if !used.contains(key) && !SILENTLY_IGNORED.contains(&key.as_str()) {
                warnings.push(format!(
                    "layer `{}` ({}): prop `{key}` is not supported yet and was ignored",
                    self.id, self.layer_type
                ));
            }
        }
    }
}

/// deck.gl's `parameters` object with luma.gl's WebGPU style names (`depthCompare`,
/// `depthWriteEnabled`, `cullMode`, `blend`, `blendColorSrcFactor`, ...) plus the WebGL era
/// `depthTest`, `depthMask` and `cull`. Unknown keys are passed to `unknown`.
pub fn render_parameters(
    map: &Map<String, Value>,
    mut unknown: impl FnMut(&str),
) -> std::result::Result<RenderParameters, String> {
    let mut parameters = RenderParameters::default();
    let mut blend = premultiplied_blend();
    let mut custom_blend = false;
    let bool_of = |key: &str, v: &Value| v.as_bool().ok_or_else(|| format!("{key}: expected a boolean"));
    fn str_of<'v>(key: &str, v: &'v Value) -> std::result::Result<&'v str, String> {
        v.as_str().ok_or_else(|| format!("{key}: expected a string"))
    }
    for (key, value) in map {
        if value.is_null() {
            continue;
        }
        match key.as_str() {
            "depthTest" => parameters.depth_test = Some(bool_of(key, value)?),
            "depthWriteEnabled" | "depthMask" => parameters.depth_write_enabled = Some(bool_of(key, value)?),
            "depthCompare" => parameters.depth_compare = Some(compare_function(str_of(key, value)?)?),
            "cullMode" => {
                parameters.cull_mode = Some(match str_of(key, value)? {
                    "none" => CullMode::None,
                    "front" => CullMode::Front,
                    "back" => CullMode::Back,
                    other => return Err(format!("cullMode: unknown value `{other}`")),
                })
            }
            "cull" => {
                parameters.cull_mode = Some(if bool_of(key, value)? {
                    CullMode::Back
                } else {
                    CullMode::None
                })
            }
            "blend" => parameters.blend = Some(bool_of(key, value)?),
            "blendColorOperation" => {
                blend.color.operation = blend_operation(str_of(key, value)?)?;
                custom_blend = true;
            }
            "blendAlphaOperation" => {
                blend.alpha.operation = blend_operation(str_of(key, value)?)?;
                custom_blend = true;
            }
            "blendColorSrcFactor" => {
                blend.color.src_factor = blend_factor(str_of(key, value)?)?;
                custom_blend = true;
            }
            "blendColorDstFactor" => {
                blend.color.dst_factor = blend_factor(str_of(key, value)?)?;
                custom_blend = true;
            }
            "blendAlphaSrcFactor" => {
                blend.alpha.src_factor = blend_factor(str_of(key, value)?)?;
                custom_blend = true;
            }
            "blendAlphaDstFactor" => {
                blend.alpha.dst_factor = blend_factor(str_of(key, value)?)?;
                custom_blend = true;
            }
            other => unknown(other),
        }
    }
    if custom_blend {
        parameters.blend_state = Some(blend);
    }
    Ok(parameters)
}

fn premultiplied_blend() -> wgpu::BlendState {
    deck_gl::luma_gl::model::premultiplied_alpha_blend()
}

fn compare_function(name: &str) -> std::result::Result<wgpu::CompareFunction, String> {
    use wgpu::CompareFunction as C;
    Ok(match name {
        "never" => C::Never,
        "less" => C::Less,
        "equal" => C::Equal,
        "less-equal" => C::LessEqual,
        "greater" => C::Greater,
        "not-equal" => C::NotEqual,
        "greater-equal" => C::GreaterEqual,
        "always" => C::Always,
        other => return Err(format!("depthCompare: unknown value `{other}`")),
    })
}

fn blend_operation(name: &str) -> std::result::Result<wgpu::BlendOperation, String> {
    use wgpu::BlendOperation as O;
    Ok(match name {
        "add" => O::Add,
        "subtract" => O::Subtract,
        "reverse-subtract" => O::ReverseSubtract,
        "min" => O::Min,
        "max" => O::Max,
        other => return Err(format!("blend operation: unknown value `{other}`")),
    })
}

fn blend_factor(name: &str) -> std::result::Result<wgpu::BlendFactor, String> {
    use wgpu::BlendFactor as F;
    Ok(match name {
        "zero" => F::Zero,
        "one" => F::One,
        "src" => F::Src,
        "one-minus-src" => F::OneMinusSrc,
        "src-alpha" => F::SrcAlpha,
        "one-minus-src-alpha" => F::OneMinusSrcAlpha,
        "dst" => F::Dst,
        "one-minus-dst" => F::OneMinusDst,
        "dst-alpha" => F::DstAlpha,
        "one-minus-dst-alpha" => F::OneMinusDstAlpha,
        "src-alpha-saturated" => F::SrcAlphaSaturated,
        "constant" => F::Constant,
        "one-minus-constant" => F::OneMinusConstant,
        other => return Err(format!("blend factor: unknown value `{other}`")),
    })
}

pub fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Conversions from evaluated JSON values to layer attribute types.
pub mod convert {
    use deck_gl::{Color, Path, Polygon, Position};
    use serde_json::Value;

    use super::describe;

    pub fn number(value: &Value) -> Result<f64, String> {
        match value {
            Value::Number(n) => n.as_f64().ok_or_else(|| "number out of range".to_string()),
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            other => Err(format!("expected a number, got {}", describe(other))),
        }
    }

    pub fn f32(value: &Value) -> Result<f32, String> {
        number(value).map(|v| v as f32)
    }

    pub fn string(value: &Value) -> Result<String, String> {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Number(n) => Ok(n.to_string()),
            other => Err(format!("expected a string, got {}", describe(other))),
        }
    }

    pub fn numbers(value: &Value, min: usize, max: usize) -> Result<Vec<f64>, String> {
        let items = value
            .as_array()
            .ok_or_else(|| format!("expected an array of numbers, got {}", describe(value)))?;
        if items.len() < min || items.len() > max {
            return Err(match (min, max) {
                (min, max) if min == max => format!("expected {min} numbers, got {}", items.len()),
                _ => format!("expected {min} to {max} numbers, got {}", items.len()),
            });
        }
        items.iter().map(number).collect()
    }

    pub fn position(value: &Value) -> Result<Position, String> {
        let v = numbers(value, 2, 3)?;
        Ok([v[0], v[1], v.get(2).copied().unwrap_or(0.0)])
    }

    pub fn color(value: &Value) -> Result<Color, String> {
        let v = numbers(value, 3, 4)?;
        let channel = |c: f64| c.round().clamp(0.0, 255.0) as u8;
        Ok([
            channel(v[0]),
            channel(v[1]),
            channel(v[2]),
            v.get(3).copied().map(channel).unwrap_or(255),
        ])
    }

    pub fn rgb(value: &Value) -> Result<[u8; 3], String> {
        let c = color(value)?;
        Ok([c[0], c[1], c[2]])
    }

    pub fn vec2(value: &Value) -> Result<[f32; 2], String> {
        let v = numbers(value, 2, 2)?;
        Ok([v[0] as f32, v[1] as f32])
    }

    pub fn vec4(value: &Value) -> Result<[f32; 4], String> {
        let n = numbers(value, 4, 4)?;
        Ok([n[0] as f32, n[1] as f32, n[2] as f32, n[3] as f32])
    }

    pub fn vec3(value: &Value) -> Result<[f32; 3], String> {
        let v = numbers(value, 3, 3)?;
        Ok([v[0] as f32, v[1] as f32, v[2] as f32])
    }

    pub fn f32_list(value: &Value) -> Result<Vec<f32>, String> {
        let items = value
            .as_array()
            .ok_or_else(|| format!("expected an array of numbers, got {}", describe(value)))?;
        items.iter().map(|v| number(v).map(|n| n as f32)).collect()
    }

    pub fn string_list(value: &Value) -> Result<Vec<String>, String> {
        let items = value
            .as_array()
            .ok_or_else(|| format!("expected an array of strings, got {}", describe(value)))?;
        items.iter().map(string).collect()
    }

    pub fn path(value: &Value) -> Result<Path, String> {
        let items = value
            .as_array()
            .ok_or_else(|| format!("expected an array of positions, got {}", describe(value)))?;
        items.iter().map(position).collect()
    }

    /// A polygon is a ring, or an array of rings where the first is the outline.
    pub fn polygon(value: &Value) -> Result<Polygon, String> {
        let items = value
            .as_array()
            .ok_or_else(|| format!("expected a polygon, got {}", describe(value)))?;
        let is_single_ring = items
            .first()
            .and_then(Value::as_array)
            .and_then(|first| first.first())
            .is_some_and(Value::is_number);
        if is_single_ring {
            Ok(vec![path(value)?])
        } else {
            items.iter().map(path).collect()
        }
    }
}

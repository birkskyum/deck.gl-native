//! Typed readers for the camelCase props of a JSON layer object, including `@@=` accessors.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

use deck_gl::glam::DMat4;
use deck_gl::{Accessor, Color, CoordinateSystem, LayerProps, Unit};
use serde_json::{Map, Value};

use crate::expression::Expr;
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
            Accessor::Func(_) => panic!("default accessors are constants or columns"),
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
    rows: Option<Rows>,
    /// True when `data` is an Arrow table, so accessors are columns rather than expressions.
    table: bool,
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
            rows: None,
            table: false,
        };
        props.mark("id");
        props.mark(TYPE_KEY);
        props
    }

    /// Attach the rows that accessor expressions evaluate against.
    pub fn set_rows(&mut self, rows: Rows) {
        self.rows = Some(rows);
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
        base.highlight_color = self.color("highlightColor", defaults.highlight_color)?;
        base.highlighted_object_index = match self.get("highlightedObjectIndex") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let index = convert::number(value).map_err(|m| self.error("highlightedObjectIndex", m))?;
                (index >= 0.0).then_some(index as u32)
            }
        };
        Ok(base)
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

    /// Report props that were never read, mirroring deck.gl's warnings about unknown props.
    pub fn finish(&self, warnings: &mut Vec<String>) {
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

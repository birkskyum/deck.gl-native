//! Layer data and accessors.
//!
//! deck.gl JS layers take a `data` array plus accessor functions. The native port keeps the
//! accessor idea but makes Apache Arrow the primary data model: a layer's data is an Arrow
//! `RecordBatch`, and an [`Accessor`] is either a constant, the name of a column in that
//! batch, or a function of the row index. Column accessors read Arrow buffers directly,
//! which is the path GeoArrow data takes with no per-row callbacks.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType;

use crate::{DeckError, Result};

/// A position in the layer's coordinate system (lng, lat, meters by default).
pub type Position = [f64; 3];
/// An RGBA color with components in 0..255.
pub type Color = [u8; 4];
/// A polygon as a list of rings. The first ring is the outer boundary, the rest are holes.
pub type Polygon = Vec<Vec<Position>>;
/// A path (polyline) as a list of positions.
pub type Path = Vec<Position>;

/// How a per-object value is obtained.
#[derive(Clone)]
pub enum Accessor<T: Clone> {
    /// The same value for every object.
    Constant(T),
    /// Read from a column of the layer's Arrow record batch.
    Column(String),
    /// Computed from the row index.
    Func(Arc<dyn Fn(usize) -> T + Send + Sync>),
}

impl<T: Clone> Accessor<T> {
    pub fn column(name: impl Into<String>) -> Self {
        Accessor::Column(name.into())
    }

    pub fn func(f: impl Fn(usize) -> T + Send + Sync + 'static) -> Self {
        Accessor::Func(Arc::new(f))
    }
}

impl<T: Clone> From<T> for Accessor<T> {
    fn from(value: T) -> Self {
        Accessor::Constant(value)
    }
}

/// Constants and columns compare by value; functions compare by identity.
impl<T: Clone + PartialEq> PartialEq for Accessor<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Accessor::Constant(a), Accessor::Constant(b)) => a == b,
            (Accessor::Column(a), Accessor::Column(b)) => a == b,
            (Accessor::Func(a), Accessor::Func(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl<T: Clone + std::fmt::Debug> std::fmt::Debug for Accessor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Accessor::Constant(v) => write!(f, "Constant({v:?})"),
            Accessor::Column(c) => write!(f, "Column({c:?})"),
            Accessor::Func(_) => write!(f, "Func(..)"),
        }
    }
}

/// The data a layer renders: an optional Arrow record batch and the number of objects.
///
/// Data is compared by column identity, so sending the same batch again costs nothing.
/// When only some rows changed, [`LayerData::with_changed_rows`] tells the layer to rewrite
/// those rows in place instead of rebuilding every attribute; appended rows grow the buffers
/// with headroom so streams of appends stay cheap.
#[derive(Clone, Debug, Default)]
pub struct LayerData {
    pub batch: Option<RecordBatch>,
    pub length: usize,
    /// For sub layers of a composite layer: the source row of each item, so picking and
    /// highlighting report the parent's rows. Mirrors deck.gl's `__source.index`.
    pub source_rows: Option<Arc<Vec<u32>>>,
    /// The rows that differ from the data the layer had before, when known. Rows outside the
    /// range are unchanged, so attributes are only rewritten for these rows. Mirrors deck.gl's
    /// `_dataDiff`.
    pub changed_rows: Option<Range<usize>>,
    /// The row of the original data that row 0 of a slice maps to, so function accessors and
    /// row indices see the original row numbers.
    pub row_offset: usize,
}

/// Batches compare by column identity first, so re-sending the same data is cheap, and by
/// value otherwise. Data that names changed rows never compares equal: the rows are meant to
/// be written.
impl PartialEq for LayerData {
    fn eq(&self, other: &Self) -> bool {
        if self.changed_rows.is_some() || other.changed_rows.is_some() {
            return false;
        }
        if self.length != other.length || self.row_offset != other.row_offset {
            return false;
        }
        let rows_equal = match (&self.source_rows, &other.source_rows) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
            _ => false,
        };
        if !rows_equal {
            return false;
        }
        match (&self.batch, &other.batch) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                let same_columns = a.num_columns() == b.num_columns()
                    && a.columns()
                        .iter()
                        .zip(b.columns())
                        .all(|(x, y)| Arc::ptr_eq(x, y));
                same_columns || a == b
            }
            _ => false,
        }
    }
}

impl LayerData {
    /// Data backed by an Arrow record batch. Column accessors read from it.
    pub fn from_batch(batch: RecordBatch) -> Self {
        Self {
            length: batch.num_rows(),
            batch: Some(batch),
            source_rows: None,
            changed_rows: None,
            row_offset: 0,
        }
    }

    /// Data with only a length. All accessors must be constants or functions.
    pub fn with_length(length: usize) -> Self {
        Self {
            batch: None,
            length,
            source_rows: None,
            changed_rows: None,
            row_offset: 0,
        }
    }

    /// Mark `rows` as the only rows that differ from the data the layer rendered before.
    /// The range may extend past the previous length to append rows. Function accessors are
    /// called for the changed rows only, so they must still be the same functions.
    pub fn with_changed_rows(mut self, rows: Range<usize>) -> Self {
        self.changed_rows = Some(rows);
        self
    }

    /// The rows at `indices`, in that order: Arrow columns are gathered without copying the
    /// values that are left out, and every row remembers where it came from, so picking and
    /// the accessors of a composite layer still report the original rows.
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        let batch = match &self.batch {
            Some(batch) => {
                let taken = arrow_select::take::take_record_batch(
                    batch,
                    &arrow_array::UInt32Array::from(indices.to_vec()),
                )
                .map_err(|e| DeckError::Data(format!("could not gather rows: {e}")))?;
                Some(taken)
            }
            None => None,
        };
        Ok(Self {
            batch,
            length: indices.len(),
            source_rows: Some(Arc::new(
                indices.iter().map(|i| self.source_row(*i as usize)).collect(),
            )),
            changed_rows: None,
            row_offset: 0,
        })
    }

    /// The rows in `range`, clamped to the data. Arrow columns are sliced without copying and
    /// function accessors keep seeing the original row numbers.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let end = range.end.min(self.length);
        let start = range.start.min(end);
        Self {
            batch: self.batch.as_ref().map(|b| b.slice(start, end - start)),
            length: end - start,
            source_rows: self
                .source_rows
                .as_ref()
                .map(|rows| Arc::new(rows.get(start..end).map(<[u32]>::to_vec).unwrap_or_default())),
            changed_rows: None,
            row_offset: self.row_offset + start,
        }
    }

    /// Map every item to a row of a parent layer's data (for composite sub layers).
    pub fn with_source_rows(mut self, rows: Arc<Vec<u32>>) -> Self {
        self.source_rows = Some(rows);
        self
    }

    /// The row reported by picking for item `index`.
    pub fn source_row(&self, index: usize) -> u32 {
        match &self.source_rows {
            Some(rows) => rows
                .get(index)
                .copied()
                .unwrap_or((self.row_offset + index) as u32),
            None => (self.row_offset + index) as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The Arrow column of the layer's table, by name.
    /// The GeoArrow extension name of a column (`geoarrow.point`, `geoarrow.multipolygon`
    /// and so on), when its field carries one.
    pub fn column_extension(&self, name: &str) -> Option<String> {
        let batch = self.batch.as_ref()?;
        let field = batch.schema_ref().field_with_name(name).ok()?;
        field
            .metadata()
            .get("ARROW:extension:name")
            .filter(|name| name.starts_with("geoarrow."))
            .cloned()
    }

    pub fn column(&self, name: &str) -> Result<&ArrayRef> {
        let batch = self.batch.as_ref().ok_or_else(|| {
            DeckError::Data(format!("column `{name}` requested but layer has no record batch"))
        })?;
        batch
            .column_by_name(name)
            .ok_or_else(|| DeckError::Data(format!("no column `{name}` in record batch")))
    }
}

/// Convert any numeric primitive array into f64 values.
fn primitive_to_f64(array: &dyn Array) -> Result<Vec<f64>> {
    macro_rules! convert {
        ($t:ty) => {
            array
                .as_primitive::<$t>()
                .values()
                .iter()
                .map(|v| *v as f64)
                .collect()
        };
    }
    Ok(match array.data_type() {
        DataType::Float64 => convert!(Float64Type),
        DataType::Float32 => convert!(Float32Type),
        DataType::Int8 => convert!(Int8Type),
        DataType::Int16 => convert!(Int16Type),
        DataType::Int32 => convert!(Int32Type),
        DataType::Int64 => convert!(Int64Type),
        DataType::UInt8 => convert!(UInt8Type),
        DataType::UInt16 => convert!(UInt16Type),
        DataType::UInt32 => convert!(UInt32Type),
        DataType::UInt64 => convert!(UInt64Type),
        other => return Err(DeckError::Data(format!("expected a numeric column, got {other}"))),
    })
}

/// Flatten a GeoArrow coordinate array into (values, width).
///
/// Both layouts of the specification are read: interleaved coordinates
/// (`FixedSizeList<Float64, 2|3>`, `xyxy`) and separated ones (`Struct<x, y, z?>`, the
/// "struct of arrays" layout), so a column written either way works the same.
fn fixed_size_list_to_f64(array: &dyn Array) -> Result<(Vec<f64>, usize)> {
    if let Some(list) = array.as_fixed_size_list_opt() {
        let width = list.value_length() as usize;
        let values = primitive_to_f64(list.values().as_ref())?;
        // Respect the list array's offset into its child values.
        let start = list.offset() * width;
        let end = start + list.len() * width;
        return Ok((values[start..end].to_vec(), width));
    }
    if let Some(separated) = separated_coords(array)? {
        return Ok(separated);
    }
    Err(DeckError::Data(format!(
        "expected interleaved (FixedSizeList) or separated (Struct of x, y and z) coordinates, got {}",
        array.data_type()
    )))
}

/// GeoArrow's separated coordinates: a struct with `x`, `y` and optionally `z` children,
/// interleaved here into (values, width).
fn separated_coords(array: &dyn Array) -> Result<Option<(Vec<f64>, usize)>> {
    let Some(fields) = array.as_struct_opt() else {
        return Ok(None);
    };
    let DataType::Struct(schema) = fields.data_type() else {
        return Ok(None);
    };
    let named = |name: &str| schema.iter().position(|f| f.name() == name);
    let (Some(x), Some(y)) = (named("x"), named("y")) else {
        return Ok(None);
    };
    let z = named("z");
    let width = if z.is_some() { 3 } else { 2 };
    let column = |index: usize| -> Result<Vec<f64>> {
        let values = primitive_to_f64(fields.column(index).as_ref())?;
        // The struct's offset applies to its children
        Ok(values[fields.offset()..fields.offset() + fields.len()].to_vec())
    };
    let xs = column(x)?;
    let ys = column(y)?;
    let zs = match z {
        Some(z) => Some(column(z)?),
        None => None,
    };
    let mut values = Vec::with_capacity(xs.len() * width);
    for row in 0..xs.len() {
        values.push(xs[row]);
        values.push(ys[row]);
        if let Some(zs) = &zs {
            values.push(zs[row]);
        }
    }
    Ok(Some((values, width)))
}

/// Resolve an accessor, reading columns through `from_column`.
pub fn resolve_with<T: Clone + Send>(
    data: &LayerData,
    accessor: &Accessor<T>,
    from_column: impl FnOnce(&ArrayRef) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    match accessor {
        Accessor::Constant(v) => Ok(vec![v.clone(); data.len()]),
        Accessor::Func(f) => Ok(resolve_function(f.as_ref(), data.row_offset, data.len())),
        Accessor::Column(name) => {
            let column = data.column(name)?;
            let values = from_column(column)?;
            if values.len() != data.len() {
                return Err(DeckError::Data(format!(
                    "column `{name}` has {} rows, expected {}",
                    values.len(),
                    data.len()
                )));
            }
            Ok(values)
        }
    }
}

/// Add the GeoArrow extension name of the column to an error, so a column of a geometry kind
/// that a layer cannot take says which kind it is.
fn in_geoarrow_column<T, U: Clone>(data: &LayerData, accessor: &Accessor<U>, result: Result<T>) -> Result<T> {
    let (Err(DeckError::Data(message)), Accessor::Column(name)) = (&result, accessor) else {
        return result;
    };
    match data.column_extension(name) {
        Some(extension) => Err(DeckError::Data(format!(
            "column `{name}` is `{extension}`: {message}"
        ))),
        None => result,
    }
}

/// Resolve positions. Columns hold GeoArrow points, either interleaved
/// (`FixedSizeList<Float32|Float64, 2|3>`) or separated (`Struct<x, y, z?>`), or WKB.
pub fn resolve_positions(data: &LayerData, accessor: &Accessor<Position>) -> Result<Vec<Position>> {
    let result = resolve_with(data, accessor, |column| {
        if let Some(points) = wkb_column(column, "point", [0.0; 3], |g| match g {
            crate::geojson::Geometry::Point(p) => Some(p),
            _ => None,
        })? {
            return Ok(points);
        }
        let (values, width) = fixed_size_list_to_f64(column.as_ref())?;
        if width != 2 && width != 3 {
            return Err(DeckError::Data(format!(
                "position column must have 2 or 3 components, got {width}"
            )));
        }
        Ok(values
            .chunks(width)
            .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
            .collect())
    });
    in_geoarrow_column(data, accessor, result)
}

/// Rows above which function accessors are evaluated on all cores.
const PARALLEL_ROWS: usize = 16_384;

/// Evaluate a function accessor for every row, in parallel for large data.
fn resolve_function<T: Clone + Send>(
    f: &(dyn Fn(usize) -> T + Send + Sync),
    offset: usize,
    len: usize,
) -> Vec<T> {
    if len < PARALLEL_ROWS {
        return (offset..offset + len).map(f).collect();
    }
    use rayon::prelude::*;
    (offset..offset + len).into_par_iter().map(f).collect()
}

/// The WKB bytes of every row of a binary column, when the column is one.
fn wkb_rows(column: &ArrayRef) -> Option<Vec<Option<&[u8]>>> {
    let rows = column.len();
    if let Some(array) = column.as_binary_opt::<i32>() {
        return Some(
            (0..rows)
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect(),
        );
    }
    if let Some(array) = column.as_binary_opt::<i64>() {
        return Some(
            (0..rows)
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect(),
        );
    }
    if let Some(array) = column.as_binary_view_opt() {
        return Some(
            (0..rows)
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect(),
        );
    }
    None
}

/// Decode every row of a WKB column with `pick`, which turns the geometry into the value the
/// accessor needs; nulls and geometries of another kind become `fallback`.
fn wkb_column<T: Clone>(
    column: &ArrayRef,
    what: &str,
    fallback: T,
    pick: impl Fn(crate::geojson::Geometry) -> Option<T>,
) -> Result<Option<Vec<T>>> {
    let Some(rows) = wkb_rows(column) else {
        return Ok(None);
    };
    let mut values = Vec::with_capacity(rows.len());
    for (row, bytes) in rows.into_iter().enumerate() {
        let value = match bytes {
            None => fallback.clone(),
            Some(bytes) => {
                let geometry = crate::wkb::wkb_geometry(bytes)
                    .map_err(|e| DeckError::Data(format!("row {row}: {e}")))?;
                pick(geometry)
                    .ok_or_else(|| DeckError::Data(format!("row {row}: WKB geometry is not a {what}")))?
            }
        };
        values.push(value);
    }
    Ok(Some(values))
}

/// Resolve scalar floats from any numeric column.
pub fn resolve_f32(data: &LayerData, accessor: &Accessor<f32>) -> Result<Vec<f32>> {
    resolve_with(data, accessor, |column| {
        Ok(primitive_to_f64(column.as_ref())?
            .into_iter()
            .map(|v| v as f32)
            .collect())
    })
}

/// Resolve strings from a `Utf8` or `LargeUtf8` column.
pub fn resolve_strings(data: &LayerData, accessor: &Accessor<String>) -> Result<Vec<String>> {
    resolve_with(data, accessor, |column| {
        if let Some(array) = column.as_string_opt::<i32>() {
            return Ok(array.iter().map(|v| v.unwrap_or_default().to_string()).collect());
        }
        if let Some(array) = column.as_string_opt::<i64>() {
            return Ok(array.iter().map(|v| v.unwrap_or_default().to_string()).collect());
        }
        Err(DeckError::Data(format!(
            "expected a string column, got {}",
            column.data_type()
        )))
    })
}

/// Resolve 2-component vectors from a `FixedSizeList` column of width 2.
pub fn resolve_vec2(data: &LayerData, accessor: &Accessor<[f32; 2]>) -> Result<Vec<[f32; 2]>> {
    resolve_with(data, accessor, |column| {
        let (values, width) = fixed_size_list_to_f64(column.as_ref())?;
        if width != 2 {
            return Err(DeckError::Data(format!("expected 2 components, got {width}")));
        }
        Ok(values.chunks(2).map(|c| [c[0] as f32, c[1] as f32]).collect())
    })
}

/// Resolve RGBA colors. Columns must be `FixedSizeList<numeric>` of width 3 or 4 with
/// Resolve 3-component vectors from a `FixedSizeList` column of width 3.
pub fn resolve_vec3(data: &LayerData, accessor: &Accessor<[f32; 3]>) -> Result<Vec<[f32; 3]>> {
    resolve_with(data, accessor, |column| {
        let (values, width) = fixed_size_list_to_f64(column.as_ref())?;
        if width != 3 {
            return Err(DeckError::Data(format!("expected 3 components, got {width}")));
        }
        Ok(values
            .chunks(3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect())
    })
}

/// Resolve 4-component vectors from a `FixedSizeList` column of width 4.
pub fn resolve_vec4(data: &LayerData, accessor: &Accessor<[f32; 4]>) -> Result<Vec<[f32; 4]>> {
    resolve_with(data, accessor, |column| {
        let (values, width) = fixed_size_list_to_f64(column.as_ref())?;
        if width != 4 {
            return Err(DeckError::Data(format!("expected 4 components, got {width}")));
        }
        Ok(values
            .chunks(4)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
            .collect())
    })
}

/// components in 0..255. Missing alpha defaults to 255.
pub fn resolve_colors(data: &LayerData, accessor: &Accessor<Color>) -> Result<Vec<Color>> {
    resolve_with(data, accessor, |column| {
        let (values, width) = fixed_size_list_to_f64(column.as_ref())?;
        if width != 3 && width != 4 {
            return Err(DeckError::Data(format!(
                "color column must have 3 or 4 components, got {width}"
            )));
        }
        Ok(values
            .chunks(width)
            .map(|c| {
                [
                    c[0].clamp(0.0, 255.0) as u8,
                    c[1].clamp(0.0, 255.0) as u8,
                    c[2].clamp(0.0, 255.0) as u8,
                    if width == 4 {
                        c[3].clamp(0.0, 255.0) as u8
                    } else {
                        255
                    },
                ]
            })
            .collect())
    })
}

/// Resolve paths. Columns must be GeoArrow linestrings (`List<FixedSizeList<Float64, 2|3>>`).
pub fn resolve_paths(data: &LayerData, accessor: &Accessor<Path>) -> Result<Vec<Path>> {
    let result = resolve_with(data, accessor, |column| {
        if let Some(paths) = wkb_column(column, "linestring", Vec::new(), |g| match g {
            crate::geojson::Geometry::LineString(path) => Some(path),
            _ => None,
        })? {
            return Ok(paths);
        }
        let (offsets, coords) = list_parts(column.as_ref())?;
        let (values, width) = fixed_size_list_to_f64(coords.as_ref())?;
        if width != 2 && width != 3 {
            return Err(DeckError::Data(format!(
                "path coordinates must have 2 or 3 components, got {width}"
            )));
        }
        Ok((0..offsets.len().saturating_sub(1))
            .map(|i| {
                values[offsets[i] * width..offsets[i + 1] * width]
                    .chunks(width)
                    .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
                    .collect()
            })
            .collect())
    });
    in_geoarrow_column(data, accessor, result)
}

/// Resolve lists of strings, such as the cells of an H3 cluster, from a `List<Utf8>` column.
pub fn resolve_string_lists(data: &LayerData, accessor: &Accessor<Vec<String>>) -> Result<Vec<Vec<String>>> {
    resolve_with(data, accessor, |column| {
        let (offsets, values) = list_parts(column.as_ref())?;
        let strings: Vec<String> = if let Some(array) = values.as_string_opt::<i32>() {
            array.iter().map(|v| v.unwrap_or_default().to_string()).collect()
        } else if let Some(array) = values.as_string_opt::<i64>() {
            array.iter().map(|v| v.unwrap_or_default().to_string()).collect()
        } else {
            return Err(DeckError::Data(format!(
                "expected a list of strings, got a list of {}",
                values.data_type()
            )));
        };
        Ok((0..offsets.len().saturating_sub(1))
            .map(|i| strings[offsets[i]..offsets[i + 1]].to_vec())
            .collect())
    })
}

/// Resolve lists of numbers, such as timestamps per path vertex, from a `List<numeric>` column.
pub fn resolve_f32_lists(data: &LayerData, accessor: &Accessor<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    resolve_with(data, accessor, |column| {
        let (offsets, values) = list_parts(column.as_ref())?;
        let values = primitive_to_f64(values.as_ref())?;
        Ok((0..offsets.len().saturating_sub(1))
            .map(|i| {
                values[offsets[i]..offsets[i + 1]]
                    .iter()
                    .map(|v| *v as f32)
                    .collect()
            })
            .collect())
    })
}

/// Resolve polygons. Columns may be GeoArrow polygons
/// (`List<List<FixedSizeList<Float64, 2|3>>>`) or single rings (`List<FixedSizeList<..>>`).
pub fn resolve_polygons(data: &LayerData, accessor: &Accessor<Polygon>) -> Result<Vec<Polygon>> {
    let result = resolve_with(data, accessor, polygons_from_column);
    in_geoarrow_column(data, accessor, result)
}

fn polygons_from_column(column: &ArrayRef) -> Result<Vec<Polygon>> {
    if let Some(polygons) = wkb_column(column, "polygon", Vec::new(), |g| match g {
        crate::geojson::Geometry::Polygon(polygon) => Some(polygon),
        _ => None,
    })? {
        return Ok(polygons);
    }
    match column.data_type() {
        DataType::List(_) | DataType::LargeList(_) => {}
        other => {
            return Err(DeckError::Data(format!(
                "polygon column must be a List, got {other}"
            )))
        }
    }
    let (outer_offsets, rings_array) = list_parts(column.as_ref())?;
    // Two shapes: List<List<FixedSizeList>> (polygon with rings) or List<FixedSizeList> (ring).
    let is_ring_list = matches!(
        rings_array.data_type(),
        DataType::List(_) | DataType::LargeList(_)
    );
    if is_ring_list {
        let (ring_offsets, coords) = list_parts(rings_array.as_ref())?;
        let (values, width) = fixed_size_list_to_f64(coords.as_ref())?;
        let mut polygons = Vec::with_capacity(outer_offsets.len().saturating_sub(1));
        for p in 0..outer_offsets.len() - 1 {
            let mut rings = Vec::new();
            for r in outer_offsets[p]..outer_offsets[p + 1] {
                let start = ring_offsets[r] * width;
                let end = ring_offsets[r + 1] * width;
                rings.push(
                    values[start..end]
                        .chunks(width)
                        .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
                        .collect(),
                );
            }
            polygons.push(rings);
        }
        Ok(polygons)
    } else {
        let (values, width) = fixed_size_list_to_f64(rings_array.as_ref())?;
        let mut polygons = Vec::with_capacity(outer_offsets.len().saturating_sub(1));
        for p in 0..outer_offsets.len() - 1 {
            let start = outer_offsets[p] * width;
            let end = outer_offsets[p + 1] * width;
            polygons.push(vec![values[start..end]
                .chunks(width)
                .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
                .collect()]);
        }
        Ok(polygons)
    }
}

/// The parts of a column of multi geometries, one entry per part.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiParts {
    Points(Vec<Position>),
    Paths(Vec<Path>),
    Polygons(Vec<Polygon>),
}

impl MultiParts {
    pub fn len(&self) -> usize {
        match self {
            Self::Points(v) => v.len(),
            Self::Paths(v) => v.len(),
            Self::Polygons(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Split a column of multi geometries into one row per part, the way
/// `@geoarrow/deck.gl-layers` does: a layer draws parts, and picking reports the row each
/// part came from.
///
/// The column is recognised by its GeoArrow extension name (`geoarrow.multipoint`,
/// `geoarrow.multilinestring`, `geoarrow.multipolygon`) or, for WKB, by the geometry in the
/// bytes. `Ok(None)` means the column holds single geometries, which layers read as they are.
///
/// The data that comes back has one row per part, with every other column gathered to match,
/// so the layer's other accessors keep working.
pub fn explode_multi(data: &LayerData, column: &str) -> Result<Option<(LayerData, MultiParts)>> {
    let array = data.column(column)?;
    let extension = data.column_extension(column);
    let mut rows: Vec<u32> = Vec::new();
    let parts = match extension.as_deref() {
        Some("geoarrow.multipoint") => {
            let (offsets, coords) = list_parts(array.as_ref())?;
            let (values, width) = fixed_size_list_to_f64(coords.as_ref())?;
            let mut points = Vec::new();
            for row in 0..offsets.len().saturating_sub(1) {
                for point in values[offsets[row] * width..offsets[row + 1] * width].chunks(width) {
                    points.push([point[0], point[1], if width == 3 { point[2] } else { 0.0 }]);
                    rows.push(row as u32);
                }
            }
            MultiParts::Points(points)
        }
        Some("geoarrow.multilinestring") => {
            let (outer, lines) = list_parts(array.as_ref())?;
            let (line_offsets, coords) = list_parts(lines.as_ref())?;
            let (values, width) = fixed_size_list_to_f64(coords.as_ref())?;
            let mut paths = Vec::new();
            for row in 0..outer.len().saturating_sub(1) {
                for line in outer[row]..outer[row + 1] {
                    paths.push(
                        values[line_offsets[line] * width..line_offsets[line + 1] * width]
                            .chunks(width)
                            .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
                            .collect(),
                    );
                    rows.push(row as u32);
                }
            }
            MultiParts::Paths(paths)
        }
        Some("geoarrow.multipolygon") => {
            let (outer, polygons) = list_parts(array.as_ref())?;
            let (polygon_offsets, rings) = list_parts(polygons.as_ref())?;
            let (ring_offsets, coords) = list_parts(rings.as_ref())?;
            let (values, width) = fixed_size_list_to_f64(coords.as_ref())?;
            let mut out = Vec::new();
            for row in 0..outer.len().saturating_sub(1) {
                for polygon in outer[row]..outer[row + 1] {
                    let mut ring_list = Vec::new();
                    for ring in polygon_offsets[polygon]..polygon_offsets[polygon + 1] {
                        ring_list.push(
                            values[ring_offsets[ring] * width..ring_offsets[ring + 1] * width]
                                .chunks(width)
                                .map(|c| [c[0], c[1], if width == 3 { c[2] } else { 0.0 }])
                                .collect(),
                        );
                    }
                    out.push(ring_list);
                    rows.push(row as u32);
                }
            }
            MultiParts::Polygons(out)
        }
        _ => match explode_wkb(array, &mut rows)? {
            Some(parts) => parts,
            None => return Ok(None),
        },
    };
    Ok(Some((data.gather(&rows)?, parts)))
}

/// The parts of a WKB column of multi geometries, or `None` when it holds single ones.
fn explode_wkb(column: &ArrayRef, rows: &mut Vec<u32>) -> Result<Option<MultiParts>> {
    let Some(bytes) = wkb_rows(column) else {
        return Ok(None);
    };
    let geometries: Vec<Option<crate::geojson::Geometry>> = bytes
        .iter()
        .map(|row| row.and_then(|b| crate::wkb::wkb_geometry(b).ok()))
        .collect();
    // The first geometry that says what the column holds decides how it is split
    let kind = geometries.iter().flatten().find_map(|g| match g {
        crate::geojson::Geometry::MultiPoint(_) => Some(0),
        crate::geojson::Geometry::MultiLineString(_) => Some(1),
        crate::geojson::Geometry::MultiPolygon(_) => Some(2),
        _ => None,
    });
    let Some(kind) = kind else {
        return Ok(None);
    };
    let mut points = Vec::new();
    let mut paths = Vec::new();
    let mut polygons = Vec::new();
    for (row, geometry) in geometries.iter().enumerate() {
        let Some(geometry) = geometry else { continue };
        let mut push = |count: usize| rows.extend(std::iter::repeat_n(row as u32, count));
        match (kind, geometry) {
            (0, crate::geojson::Geometry::MultiPoint(ps)) => {
                push(ps.len());
                points.extend(ps.iter().copied());
            }
            (0, crate::geojson::Geometry::Point(p)) => {
                push(1);
                points.push(*p);
            }
            (1, crate::geojson::Geometry::MultiLineString(ls)) => {
                push(ls.len());
                paths.extend(ls.iter().cloned());
            }
            (1, crate::geojson::Geometry::LineString(l)) => {
                push(1);
                paths.push(l.clone());
            }
            (2, crate::geojson::Geometry::MultiPolygon(ps)) => {
                push(ps.len());
                polygons.extend(ps.iter().cloned());
            }
            (2, crate::geojson::Geometry::Polygon(p)) => {
                push(1);
                polygons.push(p.clone());
            }
            // A row of another kind contributes nothing rather than a wrong shape
            _ => {}
        }
    }
    Ok(Some(match kind {
        0 => MultiParts::Points(points),
        1 => MultiParts::Paths(paths),
        _ => MultiParts::Polygons(polygons),
    }))
}

/// Offsets (as usize, relative to the child values) and child array of a list array.
fn list_parts(array: &dyn Array) -> Result<(Vec<usize>, ArrayRef)> {
    if let Some(list) = array.as_list_opt::<i32>() {
        let offsets: Vec<usize> = list.value_offsets().iter().map(|o| *o as usize).collect();
        return Ok((offsets, list.values().clone()));
    }
    if let Some(list) = array.as_list_opt::<i64>() {
        let offsets: Vec<usize> = list.value_offsets().iter().map(|o| *o as usize).collect();
        return Ok((offsets, list.values().clone()));
    }
    Err(DeckError::Data(format!(
        "expected a List array, got {}",
        array.data_type()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, ListBuilder, UInt8Builder};
    use arrow_array::{Float32Array, RecordBatch};
    use arrow_schema::{Field, Schema};

    fn batch() -> RecordBatch {
        let mut positions = FixedSizeListBuilder::new(Float64Builder::new(), 2);
        for (x, y) in [(1.0, 2.0), (3.0, 4.0)] {
            positions.values().append_value(x);
            positions.values().append_value(y);
            positions.append(true);
        }
        let mut colors = FixedSizeListBuilder::new(UInt8Builder::new(), 3);
        for c in [[255u8, 0, 0], [0, 255, 0]] {
            colors.values().append_slice(&c);
            colors.append(true);
        }
        let radius = Float32Array::from(vec![5.0f32, 6.0]);
        let schema = Schema::new(vec![
            Field::new("pos", positions.finish_cloned().data_type().clone(), false),
            Field::new("color", colors.finish_cloned().data_type().clone(), false),
            Field::new("radius", DataType::Float32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(positions.finish()),
                Arc::new(colors.finish()),
                Arc::new(radius),
            ],
        )
        .unwrap()
    }

    #[test]
    fn reads_columns() {
        let data = LayerData::from_batch(batch());
        assert_eq!(
            resolve_positions(&data, &Accessor::column("pos")).unwrap(),
            vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
        );
        assert_eq!(
            resolve_colors(&data, &Accessor::column("color")).unwrap(),
            vec![[255, 0, 0, 255], [0, 255, 0, 255]]
        );
        assert_eq!(
            resolve_f32(&data, &Accessor::column("radius")).unwrap(),
            vec![5.0, 6.0]
        );
        assert_eq!(
            resolve_f32(&data, &Accessor::Constant(1.0)).unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            resolve_f32(&data, &Accessor::func(|i| i as f32)).unwrap(),
            vec![0.0, 1.0]
        );
        assert!(resolve_f32(&data, &Accessor::column("missing")).is_err());
    }

    #[test]
    fn slices_keep_original_row_numbers() {
        let data = LayerData::from_batch(batch());
        let slice = data.slice(1..5);
        assert_eq!(slice.len(), 1);
        assert_eq!(slice.row_offset, 1);
        assert_eq!(slice.source_row(0), 1);
        assert_eq!(
            resolve_positions(&slice, &Accessor::column("pos")).unwrap(),
            vec![[3.0, 4.0, 0.0]]
        );
        assert_eq!(
            resolve_colors(&slice, &Accessor::column("color")).unwrap(),
            vec![[0, 255, 0, 255]]
        );
        assert_eq!(
            resolve_f32(&slice, &Accessor::func(|i| i as f32 * 10.0)).unwrap(),
            vec![10.0]
        );
        let mapped = LayerData {
            source_rows: Some(Arc::new(vec![7, 9])),
            ..data.clone()
        };
        assert_eq!(mapped.slice(1..2).source_row(0), 9);
        assert!(data.slice(3..9).is_empty());
        // A hint always counts as a change, so layers apply it
        assert_ne!(data.clone().with_changed_rows(0..1), data);
        assert_eq!(data.clone(), data);
    }

    #[test]
    fn reads_separated_geoarrow_coordinates() {
        use arrow_array::{Float64Array, StructArray};
        use std::collections::HashMap;
        // GeoArrow's separated layout: a struct of x, y and z columns instead of a list
        let xs = Arc::new(Float64Array::from(vec![1.0, 3.0])) as ArrayRef;
        let ys = Arc::new(Float64Array::from(vec![2.0, 4.0])) as ArrayRef;
        let zs = Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef;
        let fields = vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ];
        let points = StructArray::new(fields.clone().into(), vec![xs, ys, zs], None);
        let mut metadata = HashMap::new();
        metadata.insert("ARROW:extension:name".to_string(), "geoarrow.point".to_string());
        let schema = Schema::new(vec![
            Field::new("geometry", points.data_type().clone(), false).with_metadata(metadata)
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(points)]).unwrap();
        let data = LayerData::from_batch(batch);
        assert_eq!(
            data.column_extension("geometry").as_deref(),
            Some("geoarrow.point")
        );
        assert_eq!(
            resolve_positions(&data, &Accessor::column("geometry")).unwrap(),
            vec![[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]
        );
        // Without a z child the points are flat
        let flat = StructArray::new(
            fields[..2].to_vec().into(),
            vec![
                Arc::new(Float64Array::from(vec![5.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![6.0])) as ArrayRef,
            ],
            None,
        );
        let schema = Schema::new(vec![Field::new("p", flat.data_type().clone(), false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(flat)]).unwrap();
        let data = LayerData::from_batch(batch);
        assert_eq!(
            resolve_positions(&data, &Accessor::column("p")).unwrap(),
            vec![[5.0, 6.0, 0.0]]
        );
        assert!(data.column_extension("p").is_none());
        // A geometry column a layer cannot read says which kind it is
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "geoarrow.multipolygon".to_string(),
        );
        let values = Arc::new(Float64Array::from(vec![1.0])) as ArrayRef;
        let schema = Schema::new(vec![
            Field::new("bad", DataType::Float64, false).with_metadata(metadata)
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![values]).unwrap();
        let data = LayerData::from_batch(batch);
        let error = resolve_polygons(&data, &Accessor::column("bad"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("geoarrow.multipolygon"), "{error}");
    }

    #[test]
    fn multi_geometry_columns_explode_into_one_row_per_part() {
        use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, ListBuilder};
        use arrow_array::{Int32Array, StringArray};
        use std::collections::HashMap;
        // A GeoArrow multipoint column: the first row has two points, the second one
        let coords = FixedSizeListBuilder::new(Float64Builder::new(), 2);
        let mut points = ListBuilder::new(coords);
        for row in [vec![[1.0, 2.0], [3.0, 4.0]], vec![[5.0, 6.0]]] {
            for [x, y] in row {
                let coords = points.values();
                coords.values().append_value(x);
                coords.values().append_value(y);
                coords.append(true);
            }
            points.append(true);
        }
        let array = points.finish();
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "geoarrow.multipoint".to_string(),
        );
        let schema = Schema::new(vec![
            Field::new("geometry", array.data_type().clone(), false).with_metadata(metadata),
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(array),
                Arc::new(Int32Array::from(vec![10, 20])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let data = LayerData::from_batch(batch);
        let (exploded, parts) = explode_multi(&data, "geometry").unwrap().expect("a multi column");
        assert_eq!(
            parts,
            MultiParts::Points(vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
        );
        // Three parts, and the other columns follow the rows they came from
        assert_eq!(exploded.len(), 3);
        assert_eq!(
            resolve_f32(&exploded, &Accessor::column("id")).unwrap(),
            vec![10.0, 10.0, 20.0]
        );
        assert_eq!(
            resolve_strings(&exploded, &Accessor::column("name")).unwrap(),
            vec!["a".to_string(), "a".to_string(), "b".to_string()]
        );
        // Picking reports the rows of the original data
        assert_eq!(
            (0..3).map(|i| exploded.source_row(i)).collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        // A column of single geometries is left alone
        assert!(explode_multi(&batch_data(), "pos").unwrap().is_none());
    }

    /// The plain batch of the other tests, as layer data.
    fn batch_data() -> LayerData {
        LayerData::from_batch(batch())
    }

    #[test]
    fn wkb_columns_of_multi_geometries_explode_too() {
        use arrow_array::BinaryArray;
        // WKB multipolygon: two squares in the first row, one in the second
        let square = |x: f64, y: f64| {
            let mut ring = Vec::new();
            for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)] {
                ring.push([x + dx, y + dy]);
            }
            ring
        };
        let multi_polygon = |squares: Vec<Vec<[f64; 2]>>| {
            let mut bytes = vec![1u8];
            bytes.extend(6u32.to_le_bytes()); // MultiPolygon
            bytes.extend((squares.len() as u32).to_le_bytes());
            for ring in squares {
                bytes.push(1);
                bytes.extend(3u32.to_le_bytes()); // Polygon
                bytes.extend(1u32.to_le_bytes()); // one ring
                bytes.extend((ring.len() as u32).to_le_bytes());
                for [x, y] in ring {
                    bytes.extend(x.to_le_bytes());
                    bytes.extend(y.to_le_bytes());
                }
            }
            bytes
        };
        let first = multi_polygon(vec![square(0.0, 0.0), square(10.0, 0.0)]);
        let second = multi_polygon(vec![square(20.0, 0.0)]);
        let column = BinaryArray::from_vec(vec![first.as_slice(), second.as_slice()]);
        let schema = Schema::new(vec![Field::new("geometry", DataType::Binary, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(column)]).unwrap();
        let data = LayerData::from_batch(batch);
        let (exploded, parts) = explode_multi(&data, "geometry").unwrap().expect("a multi column");
        let MultiParts::Polygons(polygons) = parts else {
            panic!("expected polygons");
        };
        assert_eq!(polygons.len(), 3);
        assert_eq!(polygons[0][0][0], [0.0, 0.0, 0.0]);
        assert_eq!(polygons[1][0][0], [10.0, 0.0, 0.0]);
        assert_eq!(polygons[2][0][0], [20.0, 0.0, 0.0]);
        assert_eq!(exploded.len(), 3);
        assert_eq!(
            (0..3).map(|i| exploded.source_row(i)).collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
    }

    #[test]
    fn reads_geoarrow_polygons() {
        // List<List<FixedSizeList<f64, 2>>>: one polygon with an outer ring and a hole
        let coords = FixedSizeListBuilder::new(Float64Builder::new(), 2);
        let rings = ListBuilder::new(coords);
        let mut polygons = ListBuilder::new(rings);
        for ring in [
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0]],
        ] {
            let rings = polygons.values();
            let coords = rings.values();
            for [x, y] in ring {
                coords.values().append_value(x);
                coords.values().append_value(y);
                coords.append(true);
            }
            rings.append(true);
        }
        polygons.append(true);
        let array = polygons.finish();
        let schema = Schema::new(vec![Field::new("geometry", array.data_type().clone(), false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap();
        let data = LayerData::from_batch(batch);
        let result = resolve_polygons(&data, &Accessor::column("geometry")).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[0][0][1], [10.0, 0.0, 0.0]);
        assert_eq!(result[0][1][2], [4.0, 4.0, 0.0]);
    }
}

#[cfg(test)]
mod wkb_column_tests {
    use super::*;
    use arrow_array::{BinaryArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    fn le_point(x: f64, y: f64) -> Vec<u8> {
        let mut bytes = vec![1u8, 1, 0, 0, 0];
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes
    }

    #[test]
    fn positions_come_from_wkb_binary_columns() {
        let a = le_point(1.0, 2.0);
        let b = le_point(3.0, 4.0);
        let column = BinaryArray::from_opt_vec(vec![Some(a.as_slice()), None, Some(b.as_slice())]);
        let schema = Schema::new(vec![Field::new("geometry", DataType::Binary, true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(column)]).unwrap();
        let data = LayerData::from_batch(batch);
        let positions = resolve_positions(&data, &Accessor::column("geometry")).unwrap();
        assert_eq!(positions, vec![[1.0, 2.0, 0.0], [0.0; 3], [3.0, 4.0, 0.0]]);
        // a point is not a path
        assert!(resolve_paths(&data, &Accessor::column("geometry")).is_err());
    }
}

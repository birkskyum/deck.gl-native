//! Layer data and accessors.
//!
//! deck.gl JS layers take a `data` array plus accessor functions. The native port keeps the
//! accessor idea but makes Apache Arrow the primary data model: a layer's data is an Arrow
//! `RecordBatch`, and an [`Accessor`] is either a constant, the name of a column in that
//! batch, or a function of the row index. Column accessors read Arrow buffers directly,
//! which is the path GeoArrow data takes with no per-row callbacks.

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
#[derive(Clone, Debug, Default)]
pub struct LayerData {
    pub batch: Option<RecordBatch>,
    pub length: usize,
    /// For sub layers of a composite layer: the source row of each item, so picking and
    /// highlighting report the parent's rows. Mirrors deck.gl's `__source.index`.
    pub source_rows: Option<Arc<Vec<u32>>>,
}

/// Batches compare by column identity first, so re-sending the same data is cheap, and by
/// value otherwise.
impl PartialEq for LayerData {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
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
        }
    }

    /// Data with only a length. All accessors must be constants or functions.
    pub fn with_length(length: usize) -> Self {
        Self {
            batch: None,
            length,
            source_rows: None,
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
            Some(rows) => rows.get(index).copied().unwrap_or(index as u32),
            None => index as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The Arrow column of the layer's table, by name.
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

/// Flatten a `FixedSizeList<numeric>` array into (values, width).
fn fixed_size_list_to_f64(array: &dyn Array) -> Result<(Vec<f64>, usize)> {
    let list = array.as_fixed_size_list_opt().ok_or_else(|| {
        DeckError::Data(format!(
            "expected a FixedSizeList column, got {}",
            array.data_type()
        ))
    })?;
    let width = list.value_length() as usize;
    let values = primitive_to_f64(list.values().as_ref())?;
    // Respect the list array's offset into its child values.
    let start = list.offset() * width;
    let end = start + list.len() * width;
    Ok((values[start..end].to_vec(), width))
}

/// Resolve an accessor, reading columns through `from_column`.
pub fn resolve_with<T: Clone + Send>(
    data: &LayerData,
    accessor: &Accessor<T>,
    from_column: impl FnOnce(&ArrayRef) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    match accessor {
        Accessor::Constant(v) => Ok(vec![v.clone(); data.len()]),
        Accessor::Func(f) => Ok(resolve_function(f.as_ref(), data.len())),
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

/// Resolve positions. Columns must be `FixedSizeList<Float32|Float64>` of width 2 or 3.
pub fn resolve_positions(data: &LayerData, accessor: &Accessor<Position>) -> Result<Vec<Position>> {
    resolve_with(data, accessor, |column| {
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
    })
}

/// Rows above which function accessors are evaluated on all cores.
const PARALLEL_ROWS: usize = 16_384;

/// Evaluate a function accessor for every row, in parallel for large data.
fn resolve_function<T: Clone + Send>(f: &(dyn Fn(usize) -> T + Send + Sync), len: usize) -> Vec<T> {
    if len < PARALLEL_ROWS {
        return (0..len).map(f).collect();
    }
    use rayon::prelude::*;
    (0..len).into_par_iter().map(f).collect()
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
    resolve_with(data, accessor, |column| {
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
    resolve_with(data, accessor, polygons_from_column)
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

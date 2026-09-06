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
}

impl LayerData {
    /// Data backed by an Arrow record batch. Column accessors read from it.
    pub fn from_batch(batch: RecordBatch) -> Self {
        Self {
            length: batch.num_rows(),
            batch: Some(batch),
        }
    }

    /// Data with only a length. All accessors must be constants or functions.
    pub fn with_length(length: usize) -> Self {
        Self { batch: None, length }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn column(&self, name: &str) -> Result<&ArrayRef> {
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

fn resolve_with<T: Clone>(
    data: &LayerData,
    accessor: &Accessor<T>,
    from_column: impl FnOnce(&ArrayRef) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    match accessor {
        Accessor::Constant(v) => Ok(vec![v.clone(); data.len()]),
        Accessor::Func(f) => Ok((0..data.len()).map(|i| f(i)).collect()),
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

/// Resolve scalar floats from any numeric column.
pub fn resolve_f32(data: &LayerData, accessor: &Accessor<f32>) -> Result<Vec<f32>> {
    resolve_with(data, accessor, |column| {
        Ok(primitive_to_f64(column.as_ref())?
            .into_iter()
            .map(|v| v as f32)
            .collect())
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

/// Resolve polygons. Columns may be GeoArrow polygons
/// (`List<List<FixedSizeList<Float64, 2|3>>>`) or single rings (`List<FixedSizeList<..>>`).
pub fn resolve_polygons(data: &LayerData, accessor: &Accessor<Polygon>) -> Result<Vec<Polygon>> {
    resolve_with(data, accessor, polygons_from_column)
}

fn polygons_from_column(column: &ArrayRef) -> Result<Vec<Polygon>> {
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

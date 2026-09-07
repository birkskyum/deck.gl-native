//! Geospatial file formats as layer data: GeoParquet and plain Parquet (with the `parquet`
//! feature) and FlatGeobuf (with the `flatgeobuf` feature), plus the Well Known Binary
//! decoder GeoParquet geometries need.
//!
//! The WKB decoder lives in `deck_gl::wkb`. Files with geometries become GeoJSON features, so the `GeoJsonLayer` takes them as is and
//! every other layer sees one row per feature with `geometry` and `properties` for its `@@=`
//! accessors. Parquet files without geospatial metadata stay Arrow tables with `@@column:`
//! accessors.

use std::sync::Arc;

use arrow_array::RecordBatch;
use deck_gl::geojson::{Feature, FeatureCollection, Geometry};
#[cfg(feature = "parquet")]
use deck_gl::wkb::wkb_geometry;
use deck_gl::{Path, Polygon, Position};
use serde_json::{Map, Value};

use crate::{ConvertOptions, JsonError, Result};

/// A file format the loaders here read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoFormat {
    Parquet,
    FlatGeobuf,
}

impl GeoFormat {
    /// The format a `data` source names by its extension.
    pub fn from_source(source: &str) -> Option<Self> {
        let path = source.split(['?', '#']).next().unwrap_or(source);
        let extension = path.rsplit('.').next()?.to_ascii_lowercase();
        match extension.as_str() {
            "parquet" | "geoparquet" | "pq" => Some(Self::Parquet),
            "fgb" => Some(Self::FlatGeobuf),
            _ => None,
        }
    }
}

/// What a geospatial file turned into.
#[derive(Clone, Debug)]
pub enum GeoData {
    /// A Parquet file without geometries
    Table(RecordBatch),
    /// The features of a GeoParquet or FlatGeobuf file
    Features(Arc<FeatureCollection>),
}

/// Load a `data` source in one of the [`GeoFormat`]s.
pub fn load_geo(source: &str, format: GeoFormat, options: &ConvertOptions) -> Result<GeoData> {
    let bytes = crate::data::load_bytes(source, options)?;
    let decode_error = |message: String| JsonError::Load {
        url: source.to_string(),
        message,
    };
    match format {
        GeoFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                let batch = read_parquet(bytes).map_err(decode_error)?;
                match geoparquet_features(&batch).map_err(decode_error)? {
                    Some(collection) => Ok(GeoData::Features(Arc::new(collection))),
                    None => Ok(GeoData::Table(batch)),
                }
            }
            #[cfg(not(feature = "parquet"))]
            {
                let _ = bytes;
                Err(decode_error("built without the `parquet` feature".to_string()))
            }
        }
        GeoFormat::FlatGeobuf => {
            #[cfg(feature = "flatgeobuf")]
            {
                let collection = read_flatgeobuf(&bytes).map_err(decode_error)?;
                Ok(GeoData::Features(Arc::new(collection)))
            }
            #[cfg(not(feature = "flatgeobuf"))]
            {
                let _ = bytes;
                Err(decode_error("built without the `flatgeobuf` feature".to_string()))
            }
        }
    }
}

/// Read every row group of a Parquet file into one Arrow record batch.
#[cfg(feature = "parquet")]
pub fn read_parquet(bytes: Vec<u8>) -> std::result::Result<RecordBatch, String> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes)).map_err(|e| e.to_string())?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|e| e.to_string())?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    arrow_select::concat::concat_batches(&schema, &batches).map_err(|e| e.to_string())
}

/// The GeoParquet metadata of a schema: the primary geometry column and its encoding.
#[cfg(feature = "parquet")]
pub fn geoparquet_column(schema: &arrow_schema::Schema) -> Option<(String, String)> {
    let geo: Value = serde_json::from_str(schema.metadata().get("geo")?).ok()?;
    let primary = geo
        .get("primary_column")
        .and_then(Value::as_str)
        .unwrap_or("geometry");
    let encoding = geo
        .get("columns")
        .and_then(|c| c.get(primary))
        .and_then(|c| c.get("encoding"))
        .and_then(Value::as_str)
        .unwrap_or("WKB");
    Some((primary.to_string(), encoding.to_string()))
}

/// The features of a GeoParquet batch, or `None` when the batch has no geospatial metadata.
#[cfg(feature = "parquet")]
pub fn geoparquet_features(batch: &RecordBatch) -> std::result::Result<Option<FeatureCollection>, String> {
    use arrow_array::cast::AsArray;
    let Some((column, encoding)) = geoparquet_column(batch.schema_ref()) else {
        return Ok(None);
    };
    if encoding != "WKB" {
        return Err(format!(
            "GeoParquet geometry encoding `{encoding}` is not supported, only WKB"
        ));
    }
    let index = batch
        .schema_ref()
        .index_of(&column)
        .map_err(|_| format!("GeoParquet geometry column `{column}` is missing"))?;
    let geometry = batch.column(index);
    let wkb_at = |row: usize| -> std::result::Result<Option<Geometry>, String> {
        if geometry.is_null(row) {
            return Ok(None);
        }
        let bytes: &[u8] = if let Some(array) = geometry.as_binary_opt::<i32>() {
            array.value(row)
        } else if let Some(array) = geometry.as_binary_opt::<i64>() {
            array.value(row)
        } else if let Some(array) = geometry.as_binary_view_opt() {
            array.value(row)
        } else {
            return Err(format!(
                "GeoParquet geometry column `{column}` is {}, not binary",
                geometry.data_type()
            ));
        };
        wkb_geometry(bytes).map(Some)
    };
    let mut features = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut properties = Map::new();
        for (i, field) in batch.schema_ref().fields().iter().enumerate() {
            if i == index {
                continue;
            }
            if let Some(value) = column_value(batch.column(i), row) {
                properties.insert(field.name().clone(), value);
            }
        }
        features.push(Feature {
            id: None,
            geometry: wkb_at(row)?,
            properties,
        });
    }
    Ok(Some(FeatureCollection { features }))
}

/// One cell of a column as JSON, for the common scalar types; `None` for others and nulls.
#[cfg(feature = "parquet")]
fn column_value(column: &arrow_array::ArrayRef, row: usize) -> Option<Value> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::*;
    use arrow_schema::DataType;
    if column.is_null(row) {
        return None;
    }
    let number = |v: f64| serde_json::Number::from_f64(v).map(Value::Number);
    match column.data_type() {
        DataType::Utf8 => Some(Value::String(column.as_string::<i32>().value(row).to_string())),
        DataType::LargeUtf8 => Some(Value::String(column.as_string::<i64>().value(row).to_string())),
        DataType::Utf8View => Some(Value::String(column.as_string_view().value(row).to_string())),
        DataType::Boolean => Some(Value::Bool(column.as_boolean().value(row))),
        DataType::Int8 => Some(Value::from(column.as_primitive::<Int8Type>().value(row))),
        DataType::Int16 => Some(Value::from(column.as_primitive::<Int16Type>().value(row))),
        DataType::Int32 => Some(Value::from(column.as_primitive::<Int32Type>().value(row))),
        DataType::Int64 => Some(Value::from(column.as_primitive::<Int64Type>().value(row))),
        DataType::UInt8 => Some(Value::from(column.as_primitive::<UInt8Type>().value(row))),
        DataType::UInt16 => Some(Value::from(column.as_primitive::<UInt16Type>().value(row))),
        DataType::UInt32 => Some(Value::from(column.as_primitive::<UInt32Type>().value(row))),
        DataType::UInt64 => Some(Value::from(column.as_primitive::<UInt64Type>().value(row))),
        DataType::Float32 => number(column.as_primitive::<Float32Type>().value(row) as f64),
        DataType::Float64 => number(column.as_primitive::<Float64Type>().value(row)),
        _ => None,
    }
}

/// Read a FlatGeobuf file into features.
#[cfg(feature = "flatgeobuf")]
pub fn read_flatgeobuf(bytes: &[u8]) -> std::result::Result<FeatureCollection, String> {
    use flatgeobuf::FgbReader;
    use geozero::ProcessToJson;
    let reader = FgbReader::open(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut features = reader.select_all().map_err(|e| e.to_string())?;
    let json = features.to_json().map_err(|e| e.to_string())?;
    FeatureCollection::parse(&json).map_err(|e| e.to_string())
}

/// A feature as the GeoJSON object the `@@=` accessors of a row see.
pub fn feature_to_value(feature: &Feature) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String("Feature".to_string()));
    if let Some(id) = &feature.id {
        object.insert("id".to_string(), id.clone());
    }
    object.insert(
        "geometry".to_string(),
        feature.geometry.as_ref().map_or(Value::Null, geometry_to_value),
    );
    object.insert(
        "properties".to_string(),
        Value::Object(feature.properties.clone()),
    );
    Value::Object(object)
}

/// A geometry as its GeoJSON object.
pub fn geometry_to_value(geometry: &Geometry) -> Value {
    fn position(p: &Position) -> Value {
        if p[2] == 0.0 {
            Value::Array(vec![Value::from(p[0]), Value::from(p[1])])
        } else {
            Value::Array(vec![Value::from(p[0]), Value::from(p[1]), Value::from(p[2])])
        }
    }
    fn path(path: &Path) -> Value {
        Value::Array(path.iter().map(position).collect())
    }
    fn polygon(polygon: &Polygon) -> Value {
        Value::Array(polygon.iter().map(path).collect())
    }
    let (kind, coordinates) = match geometry {
        Geometry::Point(p) => ("Point", position(p)),
        Geometry::MultiPoint(points) => ("MultiPoint", path(points)),
        Geometry::LineString(line) => ("LineString", path(line)),
        Geometry::MultiLineString(lines) => {
            ("MultiLineString", Value::Array(lines.iter().map(path).collect()))
        }
        Geometry::Polygon(rings) => ("Polygon", polygon(rings)),
        Geometry::MultiPolygon(polygons) => (
            "MultiPolygon",
            Value::Array(polygons.iter().map(polygon).collect()),
        ),
        Geometry::GeometryCollection(members) => {
            let geometries = Value::Array(members.iter().map(geometry_to_value).collect());
            return serde_json::json!({"type": "GeometryCollection", "geometries": geometries});
        }
    };
    serde_json::json!({"type": kind, "coordinates": coordinates})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_come_from_extensions() {
        assert_eq!(
            GeoFormat::from_source("data/roads.parquet"),
            Some(GeoFormat::Parquet)
        );
        assert_eq!(
            GeoFormat::from_source("https://x/y.fgb?token=1"),
            Some(GeoFormat::FlatGeobuf)
        );
        assert_eq!(GeoFormat::from_source("points.geojson"), None);
    }
}

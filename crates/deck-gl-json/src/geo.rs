//! Geospatial file formats as layer data: GeoParquet and plain Parquet (with the `parquet`
//! feature) and FlatGeobuf (with the `flatgeobuf` feature), plus the Well Known Binary
//! decoder GeoParquet geometries need.
//!
//! Files with geometries become GeoJSON features, so the `GeoJsonLayer` takes them as is and
//! every other layer sees one row per feature with `geometry` and `properties` for its `@@=`
//! accessors. Parquet files without geospatial metadata stay Arrow tables with `@@column:`
//! accessors.

use std::sync::Arc;

use arrow_array::RecordBatch;
use deck_gl::geojson::{Feature, FeatureCollection, Geometry};
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

/// Decode a Well Known Binary geometry (ISO and PostGIS EWKB flavours, with or without Z, M
/// and an SRID).
pub fn wkb_geometry(bytes: &[u8]) -> std::result::Result<Geometry, String> {
    let mut reader = WkbReader { bytes, pos: 0 };
    let geometry = reader.geometry()?;
    Ok(geometry)
}

struct WkbReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl WkbReader<'_> {
    fn take(&mut self, n: usize) -> std::result::Result<&[u8], String> {
        let end = self.pos + n;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| format!("WKB ends after {} of {} bytes", self.bytes.len(), end))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> std::result::Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, little_endian: bool) -> std::result::Result<u32, String> {
        let b = self.take(4)?;
        let array = [b[0], b[1], b[2], b[3]];
        Ok(if little_endian {
            u32::from_le_bytes(array)
        } else {
            u32::from_be_bytes(array)
        })
    }

    fn f64(&mut self, little_endian: bool) -> std::result::Result<f64, String> {
        let b = self.take(8)?;
        let array = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Ok(if little_endian {
            f64::from_le_bytes(array)
        } else {
            f64::from_be_bytes(array)
        })
    }

    fn geometry(&mut self) -> std::result::Result<Geometry, String> {
        let little_endian = match self.u8()? {
            0 => false,
            1 => true,
            other => return Err(format!("WKB byte order {other} is not 0 or 1")),
        };
        let mut kind = self.u32(little_endian)?;
        let mut has_z = false;
        let mut has_m = false;
        // PostGIS EWKB flags
        if kind & 0x8000_0000 != 0 {
            has_z = true;
            kind &= !0x8000_0000;
        }
        if kind & 0x4000_0000 != 0 {
            has_m = true;
            kind &= !0x4000_0000;
        }
        if kind & 0x2000_0000 != 0 {
            kind &= !0x2000_0000;
            self.u32(little_endian)?; // SRID
        }
        // ISO WKB dimension offsets
        match kind / 1000 {
            1 => has_z = true,
            2 => has_m = true,
            3 => {
                has_z = true;
                has_m = true;
            }
            _ => {}
        }
        kind %= 1000;
        let position = |reader: &mut Self| -> std::result::Result<Position, String> {
            let x = reader.f64(little_endian)?;
            let y = reader.f64(little_endian)?;
            let z = if has_z { reader.f64(little_endian)? } else { 0.0 };
            if has_m {
                reader.f64(little_endian)?;
            }
            Ok([x, y, z])
        };
        let path = |reader: &mut Self| -> std::result::Result<Path, String> {
            let count = reader.u32(little_endian)? as usize;
            (0..count).map(|_| position(reader)).collect()
        };
        let polygon = |reader: &mut Self| -> std::result::Result<Polygon, String> {
            let rings = reader.u32(little_endian)? as usize;
            (0..rings).map(|_| path(reader)).collect()
        };
        let members = |reader: &mut Self| -> std::result::Result<Vec<Geometry>, String> {
            let count = reader.u32(little_endian)? as usize;
            (0..count).map(|_| reader.geometry()).collect()
        };
        Ok(match kind {
            1 => Geometry::Point(position(self)?),
            2 => Geometry::LineString(path(self)?),
            3 => Geometry::Polygon(polygon(self)?),
            4 => Geometry::MultiPoint(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::Point(p) => Ok(p),
                        other => Err(format!("WKB MultiPoint holds a {}", geometry_name(&other))),
                    })
                    .collect::<std::result::Result<_, _>>()?,
            ),
            5 => Geometry::MultiLineString(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::LineString(l) => Ok(l),
                        other => Err(format!("WKB MultiLineString holds a {}", geometry_name(&other))),
                    })
                    .collect::<std::result::Result<_, _>>()?,
            ),
            6 => Geometry::MultiPolygon(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::Polygon(p) => Ok(p),
                        other => Err(format!("WKB MultiPolygon holds a {}", geometry_name(&other))),
                    })
                    .collect::<std::result::Result<_, _>>()?,
            ),
            7 => Geometry::GeometryCollection(members(self)?),
            other => return Err(format!("WKB geometry type {other} is not supported")),
        })
    }
}

fn geometry_name(geometry: &Geometry) -> &'static str {
    match geometry {
        Geometry::Point(_) => "Point",
        Geometry::MultiPoint(_) => "MultiPoint",
        Geometry::LineString(_) => "LineString",
        Geometry::MultiLineString(_) => "MultiLineString",
        Geometry::Polygon(_) => "Polygon",
        Geometry::MultiPolygon(_) => "MultiPolygon",
        Geometry::GeometryCollection(_) => "GeometryCollection",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_point(x: f64, y: f64) -> Vec<u8> {
        let mut bytes = vec![1u8, 1, 0, 0, 0];
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes
    }

    #[test]
    fn decodes_wkb_points_polygons_and_ewkb_flags() {
        assert_eq!(
            wkb_geometry(&le_point(1.5, -2.0)).unwrap(),
            Geometry::Point([1.5, -2.0, 0.0])
        );
        // big endian polygon with one ring of four points
        let mut bytes = vec![0u8, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4];
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)] {
            bytes.extend_from_slice(&f64::to_be_bytes(x));
            bytes.extend_from_slice(&f64::to_be_bytes(y));
        }
        match wkb_geometry(&bytes).unwrap() {
            Geometry::Polygon(rings) => assert_eq!(rings[0][2], [1.0, 1.0, 0.0]),
            other => panic!("{other:?}"),
        }
        // EWKB point with Z and an SRID
        let mut ewkb = vec![1u8];
        ewkb.extend_from_slice(&(1u32 | 0x8000_0000 | 0x2000_0000).to_le_bytes());
        ewkb.extend_from_slice(&4326u32.to_le_bytes());
        for v in [3.0f64, 4.0, 5.0] {
            ewkb.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(wkb_geometry(&ewkb).unwrap(), Geometry::Point([3.0, 4.0, 5.0]));
        // ISO WKB multipoint with Z (type 1004)
        let mut iso = vec![1u8];
        iso.extend_from_slice(&1004u32.to_le_bytes());
        iso.extend_from_slice(&1u32.to_le_bytes());
        iso.push(1);
        iso.extend_from_slice(&1001u32.to_le_bytes());
        for v in [1.0f64, 2.0, 3.0] {
            iso.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            wkb_geometry(&iso).unwrap(),
            Geometry::MultiPoint(vec![[1.0, 2.0, 3.0]])
        );
        assert!(wkb_geometry(&[1, 9, 0, 0, 0]).unwrap_err().contains("type 9"));
        assert!(wkb_geometry(&le_point(1.0, 2.0)[..10])
            .unwrap_err()
            .contains("ends"));
    }

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

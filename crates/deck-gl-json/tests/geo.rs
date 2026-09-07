//! GeoParquet, Parquet and FlatGeobuf files as layer data.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float64Builder};
use arrow_array::{Array, BinaryArray, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use deck_gl::data::{resolve_paths, resolve_positions};
use deck_gl::geojson::Geometry;
use deck_gl_json::geo::{geoparquet_features, read_parquet};
use deck_gl_json::JsonConverter;
use deck_gl_layers::{GeoJsonLayer, PathLayer, ScatterplotLayer};
use serde_json::json;

fn temp_file(name: &str, bytes: &[u8]) -> String {
    let path = std::env::temp_dir().join(format!("deck-gl-native-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut bytes = vec![1u8, 1, 0, 0, 0];
    bytes.extend_from_slice(&x.to_le_bytes());
    bytes.extend_from_slice(&y.to_le_bytes());
    bytes
}

fn wkb_polygon(ring: &[(f64, f64)]) -> Vec<u8> {
    let mut bytes = vec![1u8, 3, 0, 0, 0, 1, 0, 0, 0];
    bytes.extend_from_slice(&(ring.len() as u32).to_le_bytes());
    for (x, y) in ring {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
    }
    bytes
}

fn parquet_bytes(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut out, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    out
}

#[test]
fn geoparquet_features_feed_geojson_and_row_layers() {
    let geo = json!({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point", "Polygon"]}}
    });
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("geometry", DataType::Binary, true),
        ],
        HashMap::from([("geo".to_string(), geo.to_string())]),
    ));
    let square = wkb_polygon(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]);
    let first = wkb_point(10.0, 20.0);
    let second = wkb_point(30.0, 40.0);
    let geometries: Vec<&[u8]> = vec![&first, &second, &square];
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Float64Array::from(vec![1.0, 2.5, 3.0])),
            Arc::new(BinaryArray::from_vec(geometries)),
        ],
    )
    .unwrap();
    let bytes = parquet_bytes(&batch);
    let path = temp_file("features.parquet", &bytes);

    // the reader decodes every geometry and keeps the other columns as properties
    let decoded = read_parquet(bytes).unwrap();
    let collection = geoparquet_features(&decoded).unwrap().expect("geo metadata");
    assert_eq!(collection.features.len(), 3);
    assert_eq!(collection.features[2].properties["name"], "c");
    assert!(
        matches!(collection.features[2].geometry, Some(Geometry::Polygon(ref rings)) if rings[0].len() == 4)
    );

    // a GeoJSON layer takes the file as is; a point layer reads the first two point rows
    let mut warnings = Vec::new();
    let mut layers = JsonConverter::new()
        .convert_layers(
            &json!([
                {"@@type": "GeoJsonLayer", "id": "geo", "data": path},
                {
                    "@@type": "ScatterplotLayer", "id": "points", "data": path,
                    "getPosition": "@@=geometry.type == 'Point' ? geometry.coordinates : [0, 0]",
                    "getRadius": "@@=properties.value"
                }
            ]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(layers[0].as_any_mut().downcast_mut::<GeoJsonLayer>().is_some());
    let points = layers[1].as_any_mut().downcast_mut::<ScatterplotLayer>().unwrap();
    let props = points.props();
    assert_eq!(props.data.len(), 3);
    let positions = resolve_positions(&props.data, &props.get_position).unwrap();
    assert_eq!(positions[1], [30.0, 40.0, 0.0]);
    assert_eq!(positions[2], [0.0, 0.0, 0.0]);
}

#[test]
fn plain_parquet_is_an_arrow_table() {
    let mut positions = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    for (x, y) in [(1.0, 2.0), (3.0, 4.0)] {
        positions.values().append_value(x);
        positions.values().append_value(y);
        positions.append(true);
    }
    let positions = positions.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("position", positions.data_type().clone(), false),
        Field::new("size", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(positions), Arc::new(Float64Array::from(vec![5.0, 6.0]))],
    )
    .unwrap();
    let path = temp_file("table.parquet", &parquet_bytes(&batch));
    let mut warnings = Vec::new();
    let mut layers = JsonConverter::new()
        .convert_layers(
            &json!([{
                "@@type": "ScatterplotLayer", "id": "table", "data": path,
                "getPosition": "@@column:position", "getRadius": "@@column:size"
            }]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let points = layers[0].as_any_mut().downcast_mut::<ScatterplotLayer>().unwrap();
    let props = points.props();
    assert_eq!(props.data.len(), 2);
    assert_eq!(
        resolve_positions(&props.data, &props.get_position).unwrap()[1],
        [3.0, 4.0, 0.0]
    );
    // a GeoJSON layer cannot use a table without geometries
    let error = match JsonConverter::new()
        .convert_layers(&json!([{"@@type": "GeoJsonLayer", "data": path}]), &mut warnings)
    {
        Ok(_) => panic!("a table without geometries was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("no GeoParquet geometry column"), "{error}");
}

#[test]
fn flatgeobuf_features_feed_geojson_and_path_layers() {
    use flatgeobuf::{FgbWriter, GeometryType};
    use geozero::geojson::GeoJsonReader;
    use geozero::GeozeroDatasource;
    let collection = json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "road", "lanes": 2},
             "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]}},
            {"type": "Feature", "properties": {"name": "track", "lanes": 1},
             "geometry": {"type": "LineString", "coordinates": [[5.0, 5.0], [6.0, 6.0]]}}
        ]
    })
    .to_string();
    let mut fgb = FgbWriter::create("roads", GeometryType::LineString).unwrap();
    GeoJsonReader(collection.as_bytes()).process(&mut fgb).unwrap();
    let mut bytes = Vec::new();
    fgb.write(&mut bytes).unwrap();
    let path = temp_file("roads.fgb", &bytes);

    let mut warnings = Vec::new();
    let mut layers = JsonConverter::new()
        .convert_layers(
            &json!([
                {"@@type": "GeoJsonLayer", "id": "geo", "data": path},
                {
                    "@@type": "PathLayer", "id": "roads", "data": path,
                    "getPath": "@@=geometry.coordinates", "getWidth": "@@=properties.lanes"
                }
            ]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(layers[0].as_any_mut().downcast_mut::<GeoJsonLayer>().is_some());
    let roads = layers[1].as_any_mut().downcast_mut::<PathLayer>().unwrap();
    let props = roads.props();
    assert_eq!(props.data.len(), 2);
    // the file's spatial index orders features by location, so compare as sets
    let mut paths = resolve_paths(&props.data, &props.get_path).unwrap();
    paths.sort_by_key(|p| p.len());
    assert_eq!(paths[0], vec![[5.0, 5.0, 0.0], [6.0, 6.0, 0.0]]);
    assert_eq!(paths[1].len(), 3);
}

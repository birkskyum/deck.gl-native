//! Conversion tests that need no GPU.

use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32, resolve_polygons, resolve_positions};
use deck_gl::{Accessor, CoordinateSystem, LayerData};
use deck_gl_json::props::{convert, Props};
use deck_gl_json::{JsonConverter, JsonError};
use serde_json::{json, Value};

fn props_with_rows(object: &Value, rows: Vec<Value>) -> (Props<'_>, LayerData) {
    let length = rows.len();
    let mut props = Props::new("TestLayer", object.as_object().unwrap());
    props.set_rows(Arc::new(rows));
    (props, LayerData::with_length(length))
}

#[test]
fn converts_a_description_with_view_state_and_layers() {
    let spec = json!({
        "initialViewState": {"longitude": -122.4, "latitude": 37.8, "zoom": 12, "pitch": 30},
        "layers": [
            {
                "@@type": "ScatterplotLayer",
                "id": "points",
                "data": [{"position": [1, 2]}],
                "getPosition": "@@=position"
            },
            null,
            [{
                "@@type": "LineLayer",
                "id": "lines",
                "data": [],
                "getSourcePosition": "@@=start",
                "getTargetPosition": "@@=end"
            }]
        ]
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    let view = deck.view_state.unwrap();
    assert_eq!(
        (view.longitude, view.latitude, view.zoom, view.pitch, view.bearing),
        (-122.4, 37.8, 12.0, 30.0, 0.0)
    );
    let ids: Vec<&str> = deck.layers.iter().map(|layer| layer.id()).collect();
    assert_eq!(ids, ["points", "lines"]);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
}

#[test]
fn a_bare_layer_array_has_no_view_state() {
    let deck = JsonConverter::new()
        .parse(r#"[{"@@type": "ScatterplotLayer", "data": []}]"#)
        .unwrap();
    assert!(deck.view_state.is_none());
    assert_eq!(deck.layers[0].id(), "ScatterplotLayer");
}

#[test]
fn accessors_evaluate_expressions_per_row() {
    let object = json!({
        "getPosition": "@@=[lng, lat]",
        "getRadius": "@@=size * 2",
        "getFillColor": "@@=size > 5 ? [255, 0, 0] : [0, 0, 255, 128]",
        "getLineColor": [1, 2, 3],
        "getLineWidth": 4
    });
    let rows = vec![
        json!({"lng": 1.0, "lat": 2.0, "size": 3}),
        json!({"lng": 4.0, "lat": 5.0, "size": 10}),
    ];
    let (props, data) = props_with_rows(&object, rows);

    let positions = props
        .accessor("getPosition", "position", convert::position)
        .unwrap();
    assert_eq!(
        resolve_positions(&data, &positions).unwrap(),
        [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]
    );
    let radius = props.accessor("getRadius", "-", convert::f32).unwrap();
    assert_eq!(resolve_f32(&data, &radius).unwrap(), [6.0, 20.0]);
    let fill = props.accessor("getFillColor", "-", convert::color).unwrap();
    assert_eq!(
        resolve_colors(&data, &fill).unwrap(),
        [[0, 0, 255, 128], [255, 0, 0, 255]]
    );
    let line = props.accessor("getLineColor", "-", convert::color).unwrap();
    assert_eq!(
        resolve_colors(&data, &line).unwrap(),
        [[1, 2, 3, 255], [1, 2, 3, 255]]
    );
    let width = props.accessor("getLineWidth", "-", convert::f32).unwrap();
    assert_eq!(resolve_f32(&data, &width).unwrap(), [4.0, 4.0]);
}

#[test]
fn default_accessors_read_the_deck_gl_field_names() {
    let object = json!({});
    let rows = vec![json!({"sourcePosition": [0, 0], "targetPosition": [1, 1, 5]})];
    let (props, data) = props_with_rows(&object, rows);
    let source = props
        .accessor("getSourcePosition", "sourcePosition", convert::position)
        .unwrap();
    assert_eq!(resolve_positions(&data, &source).unwrap(), [[0.0, 0.0, 0.0]]);
    let target = props
        .accessor("getTargetPosition", "targetPosition", convert::position)
        .unwrap();
    assert_eq!(resolve_positions(&data, &target).unwrap(), [[1.0, 1.0, 5.0]]);
    let width = props
        .accessor("getWidth", &Accessor::Constant(2.5f32), convert::f32)
        .unwrap();
    assert_eq!(resolve_f32(&data, &width).unwrap(), [2.5]);
}

#[test]
fn polygons_accept_a_ring_or_rings() {
    let object = json!({"getPolygon": "@@=polygon"});
    let rows = vec![
        json!({"polygon": [[0, 0], [1, 0], [1, 1]]}),
        json!({"polygon": [[[0, 0], [2, 0], [2, 2]], [[0.5, 0.5], [1, 0.5], [1, 1]]]}),
    ];
    let (props, data) = props_with_rows(&object, rows);
    let accessor = props.accessor("getPolygon", "polygon", convert::polygon).unwrap();
    let polygons = resolve_polygons(&data, &accessor).unwrap();
    assert_eq!(polygons[0].len(), 1);
    assert_eq!(polygons[0][0].len(), 3);
    assert_eq!(polygons[1].len(), 2);
    assert_eq!(polygons[1][1][0], [0.5, 0.5, 0.0]);
}

#[test]
fn reports_bad_expressions_with_the_prop_name() {
    let spec = json!([{
        "@@type": "ScatterplotLayer",
        "id": "bad",
        "data": [{"x": 1}],
        "getPosition": "@@=Math.max(x)"
    }]);
    match JsonConverter::new().convert(&spec).unwrap_err() {
        JsonError::Prop { layer, prop, message } => {
            assert_eq!(layer, "bad");
            assert_eq!(prop, "getPosition");
            assert!(message.contains("function calls"), "{message}");
        }
        other => panic!("unexpected error {other}"),
    }
}

#[test]
fn reports_row_conversion_errors() {
    let spec = json!([{
        "@@type": "ScatterplotLayer",
        "id": "rows",
        "data": [{"position": [1, 2]}, {"position": "nope"}],
        "getPosition": "@@=position"
    }]);
    let error = JsonConverter::new().convert(&spec).unwrap_err().to_string();
    assert!(error.contains("row 1"), "{error}");
    assert!(error.contains("getPosition"), "{error}");
}

#[test]
fn warns_about_unknown_layers_and_props() {
    let spec = json!([
        {"@@type": "HeatmapLayer", "id": "heat"},
        {"@@type": "ScatterplotLayer", "id": "p", "data": [], "bogusProp": 1, "onHover": "ignored"}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert_eq!(deck.warnings.len(), 2, "{:?}", deck.warnings);
    assert!(deck.warnings[0].contains("HeatmapLayer"));
    assert!(deck.warnings[1].contains("bogusProp"));
}

#[test]
fn base_props_and_enumerations() {
    let spec = json!([{
        "@@type": "ScatterplotLayer",
        "id": "p",
        "data": [],
        "opacity": 0.5,
        "pickable": true,
        "visible": false,
        "coordinateSystem": "@@#COORDINATE_SYSTEM.METER_OFFSETS",
        "coordinateOrigin": [-122.4, 37.8],
        "highlightColor": [1, 2, 3, 4]
    }]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    let props = deck.layers[0].props();
    assert_eq!(props.opacity, 0.5);
    assert!(props.pickable);
    assert!(!props.visible);
    assert_eq!(props.coordinate_system, CoordinateSystem::MeterOffsets);
    assert_eq!(props.coordinate_origin, [-122.4, 37.8, 0.0]);
    assert_eq!(props.highlight_color, [1, 2, 3, 4]);

    let numeric = json!([{"@@type": "ScatterplotLayer", "data": [], "coordinateSystem": 1}]);
    let deck = JsonConverter::new().convert(&numeric).unwrap();
    assert_eq!(deck.layers[0].props().coordinate_system, CoordinateSystem::LngLat);
}

#[test]
fn geojson_features_are_the_accessor_rows() {
    let collection = json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"height": 12}, "geometry": {"type": "Point", "coordinates": [1, 2]}},
            {"type": "Feature", "properties": {"height": 30}, "geometry": {"type": "Point", "coordinates": [3, 4]}}
        ]
    });
    let spec = json!({"layers": [{
        "@@type": "GeoJsonLayer",
        "id": "g",
        "data": collection,
        "getElevation": "@@=properties.height"
    }]});
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);

    // Other layers can read GeoJSON too: the features become the rows.
    let object = json!({"data": collection, "getPosition": "@@=geometry.coordinates"});
    let rows = deck_gl_json::data::rows_from_value(std::borrow::Cow::Borrowed(&object["data"])).unwrap();
    let mut props = Props::new("ScatterplotLayer", object.as_object().unwrap());
    props.set_rows(rows);
    let accessor = props
        .accessor("getPosition", "position", convert::position)
        .unwrap();
    assert_eq!(
        resolve_positions(&LayerData::with_length(2), &accessor).unwrap(),
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
    );
}

#[test]
fn loads_data_and_images_relative_to_the_spec_file() {
    let dir = std::env::temp_dir().join(format!("deck-gl-json-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("points.json"),
        r#"[{"position": [1, 2]}, {"position": [3, 4]}]"#,
    )
    .unwrap();
    let mut png = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut png),
        &[
            255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ],
        2,
        2,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();
    std::fs::write(dir.join("tile.png"), png).unwrap();
    std::fs::write(
        dir.join("spec.json"),
        r#"{"layers": [
            {"@@type": "ScatterplotLayer", "id": "points", "data": "points.json"},
            {"@@type": "BitmapLayer", "id": "tile", "image": "tile.png", "bounds": [[0, 0], [0, 1], [1, 1], [1, 0]]}
        ]}"#,
    )
    .unwrap();

    let deck = JsonConverter::parse_file(dir.join("spec.json")).unwrap();
    assert_eq!(deck.layers.len(), 2);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);

    let missing = JsonConverter::parse_file(dir.join("missing.json")).unwrap_err();
    assert!(matches!(missing, JsonError::Load { .. }));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn text_layer_props_and_font_settings() {
    let spec = json!([{
        "@@type": "TextLayer",
        "id": "labels",
        "data": [{"name": "Park", "coordinates": [-122.5, 37.77]}],
        "getText": "@@=name",
        "getPosition": "@@=coordinates",
        "getSize": 24,
        "getTextAnchor": "start",
        "getAlignmentBaseline": "@@=name == 'Park' ? 'top' : 'bottom'",
        "background": true,
        "backgroundPadding": [4, 2],
        "backgroundBorderRadius": 3,
        "characterSet": "auto",
        "fontFamily": "Monaco, monospace",
        "fontWeight": "bold",
        "fontSettings": {"sdf": true, "fontSize": 48, "buffer": 6},
        "outlineWidth": 2,
        "wordBreak": "break-all",
        "maxWidth": 12
    }]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);

    let bad = json!([{"@@type": "TextLayer", "data": [], "getTextAnchor": "left"}]);
    let error = JsonConverter::new().convert(&bad).unwrap_err().to_string();
    assert!(
        error.contains("getTextAnchor") && error.contains("left"),
        "{error}"
    );
}

#[test]
fn aggregation_layers_props() {
    let spec = json!([
        {
            "@@type": "HexagonLayer",
            "id": "hex",
            "data": [{"coordinates": [-122.4, 37.8], "w": 3}, {"coordinates": [-122.41, 37.81], "w": 1}],
            "getPosition": "@@=coordinates",
            "getColorWeight": "@@=w",
            "colorAggregation": "MEAN",
            "colorScaleType": "quantile",
            "radius": 500,
            "extruded": true,
            "elevationRange": [0, 2000],
            "colorRange": [[0, 0, 0], [255, 255, 255]],
            "upperPercentile": 90,
            "gpuAggregation": true
        },
        {"@@type": "GridLayer", "id": "grid", "data": [], "cellSize": 200, "colorDomain": [0, 10]},
        {"@@type": "GridCellLayer", "id": "cells", "data": [{"position": [0, 0], "h": 5}], "cellSize": 100, "getElevation": "@@=h * 10"}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 3);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let bad = json!([{"@@type": "HexagonLayer", "data": [], "colorAggregation": "MEDIAN"}]);
    let error = JsonConverter::new().convert(&bad).unwrap_err().to_string();
    assert!(
        error.contains("colorAggregation") && error.contains("MEDIAN"),
        "{error}"
    );
}

fn points_table() -> arrow_array::RecordBatch {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Float64Builder, UInt8Builder};
    use arrow_array::Array;
    use arrow_schema::{DataType, Field, Schema};
    let mut positions = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    let mut colors = FixedSizeListBuilder::new(UInt8Builder::new(), 4);
    let mut radius = Float32Builder::new();
    for (i, (lng, lat)) in [(-122.4, 37.8), (-122.41, 37.79), (-122.39, 37.81)]
        .iter()
        .enumerate()
    {
        positions.values().append_value(*lng);
        positions.values().append_value(*lat);
        positions.append(true);
        for c in [255, 0, i as u8 * 100, 255] {
            colors.values().append_value(c);
        }
        colors.append(true);
        radius.append_value(10.0 * (i + 1) as f32);
    }
    let positions = positions.finish();
    let colors = colors.finish();
    let radius = radius.finish();
    let schema = Schema::new(vec![
        Field::new("position", positions.data_type().clone(), false),
        Field::new("color", colors.data_type().clone(), false),
        Field::new("radius", DataType::Float32, false),
    ]);
    arrow_array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(positions), Arc::new(colors), Arc::new(radius)],
    )
    .unwrap()
}

#[test]
fn arrow_tables_feed_layers_through_column_accessors() {
    let converter = JsonConverter::new().with_table("points", points_table());
    let spec = json!([{
        "@@type": "ScatterplotLayer",
        "id": "points",
        "data": "@@table:points",
        "getPosition": "@@column:position",
        "getFillColor": "@@=color",
        "getRadius": "@@column:radius",
        "getLineColor": [0, 0, 0]
    }]);
    let deck = converter.convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);

    // The accessors resolve straight from the columns
    let mut props = Props::new("ScatterplotLayer", spec[0].as_object().unwrap());
    props.set_table();
    let data = LayerData::from_batch(points_table());
    let positions = props
        .accessor("getPosition", "position", convert::position)
        .unwrap();
    assert_eq!(
        resolve_positions(&data, &positions).unwrap()[2],
        [-122.39, 37.81, 0.0]
    );
    let colors = props.accessor("getFillColor", "-", convert::color).unwrap();
    assert_eq!(resolve_colors(&data, &colors).unwrap()[1], [255, 0, 100, 255]);
    let radius = props.accessor("getRadius", "-", convert::f32).unwrap();
    assert_eq!(resolve_f32(&data, &radius).unwrap(), [10.0, 20.0, 30.0]);

    let unknown = json!([{"@@type": "ScatterplotLayer", "data": "@@table:missing"}]);
    let error = converter.convert(&unknown).unwrap_err().to_string();
    assert!(error.contains("missing"), "{error}");

    let expression =
        json!([{"@@type": "ScatterplotLayer", "data": "@@table:points", "getRadius": "@@=radius * 2"}]);
    let error = converter.convert(&expression).unwrap_err().to_string();
    assert!(error.contains("column name"), "{error}");

    let geojson = json!([{"@@type": "GeoJsonLayer", "data": "@@table:points"}]);
    let error = converter.convert(&geojson).unwrap_err().to_string();
    assert!(error.contains("GeoJSON"), "{error}");
}

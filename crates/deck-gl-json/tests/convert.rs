//! Conversion tests that need no GPU.

use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32, resolve_polygons, resolve_positions};
use deck_gl::wgpu;
use deck_gl::{
    Accessor, AnyViewState, CoordinateSystem, CullMode, Extent, LayerData, Material, OrbitAxis,
    OrthographicViewProps, OrthographicViewState, View,
};
use deck_gl_json::props::{convert, Props};
use deck_gl_json::{JsonConverter, JsonError};
use deck_gl_layers::{
    BrushingExtension, BrushingTarget, ClipExtension, CollisionFilterExtension, DataFilterExtension,
    MaskExtension, PathStyleExtension, PathStyleTarget,
};
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
        {"@@type": "A5Layer", "id": "pentagons"},
        {"@@type": "ScatterplotLayer", "id": "p", "data": [], "bogusProp": 1, "onHover": "ignored"}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert_eq!(deck.warnings.len(), 2, "{:?}", deck.warnings);
    assert!(deck.warnings[0].contains("A5Layer"));
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
        {"@@type": "ScreenGridLayer", "id": "screen", "data": [{"position": [1, 2]}], "cellSizePixels": 40, "aggregation": "COUNT", "colorScaleType": "quantize"},
        {"@@type": "GridCellLayer", "id": "cells", "data": [{"position": [0, 0], "h": 5}], "cellSize": 100, "getElevation": "@@=h * 10"},
        {"@@type": "HeatmapLayer", "id": "heat", "data": [{"position": [0, 0], "w": 2}], "getWeight": "@@=w", "radiusPixels": 20, "aggregation": "MEAN", "threshold": 0.1, "colorDomain": [0, 5], "debounceTimeout": 100},
        {"@@type": "ContourLayer", "id": "contours", "data": [{"position": [0, 0]}], "cellSize": 200, "aggregation": "MAX", "contours": [{"threshold": 1, "color": [255, 0, 0], "strokeWidth": 2}, {"threshold": [1, 5], "color": [0, 255, 0, 128], "zIndex": 3}]}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 6);
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

#[test]
fn trips_and_great_circle_layers() {
    let spec = json!([
        {
            "@@type": "TripsLayer",
            "id": "trips",
            "data": [{"path": [[-122.4, 37.8], [-122.41, 37.81]], "timestamps": [0, 60]}],
            "getPath": "@@=path",
            "getTimestamps": "@@=timestamps",
            "currentTime": 30,
            "trailLength": 100,
            "widthMinPixels": 2
        },
        {
            "@@type": "GreatCircleLayer",
            "id": "routes",
            "data": [{"from": [-122.4, 37.8], "to": [2.35, 48.85]}],
            "getSourcePosition": "@@=from",
            "getTargetPosition": "@@=to"
        }
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 2);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
}

#[test]
fn lighting_effects_and_materials() {
    let spec = json!({
        "effects": [
            {
                "@@type": "LightingEffect",
                "ambient": {"@@type": "AmbientLight", "color": [255, 200, 200], "intensity": 0.5},
                "sun": {"@@type": "DirectionalLight", "intensity": 2.0, "direction": [-3, -9, -1]},
                "lamp": {"@@type": "PointLight", "position": [1, 2, 3], "attenuation": [1, 0.1, 0]}
            },
            {"@@type": "WaterEffect"}
        ],
        "layers": [
            {"@@type": "SolidPolygonLayer", "id": "lit", "data": []},
            {"@@type": "SolidPolygonLayer", "id": "flat", "data": [], "material": false},
            {
                "@@type": "ColumnLayer",
                "id": "shiny",
                "data": [],
                "material": {"ambient": 0.64, "shininess": 64, "specularColor": [51, 51, 51]}
            }
        ]
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert!(!deck.repeat);
    let lighting = deck.lighting.expect("lighting effect");
    assert_eq!(lighting.ambient.color, [255.0, 200.0, 200.0]);
    assert_eq!(lighting.ambient.intensity, 0.5);
    assert_eq!(lighting.directional.len(), 1);
    assert_eq!(lighting.directional[0].direction, [-3.0, -9.0, -1.0]);
    assert_eq!(lighting.directional[0].intensity, 2.0);
    assert_eq!(lighting.point.len(), 1);
    assert_eq!(lighting.point[0].attenuation, [1.0, 0.1, 0.0]);
    assert_eq!(deck.warnings.len(), 1, "{:?}", deck.warnings);
    assert!(deck.warnings[0].contains("WaterEffect"));

    let material = |index: usize| deck.layers[index].props().material;
    assert_eq!(material(0), Material::default());
    assert_eq!(material(1), Material::unlit());
    let shiny = material(2);
    assert!(!shiny.unlit);
    assert_eq!((shiny.ambient, shiny.diffuse, shiny.shininess), (0.64, 0.6, 64.0));
    assert_eq!(shiny.specular_color, [51.0, 51.0, 51.0]);

    let no_ambient =
        json!({"effects": [{"@@type": "LightingEffect", "sun": {"@@type": "DirectionalLight"}}]});
    let deck = JsonConverter::new().convert(&no_ambient).unwrap();
    assert_eq!(deck.lighting.unwrap().ambient.intensity, 0.0);

    let bad = json!([{"@@type": "ColumnLayer", "data": [], "material": "shiny"}]);
    let error = JsonConverter::new().convert(&bad).unwrap_err().to_string();
    assert!(error.contains("material"), "{error}");
}

#[test]
fn render_parameters() {
    let spec = json!([
        {
            "@@type": "ScatterplotLayer",
            "id": "glow",
            "data": [],
            "parameters": {
                "depthTest": false,
                "cullMode": "back",
                "blendColorSrcFactor": "one",
                "blendColorDstFactor": "one",
                "blendAlphaOperation": "max",
                "stencilTest": true
            }
        },
        {"@@type": "ScatterplotLayer", "id": "plain", "data": [], "parameters": {"depthCompare": "always", "depthWriteEnabled": false, "blend": false}}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    let glow = deck.layers[0].props().parameters;
    assert_eq!(glow.depth_test, Some(false));
    assert_eq!(glow.cull_mode, Some(CullMode::Back));
    let blend = glow.blend_state.expect("custom blend");
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
    let plain = deck.layers[1].props().parameters;
    assert_eq!(plain.depth_compare, Some(wgpu::CompareFunction::Always));
    assert_eq!(plain.depth_write_enabled, Some(false));
    assert_eq!(plain.blend, Some(false));
    assert!(plain.blend_state.is_none());
    assert_eq!(deck.warnings.len(), 1, "{:?}", deck.warnings);
    assert!(deck.warnings[0].contains("stencilTest"), "{:?}", deck.warnings);

    let bad =
        json!([{"@@type": "ScatterplotLayer", "data": [], "parameters": {"depthCompare": "sometimes"}}]);
    let error = JsonConverter::new().convert(&bad).unwrap_err().to_string();
    assert!(
        error.contains("depthCompare") && error.contains("sometimes"),
        "{error}"
    );
}

#[test]
fn map_view_repeat() {
    let spec = json!({
        "views": [{"@@type": "MapView", "repeat": true, "controller": true}, {"@@type": "SomeOtherView"}],
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert!(deck.repeat);
    assert_eq!(deck.warnings.len(), 1, "{:?}", deck.warnings);
    assert!(deck.warnings[0].contains("SomeOtherView"));
}

#[test]
fn non_map_views_and_their_view_states() {
    let spec = json!({
        "views": [{"@@type": "OrbitView", "orbitAxis": "Y", "fovy": 40}],
        "initialViewState": {"target": [1, 2, 3], "zoom": 2, "rotationOrbit": 30, "rotationX": 15},
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    match deck.view {
        View::Orbit(props) => assert_eq!((props.orbit_axis, props.fovy), (OrbitAxis::Y, 40.0)),
        other => panic!("{other:?}"),
    }
    match deck.camera {
        Some(AnyViewState::Orbit(state)) => {
            assert_eq!(state.target, [1.0, 2.0, 3.0]);
            assert_eq!(
                (state.zoom, state.rotation_orbit, state.rotation_x),
                (2.0, 30.0, 15.0)
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(deck.view_state.is_none());

    let spec = json!({
        "views": [{"@@type": "OrthographicView", "flipY": false}],
        "initialViewState": {"target": [5, 6], "zoom": 1, "zoomX": 3},
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(
        deck.view,
        View::Orthographic(OrthographicViewProps {
            flip_y: false,
            ..Default::default()
        })
    );
    assert_eq!(
        deck.camera,
        Some(AnyViewState::Orthographic(OrthographicViewState {
            target: [5.0, 6.0, 0.0],
            zoom: 1.0,
            zoom_x: Some(3.0),
            zoom_y: None,
        }))
    );

    let spec = json!({
        "views": [{"@@type": "GlobeView", "altitude": 2}],
        "initialViewState": {"longitude": 10, "latitude": 20, "zoom": 1},
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    match (deck.view, deck.camera) {
        (View::Globe(props), Some(AnyViewState::Globe(state))) => {
            assert_eq!(props.altitude, 2.0);
            assert_eq!((state.longitude, state.latitude, state.zoom), (10.0, 20.0, 1.0));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        deck.view_state.map(|v| v.zoom),
        Some(1.0),
        "globe states are map states"
    );

    let spec = json!({
        "views": [{"@@type": "FirstPersonView"}],
        "initialViewState": {"longitude": 10, "latitude": 20, "position": [0, 0, 50], "bearing": 90},
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    match deck.camera {
        Some(AnyViewState::FirstPerson(state)) => {
            assert_eq!((state.longitude, state.latitude), (Some(10.0), Some(20.0)));
            assert_eq!(
                (state.position, state.bearing, state.pitch),
                ([0.0, 0.0, 50.0], 90.0, 0.0)
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn several_views_with_rectangles_and_their_own_view_states() {
    let spec = json!({
        "views": [
            {"@@type": "MapView", "id": "main", "controller": true},
            {"@@type": "OrthographicView", "id": "inset", "x": "70%", "y": 10, "width": "30%", "height": "25%", "padding": {"left": 4, "bottom": "2%"}}
        ],
        "initialViewState": {
            "main": {"longitude": 1, "latitude": 2, "zoom": 3},
            "inset": {"target": [5, 6], "zoom": 1}
        },
        "layers": []
    });
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    assert_eq!(deck.views.len(), 2);
    assert_eq!(deck.views[1].id, "inset");
    assert_eq!(deck.views[1].x, Extent::Percent(70.0));
    assert_eq!(deck.views[1].y, Extent::Pixels(10.0));
    assert_eq!(deck.views[1].padding.unwrap().bottom, Extent::Percent(2.0));
    assert_eq!(deck.view, View::Map);
    assert_eq!(deck.view_state.map(|v| v.zoom), Some(3.0));
    assert!(
        matches!(deck.cameras.get("inset"), Some(AnyViewState::Orthographic(s)) if s.target == [5.0, 6.0, 0.0])
    );
    // One full size view stays a plain deck
    let spec = json!({"views": [{"@@type": "MapView", "id": "main"}], "initialViewState": {"zoom": 2}, "layers": []});
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert!(deck.views.is_empty());
    assert_eq!(deck.view_state.map(|v| v.zoom), Some(2.0));
}

#[test]
fn tile_layer_from_a_url_template() {
    let spec = json!([{
        "@@type": "TileLayer",
        "id": "osm",
        "data": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "minZoom": 0,
        "maxZoom": 19,
        "tileSize": 256,
        "refinementStrategy": "no-overlap",
        "renderSubLayers": "ignored"
    }]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let bad = json!([{"@@type": "TileLayer", "data": "tiles/{z}.png"}]);
    let error = JsonConverter::new().convert(&bad).unwrap_err().to_string();
    assert!(error.contains("{x}"), "{error}");
}

#[test]
fn csv_and_ndjson_files_load_as_rows() {
    let dir = std::env::temp_dir().join(format!("deck-gl-json-tabular-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("stops.csv"),
        "lng,lat,riders\n-122.4,37.8,10\n-122.41,37.79,20\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("stops.ndjson"),
        "{\"lng\": 1, \"lat\": 2, \"riders\": 3}\n{\"lng\": 4, \"lat\": 5, \"riders\": 6}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("spec.json"),
        r#"{"layers": [
            {"@@type": "ScatterplotLayer", "id": "csv", "data": "stops.csv", "getPosition": "@@=[lng, lat]", "getRadius": "@@=riders * 2"},
            {"@@type": "ScatterplotLayer", "id": "ndjson", "data": "stops.ndjson", "getPosition": "@@=[lng, lat]", "getRadius": "@@=riders"}
        ]}"#,
    )
    .unwrap();
    let mut deck = JsonConverter::parse_file(dir.join("spec.json")).unwrap();
    assert_eq!(deck.layers.len(), 2);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let radii = |layer: &mut Box<dyn deck_gl::Layer>| {
        let layer = layer
            .as_any_mut()
            .downcast_mut::<deck_gl_layers::ScatterplotLayer>()
            .unwrap();
        let props = layer.props();
        resolve_f32(&props.data, &props.get_radius).unwrap()
    };
    assert_eq!(radii(&mut deck.layers[0]), [20.0, 40.0]);
    assert_eq!(radii(&mut deck.layers[1]), [3.0, 6.0]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn geo_cell_layers() {
    let spec = json!([
        {"@@type": "H3HexagonLayer", "id": "h3", "data": [{"hexagon": "8928308280fffff", "n": 2}], "getElevation": "@@=n * 100", "coverage": 0.9},
        {"@@type": "S2Layer", "id": "s2", "data": [{"token": "80858004"}], "extruded": true},
        {"@@type": "GeohashLayer", "id": "geohash", "data": [{"geohash": "9q8yy"}]},
        {"@@type": "QuadkeyLayer", "id": "quadkey", "data": [{"quadkey": "0230"}], "getFillColor": [1, 2, 3]}
    ]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 4);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let ids: Vec<&str> = deck.layers.iter().map(|l| l.id()).collect();
    assert_eq!(ids, ["h3", "s2", "geohash", "quadkey"]);
}

#[test]
fn mvt_layer_from_a_url_template() {
    let spec = json!([{
        "@@type": "MVTLayer",
        "id": "vector",
        "data": "https://example.com/tiles/{z}/{x}/{y}.pbf",
        "minZoom": 0,
        "maxZoom": 14,
        "layers": ["water", "roads"],
        "getFillColor": [0, 80, 200],
        "getLineColor": [255, 255, 255],
        "lineWidthMinPixels": 1,
        "binary": true
    }]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
}

#[test]
fn wms_layer_from_an_endpoint() {
    let spec = json!([{
        "@@type": "WMSLayer",
        "id": "wms",
        "data": "https://ows.example/wms",
        "serviceType": "wms",
        "layers": ["OSM-WMS"],
        "srs": "EPSG:4326",
        "opacity": 0.8,
        "onImageLoad": "ignored"
    }]);
    let deck = JsonConverter::new().convert(&spec).unwrap();
    assert_eq!(deck.layers.len(), 1);
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let bad = json!([{"@@type": "WMSLayer", "data": "https://ows.example/wms", "srs": "EPSG:2154"}]);
    assert!(JsonConverter::new()
        .convert(&bad)
        .unwrap_err()
        .to_string()
        .contains("srs"));
}

#[test]
fn extensions_parse_the_data_filter_with_named_categories() {
    let spec = json!({
        "@@type": "ScatterplotLayer",
        "id": "filtered",
        "data": [
            {"position": [1, 2], "value": 10, "kind": "bus"},
            {"position": [1, 2], "value": 50, "kind": "tram"},
            {"position": [1, 2], "value": 90, "kind": "metro"}
        ],
        "getPosition": "@@=position",
        "extensions": [{"@@type": "DataFilterExtension", "filterSize": 1, "categorySize": 1}],
        "getFilterValue": "@@=value",
        "filterRange": [20, 80],
        "filterSoftRange": [30, 70],
        "getFilterCategory": "@@=kind",
        "filterCategories": ["bus", "metro"],
        "filterTransformSize": false
    });
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(&json!([spec]), &mut warnings)
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let filter = layers[0]
        .props()
        .extensions
        .get::<DataFilterExtension>()
        .expect("data filter");
    assert_eq!(filter.filter_range, vec![[20.0, 80.0]]);
    assert_eq!(filter.filter_soft_range, Some(vec![[30.0, 70.0]]));
    assert!(!filter.filter_transform_size);
    let data = LayerData::with_length(3);
    let values = filter.get_filter_value.resolve(&data).unwrap();
    assert_eq!(
        values.iter().map(|v| v[0]).collect::<Vec<_>>(),
        [10.0, 50.0, 90.0]
    );
    // categories get keys in order of appearance: bus 0, tram 1, metro 2
    let keys = filter
        .get_filter_category
        .as_ref()
        .unwrap()
        .resolve(&data)
        .unwrap();
    assert_eq!(keys.iter().map(|k| k[0]).collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(filter.filter_categories, vec![vec![0, 2]]);
    // the range keeps the middle row only and the categories drop that one
    assert_eq!(filter.count_filtered(&data).unwrap(), 0);

    // an extension that does not exist yet warns and is skipped
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([{"@@type": "ScatterplotLayer", "data": [], "extensions": [{"@@type": "TerrainExtension"}]}]),
            &mut warnings,
        )
        .unwrap();
    assert!(layers[0].props().extensions.is_empty());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
}

#[test]
fn extensions_parse_brushing_and_clip() {
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([
                {
                    "@@type": "ArcLayer", "id": "arcs", "data": [],
                    "extensions": [{"@@type": "BrushingExtension"}, {"@@type": "ClipExtension"}],
                    "brushingRadius": 500, "brushingTarget": "source_target",
                    "clipBounds": [-1, -2, 3, 4]
                },
                {
                    "@@type": "PathLayer", "id": "paths", "data": [],
                    "extensions": ["ClipExtension"]
                }
            ]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let arcs = layers[0].props();
    let brushing = arcs.extensions.get::<BrushingExtension>().unwrap();
    assert_eq!(brushing.brushing_radius, 500.0);
    assert_eq!(brushing.brushing_target, BrushingTarget::SourceTarget);
    let clip = arcs.extensions.get::<ClipExtension>().unwrap();
    assert_eq!(clip.clip_bounds, [-1.0, -2.0, 3.0, 4.0]);
    assert!(clip.clip_by_instance, "arcs clip by their anchors");
    let paths = layers[1].props().extensions.get::<ClipExtension>().unwrap();
    assert!(!paths.clip_by_instance, "paths clip by geometry");
}

#[test]
fn mask_operation_and_extension() {
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([
                {"@@type": "SolidPolygonLayer", "id": "geofence", "data": [], "operation": "mask"},
                {
                    "@@type": "ScatterplotLayer", "id": "points", "data": [],
                    "extensions": [{"@@type": "MaskExtension"}], "maskId": "geofence", "maskInverted": true
                }
            ]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(layers[0].props().operation, deck_gl::Operation::MASK);
    let mask = layers[1].props().extensions.get::<MaskExtension>().unwrap();
    assert_eq!(mask.mask_id, "geofence");
    assert!(mask.mask_inverted);
    assert!(mask.mask_by_instance);
}

#[test]
fn collision_filter_extension_props() {
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([{
                "@@type": "TextLayer", "id": "labels", "data": [{"position": [1, 2], "name": "a", "rank": 3}],
                "getPosition": "@@=position", "getText": "@@=name",
                "extensions": [{"@@type": "CollisionFilterExtension"}],
                "getCollisionPriority": "@@=rank", "collisionGroup": "labels"
            }]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let collision = layers[0]
        .props()
        .extensions
        .get::<CollisionFilterExtension>()
        .unwrap();
    assert_eq!(collision.collision_group, "labels");
    assert!(collision.collision_enabled);
    assert_eq!(
        resolve_f32(&LayerData::with_length(1), &collision.get_collision_priority).unwrap(),
        [3.0]
    );
}

#[test]
fn path_style_extension_props_and_target() {
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([
                {
                    "@@type": "PathLayer", "id": "dashed", "data": [],
                    "extensions": [{"@@type": "PathStyleExtension", "dash": true, "offset": true}],
                    "getDashArray": [4, 2], "getOffset": 1, "dashJustified": true
                },
                {
                    "@@type": "ScatterplotLayer", "id": "rings", "data": [],
                    "extensions": [{"@@type": "PathStyleExtension", "dash": true}]
                }
            ]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let dashed = layers[0].props().extensions.get::<PathStyleExtension>().unwrap();
    assert!(dashed.dash && dashed.offset && dashed.dash_justified);
    assert_eq!(dashed.target, PathStyleTarget::Path);
    assert_eq!(
        resolve_f32(&LayerData::with_length(1), &dashed.get_offset).unwrap(),
        [1.0]
    );
    let rings = layers[1].props().extensions.get::<PathStyleExtension>().unwrap();
    assert_eq!(rings.target, PathStyleTarget::Scatterplot);
}

#[test]
fn transitions_prop_parses_durations_easings_and_springs() {
    use deck_gl::{EasingKind, PropTransition};
    let mut warnings = Vec::new();
    let layers = JsonConverter::new()
        .convert_layers(
            &json!([{
                "@@type": "ScatterplotLayer", "id": "animated", "data": [],
                "transitions": {
                    "getRadius": 300,
                    "radiusScale": {"duration": 500, "easing": "easeInOut"},
                    "getPosition": {"type": "spring", "stiffness": 0.1}
                }
            }]),
            &mut warnings,
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let transitions = &layers[0].props().transitions;
    assert_eq!(
        transitions.get("getRadius"),
        Some(&PropTransition::interpolation(300.0))
    );
    assert_eq!(
        transitions.get("radiusScale"),
        Some(&PropTransition::Interpolation {
            duration_ms: 500.0,
            easing: EasingKind::EaseInOut
        })
    );
    assert_eq!(
        transitions.get("getPosition"),
        Some(&PropTransition::Spring {
            stiffness: 0.1,
            damping: 0.5
        })
    );
    assert_eq!(
        transitions.for_attribute("radius"),
        Some(PropTransition::interpolation(300.0))
    );
}

#[test]
fn simple_mesh_layer_reads_inline_meshes_and_obj_files() {
    use deck_gl_layers::SimpleMeshLayer;
    let dir = std::env::temp_dir().join(format!("deckgl-mesh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tri.obj"),
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n",
    )
    .unwrap();
    let converter = JsonConverter::with_base_dir(&dir);
    let spec = json!({
        "layers": [
            {
                "@@type": "SimpleMeshLayer",
                "id": "inline",
                "mesh": {
                    "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                    "texCoords": [0, 0, 1, 0, 0, 1, 1, 1],
                    "indices": [0, 1, 2, 1, 3, 2]
                },
                "data": [{"position": [1, 2], "yaw": 45}],
                "getPosition": "@@=position",
                "getOrientation": "@@=[0, yaw, 0]",
                "getScale": [2, 2, 2],
                "sizeScale": 10,
                "wireframe": true
            },
            {
                "@@type": "SimpleMeshLayer",
                "id": "obj",
                "mesh": "tri.obj",
                "data": [{"position": [1, 2]}],
                "getPosition": "@@=position",
                "getTransformMatrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
            }
        ]
    });
    let mut deck = converter.convert(&spec).unwrap();
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    assert_eq!(deck.layers.len(), 2);
    let props = |layer: &mut Box<dyn deck_gl::Layer>| {
        layer
            .as_any_mut()
            .downcast_mut::<SimpleMeshLayer>()
            .unwrap()
            .props()
            .clone()
    };
    let inline = props(&mut deck.layers[0]);
    let mesh = inline.mesh.as_ref().unwrap();
    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.tex_coords.as_ref().unwrap()[3], [1.0, 1.0]);
    assert_eq!(mesh.indices.as_ref().unwrap().len(), 6);
    assert!(mesh.normals.is_none());
    assert_eq!(inline.size_scale, 10.0);
    assert!(inline.wireframe);
    assert_eq!(
        deck_gl::data::resolve_vec3(&inline.data, &inline.get_orientation).unwrap(),
        vec![[0.0, 45.0, 0.0]]
    );
    assert_eq!(inline.get_scale, deck_gl::Accessor::Constant([2.0, 2.0, 2.0]));
    let obj = props(&mut deck.layers[1]);
    let mesh = obj.mesh.as_ref().unwrap();
    assert_eq!(mesh.vertex_count(), 3);
    assert!(mesh.has_normals());
    let matrix = obj.get_transform_matrix.as_ref().unwrap();
    assert!(matches!(matrix, deck_gl::Accessor::Constant(m) if m[12..15] == [5.0, 6.0, 7.0]));
    // A mesh whose indices point past its vertices is rejected
    let bad = json!([{
        "@@type": "SimpleMeshLayer",
        "mesh": {"positions": [0, 0, 0], "indices": [0, 1, 2]},
        "data": []
    }]);
    let error = converter.convert(&bad).unwrap_err().to_string();
    assert!(error.contains("out of range"), "{error}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scenegraph_layer_reads_gltf_files_with_external_buffers() {
    use deck_gl_layers::{ScenegraphLayer, ScenegraphLighting};
    let dir = std::env::temp_dir().join(format!("deckgl-gltf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let bin: Vec<u8> = positions.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join("model.bin"), bin).unwrap();
    std::fs::write(
        dir.join("model.gltf"),
        r#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"translation":[0,0,5]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],"bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36,"uri":"model.bin"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,0.5,0,1]}}]}"#,
    )
    .unwrap();
    let converter = JsonConverter::with_base_dir(&dir);
    let spec = json!([{
        "@@type": "ScenegraphLayer",
        "id": "planes",
        "scenegraph": "model.gltf",
        "data": [{"position": [1, 2], "heading": 30}],
        "getPosition": "@@=position",
        "getOrientation": "@@=[0, heading, 90]",
        "sizeScale": 5,
        "sizeMinPixels": 2,
        "sizeMaxPixels": 100,
        "_lighting": "pbr"
    }]);
    let mut deck = converter.convert(&spec).unwrap();
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let layer = deck.layers[0]
        .as_any_mut()
        .downcast_mut::<ScenegraphLayer>()
        .expect("a scenegraph layer");
    let props = layer.props();
    let scene = props.scenegraph.as_ref().unwrap();
    assert_eq!(scene.primitives.len(), 1);
    assert_eq!(scene.primitives[0].mesh.positions[1], [1.0, 0.0, 0.0]);
    assert_eq!(scene.primitives[0].base_color, [1.0, 0.5, 0.0, 1.0]);
    assert_eq!(
        scene.primitives[0].model_matrix.w_axis.z, 5.0,
        "the node translation"
    );
    assert_eq!(props.size_scale, 5.0);
    assert_eq!((props.size_min_pixels, props.size_max_pixels), (2.0, 100.0));
    assert_eq!(props.lighting, ScenegraphLighting::Pbr);
    assert_eq!(
        deck_gl::data::resolve_vec3(&props.data, &props.get_orientation).unwrap(),
        vec![[0.0, 30.0, 90.0]]
    );
    let bad =
        json!([{"@@type": "ScenegraphLayer", "scenegraph": "model.gltf", "data": [], "_lighting": "neon"}]);
    let error = converter.convert(&bad).unwrap_err().to_string();
    assert!(error.contains("_lighting"), "{error}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn post_process_effects_are_read_from_effects() {
    use deck_gl::UniformValue;
    let deck = JsonConverter::new()
        .convert(&json!({
            "layers": [],
            "effects": [
                {"@@type": "PostProcessEffect", "module": "vignette", "props": {"radius": 0.8}},
                {"@@type": "PostProcessEffect", "module": "zoomBlur", "center": [0.2, 0.7], "strength": 0.5},
                {"@@type": "PostProcessEffect", "module": "edgeWork", "radius": 3}
            ]
        }))
        .unwrap();
    assert_eq!(deck.post_process.len(), 3);
    assert_eq!(deck.post_process[0].module.name, "vignette");
    assert_eq!(deck.post_process[0].prop("radius"), Some(UniformValue::F32(0.8)));
    assert_eq!(
        deck.post_process[0].prop("amount"),
        Some(UniformValue::F32(0.5)),
        "the default"
    );
    assert_eq!(
        deck.post_process[1].prop("center"),
        Some(UniformValue::Vec2([0.2, 0.7]))
    );
    assert_eq!(
        deck.post_process[1].prop("strength"),
        Some(UniformValue::F32(0.5))
    );
    assert_eq!(deck.post_process[2].prop("radius"), Some(UniformValue::F32(3.0)));
    let unknown = JsonConverter::new()
        .convert(&json!({"layers": [], "effects": [{"@@type": "PostProcessEffect", "module": "sparkle"}]}))
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("sparkle"), "{unknown}");
    let bad_prop = JsonConverter::new()
        .convert(
            &json!({"layers": [], "effects": [{"@@type": "PostProcessEffect", "module": "sepia", "hue": 1}]}),
        )
        .unwrap_err()
        .to_string();
    assert!(bad_prop.contains("no prop `hue`"), "{bad_prop}");
}

#[test]
fn directional_lights_read_shadow_flags() {
    let deck = JsonConverter::new()
        .convert(&json!({
            "layers": [{"@@type": "SolidPolygonLayer", "id": "a", "data": [], "shadowEnabled": false}],
            "effects": [{
                "@@type": "LightingEffect",
                "shadowColor": [0, 0, 60, 128],
                "ambient": {"@@type": "AmbientLight", "intensity": 0.4},
                "sun": {"@@type": "DirectionalLight", "direction": [-1, -3, -1], "_shadow": true},
                "fill": {"@@type": "DirectionalLight", "direction": [1, 3, 1]}
            }]
        }))
        .unwrap();
    let lighting = deck.lighting.expect("lighting");
    assert_eq!(lighting.directional.len(), 2);
    // Lights come out in the JSON object's key order, so find them by their flag
    let casting: Vec<[f32; 3]> = lighting
        .directional
        .iter()
        .filter(|l| l.shadow)
        .map(|l| l.direction)
        .collect();
    assert_eq!(casting, vec![[-1.0, -3.0, -1.0]], "only the sun casts shadows");
    assert_eq!(lighting.shadow_color, [0.0, 0.0, 60.0 / 255.0, 128.0 / 255.0]);
    assert!(!deck.layers[0].props().shadow_enabled, "shadowEnabled false");
    // Without the flag nothing casts shadows and the colour keeps deck.gl's default
    let plain = JsonConverter::new()
        .convert(&json!({
            "layers": [],
            "effects": [{"@@type": "LightingEffect", "sun": {"@@type": "DirectionalLight"}}]
        }))
        .unwrap()
        .lighting
        .unwrap();
    assert!(!plain.directional[0].shadow);
    assert_eq!(plain.shadow_color, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn terrain_layer_reads_tiles_decoders_and_bounds() {
    use deck_gl_layers::{ElevationDecoder, TerrainLayer};
    let mut deck = JsonConverter::new()
        .convert(&json!([
            {
                "@@type": "TerrainLayer",
                "id": "tiled",
                "elevationData": "https://example.com/{z}/{x}/{y}.png",
                "texture": "https://example.com/sat/{z}/{x}/{y}.jpg",
                "elevationDecoder": "terrarium",
                "meshMaxError": 6,
                "maxZoom": 12,
                "color": [200, 190, 180],
                "wireframe": true
            },
            {
                "@@type": "TerrainLayer",
                "id": "single",
                "elevationData": "https://example.com/height.png",
                "bounds": [-122.5, 37.7, -122.3, 37.9],
                "elevationDecoder": {"rScaler": 2, "offset": -100}
            }
        ]))
        .unwrap();
    assert!(deck.warnings.is_empty(), "{:?}", deck.warnings);
    let props = |layer: &mut Box<dyn deck_gl::Layer>| {
        layer
            .as_any_mut()
            .downcast_mut::<TerrainLayer>()
            .expect("a terrain layer")
            .props()
            .clone()
    };
    let tiled = props(&mut deck.layers[0]);
    assert_eq!(tiled.elevation_decoder, ElevationDecoder::terrarium());
    assert_eq!(tiled.mesh_max_error, 6.0);
    assert_eq!(tiled.max_zoom, Some(12));
    assert_eq!(tiled.color, [200, 190, 180, 255]);
    assert!(tiled.wireframe);
    assert_eq!(tiled.texture.len(), 1);
    let single = props(&mut deck.layers[1]);
    assert_eq!(single.bounds, Some([-122.5, 37.7, -122.3, 37.9]));
    assert_eq!(single.elevation_decoder.r_scaler, 2.0);
    assert_eq!(single.elevation_decoder.offset, -100.0);
    assert_eq!(single.elevation_decoder.g_scaler, 0.0, "the default for the rest");
    // A terrain layer without elevation data, or with an unknown decoder, is an error
    for bad in [
        json!([{"@@type": "TerrainLayer", "id": "x"}]),
        json!([{"@@type": "TerrainLayer", "id": "x", "elevationData": "u", "elevationDecoder": "moon"}]),
    ] {
        assert!(JsonConverter::new().convert(&bad).is_err(), "{bad}");
    }
}

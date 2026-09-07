//! GPU integration tests: render each layer headlessly and check pixels.
//! Skipped (with a message) when no GPU adapter is available.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::luma_gl::device::{
    create_headless_context, create_render_texture, read_texture_rgba8, HeadlessContext,
};
use deck_gl::luma_gl::{Model, RenderTarget, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::wgpu;
use deck_gl::{
    same_extension, Accessor, ClickCallback, Deck, DeckProps, ExtensionAttribute, ExtensionShaders,
    Extensions, HoverCallback, Layer, LayerContext, LayerData, LayerExtension, LayerProps, Material,
    Operation, Path, PickingInfo, RenderParameters, Unit, ViewState, Viewport,
};
use deck_gl_layers::{
    AggregationOperation, AggregationProps, ArcLayer, ArcLayerProps, BitmapImage, BitmapLayer,
    BitmapLayerProps, BrushingExtension, ClipExtension, CollisionFilterExtension, ColumnLayer,
    ColumnLayerProps, Contour, ContourLayer, ContourLayerProps, DataFilterExtension, FillPattern,
    FillPatternAtlas, FillStyleExtension, FilterCategories, GeoJsonLayer, GeoJsonLayerProps, GridLayer,
    GridLayerProps, HeatmapAggregation, HeatmapLayer, HeatmapLayerProps, HexagonLayer, HexagonLayerProps,
    IconAtlas, IconLayer, IconLayerProps, IconMapping, LineLayer, LineLayerProps, MaskExtension, PathLayer,
    PathLayerProps, PathStyleExtension, PathStyleTarget, PointCloudLayer, PointCloudLayerProps, PolygonLayer,
    PolygonLayerProps, ScatterplotLayer, ScatterplotLayerProps, ScreenGridLayer, ScreenGridLayerProps,
    SolidPolygonLayer, SolidPolygonLayerProps, TextLayer, TextLayerProps, TripsLayer, TripsLayerProps,
};

const SIZE: u32 = 64;
const CENTER: [f64; 3] = [-122.4, 37.8, 0.0];

fn context() -> Option<HeadlessContext> {
    match create_headless_context() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn make_deck(ctx: &HeadlessContext, layers: Vec<Box<dyn Layer>>) -> Deck {
    Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude: CENTER[0],
                latitude: CENTER[1],
                zoom: 14.0,
                pitch: 0.0,
                bearing: 0.0,
            },
            layers,
            ..Default::default()
        },
    )
    .expect("deck")
}

/// Render layers on a transparent background and return RGBA pixels.
fn render(ctx: &HeadlessContext, layers: Vec<Box<dyn Layer>>) -> Vec<u8> {
    let target = RenderTarget::default();
    let color = create_render_texture(&ctx.device, "color", SIZE, SIZE, target.color_format);
    let depth = create_render_texture(&ctx.device, "depth", SIZE, SIZE, target.depth_format.unwrap());
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude: CENTER[0],
                latitude: CENTER[1],
                zoom: 14.0,
                pitch: 0.0,
                bearing: 0.0,
            },
            layers,
            ..Default::default()
        },
    )
    .expect("deck");
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(wgpu::Color::TRANSPARENT),
    )
    .expect("render");
    ctx.queue.submit([encoder.finish()]);
    read_texture_rgba8(&ctx.device, &ctx.queue, &color).expect("readback")
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn assert_pixel(pixels: &[u8], x: u32, y: u32, expected: [u8; 4], tolerance: u8) {
    let actual = pixel(pixels, x, y);
    for c in 0..4 {
        assert!(
            actual[c].abs_diff(expected[c]) <= tolerance,
            "pixel ({x}, {y}) = {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn scatterplot_draws_a_circle_at_the_view_center() {
    let Some(ctx) = context() else { return };
    let layer = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps::new("points"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(12.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 1);
    // Inside the circle, 8 pixels from the center
    assert_pixel(&pixels, c + 8, c, [255, 0, 0, 255], 1);
    // Outside the circle, 16 pixels from the center, and in the corner
    assert_pixel(&pixels, c + 16, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, 1, 1, [0, 0, 0, 0], 0);
}

#[test]
fn line_layer_draws_a_horizontal_line() {
    let Some(ctx) = context() else { return };
    let layer = LineLayer::new(LineLayerProps {
        base: LayerProps::new("lines"),
        data: LayerData::with_length(1),
        get_source_position: Accessor::Constant([CENTER[0] - 0.01, CENTER[1], 0.0]),
        get_target_position: Accessor::Constant([CENTER[0] + 0.01, CENTER[1], 0.0]),
        get_color: Accessor::Constant([0, 255, 0, 255]),
        get_width: Accessor::Constant(6.0),
        width_units: Unit::Pixels,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    assert_pixel(&pixels, c, c, [0, 255, 0, 255], 1);
    assert_pixel(&pixels, 4, c, [0, 255, 0, 255], 1);
    assert_pixel(&pixels, SIZE - 4, c, [0, 255, 0, 255], 1);
    assert_pixel(&pixels, c, c - 10, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c, c + 10, [0, 0, 0, 0], 0);
}

#[test]
fn solid_polygon_fills_a_square_with_a_hole() {
    let Some(ctx) = context() else { return };
    // At zoom 14 one degree of longitude is about 23300 pixels, so the outer square spans
    // about 56 pixels and the hole about 19 pixels of the 64 pixel viewport.
    let d = 0.0012;
    let h = 0.0004;
    let polygon = vec![
        vec![
            [CENTER[0] - d, CENTER[1] - d, 0.0],
            [CENTER[0] + d, CENTER[1] - d, 0.0],
            [CENTER[0] + d, CENTER[1] + d, 0.0],
            [CENTER[0] - d, CENTER[1] + d, 0.0],
        ],
        vec![
            [CENTER[0] - h, CENTER[1] - h, 0.0],
            [CENTER[0] + h, CENTER[1] - h, 0.0],
            [CENTER[0] + h, CENTER[1] + h, 0.0],
            [CENTER[0] - h, CENTER[1] + h, 0.0],
        ],
    ];
    let polygon = Arc::new(polygon);
    let layer = SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps::new("polygons"),
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*polygon).clone()),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    // The hole is transparent, the ring around it is filled, the corners are outside
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c, c + 6, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 14, c, [0, 0, 255, 255], 1);
    assert_pixel(&pixels, c - 14, c, [0, 0, 255, 255], 1);
    assert_pixel(&pixels, c, c - 20, [0, 0, 255, 255], 1);
    assert_pixel(&pixels, 1, 1, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, SIZE - 2, SIZE - 2, [0, 0, 0, 0], 0);
}

#[test]
fn opacity_and_depth_composite_layers_in_order() {
    let Some(ctx) = context() else { return };
    let bottom = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps::new("bottom"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(20.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let top = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            opacity: 0.5,
            ..LayerProps::new("top")
        },
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(10.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(bottom), Box::new(top)]);
    let c = SIZE / 2;
    // Gamma-adjusted opacity: 0.5^(1/2.2) = 0.73, blended over red
    let a = 0.5f32.powf(1.0 / 2.2);
    let expected = [(255.0 * (1.0 - a)) as u8, 0, (255.0 * a) as u8, 255];
    assert_pixel(&pixels, c, c, expected, 3);
    assert_pixel(&pixels, c + 15, c, [255, 0, 0, 255], 1);
}

#[test]
fn later_layer_wins_on_a_shared_surface() {
    // Two coplanar layers: a filled square, then circles on top of it at the same elevation.
    // deck.gl's per-layer polygon offset must make the circles win everywhere, with no
    // z-fighting speckle.
    let Some(ctx) = context() else { return };
    let d = 0.0012;
    let polygon = Arc::new(vec![vec![
        [CENTER[0] - d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] + d, 0.0],
        [CENTER[0] - d, CENTER[1] + d, 0.0],
    ]]);
    let square = SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps::new("square"),
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*polygon).clone()),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        ..Default::default()
    });
    let circle = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps::new("circle"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(16.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(square), Box::new(circle)]);
    let c = SIZE / 2;
    // Every pixel well inside the circle must be red, none may show the square through it.
    for dy in -10i32..=10 {
        for dx in -10i32..=10 {
            assert_pixel(
                &pixels,
                (c as i32 + dx) as u32,
                (c as i32 + dy) as u32,
                [255, 0, 0, 255],
                1,
            );
        }
    }
    assert_pixel(&pixels, c + 24, c, [0, 0, 255, 255], 1);
}

#[test]
fn path_layer_draws_a_polyline_with_a_corner() {
    let Some(ctx) = context() else { return };
    // An L shape: west to center, then north. Width 6 pixels.
    let d = 0.001;
    let path = Arc::new(vec![
        [CENTER[0] - d, CENTER[1], 0.0],
        [CENTER[0], CENTER[1], 0.0],
        [CENTER[0], CENTER[1] + d, 0.0],
    ]);
    let layer = PathLayer::new(PathLayerProps {
        base: LayerProps::new("path"),
        data: LayerData::with_length(1),
        get_path: Accessor::func(move |_| (*path).clone()),
        get_color: Accessor::Constant([255, 0, 255, 255]),
        get_width: Accessor::Constant(6.0),
        width_units: Unit::Pixels,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    // Along the horizontal leg, on the vertical leg (north is up on screen), and at the corner
    assert_pixel(&pixels, c - 12, c, [255, 0, 255, 255], 1);
    assert_pixel(&pixels, c, c - 12, [255, 0, 255, 255], 1);
    assert_pixel(&pixels, c, c, [255, 0, 255, 255], 1);
    // Off the path: east of the corner and south of the horizontal leg
    assert_pixel(&pixels, c + 12, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c - 12, c + 10, [0, 0, 0, 0], 0);
}

#[test]
fn arc_layer_with_zero_height_is_a_straight_line() {
    let Some(ctx) = context() else { return };
    let d = 0.001;
    let layer = ArcLayer::new(ArcLayerProps {
        base: LayerProps::new("arcs"),
        data: LayerData::with_length(1),
        get_source_position: Accessor::Constant([CENTER[0] - d, CENTER[1], 0.0]),
        get_target_position: Accessor::Constant([CENTER[0] + d, CENTER[1], 0.0]),
        get_source_color: Accessor::Constant([255, 0, 0, 255]),
        get_target_color: Accessor::Constant([255, 0, 0, 255]),
        get_width: Accessor::Constant(6.0),
        get_height: Accessor::Constant(0.0),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, c - 12, c, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, c, c - 10, [0, 0, 0, 0], 0);
}

#[test]
fn polygon_layer_fills_and_strokes() {
    let Some(ctx) = context() else { return };
    let d = 0.0008;
    let polygon = Arc::new(vec![vec![
        [CENTER[0] - d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] + d, 0.0],
        [CENTER[0] - d, CENTER[1] + d, 0.0],
    ]]);
    let layer = PolygonLayer::new(PolygonLayerProps {
        base: LayerProps::new("polygons"),
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*polygon).clone()),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        get_line_color: Accessor::Constant([255, 0, 0, 255]),
        get_line_width: Accessor::Constant(6.0),
        line_width_units: Unit::Pixels,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    // Fill in the middle, stroke centered on the edges (19 pixels out in x, 24 in y because of the
    // latitude scale), nothing outside
    assert_pixel(&pixels, c, c, [0, 0, 255, 255], 1);
    assert_pixel(&pixels, c + 19, c, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, c, c - 24, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, 2, 2, [0, 0, 0, 0], 0);
}

#[test]
fn pick_finds_the_object_and_layer_under_a_pixel() {
    let Some(ctx) = context() else { return };
    // 0.0007 degrees is about 16 pixels at zoom 14; the viewport is 64 pixels wide
    let d = 0.0007;
    // Two circles in one layer, a polygon in another, drawn after the circles.
    let circles = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("circles")
        },
        data: LayerData::with_length(2),
        get_position: Accessor::func(move |i| [CENTER[0] + if i == 0 { -d } else { d }, CENTER[1], 0.0]),
        get_radius: Accessor::Constant(8.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let s = 0.0002;
    let square = Arc::new(vec![vec![
        [CENTER[0] - s, CENTER[1] - s, 0.0],
        [CENTER[0] + s, CENTER[1] - s, 0.0],
        [CENTER[0] + s, CENTER[1] + s, 0.0],
        [CENTER[0] - s, CENTER[1] + s, 0.0],
    ]]);
    let polygon = PolygonLayer::new(PolygonLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("square")
        },
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*square).clone()),
        stroked: false,
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(circles), Box::new(polygon)]);
    let c = SIZE as f64 / 2.0;
    let left = deck.pick(c - 16.0, c).expect("pick").expect("hit");
    assert_eq!(left.layer_id, "circles");
    assert_eq!(left.index, 0);
    let right = deck.pick(c + 16.0, c).expect("pick").expect("hit");
    assert_eq!((right.layer_id.as_str(), right.index), ("circles", 1));
    assert!(
        (right.coordinate[0] - (CENTER[0] + d)).abs() < 2e-4,
        "{:?}",
        right.coordinate
    );
    let middle: PickingInfo = deck.pick(c, c).expect("pick").expect("hit");
    assert_eq!((middle.layer_id.as_str(), middle.index), ("square", 0));
    assert_eq!(deck.pick(2.0, 2.0).expect("pick"), None);
    assert_eq!(deck.pick(-1.0, 5.0).expect("pick"), None);
}

#[test]
fn highlighted_object_is_tinted() {
    let Some(ctx) = context() else { return };
    let layer = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            pickable: true,
            highlighted_object_index: Some(0),
            highlight_color: [0, 0, 255, 255],
            ..LayerProps::new("points")
        },
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(12.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    assert_pixel(&pixels, c, c, [0, 0, 255, 255], 1);
}

#[test]
fn geojson_layer_renders_polygons_lines_and_points_with_feature_accessors() {
    let Some(ctx) = context() else { return };
    let (x, y) = (CENTER[0], CENTER[1]);
    let d = 0.0004;
    let text = format!(
        r#"{{"type":"FeatureCollection","features":[
            {{"type":"Feature","properties":{{"kind":"poly"}},"geometry":{{"type":"Polygon","coordinates":[[[{x0},{y0}],[{x1},{y0}],[{x1},{y1}],[{x0},{y1}],[{x0},{y0}]]]}}}},
            {{"type":"Feature","properties":{{"kind":"line"}},"geometry":{{"type":"LineString","coordinates":[[{lx0},{y}],[{lx1},{y}]]}}}},
            {{"type":"Feature","properties":{{"kind":"point"}},"geometry":{{"type":"Point","coordinates":[{px},{y}]}}}}
        ]}}"#,
        x0 = x - d,
        x1 = x + d,
        y0 = y - d,
        y1 = y + d,
        lx0 = x - 0.0015,
        lx1 = x - 0.0007,
        px = x + 0.0011,
        y = y
    );
    let data = Arc::new(deck_gl::FeatureCollection::parse(&text).unwrap());
    let features = data.clone();
    let layer = GeoJsonLayer::new(GeoJsonLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("geojson")
        },
        data,
        stroked: true,
        get_fill_color: Accessor::func(move |i| match features.features[i].string("kind") {
            Some("poly") => [0, 0, 255, 255],
            _ => [255, 0, 0, 255],
        }),
        get_line_color: Accessor::Constant([0, 255, 0, 255]),
        get_line_width: Accessor::Constant(4.0),
        line_width_units: Unit::Pixels,
        get_point_radius: Accessor::Constant(6.0),
        point_radius_units: Unit::Pixels,
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    let c = SIZE as f64 / 2.0;
    // Polygon fill in the middle, line to the west, point to the east
    let hit = deck.pick(c, c).unwrap().expect("polygon");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("geojson", 0));
    let hit = deck.pick(c - 26.0, c).unwrap().expect("line");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("geojson", 1));
    let hit = deck.pick(c + 26.0, c).unwrap().expect("point");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("geojson", 2));
    assert!(deck.pick(c, 3.0).unwrap().is_none());
}

#[test]
fn bitmap_layer_drapes_an_image_over_bounds() {
    let Some(ctx) = context() else { return };
    // 2x2 image: red, green on the top row; blue, white on the bottom row
    let image = BitmapImage::new(
        2,
        2,
        vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255],
    );
    let d = 0.0008;
    let layer = BitmapLayer::new(BitmapLayerProps {
        base: LayerProps::new("bitmap"),
        image: Some(image),
        bounds: [CENTER[0] - d, CENTER[1] - d, CENTER[0] + d, CENTER[1] + d],
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    // Top left of the image is the north west corner: up and left on screen. Sample near the
    // corners, outside the bilinear blend between the two texel centers.
    assert_pixel(&pixels, c - 15, c - 19, [255, 0, 0, 255], 2);
    assert_pixel(&pixels, c + 15, c - 19, [0, 255, 0, 255], 2);
    assert_pixel(&pixels, c - 15, c + 19, [0, 0, 255, 255], 2);
    assert_pixel(&pixels, c + 15, c + 19, [255, 255, 255, 255], 2);
    assert_pixel(&pixels, 1, 1, [0, 0, 0, 0], 0);
}

#[test]
fn column_layer_draws_flat_disks_and_extruded_columns() {
    let Some(ctx) = context() else { return };
    let flat = ColumnLayer::new(ColumnLayerProps {
        base: LayerProps::new("disks"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant([CENTER[0] - 0.0007, CENTER[1], 0.0]),
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        radius: 10.0,
        radius_units: Unit::Pixels,
        extruded: false,
        ..Default::default()
    });
    let extruded = ColumnLayer::new(ColumnLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("columns")
        },
        data: LayerData::with_length(1),
        get_position: Accessor::Constant([CENTER[0] + 0.0007, CENTER[1], 0.0]),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        get_elevation: Accessor::Constant(200.0),
        radius: 10.0,
        radius_units: Unit::Pixels,
        wireframe: true,
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(flat), Box::new(extruded)]);
    let c = SIZE as f64 / 2.0;
    let hit = deck.pick(c + 16.0, c).unwrap().expect("column");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("columns", 0));

    let flat = ColumnLayer::new(ColumnLayerProps {
        base: LayerProps::new("disks"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        radius: 10.0,
        radius_units: Unit::Pixels,
        extruded: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(flat)]);
    let c = SIZE / 2;
    // Flat disks are unlit, so the fill color comes through exactly
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, c + 6, c, [255, 0, 0, 255], 1);
    assert_pixel(&pixels, c + 14, c, [0, 0, 0, 0], 0);
}

#[test]
fn material_controls_shading_of_extruded_columns() {
    let Some(ctx) = context() else { return };
    let column = |material: Material| {
        Box::new(ColumnLayer::new(ColumnLayerProps {
            base: LayerProps {
                material,
                ..LayerProps::new("column")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_fill_color: Accessor::Constant([20, 40, 60, 255]),
            get_elevation: Accessor::Constant(200.0),
            radius: 12.0,
            radius_units: Unit::Pixels,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let c = SIZE / 2;
    // Unlit columns keep the fill colour exactly, like deck.gl's `material: false`
    let unlit = render(&ctx, vec![column(Material::unlit())]);
    assert_pixel(&unlit, c, c, [20, 40, 60, 255], 1);
    // The default material shades the lit top face
    let lit = render(&ctx, vec![column(Material::default())]);
    let shaded = pixel(&lit, c, c);
    assert_eq!(shaded[3], 255);
    assert_ne!(&shaded[..3], &[20, 40, 60], "lighting should change the colour");
    // A brighter ambient term brightens the face
    let bright = render(
        &ctx,
        vec![column(Material {
            ambient: 1.0,
            ..Material::default()
        })],
    );
    let brighter = pixel(&bright, c, c);
    let sum = |p: [u8; 4]| p[..3].iter().map(|&v| v as u32).sum::<u32>();
    assert!(
        sum(brighter) > sum(shaded),
        "ambient 1.0 {brighter:?} should be brighter than the default {shaded:?}"
    );
}

#[test]
fn render_parameters_override_depth_test_and_blending() {
    let Some(ctx) = context() else { return };
    let column = || {
        Box::new(ColumnLayer::new(ColumnLayerProps {
            base: LayerProps {
                material: Material::unlit(),
                ..LayerProps::new("column")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_fill_color: Accessor::Constant([0, 0, 255, 255]),
            get_elevation: Accessor::Constant(200.0),
            radius: 14.0,
            radius_units: Unit::Pixels,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let disk = |parameters: RenderParameters, color: [u8; 4]| {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                parameters,
                ..LayerProps::new("disk")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_fill_color: Accessor::Constant(color),
            get_radius: Accessor::Constant(8.0),
            radius_units: Unit::Pixels,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let c = SIZE / 2;
    // The ground level disk sits under the column top, so the depth test hides it
    let hidden = render(
        &ctx,
        vec![column(), disk(RenderParameters::default(), [255, 0, 0, 255])],
    );
    assert_pixel(&hidden, c, c, [0, 0, 255, 255], 1);
    // depthTest: false draws it regardless of depth
    let on_top = render(
        &ctx,
        vec![
            column(),
            disk(
                RenderParameters {
                    depth_test: Some(false),
                    ..Default::default()
                },
                [255, 0, 0, 255],
            ),
        ],
    );
    assert_pixel(&on_top, c, c, [255, 0, 0, 255], 1);
    // depthCompare: always does the same
    let always = render(
        &ctx,
        vec![
            column(),
            disk(
                RenderParameters {
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    ..Default::default()
                },
                [255, 0, 0, 255],
            ),
        ],
    );
    assert_pixel(&always, c, c, [255, 0, 0, 255], 1);
    // A translucent disk blends over the column by default, and replaces it with blend: false
    let translucent = RenderParameters {
        depth_test: Some(false),
        ..Default::default()
    };
    let blended = render(&ctx, vec![column(), disk(translucent, [255, 0, 0, 128])]);
    assert_pixel(&blended, c, c, [128, 0, 127, 255], 2);
    let replaced = render(
        &ctx,
        vec![
            column(),
            disk(
                RenderParameters {
                    blend: Some(false),
                    ..translucent
                },
                [255, 0, 0, 128],
            ),
        ],
    );
    assert_pixel(&replaced, c, c, [128, 0, 0, 128], 2);
    // Additive blending sums the colours
    let additive = render(
        &ctx,
        vec![
            column(),
            disk(
                RenderParameters {
                    blend_state: Some(deck_gl::parameters::additive_blend()),
                    ..translucent
                },
                [255, 0, 0, 128],
            ),
        ],
    );
    assert_pixel(&additive, c, c, [128, 0, 255, 255], 2);
}

#[test]
fn snapshot_reads_back_the_rendered_frame() {
    let Some(ctx) = context() else { return };
    let layer = || {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new("disk"),
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_fill_color: Accessor::Constant([255, 0, 0, 255]),
            get_radius: Accessor::Constant(8.0),
            radius_units: Unit::Pixels,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let pixels = render(&ctx, vec![layer()]);
    let mut deck = make_deck(&ctx, vec![layer()]);
    let snapshot = deck.snapshot(None).unwrap();
    assert_eq!((snapshot.width, snapshot.height), (SIZE, SIZE));
    assert_eq!(snapshot.rgba, pixels, "snapshot matches a manual render");
    assert_eq!(snapshot.pixel(SIZE / 2, SIZE / 2), [255, 0, 0, 255]);
    // A clear colour fills the background and the deck can be snapshotted again
    let cleared = deck
        .snapshot(Some(wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        }))
        .unwrap();
    assert_eq!(cleared.pixel(1, 1), [0, 0, 255, 255]);
    assert_eq!(cleared.pixel(SIZE / 2, SIZE / 2), [255, 0, 0, 255]);
}

#[test]
fn hover_and_click_callbacks_with_auto_highlight() {
    use std::sync::Mutex;
    let Some(ctx) = context() else { return };
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log = |events: &Arc<Mutex<Vec<String>>>, entry: String| events.lock().unwrap().push(entry);
    let hover_log = events.clone();
    let click_log = events.clone();
    let layer = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            pickable: true,
            auto_highlight: true,
            highlight_color: [0, 0, 255, 255],
            on_hover: Some(HoverCallback::new(move |info| {
                log(
                    &hover_log,
                    match info {
                        Some(hit) => format!("hover {} {}", hit.layer_id, hit.index),
                        None => "leave".to_string(),
                    },
                )
            })),
            on_click: Some(ClickCallback::new(move |hit| {
                log(&click_log, format!("click {} {}", hit.layer_id, hit.index))
            })),
            ..LayerProps::new("points")
        },
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_radius: Accessor::Constant(12.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    let deck_log = events.clone();
    deck.set_on_hover(Some(HoverCallback::new(move |info| {
        log(&deck_log, format!("deck hover {}", info.is_some()))
    })));
    let c = SIZE as f64 / 2.0;
    deck.update().unwrap();
    assert_eq!(
        deck.snapshot(None).unwrap().pixel(SIZE / 2, SIZE / 2),
        [255, 0, 0, 255]
    );

    let hit = deck.pointer_move(c, c).unwrap().expect("hit");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("points", 0));
    assert_eq!(deck.hovered().map(|h| h.index), Some(0));
    // Moving within the same object does not fire again
    deck.pointer_move(c + 1.0, c).unwrap();
    // The auto highlight tints the hovered object
    assert_eq!(
        deck.snapshot(None).unwrap().pixel(SIZE / 2, SIZE / 2),
        [0, 0, 255, 255]
    );
    deck.click(c, c).unwrap();
    assert!(deck.pointer_move(1.0, 1.0).unwrap().is_none());
    assert!(deck.hovered().is_none());
    assert_eq!(
        deck.snapshot(None).unwrap().pixel(SIZE / 2, SIZE / 2),
        [255, 0, 0, 255]
    );
    deck.pointer_move(c, c).unwrap();
    deck.pointer_leave();
    assert_eq!(
        *events.lock().unwrap(),
        [
            "hover points 0",
            "deck hover true",
            "click points 0",
            "leave",
            "deck hover false",
            "hover points 0",
            "deck hover true",
            "leave",
            "deck hover false",
        ]
    );
}

/// A deck looking at `longitude` on the equator with the test size.
fn deck_at(
    ctx: &HeadlessContext,
    longitude: f64,
    zoom: f64,
    repeat: bool,
    layers: Vec<Box<dyn Layer>>,
) -> Deck {
    Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude,
                latitude: 0.0,
                zoom,
                pitch: 0.0,
                bearing: 0.0,
            },
            layers,
            repeat,
            ..Default::default()
        },
    )
    .expect("deck")
}

#[test]
fn repeat_draws_world_copies_across_the_antimeridian() {
    let Some(ctx) = context() else { return };
    let points = || {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                pickable: true,
                ..LayerProps::new("points")
            },
            data: LayerData::with_length(2),
            // One point just west of the antimeridian, one just east of it (12 px at zoom 14)
            get_position: Accessor::Func(Arc::new(|i| {
                if i == 0 {
                    [179.9995, 0.0, 0.0]
                } else {
                    [-179.9995, 0.0, 0.0]
                }
            })),
            get_fill_color: Accessor::Constant([255, 0, 0, 255]),
            get_radius: Accessor::Constant(4.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let c = SIZE / 2;
    let plain = deck_at(&ctx, 180.0, 14.0, false, vec![points()])
        .snapshot(None)
        .unwrap();
    assert_eq!(
        plain.pixel(c - 12, c),
        [255, 0, 0, 255],
        "west point is in this world"
    );
    assert_eq!(plain.pixel(c + 12, c), [0, 0, 0, 0], "east point is a world away");
    let repeated = deck_at(&ctx, 180.0, 14.0, true, vec![points()])
        .snapshot(None)
        .unwrap();
    assert_eq!(repeated.pixel(c - 12, c), [255, 0, 0, 255]);
    assert_eq!(
        repeated.pixel(c + 12, c),
        [255, 0, 0, 255],
        "the next world copy shows it"
    );
    // Picking still works on the main viewport
    let mut deck = deck_at(&ctx, 180.0, 14.0, true, vec![points()]);
    deck.snapshot(None).unwrap();
    let hit = deck.pick(c as f64 - 12.0, c as f64).unwrap().expect("hit");
    assert_eq!(hit.index, 0);
}

#[test]
fn line_layer_wrap_longitude_takes_the_shortest_path() {
    let Some(ctx) = context() else { return };
    let line = |wrap_longitude: bool| {
        Box::new(LineLayer::new(LineLayerProps {
            base: LayerProps {
                wrap_longitude,
                ..LayerProps::new("line")
            },
            data: LayerData::with_length(1),
            get_source_position: Accessor::Constant([179.5, 0.0, 0.0]),
            get_target_position: Accessor::Constant([-179.5, 0.0, 0.0]),
            get_color: Accessor::Constant([0, 255, 0, 255]),
            get_width: Accessor::Constant(4.0),
            width_units: Unit::Pixels,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let c = SIZE / 2;
    // Without wrapping the line runs the long way round, through longitude 0
    let long_way = deck_at(&ctx, 180.0, 8.0, true, vec![line(false)])
        .snapshot(None)
        .unwrap();
    assert_eq!(long_way.pixel(c - 4, c), [0, 0, 0, 0]);
    assert_eq!(long_way.pixel(c + 4, c), [0, 0, 0, 0]);
    // With wrapping it crosses the antimeridian: the west half is drawn in this world and
    // the east half needs the repeated world copy
    let short = deck_at(&ctx, 180.0, 8.0, false, vec![line(true)])
        .snapshot(None)
        .unwrap();
    assert_eq!(short.pixel(c - 4, c), [0, 255, 0, 255]);
    assert_eq!(short.pixel(c + 4, c), [0, 0, 0, 0]);
    let short_repeated = deck_at(&ctx, 180.0, 8.0, true, vec![line(true)])
        .snapshot(None)
        .unwrap();
    assert_eq!(short_repeated.pixel(c - 4, c), [0, 255, 0, 255]);
    assert_eq!(short_repeated.pixel(c + 4, c), [0, 255, 0, 255]);
}

#[test]
fn heatmap_layer_colours_dense_points_and_follows_the_view() {
    let Some(ctx) = context() else { return };
    let heatmap = |aggregation: HeatmapAggregation| {
        Box::new(HeatmapLayer::new(HeatmapLayerProps {
            base: LayerProps::new("heat"),
            data: LayerData::with_length(3),
            get_position: Accessor::Constant(CENTER),
            get_weight: Accessor::Constant(1.0),
            radius_pixels: 10.0,
            color_range: vec![[0, 0, 255, 255], [255, 0, 0, 255]],
            aggregation,
            weights_texture_size: 256,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let c = SIZE / 2;
    for aggregation in [HeatmapAggregation::Sum, HeatmapAggregation::Mean] {
        let mut deck = make_deck(&ctx, vec![heatmap(aggregation)]);
        let shot = deck.snapshot(None).unwrap();
        // The densest spot is the last colour of the range at full opacity
        assert_eq!(shot.pixel(c, c), [255, 0, 0, 255], "{aggregation:?}");
        // Beyond the radius nothing is drawn
        assert_eq!(shot.pixel(c + 20, c), [0, 0, 0, 0], "{aggregation:?}");
        // Off centre the weight fades through the range towards blue
        let edge = shot.pixel(c + 6, c);
        assert!(edge[3] > 0 && edge[2] > edge[0], "{aggregation:?}: {edge:?}");
        // Moving the view far away re-aggregates: nothing left on screen
        deck.set_view_state(ViewState {
            longitude: CENTER[0] + 1.0,
            latitude: CENTER[1],
            zoom: 14.0,
            pitch: 0.0,
            bearing: 0.0,
        });
        let moved = deck.snapshot(None).unwrap();
        assert_eq!(moved.pixel(c, c), [0, 0, 0, 0], "{aggregation:?}");
    }
}

#[test]
fn cartesian_positions_with_a_model_matrix_land_where_expected() {
    use deck_gl::glam::{DMat4, DVec3};
    use deck_gl::math_gl::web_mercator::lng_lat_to_world;
    let Some(ctx) = context() else { return };
    let common = lng_lat_to_world([CENTER[0], CENTER[1]]);
    for (label, position, matrix) in [
        ("plain", [common[0], common[1], 0.0], None),
        (
            "matrix",
            [1.0, 1.0, 0.0],
            Some(DMat4::from_translation(DVec3::new(
                common[0] - 1.0,
                common[1] - 1.0,
                0.0,
            ))),
        ),
    ] {
        let layer = ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                coordinate_system: deck_gl::CoordinateSystem::Cartesian,
                model_matrix: matrix,
                ..LayerProps::new("probe")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(position),
            get_fill_color: Accessor::Constant([255, 0, 0, 255]),
            get_radius: Accessor::Constant(2.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        });
        let shot = make_deck(&ctx, vec![Box::new(layer)]).snapshot(None).unwrap();
        let hits: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| shot.pixel(x, y)[3] > 0)
            .collect();
        assert!(!hits.is_empty(), "{label}: nothing drawn");
        let n = hits.len() as f64;
        let (sx, sy) = hits
            .iter()
            .fold((0.0, 0.0), |a, h| (a.0 + h.0 as f64, a.1 + h.1 as f64));
        let centre = (sx / n, sy / n);
        let expected = (SIZE as f64 / 2.0 - 0.5, SIZE as f64 / 2.0 - 0.5);
        assert!(
            (centre.0 - expected.0).abs() < 0.6 && (centre.1 - expected.1).abs() < 0.6,
            "{label}: drawn at {centre:?}"
        );
    }
}

#[test]
fn offset_coordinate_systems_place_points_relative_to_the_origin() {
    use deck_gl::CoordinateSystem;
    let Some(ctx) = context() else { return };
    // At zoom 14 near the view centre one pixel is about 3.75 m
    let meters_per_pixel = 40_075_016.686 * CENTER[1].to_radians().cos() / (512.0 * 2f64.powi(14));
    let cases = [
        (
            "meter offsets",
            CoordinateSystem::MeterOffsets,
            [12.0 * meters_per_pixel, 0.0, 0.0],
            (12, 0),
        ),
        (
            "meter offsets north",
            CoordinateSystem::MeterOffsets,
            [0.0, 8.0 * meters_per_pixel, 0.0],
            (0, -8),
        ),
        (
            "lnglat offsets",
            CoordinateSystem::LngLatOffsets,
            [-0.0005, 0.0, 0.0],
            (-12, 0),
        ),
    ];
    for (label, coordinate_system, position, (dx, dy)) in cases {
        let layer = ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                coordinate_system,
                coordinate_origin: CENTER,
                ..LayerProps::new("offsets")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(position),
            get_fill_color: Accessor::Constant([255, 0, 0, 255]),
            get_radius: Accessor::Constant(2.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        });
        let shot = make_deck(&ctx, vec![Box::new(layer)]).snapshot(None).unwrap();
        let hits: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| shot.pixel(x, y)[3] > 0)
            .collect();
        assert!(!hits.is_empty(), "{label}: nothing drawn");
        let n = hits.len() as f64;
        let (sx, sy) = hits
            .iter()
            .fold((0.0, 0.0), |a, h| (a.0 + h.0 as f64, a.1 + h.1 as f64));
        let centre = (sx / n, sy / n);
        let expected = (
            SIZE as f64 / 2.0 - 0.5 + dx as f64,
            SIZE as f64 / 2.0 - 0.5 + dy as f64,
        );
        assert!(
            (centre.0 - expected.0).abs() < 1.0 && (centre.1 - expected.1).abs() < 1.0,
            "{label}: drawn at {centre:?}, expected {expected:?}"
        );
    }
}

#[test]
fn orthographic_and_orbit_views_draw_cartesian_data() {
    use deck_gl::{
        AnyViewState, OrbitViewProps, OrbitViewState, OrthographicViewProps, OrthographicViewState, View,
    };
    let Some(ctx) = context() else { return };
    let point = |position: [f64; 3], color: [u8; 4]| {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new(format!("point-{}", color[0])),
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(position),
            get_fill_color: Accessor::Constant(color),
            get_radius: Accessor::Constant(3.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let deck_with = |view: View, camera: AnyViewState, layers: Vec<Box<dyn Layer>>| {
        let mut deck = Deck::new(
            &ctx.device,
            &ctx.queue,
            RenderTarget::default(),
            DeckProps {
                width: SIZE,
                height: SIZE,
                view,
                layers,
                ..Default::default()
            },
        )
        .expect("deck");
        deck.set_any_view_state(camera);
        deck
    };
    let c = SIZE / 2;
    // Orthographic: the target sits at the centre, zoom 2 makes one unit four pixels, +y is down
    let mut deck = deck_with(
        View::Orthographic(OrthographicViewProps::default()),
        AnyViewState::Orthographic(OrthographicViewState {
            target: [100.0, 50.0, 0.0],
            zoom: 2.0,
            ..Default::default()
        }),
        vec![
            point([100.0, 50.0, 0.0], [255, 0, 0, 255]),
            point([104.0, 53.0, 0.0], [0, 255, 0, 255]),
        ],
    );
    let shot = deck.snapshot(None).unwrap();
    assert_eq!(shot.pixel(c, c), [255, 0, 0, 255]);
    assert_eq!(shot.pixel(c + 16, c + 12), [0, 255, 0, 255]);
    // Orbit: looking straight down at the target, +y is up on screen
    let mut deck = deck_with(
        View::Orbit(OrbitViewProps::default()),
        AnyViewState::Orbit(OrbitViewState {
            target: [10.0, 10.0, 0.0],
            zoom: 3.0,
            rotation_x: 90.0,
            rotation_orbit: 0.0,
        }),
        vec![
            point([10.0, 10.0, 0.0], [255, 0, 0, 255]),
            point([12.0, 11.0, 0.0], [0, 0, 255, 255]),
        ],
    );
    let shot = deck.snapshot(None).unwrap();
    assert_eq!(shot.pixel(c, c), [255, 0, 0, 255]);
    assert_eq!(shot.pixel(c + 16, c - 8), [0, 0, 255, 255]);
}

#[test]
fn globe_view_draws_points_on_the_sphere() {
    use deck_gl::{AnyViewState, GlobeViewProps, View};
    let Some(ctx) = context() else { return };
    let point = |position: [f64; 3], color: [u8; 4]| {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new(format!("point-{}", color[0])),
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(position),
            get_fill_color: Accessor::Constant(color),
            get_radius: Accessor::Constant(3.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let view_state = ViewState {
        longitude: 10.0,
        latitude: 40.0,
        zoom: 1.0,
        pitch: 0.0,
        bearing: 0.0,
    };
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            view: View::Globe(GlobeViewProps::default()),
            view_state,
            layers: vec![
                point([10.0, 40.0, 0.0], [255, 0, 0, 255]),
                point([10.0, 43.0, 0.0], [0, 0, 255, 255]),
            ],
            ..Default::default()
        },
    )
    .expect("deck");
    assert!(matches!(deck.any_view_state(), AnyViewState::Globe(_)));
    let shot = deck.snapshot(None).unwrap();
    let c = SIZE / 2;
    assert_eq!(shot.pixel(c, c), [255, 0, 0, 255], "the target is at the centre");
    // The northern point lies above the centre, where the viewport projects it
    let expected = deck
        .viewport()
        .project(deck_gl::glam::DVec3::new(10.0, 43.0, 0.0), true);
    assert_eq!(
        shot.pixel(expected.x.round() as u32, expected.y.round() as u32),
        [0, 0, 255, 255],
        "{expected:?}"
    );
    assert!(expected.y < c as f64);
}

#[test]
fn several_views_draw_in_their_rectangles_with_a_layer_filter() {
    use deck_gl::{
        AnyViewState, DeckView, Extent, LayerFilter, OrthographicViewProps, OrthographicViewState, View,
    };
    let Some(ctx) = context() else { return };
    let point = |id: &str, position: [f64; 3], color: [u8; 4]| {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                pickable: true,
                ..LayerProps::new(id)
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(position),
            get_fill_color: Accessor::Constant(color),
            get_radius: Accessor::Constant(3.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        })) as Box<dyn Layer>
    };
    let ortho = View::Orthographic(OrthographicViewProps::default());
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            layers: vec![
                point("red", [0.0, 0.0, 0.0], [255, 0, 0, 255]),
                point("blue", [0.0, 0.0, 0.0], [0, 0, 255, 255]),
            ],
            ..Default::default()
        },
    )
    .expect("deck");
    // Left half and right half, both looking at the origin
    deck.set_views(vec![
        DeckView::new("left", ortho).with_rect(
            Extent::Pixels(0.0),
            Extent::Pixels(0.0),
            Extent::Percent(50.0),
            Extent::Percent(100.0),
        ),
        DeckView::new("right", ortho).with_rect(
            Extent::Percent(50.0),
            Extent::Pixels(0.0),
            Extent::Percent(50.0),
            Extent::Percent(100.0),
        ),
    ]);
    deck.set_view_state_for(
        "left",
        AnyViewState::Orthographic(OrthographicViewState::default()),
    );
    deck.set_view_state_for(
        "right",
        AnyViewState::Orthographic(OrthographicViewState::default()),
    );
    // The blue point only shows on the right
    deck.set_layer_filter(Some(LayerFilter::new(|layer, view| {
        layer != "blue" || view == "right"
    })));
    let shot = deck.snapshot(None).unwrap();
    let (q, c) = (SIZE / 4, SIZE / 2);
    assert_eq!(shot.pixel(q, c), [255, 0, 0, 255], "left view shows red");
    assert_eq!(
        shot.pixel(q + c, c),
        [0, 0, 255, 255],
        "right view shows blue on top"
    );
    assert_eq!(shot.pixel(c - 1, 2), [0, 0, 0, 0]);
    // Picking reports the view and unprojects with its viewport
    let hit = deck.pick(q as f64 + c as f64, c as f64).unwrap().expect("hit");
    assert_eq!((hit.layer_id.as_str(), hit.view_id.as_str()), ("blue", "right"));
    assert!(
        hit.coordinate[0].abs() < 1.0 && hit.coordinate[1].abs() < 1.0,
        "{:?}",
        hit.coordinate
    );
    let hit = deck.pick(q as f64, c as f64).unwrap().expect("hit");
    assert_eq!((hit.layer_id.as_str(), hit.view_id.as_str()), ("red", "left"));
    // Back to a single view
    deck.set_views(Vec::new());
    deck.set_layer_filter(None);
    assert_eq!(deck.viewports().len(), 1);
}

#[test]
fn tile_layer_loads_tiles_in_the_background_and_draws_them() {
    use deck_gl_layers::{TileData, TileIndex, TileLayer, TileLayerProps, TileLoader};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(ctx) = context() else { return };
    // Each tile is a solid colour telling its index
    let loads = Arc::new(AtomicUsize::new(0));
    let counter = loads.clone();
    let loader = TileLoader::new(move |index, _bounds| {
        counter.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let color = [
            (index.x as u8 + 1) * 40,
            (index.y as u8 + 1) * 40,
            index.z as u8 * 60,
            255,
        ];
        let rgba: Vec<u8> = color.iter().copied().cycle().take(4 * 4).collect();
        Ok(Some(Arc::new(BitmapImage {
            width: 2,
            height: 2,
            rgba: Arc::new(rgba),
        }) as TileData))
    });
    let layer = TileLayer::new(TileLayerProps {
        base: LayerProps::new("tiles"),
        get_tile_data: Some(loader),
        max_requests: 2,
        ..Default::default()
    });
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude: 90.0,
                latitude: 45.0,
                zoom: 1.0,
                pitch: 0.0,
                bearing: 0.0,
            },
            layers: vec![Box::new(layer)],
            ..Default::default()
        },
    )
    .expect("deck");
    // The first frame starts the loads; nothing is drawn yet
    let first = deck.snapshot(None).unwrap();
    assert_eq!(first.pixel(SIZE / 2, SIZE / 2), [0, 0, 0, 0]);
    // Wait for the loads to finish, then the tile under the centre appears
    let mut shot = first;
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        shot = deck.snapshot(None).unwrap();
        if shot.pixel(SIZE / 2, SIZE / 2)[3] > 0 {
            break;
        }
    }
    // Longitude 90, latitude 45 lies in tile x 1, y 0 of zoom 1
    let index = TileIndex::new(1, 0, 1);
    let expected = [(index.x as u8 + 1) * 40, (index.y as u8 + 1) * 40, 60, 255];
    assert_eq!(shot.pixel(SIZE / 2, SIZE / 2), expected);
    let tile_layer = deck
        .layer_mut("tiles")
        .and_then(|l| l.as_any_mut().downcast_mut::<TileLayer>())
        .unwrap();
    assert!(tile_layer.is_loaded());
    assert!(loads.load(Ordering::SeqCst) >= 1);
}

#[test]
fn geo_cell_layers_fill_their_cells() {
    use deck_gl_layers::{geohash_bounds, GeoCellLayer, GeoCellLayerProps};
    let Some(ctx) = context() else { return };
    // The geohash of the view centre at 7 characters is about 150 m wide: it fills the middle
    let cell = geohash_encode(CENTER[0], CENTER[1], 7);
    let bounds = geohash_bounds(&cell).unwrap();
    assert!(bounds[1] < CENTER[0] && bounds[3] > CENTER[0]);
    let mut props = GeoCellLayerProps::geohash();
    props.polygon.data = LayerData::with_length(1);
    props.polygon.get_fill_color = Accessor::Constant([0, 200, 0, 255]);
    props.get_cell = Accessor::Constant(cell.clone());
    let layer = GeoCellLayer::new(props);
    let shot = make_deck(&ctx, vec![Box::new(layer)]).snapshot(None).unwrap();
    let c = SIZE / 2;
    assert_eq!(shot.pixel(c, c), [0, 200, 0, 255], "cell {cell} fills the centre");
    // An H3 cell of resolution 9 around the centre, extruded by default
    let mut props = GeoCellLayerProps::h3();
    props.polygon.data = LayerData::with_length(1);
    props.polygon.get_fill_color = Accessor::Constant([200, 0, 0, 255]);
    props.polygon.get_elevation = Accessor::Constant(10.0);
    props.polygon.base.material = Material::unlit();
    props.get_cell = Accessor::Constant("8928308280fffff".to_string());
    let layer = GeoCellLayer::new(props);
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    // Look at the cell's own centre
    let centre = deck_gl_layers::h3_polygon("8928308280fffff", 1.0).unwrap();
    let (lng, lat) = (
        centre.iter().map(|p| p[0]).sum::<f64>() / centre.len() as f64,
        centre.iter().map(|p| p[1]).sum::<f64>() / centre.len() as f64,
    );
    deck.set_view_state(ViewState {
        longitude: lng,
        latitude: lat,
        zoom: 14.0,
        pitch: 0.0,
        bearing: 0.0,
    });
    let shot = deck.snapshot(None).unwrap();
    assert_eq!(shot.pixel(c, c), [200, 0, 0, 255]);
}

/// Encode a geohash (the inverse of the layer's decoder), for the test above.
fn geohash_encode(lng: f64, lat: f64, length: usize) -> String {
    const BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
    let (mut min_lat, mut max_lat, mut min_lng, mut max_lng) = (-90.0f64, 90.0f64, -180.0f64, 180.0f64);
    let mut is_lng = true;
    let mut out = String::new();
    let mut value = 0u32;
    let mut bits = 0;
    while out.len() < length {
        if is_lng {
            let mid = (min_lng + max_lng) / 2.0;
            value = value * 2
                + if lng >= mid {
                    min_lng = mid;
                    1
                } else {
                    max_lng = mid;
                    0
                };
        } else {
            let mid = (min_lat + max_lat) / 2.0;
            value = value * 2
                + if lat >= mid {
                    min_lat = mid;
                    1
                } else {
                    max_lat = mid;
                    0
                };
        }
        is_lng = !is_lng;
        bits += 1;
        if bits == 5 {
            out.push(BASE32[value as usize] as char);
            value = 0;
            bits = 0;
        }
    }
    out
}

#[test]
fn mvt_layer_draws_decoded_vector_tiles() {
    use deck_gl_layers::{
        encode_tile, mvt_loader_with, GeoJsonLayerProps, MvtLayer, MvtLayerProps, TileLayerProps,
    };
    use std::collections::HashMap;
    let Some(ctx) = context() else { return };
    // Every tile holds one polygon covering its whole area, tagged with the tile's z
    let loader = mvt_loader_with(|index| {
        let mut props = HashMap::new();
        props.insert("z".to_string(), serde_json::Value::from(index.z));
        Ok(Some(encode_tile(&[(
            "fill",
            4096,
            vec![(
                3,
                vec![vec![[0, 0], [4096, 0], [4096, 4096], [0, 4096], [0, 0]]],
                props,
            )],
        )])))
    });
    let layer = MvtLayer::new(MvtLayerProps {
        tiles: TileLayerProps {
            base: LayerProps::new("vector"),
            get_tile_data: Some(loader),
            ..Default::default()
        },
        geojson: GeoJsonLayerProps {
            get_fill_color: Accessor::Constant([0, 0, 255, 255]),
            stroked: false,
            ..Default::default()
        },
        layers: Some(vec!["fill".to_string()]),
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    let mut shot = deck.snapshot(None).unwrap();
    for _ in 0..100 {
        if shot.pixel(SIZE / 2, SIZE / 2)[3] > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        shot = deck.snapshot(None).unwrap();
    }
    assert_eq!(shot.pixel(SIZE / 2, SIZE / 2), [0, 0, 255, 255]);
    assert_eq!(shot.pixel(1, 1), [0, 0, 255, 255], "tiles cover the whole view");
}

#[test]
fn wms_layer_requests_an_image_for_the_view() {
    use deck_gl_layers::{WmsFetch, WmsLayer, WmsLayerProps};
    use std::sync::Mutex;
    let Some(ctx) = context() else { return };
    // The stub server records the URL and returns a 2 x 2 green PNG
    let urls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = urls.clone();
    let mut png = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut png),
        &[0u8, 200, 0, 255, 0, 200, 0, 255, 0, 200, 0, 255, 0, 200, 0, 255],
        2,
        2,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();
    let layer = WmsLayer::new(WmsLayerProps {
        base: LayerProps::new("wms"),
        data: "https://wms.example/ows".to_string(),
        layers: vec!["osm".to_string()],
        debounce_ms: 0.0,
        fetch: Some(WmsFetch::new(move |url| {
            seen.lock().unwrap().push(url.to_string());
            Ok(png.clone())
        })),
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    let mut shot = deck.snapshot(None).unwrap();
    for _ in 0..100 {
        if shot.pixel(SIZE / 2, SIZE / 2)[3] > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        shot = deck.snapshot(None).unwrap();
    }
    assert_eq!(shot.pixel(SIZE / 2, SIZE / 2), [0, 200, 0, 255]);
    assert_eq!(shot.pixel(1, 1), [0, 200, 0, 255], "the image covers the view");
    let requested = urls.lock().unwrap().clone();
    assert_eq!(requested.len(), 1, "{requested:?}");
    assert!(
        requested[0].contains("REQUEST=GetMap&LAYERS=osm") && requested[0].contains("WIDTH=64&HEIGHT=64"),
        "{requested:?}"
    );
    // A moved view asks for a new image
    deck.set_view_state(ViewState {
        longitude: CENTER[0] + 0.01,
        latitude: CENTER[1],
        zoom: 14.0,
        pitch: 0.0,
        bearing: 0.0,
    });
    for _ in 0..100 {
        deck.snapshot(None).unwrap();
        if urls.lock().unwrap().len() >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(urls.lock().unwrap().len(), 2);
}

#[test]
fn frame_stats_count_layers_draws_and_uploads() {
    let Some(ctx) = context() else { return };
    let layer = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps::new("points"),
        data: LayerData::with_length(7),
        get_position: Accessor::Constant(CENTER),
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        get_radius: Accessor::Constant(2.0),
        radius_units: Unit::Pixels,
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    assert_eq!(deck.stats().frame, 0);
    deck.snapshot(None).unwrap();
    let first = deck.stats();
    assert_eq!(
        (first.frame, first.layers, first.draw_calls, first.instances),
        (1, 1, 1, 7)
    );
    assert!(first.uploaded_bytes > 0, "attributes were uploaded: {first:?}");
    deck.snapshot(None).unwrap();
    let second = deck.stats();
    assert_eq!((second.frame, second.draw_calls), (2, 1));
    assert_eq!(second.uploaded_bytes, 0, "nothing changed: {second:?}");
}

#[test]
fn prop_changes_upload_only_what_changed() {
    let Some(ctx) = context() else { return };
    let props = |radius_scale: f32, color: [u8; 4]| ScatterplotLayerProps {
        base: LayerProps::new("points"),
        data: LayerData::with_length(100),
        get_position: Accessor::Constant(CENTER),
        get_fill_color: Accessor::Constant(color),
        get_radius: Accessor::Constant(2.0),
        radius_units: Unit::Pixels,
        radius_scale,
        ..Default::default()
    };
    let mut deck = make_deck(
        &ctx,
        vec![Box::new(ScatterplotLayer::new(props(1.0, [255, 0, 0, 255])))],
    );
    deck.snapshot(None).unwrap();
    let full = deck.stats().uploaded_bytes;
    assert!(full > 0);
    // A uniform only change uploads nothing
    deck.set_layers(vec![Box::new(ScatterplotLayer::new(props(
        3.0,
        [255, 0, 0, 255],
    )))]);
    let shot = deck.snapshot(None).unwrap();
    assert_eq!(deck.stats().uploaded_bytes, 0, "{:?}", deck.stats());
    assert_eq!(
        shot.pixel(SIZE / 2 + 4, SIZE / 2),
        [255, 0, 0, 255],
        "the larger radius shows"
    );
    // A colour change uploads the colour buffer only: 4 bytes per instance
    deck.set_layers(vec![Box::new(ScatterplotLayer::new(props(
        3.0,
        [0, 0, 255, 255],
    )))]);
    let shot = deck.snapshot(None).unwrap();
    assert_eq!(deck.stats().uploaded_bytes, 400, "{:?}", deck.stats());
    assert_eq!(shot.pixel(SIZE / 2, SIZE / 2), [0, 0, 255, 255]);
    // The same for a line layer's width
    let line = |width: f32| LineLayerProps {
        base: LayerProps::new("line"),
        data: LayerData::with_length(10),
        get_source_position: Accessor::Constant([CENTER[0] - 0.001, CENTER[1], 0.0]),
        get_target_position: Accessor::Constant([CENTER[0] + 0.001, CENTER[1], 0.0]),
        get_color: Accessor::Constant([0, 255, 0, 255]),
        get_width: Accessor::Constant(width),
        width_units: Unit::Pixels,
        ..Default::default()
    };
    let mut deck = make_deck(&ctx, vec![Box::new(LineLayer::new(line(2.0)))]);
    deck.snapshot(None).unwrap();
    deck.set_layers(vec![Box::new(LineLayer::new(line(5.0)))]);
    deck.snapshot(None).unwrap();
    assert_eq!(deck.stats().uploaded_bytes, 40, "widths only: {:?}", deck.stats());
}

#[test]
fn arrow_columns_upload_without_conversion() {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, UInt8Builder};
    use arrow_array::{Array, RecordBatch};
    use arrow_schema::{Field, Schema};
    let Some(ctx) = context() else { return };
    let batch = |count: usize| {
        let mut positions = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        let mut colors = FixedSizeListBuilder::new(UInt8Builder::new(), 4);
        for i in 0..count {
            positions.values().append_value(CENTER[0] as f32);
            positions.values().append_value(CENTER[1] as f32);
            positions.values().append_value(0.0);
            positions.append(true);
            colors
                .values()
                .append_slice(&[0, 0, 255, if i == 0 { 255 } else { 0 }]);
            colors.append(true);
        }
        let positions = positions.finish();
        let colors = colors.finish();
        let schema = Schema::new(vec![
            Field::new("position", positions.data_type().clone(), false),
            Field::new("color", colors.data_type().clone(), false),
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(positions), Arc::new(colors)]).unwrap()
    };
    let uploaded = |count: usize| {
        let layer = ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new("arrow"),
            data: LayerData::from_batch(batch(count)),
            get_position: Accessor::column("position"),
            get_fill_color: Accessor::column("color"),
            get_radius: Accessor::Constant(3.0),
            radius_units: Unit::Pixels,
            antialiasing: false,
            ..Default::default()
        });
        let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
        let shot = deck.snapshot(None).unwrap();
        assert_eq!(shot.pixel(SIZE / 2, SIZE / 2), [0, 0, 255, 255]);
        deck.stats().uploaded_bytes
    };
    // Per row: positions high and low (12 bytes each), fill and line colours (4 each) and the
    // 20 byte instance data; the quad geometry of the layer is the same for both
    assert_eq!(uploaded(100) - uploaded(50), 50 * (12 + 12 + 4 + 4 + 20));
}

#[test]
fn contour_layer_draws_isolines_and_isobands() {
    use deck_gl::math_gl::web_mercator::{get_distance_scales, lng_lat_to_world, world_to_lng_lat};
    let Some(ctx) = context() else { return };
    // A 3 x 3 block of cells of 40 m around the view centre, one point per cell
    let cell_size = 40.0;
    let scales = get_distance_scales(CENTER[0], CENTER[1], false);
    let size = [
        scales.units_per_meter.x * cell_size,
        scales.units_per_meter.y * cell_size,
    ];
    let centroid = lng_lat_to_world([CENTER[0], CENTER[1]]);
    let origin = [
        (centroid[0] / size[0]).floor() * size[0],
        (centroid[1] / size[1]).floor() * size[1],
    ];
    let positions: Vec<[f64; 3]> = (-1..=1)
        .flat_map(|i| {
            (-1..=1).map(move |j| {
                let p = world_to_lng_lat([
                    origin[0] + (i as f64 + 0.5) * size[0],
                    origin[1] + (j as f64 + 0.5) * size[1],
                ]);
                [p[0], p[1], 0.0]
            })
        })
        .collect();
    let positions = Arc::new(positions);
    let props = |contours: Vec<Contour>| {
        let positions = positions.clone();
        ContourLayerProps {
            base: LayerProps::new("contours"),
            data: LayerData::with_length(9),
            get_position: Accessor::Func(Arc::new(move |i| positions[i])),
            cell_size,
            contours,
            ..Default::default()
        }
    };
    // The layer's grid origin is a whole number of cells from the world origin, so take it
    // from the layer rather than recomputing it with a slightly different cell size
    let grid = ContourLayer::aggregate(&props(Vec::new())).unwrap();
    assert_eq!((grid.x_range, grid.y_range), ([-1, 2], [-1, 2]));
    let (origin, size) = (grid.cell_origin_common, grid.cell_size_common);
    // Lattice point (lx, ly) in cell units to a pixel
    let pixel_at = |lx: f64, ly: f64| {
        let scale = 2f64.powi(14);
        let x = SIZE as f64 / 2.0 + (origin[0] + lx * size[0] - centroid[0]) * scale;
        let y = SIZE as f64 / 2.0 - (origin[1] + ly * size[1] - centroid[1]) * scale;
        (x.round() as u32, y.round() as u32)
    };
    // Cell values sit at cell centres: the block spans lattice -0.5 to 2.5 with its middle at
    // (0.5, 0.5), and the isoline crosses halfway between the last centre (1.5) and the empty one
    let (cx, cy) = pixel_at(0.5, 0.5);
    let (rx, ry) = pixel_at(2.0, 0.5);

    let bands = make_deck(
        &ctx,
        vec![Box::new(ContourLayer::new(props(vec![
            Contour::band(0.5, 10.0).with_color([255, 0, 0, 255])
        ])))],
    )
    .snapshot(None)
    .unwrap();
    assert_eq!(bands.pixel(cx, cy), [255, 0, 0, 255], "inside the band");
    assert_eq!(bands.pixel(1, 1), [0, 0, 0, 0], "far outside");

    let lines = make_deck(
        &ctx,
        vec![Box::new(ContourLayer::new(props(vec![Contour::line(0.5)
            .with_color([0, 255, 0, 255])
            .with_stroke_width(6.0)])))],
    )
    .snapshot(None)
    .unwrap();
    assert_eq!(lines.pixel(cx, cy), [0, 0, 0, 0], "no line through the middle");
    assert_eq!(
        lines.pixel(rx, ry),
        [0, 255, 0, 255],
        "the isoline rings the block"
    );
}

#[test]
fn point_cloud_layer_draws_lit_points() {
    let Some(ctx) = context() else { return };
    let layer = PointCloudLayer::new(PointCloudLayerProps {
        base: LayerProps::new("cloud"),
        data: LayerData::with_length(1),
        get_position: Accessor::Constant(CENTER),
        get_color: Accessor::Constant([255, 0, 0, 255]),
        point_size: 8.0,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    let p = pixel(&pixels, c, c);
    assert!(
        p[3] == 255 && p[0] > 120 && p[0] > p[1] + 40 && p[0] > p[2] + 40,
        "{p:?}"
    );
    assert_pixel(&pixels, c + 12, c, [0, 0, 0, 0], 0);
}

#[test]
fn icon_layer_draws_masked_icons_from_an_atlas() {
    let Some(ctx) = context() else { return };
    // 16x8 atlas: left half is an opaque white square icon, right half is transparent
    let mut rgba = vec![0u8; 16 * 8 * 4];
    for y in 0..8 {
        for x in 0..8 {
            let i = (y * 16 + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let mut mapping = std::collections::HashMap::new();
    mapping.insert("square".to_string(), IconMapping::new(0, 0, 8, 8).mask());
    mapping.insert("blank".to_string(), IconMapping::new(8, 0, 8, 8));
    let atlas = Arc::new(IconAtlas {
        image: BitmapImage::new(16, 8, rgba),
        mapping,
    });
    let layer = IconLayer::new(IconLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("icons")
        },
        data: LayerData::with_length(2),
        atlas: Some(atlas),
        get_position: Accessor::func(|i| [CENTER[0] + if i == 0 { -0.0007 } else { 0.0007 }, CENTER[1], 0.0]),
        get_icon: Accessor::func(|i| {
            if i == 0 {
                "square".to_string()
            } else {
                "blank".to_string()
            }
        }),
        get_color: Accessor::Constant([0, 128, 255, 255]),
        get_size: Accessor::Constant(16.0),
        ..Default::default()
    });
    let mut deck = make_deck(&ctx, vec![Box::new(layer)]);
    let c = SIZE as f64 / 2.0;
    let hit = deck.pick(c - 16.0, c).unwrap().expect("icon");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("icons", 0));
    // The blank icon is fully transparent and discarded, so nothing is picked there
    assert!(deck.pick(c + 16.0, c).unwrap().is_none());
    assert_eq!(
        IconAtlas::mapping_from_json(
            r#"{"m": {"x": 1, "y": 2, "width": 3, "height": 4, "anchorY": 4, "mask": true}}"#
        )
        .unwrap()["m"],
        IconMapping::new(1, 2, 3, 4)
            .mask()
            .anchor(1.5, 4.0)
            .into_anchor_x_default()
    );
}

trait AnchorFix {
    fn into_anchor_x_default(self) -> Self;
}
impl AnchorFix for IconMapping {
    fn into_anchor_x_default(mut self) -> Self {
        self.anchor_x = None;
        self
    }
}

#[test]
fn text_layer_draws_glyphs_and_picks_the_label() {
    let Some(ctx) = context() else { return };
    let layer = TextLayer::new(TextLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("labels")
        },
        data: LayerData::with_length(1),
        get_text: Accessor::Constant("HI".to_string()),
        get_position: Accessor::Constant(CENTER),
        get_size: Accessor::Constant(40.0),
        get_color: Accessor::Constant([255, 0, 0, 255]),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let c = SIZE / 2;
    // Two 40 px glyphs centred on the middle: red ink somewhere in the middle rows, none at
    // the far edges.
    let ink = |y: u32| (0..SIZE).filter(|x| pixel(&pixels, *x, y)[3] > 200).count();
    assert!(ink(c) > 4, "middle row has ink: {}", ink(c));
    let red = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|(x, y)| {
            let p = pixel(&pixels, *x, *y);
            p[3] > 200 && p[0] > 200 && p[1] < 50
        })
        .count();
    assert!(red > 40, "red glyph pixels {red}");
    assert_eq!(ink(0), 0);
    assert_eq!(ink(SIZE - 1), 0);

    // Picking on an inked pixel returns the label
    let mut deck = make_deck(
        &ctx,
        vec![Box::new(TextLayer::new(TextLayerProps {
            base: LayerProps {
                pickable: true,
                ..LayerProps::new("labels")
            },
            data: LayerData::with_length(1),
            get_text: Accessor::Constant("HI".to_string()),
            get_position: Accessor::Constant(CENTER),
            get_size: Accessor::Constant(40.0),
            ..Default::default()
        }))],
    );
    let inked = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .find(|(x, y)| pixel(&pixels, *x, *y)[3] > 200)
        .expect("an inked pixel");
    let hit = deck.pick(inked.0 as f64 + 0.5, inked.1 as f64 + 0.5).unwrap();
    assert_eq!(
        hit.map(|h: PickingInfo| (h.layer_id, h.index)),
        Some(("labels".to_string(), 0))
    );
}

#[test]
fn text_layer_background_and_sdf_outline() {
    let Some(ctx) = context() else { return };
    let layer = TextLayer::new(TextLayerProps {
        base: LayerProps::new("labels"),
        data: LayerData::with_length(1),
        get_text: Accessor::Constant("A".to_string()),
        get_position: Accessor::Constant(CENTER),
        get_size: Accessor::Constant(32.0),
        get_color: Accessor::Constant([255, 255, 255, 255]),
        background: true,
        get_background_color: Accessor::Constant([0, 0, 255, 255]),
        background_padding: [4.0, 4.0, 4.0, 4.0],
        font: deck_gl_layers::FontSettings {
            sdf: true,
            ..Default::default()
        },
        outline_width: 6.0,
        outline_color: [255, 0, 0, 255],
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let count = |f: &dyn Fn([u8; 4]) -> bool| {
        (0..SIZE * SIZE)
            .filter(|i| f(pixel(&pixels, i % SIZE, i / SIZE)))
            .count()
    };
    let blue = count(&|p| p[2] > 200 && p[0] < 50 && p[3] > 200);
    let white = count(&|p| p[0] > 200 && p[1] > 200 && p[2] > 200);
    let red = count(&|p| p[0] > 150 && p[1] < 100 && p[2] < 100);
    assert!(blue > 100, "background box {blue}");
    assert!(white > 10, "white fill {white}");
    assert!(red > 10, "red outline {red}");
}

/// Points clustered in two spots near the view centre: a dense cluster on the left, a sparse
/// one on the right.
fn clustered_points() -> (LayerData, Accessor<[f64; 3]>) {
    let mut points = Vec::new();
    for i in 0..40 {
        let t = i as f64 / 40.0;
        points.push([
            CENTER[0] - 0.0012 + t * 0.0002,
            CENTER[1] + (t * 7.0).sin() * 0.0001,
            0.0,
        ]);
    }
    for i in 0..4 {
        points.push([CENTER[0] + 0.0012, CENTER[1] + i as f64 * 0.00005, 0.0]);
    }
    let positions = std::sync::Arc::new(points);
    let n = positions.len();
    (LayerData::with_length(n), Accessor::func(move |i| positions[i]))
}

#[test]
fn hexagon_layer_aggregates_and_colors_bins() {
    let Some(ctx) = context() else { return };
    let (data, get_position) = clustered_points();
    let props = HexagonLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("hexagons")
        },
        data,
        radius: 60.0,
        aggregation: AggregationProps {
            get_position,
            color_aggregation: AggregationOperation::Count,
            ..Default::default()
        },
    };
    let aggregation = HexagonLayer::aggregate(&props).unwrap();
    assert!(
        aggregation.bins.len() >= 2 && aggregation.bins.len() <= 6,
        "{} bins",
        aggregation.bins.len()
    );
    let total: usize = aggregation.bins.iter().map(|b| b.count).sum();
    assert_eq!(total, 44);
    let densest = aggregation.bins.iter().max_by_key(|b| b.count).unwrap();
    assert!(
        densest.position[0] < CENTER[0],
        "the dense cluster is on the left"
    );
    assert_eq!(aggregation.color_domain[1], densest.count as f32);

    let layer = HexagonLayer::new(props.clone());
    let pixels = render(&ctx, vec![Box::new(layer)]);
    let mut deck = make_deck(&ctx, vec![Box::new(HexagonLayer::new(props))]);
    let at = |position: [f64; 2]| {
        let p = deck
            .viewport()
            .project(deck_gl::glam::DVec3::new(position[0], position[1], 0.0), true);
        (p.x.round() as u32, p.y.round() as u32)
    };
    // the densest bin gets the last colour of the range, the sparsest the first
    let sparsest = aggregation.bins.iter().min_by_key(|b| b.count).unwrap();
    let (dx, dy) = at(densest.position);
    let (sx, sy) = at(sparsest.position);
    assert_pixel(&pixels, dx, dy, [189, 0, 38, 255], 2);
    assert_pixel(&pixels, sx, sy, [255, 255, 178, 255], 2);

    // picking returns a bin index
    let hit = deck
        .pick(dx as f64, dy as f64)
        .unwrap()
        .expect("a bin under the cursor");
    assert_eq!(hit.layer_id, "hexagons");
    assert_eq!(aggregation.bins[hit.index as usize].count, densest.count);
}

#[test]
fn grid_layer_extrudes_cells_by_weight() {
    let Some(ctx) = context() else { return };
    let (data, get_position) = clustered_points();
    let props = GridLayerProps {
        base: LayerProps::new("grid"),
        data,
        cell_size: 80.0,
        aggregation: AggregationProps {
            get_position,
            extruded: true,
            elevation_range: [0.0, 500.0],
            color_range: vec![[0, 0, 255, 255], [255, 0, 0, 255]],
            ..Default::default()
        },
    };
    let aggregation = GridLayer::aggregate(&props).unwrap();
    assert!(aggregation.bins.len() >= 2, "{} bins", aggregation.bins.len());
    let tallest = aggregation
        .cells
        .iter()
        .map(|c| c.elevation)
        .fold(0.0f32, f32::max);
    assert!(
        (tallest - 500.0).abs() < 1e-3,
        "densest cell reaches the top of the range: {tallest}"
    );
    let pixels = render(&ctx, vec![Box::new(GridLayer::new(props))]);
    let colored = (0..SIZE * SIZE)
        .filter(|i| pixel(&pixels, i % SIZE, i / SIZE)[3] > 200)
        .count();
    assert!(colored > 50, "cells cover pixels: {colored}");
}

#[test]
fn trips_layer_shows_only_the_travelled_part() {
    let Some(ctx) = context() else { return };
    // A horizontal path across the view, timestamps 0 on the left to 100 on the right
    let path: Path = vec![
        [CENTER[0] - 0.002, CENTER[1], 0.0],
        [CENTER[0] + 0.002, CENTER[1], 0.0],
    ];
    let make = |current_time: f32, fade: bool| {
        let path = path.clone();
        TripsLayer::new(TripsLayerProps {
            path: PathLayerProps {
                base: LayerProps::new("trips"),
                data: LayerData::with_length(1),
                get_path: Accessor::func(move |_| path.clone()),
                get_color: Accessor::Constant([0, 255, 0, 255]),
                get_width: Accessor::Constant(6.0),
                width_units: Unit::Pixels,
                ..Default::default()
            },
            get_timestamps: Accessor::Constant(vec![0.0, 100.0]),
            current_time,
            trail_length: 1000.0,
            fade_trail: fade,
        })
    };
    let pixels = render(&ctx, vec![Box::new(make(50.0, false))]);
    let c = SIZE / 2;
    assert_pixel(&pixels, c - 12, c, [0, 255, 0, 255], 1);
    assert_pixel(&pixels, c + 12, c, [0, 0, 0, 0], 0);

    // With fading, older parts of the trail are more transparent
    let faded = render(
        &ctx,
        vec![Box::new(TripsLayer::new(TripsLayerProps {
            trail_length: 60.0,
            ..make(60.0, true).props().clone()
        }))],
    );
    let old = pixel(&faded, c - 20, c);
    let recent = pixel(&faded, c + 2, c);
    assert!(
        recent[3] > old[3] + 50,
        "recent {recent:?} is more opaque than old {old:?}"
    );
}

/// Render into a 4x multisampled target: edges get intermediate coverage and picking still
/// works through its single sample pass.
#[test]
fn multisampled_target_antialiases_and_picks() {
    let Some(ctx) = context() else { return };
    let target = RenderTarget {
        sample_count: 4,
        ..RenderTarget::default()
    };
    let make_layer = || -> Box<dyn Layer> {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                pickable: true,
                ..LayerProps::new("points")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_radius: Accessor::Constant(20.0),
            radius_units: Unit::Pixels,
            get_fill_color: Accessor::Constant([255, 0, 0, 255]),
            antialiasing: false,
            ..Default::default()
        }))
    };
    let color = create_render_texture(&ctx.device, "color", SIZE, SIZE, target.color_format);
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude: CENTER[0],
                latitude: CENTER[1],
                zoom: 14.0,
                pitch: 0.0,
                bearing: 0.0,
            },
            layers: vec![make_layer()],
            ..Default::default()
        },
    )
    .expect("deck");
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        None,
        Some(wgpu::Color::TRANSPARENT),
    )
    .expect("render");
    ctx.queue.submit([encoder.finish()]);
    let pixels = read_texture_rgba8(&ctx.device, &ctx.queue, &color).expect("readback");
    let c = SIZE / 2;
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 1);
    // Along the circle's edge some pixels are partially covered
    let partial = (0..SIZE * SIZE)
        .map(|i| pixel(&pixels, i % SIZE, i / SIZE)[3])
        .filter(|a| *a > 20 && *a < 235)
        .count();
    assert!(partial > 20, "partially covered edge pixels: {partial}");
    let hit = deck.pick(c as f64, c as f64).unwrap();
    assert_eq!(hit.map(|h| h.index), Some(0));

    // Loading a multisampled target's contents is refused
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let error = deck
        .render_with(
            &mut encoder,
            &color.create_view(&Default::default()),
            None,
            wgpu::LoadOp::Load,
            wgpu::LoadOp::Clear(1.0),
        )
        .unwrap_err();
    assert!(error.to_string().contains("multisampled"), "{error}");
}

#[test]
fn screen_grid_layer_bins_in_screen_space_and_follows_the_view() {
    let Some(ctx) = context() else { return };
    let (data, get_position) = clustered_points();
    let props = ScreenGridLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("screen")
        },
        data,
        cell_size_pixels: 16.0,
        cell_margin_pixels: 1.0,
        aggregation: AggregationOperation::Count,
        get_position,
        ..Default::default()
    };
    let mut deck = make_deck(&ctx, vec![Box::new(ScreenGridLayer::new(props.clone()))]);
    deck.update().unwrap();
    let layer = deck
        .layer_mut("screen")
        .and_then(|l| l.as_any_mut().downcast_mut::<ScreenGridLayer>())
        .unwrap();
    let bins = layer.bins().to_vec();
    assert!(!bins.is_empty());
    let total: usize = bins.iter().map(|b| b.count).sum();
    assert_eq!(total, 44, "every point on screen lands in a cell");
    let densest = bins.iter().max_by_key(|b| b.count).unwrap().clone();
    assert!(densest.col < 2, "the dense cluster is on the left: {densest:?}");

    let pixels = render(&ctx, vec![Box::new(ScreenGridLayer::new(props.clone()))]);
    let center_of = |b: &deck_gl_layers::ScreenGridBin| (b.col * 16 + 8, b.row * 16 + 8);
    let (x, y) = center_of(&densest);
    let dense = pixel(&pixels, x, y);
    assert_eq!(dense, [189, 0, 38, 255], "densest cell has the last colour");
    let hit = deck.pick(x as f64, y as f64).unwrap().expect("a cell");
    assert_eq!(bins[hit.index as usize].count, densest.count);

    // Zooming out re-aggregates: the clusters collapse into fewer cells
    deck.set_view_state(ViewState {
        zoom: 13.0,
        ..*deck.view_state()
    });
    deck.update().unwrap();
    let zoomed: Vec<(u32, u32)> = deck
        .layer_mut("screen")
        .and_then(|l| l.as_any_mut().downcast_mut::<ScreenGridLayer>())
        .unwrap()
        .bins()
        .iter()
        .map(|b| (b.col, b.row))
        .collect();
    let before: Vec<(u32, u32)> = bins.iter().map(|b| (b.col, b.row)).collect();
    assert!(!zoomed.is_empty());
    assert_ne!(zoomed, before, "cells move when the view changes");
}

/// A test extension: a per object `tintValues` attribute travels to the fragment stage as a
/// varying and scales the colour there, together with a uniform from the extension's module.
#[derive(Debug, PartialEq)]
struct TintExtension {
    get_tint: Accessor<f32>,
    scale: f32,
}

const TINT_MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "tint",
    source:
        "struct TintUniforms { scale: f32, };\n@group(0) @binding(auto) var<uniform> tint: TintUniforms;\n",
};

impl LayerExtension for TintExtension {
    fn name(&self) -> &'static str {
        "TintExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders {
            modules: vec![TINT_MODULE],
            injections: vec![
                ShaderInjection::new("vs:#main-start", "tint_value = tintValues;"),
                ShaderInjection::new(
                    "fs:DECKGL_FILTER_COLOR",
                    "color = vec4<f32>(color.rgb * tint_value * tint.scale, color.a);",
                ),
            ],
            attributes: vec![ExtensionAttribute {
                name: "tintValues",
                format: wgpu::VertexFormat::Float32,
            }],
            varyings: vec![ShaderField {
                name: "tint_value",
                ty: "f32",
            }],
        }
    }

    fn attributes(&self, _data: &LayerData) -> deck_gl::Result<Vec<(&'static str, AttributeSource)>> {
        Ok(vec![(
            "tintValues",
            AttributeSource::Floats(self.get_tint.clone()),
        )])
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        _ctx: &LayerContext,
        _viewport: &Viewport,
        _props: &LayerProps,
    ) -> deck_gl::Result<()> {
        model.uniforms("tint")?.set_f32("scale", self.scale)?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn tinted_circles(scale: f32) -> Box<dyn Layer> {
    let d = 0.0007;
    Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(TintExtension {
                get_tint: Accessor::func(|i| if i == 0 { 1.0 } else { 0.0 }),
                scale,
            }),
            ..LayerProps::new("tinted")
        },
        data: LayerData::with_length(2),
        get_position: Accessor::func(move |i| [CENTER[0] + if i == 0 { -d } else { d }, CENTER[1], 0.0]),
        get_radius: Accessor::Constant(8.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([200, 100, 0, 255]),
        antialiasing: false,
        ..Default::default()
    }))
}

#[test]
fn extensions_add_attributes_varyings_uniforms_and_hook_code() {
    let Some(ctx) = context() else { return };
    let pixels = render(&ctx, vec![tinted_circles(0.5)]);
    let c = SIZE / 2;
    // tint 1 times the uniform scale of 0.5
    assert_pixel(&pixels, c - 16, c, [100, 50, 0, 255], 2);
    // tint 0
    assert_pixel(&pixels, c + 16, c, [0, 0, 0, 255], 2);

    // A new extension instance with other options rebuilds the model with the new uniforms.
    let mut deck = make_deck(&ctx, vec![tinted_circles(0.5)]);
    deck.set_layers(vec![tinted_circles(1.0)]);
    let snapshot = deck.snapshot(Some(wgpu::Color::TRANSPARENT)).expect("snapshot");
    assert_pixel(&snapshot.rgba, c - 16, c, [200, 100, 0, 255], 2);
}

fn filtered_circles(filter: DataFilterExtension) -> Box<dyn Layer> {
    let d = 0.0007;
    Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(filter),
            ..LayerProps::new("filtered")
        },
        data: LayerData::with_length(3),
        get_position: Accessor::func(move |i| [CENTER[0] + (i as f64 - 1.0) * d, CENTER[1], 0.0]),
        get_radius: Accessor::Constant(6.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    }))
}

#[test]
fn data_filter_extension_hides_and_fades_objects() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    // values 0, 1 and 2 with the range [0.5, 1.5]: only the middle circle is drawn
    let value = Accessor::func(|i| i as f32);
    let pixels = render(
        &ctx,
        vec![filtered_circles(DataFilterExtension::new(
            value.clone(),
            [0.5, 1.5],
        ))],
    );
    assert_pixel(&pixels, c - 16, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [0, 0, 0, 0], 0);

    // a soft range of [1, 1] inside [-1, 3] gives the outer circles a filter value of 0.5:
    // half the opacity with the colour transform
    let faded = DataFilterExtension {
        filter_soft_range: Some(vec![[1.0, 1.0]]),
        filter_transform_size: false,
        ..DataFilterExtension::new(value.clone(), [-1.0, 3.0])
    };
    let pixels = render(&ctx, vec![filtered_circles(faded.clone())]);
    assert_pixel(&pixels, c - 16, c, [128, 0, 0, 128], 2);
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [128, 0, 0, 128], 2);
    // or half the radius with the size transform: 3 instead of 6 pixels
    let shrunk = DataFilterExtension {
        filter_transform_size: true,
        filter_transform_color: false,
        ..faded
    };
    let pixels = render(&ctx, vec![filtered_circles(shrunk)]);
    assert_pixel(&pixels, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c - 16 + 5, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 5, c, [255, 0, 0, 255], 0);

    // disabled: everything is drawn
    let pixels = render(
        &ctx,
        vec![filtered_circles(DataFilterExtension {
            filter_enabled: false,
            ..DataFilterExtension::new(value, [0.5, 1.5])
        })],
    );
    assert_pixel(&pixels, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [255, 0, 0, 255], 0);
}

#[test]
fn data_filter_extension_keeps_the_selected_categories() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let filter = DataFilterExtension {
        get_filter_category: Some(FilterCategories::One(Accessor::func(|i| i as u32))),
        filter_categories: vec![vec![0, 2]],
        ..Default::default()
    };
    assert_eq!(filter.count_filtered(&LayerData::with_length(3)).unwrap(), 2);
    let pixels = render(&ctx, vec![filtered_circles(filter)]);
    assert_pixel(&pixels, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 16, c, [255, 0, 0, 255], 0);
}

fn three_circles(id: &str, extensions: Extensions) -> Box<dyn Layer> {
    let d = 0.0007;
    Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            extensions,
            ..LayerProps::new(id)
        },
        data: LayerData::with_length(3),
        get_position: Accessor::func(move |i| [CENTER[0] + (i as f64 - 1.0) * d, CENTER[1], 0.0]),
        get_radius: Accessor::Constant(6.0),
        radius_units: Unit::Pixels,
        get_fill_color: Accessor::Constant([255, 0, 0, 255]),
        antialiasing: false,
        ..Default::default()
    }))
}

#[test]
fn brushing_extension_shows_objects_near_the_pointer() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    // the circles are about 62 metres apart: a 30 metre brush keeps one
    let layer = three_circles("brushed", Extensions::from_one(BrushingExtension::new(30.0)));
    let mut deck = make_deck(&ctx, vec![layer]);
    deck.pointer_move((c - 16) as f64, c as f64).expect("pointer");
    let snapshot = deck.snapshot(Some(wgpu::Color::TRANSPARENT)).expect("snapshot");
    assert_pixel(&snapshot.rgba, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&snapshot.rgba, c, c, [0, 0, 0, 0], 0);
    assert_pixel(&snapshot.rgba, c + 16, c, [0, 0, 0, 0], 0);
    // without a pointer everything is drawn
    deck.pointer_leave();
    let snapshot = deck.snapshot(Some(wgpu::Color::TRANSPARENT)).expect("snapshot");
    assert_pixel(&snapshot.rgba, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&snapshot.rgba, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&snapshot.rgba, c + 16, c, [255, 0, 0, 255], 0);
}

#[test]
fn clip_extension_clips_by_anchor_or_by_geometry() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let d = 0.0007;
    // by instance: the left circle's centre is outside the bounds
    let bounds = [
        CENTER[0] - d / 2.0,
        CENTER[1] - 1.0,
        CENTER[0] + 2.0 * d,
        CENTER[1] + 1.0,
    ];
    let layer = three_circles("clipped", Extensions::from_one(ClipExtension::new(bounds, true)));
    let pixels = render(&ctx, vec![layer]);
    assert_pixel(&pixels, c - 16, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [255, 0, 0, 255], 0);

    // by geometry: a square is trimmed to its left half
    let square = Arc::new(vec![vec![
        [CENTER[0] - d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] + d, 0.0],
        [CENTER[0] - d, CENTER[1] + d, 0.0],
    ]]);
    let half = [CENTER[0] - d, CENTER[1] - d, CENTER[0], CENTER[1] + d];
    let layer = SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(ClipExtension::new(half, false)),
            ..LayerProps::new("trimmed")
        },
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*square).clone()),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(layer)]);
    assert_pixel(&pixels, c - 8, c, [0, 0, 255, 255], 0);
    assert_pixel(&pixels, c - 2, c, [0, 0, 255, 255], 0);
    assert_pixel(&pixels, c + 2, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 8, c, [0, 0, 0, 0], 0);
}

#[test]
fn mask_extension_keeps_what_the_mask_layer_covers() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let d = 0.0007;
    // the mask covers the centre and the right circle
    let fence = Arc::new(vec![vec![
        [CENTER[0] - d / 2.0, CENTER[1] - d, 0.0],
        [CENTER[0] + 2.0 * d, CENTER[1] - d, 0.0],
        [CENTER[0] + 2.0 * d, CENTER[1] + d, 0.0],
        [CENTER[0] - d / 2.0, CENTER[1] + d, 0.0],
    ]]);
    let mask_layer = || -> Box<dyn Layer> {
        let fence = fence.clone();
        Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
            base: LayerProps {
                operation: Operation::MASK,
                ..LayerProps::new("fence")
            },
            data: LayerData::with_length(1),
            get_polygon: Accessor::func(move |_| (*fence).clone()),
            get_fill_color: Accessor::Constant([0, 255, 0, 255]),
            ..Default::default()
        }))
    };
    let masked = three_circles("masked", Extensions::from_one(MaskExtension::new("fence")));
    let pixels = render(&ctx, vec![mask_layer(), masked]);
    assert_pixel(&pixels, c - 16, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [255, 0, 0, 255], 0);
    // the mask layer itself is not drawn
    assert_pixel(&pixels, c + 8, c + 8, [0, 0, 0, 0], 0);

    let inverted = three_circles(
        "masked",
        Extensions::from_one(MaskExtension {
            mask_inverted: true,
            ..MaskExtension::new("fence")
        }),
    );
    let pixels = render(&ctx, vec![mask_layer(), inverted]);
    assert_pixel(&pixels, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 16, c, [0, 0, 0, 0], 0);

    // by geometry: a square larger than the mask is trimmed to it
    let square = Arc::new(vec![vec![
        [CENTER[0] - 2.0 * d, CENTER[1] - 2.0 * d, 0.0],
        [CENTER[0] + 2.0 * d, CENTER[1] - 2.0 * d, 0.0],
        [CENTER[0] + 2.0 * d, CENTER[1] + 2.0 * d, 0.0],
        [CENTER[0] - 2.0 * d, CENTER[1] + 2.0 * d, 0.0],
    ]]);
    let trimmed = SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(MaskExtension {
                mask_by_instance: false,
                ..MaskExtension::new("fence")
            }),
            ..LayerProps::new("trimmed")
        },
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*square).clone()),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![mask_layer(), Box::new(trimmed)]);
    assert_pixel(&pixels, c - 16, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c - 4, c, [0, 0, 255, 255], 0);
    assert_pixel(&pixels, c + 16, c, [0, 0, 255, 255], 0);
    assert_pixel(&pixels, c + 16, c - 24, [0, 0, 0, 0], 0);
}

#[test]
fn collision_filter_extension_keeps_the_highest_priority_object() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let d = 0.0007;
    // two circles on the same spot (red wins on priority) and one apart
    let circles = |enabled: bool| -> Box<dyn Layer> {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                extensions: Extensions::from_one(CollisionFilterExtension {
                    collision_enabled: enabled,
                    ..CollisionFilterExtension::new(Accessor::func(|i| if i == 0 { 10.0 } else { 0.0 }))
                }),
                ..LayerProps::new("labels")
            },
            data: LayerData::with_length(3),
            get_position: Accessor::func(move |i| [CENTER[0] + if i == 2 { d } else { 0.0 }, CENTER[1], 0.0]),
            get_radius: Accessor::Constant(6.0),
            radius_units: Unit::Pixels,
            get_fill_color: Accessor::func(|i| match i {
                0 => [255, 0, 0, 255],
                1 => [0, 0, 255, 255],
                _ => [0, 255, 0, 255],
            }),
            antialiasing: false,
            ..Default::default()
        }))
    };
    let pixels = render(&ctx, vec![circles(true)]);
    assert_pixel(&pixels, c, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c + 16, c, [0, 255, 0, 255], 0);
    // without the filter the later object draws on top
    let pixels = render(&ctx, vec![circles(false)]);
    assert_pixel(&pixels, c, c, [0, 0, 255, 255], 0);
}

#[test]
fn extension_attributes_expand_over_tessellated_paths_and_polygons() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let d = 0.0007;
    // a degree of latitude spans more pixels than a degree of longitude here: 16 pixels
    let dy = d * CENTER[1].to_radians().cos();
    // two horizontal paths, the lower one filtered out
    let paths = PathLayer::new(PathLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(DataFilterExtension::new(
                Accessor::func(|i| i as f32),
                [-0.5, 0.5],
            )),
            ..LayerProps::new("paths")
        },
        data: LayerData::with_length(2),
        get_path: Accessor::func(move |i| {
            let y = CENTER[1] + if i == 0 { dy } else { -dy };
            vec![[CENTER[0] - 2.0 * d, y, 0.0], [CENTER[0] + 2.0 * d, y, 0.0]]
        }),
        get_width: Accessor::Constant(4.0),
        width_units: Unit::Pixels,
        get_color: Accessor::Constant([255, 0, 0, 255]),
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(paths)]);
    assert_pixel(&pixels, c, c - 16, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c, c + 16, [0, 0, 0, 0], 0);

    // two squares side by side, the right one filtered out, through the composite polygon layer
    let squares = PolygonLayer::new(PolygonLayerProps {
        base: LayerProps {
            extensions: Extensions::from_one(DataFilterExtension::new(
                Accessor::func(|i| i as f32),
                [-0.5, 0.5],
            )),
            ..LayerProps::new("squares")
        },
        data: LayerData::with_length(2),
        get_polygon: Accessor::func(move |i| {
            let x = CENTER[0] + if i == 0 { -d } else { d };
            vec![vec![
                [x - d / 2.0, CENTER[1] - d / 2.0, 0.0],
                [x + d / 2.0, CENTER[1] - d / 2.0, 0.0],
                [x + d / 2.0, CENTER[1] + d / 2.0, 0.0],
                [x - d / 2.0, CENTER[1] + d / 2.0, 0.0],
            ]]
        }),
        get_fill_color: Accessor::Constant([0, 0, 255, 255]),
        stroked: false,
        ..Default::default()
    });
    let pixels = render(&ctx, vec![Box::new(squares)]);
    assert_pixel(&pixels, c - 16, c, [0, 0, 255, 255], 0);
    assert_pixel(&pixels, c + 16, c, [0, 0, 0, 0], 0);
}

#[test]
fn fill_style_extension_tiles_a_pattern_over_polygons() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let d = 0.0007;
    let square = Arc::new(vec![vec![
        [CENTER[0] - d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] - d, 0.0],
        [CENTER[0] + d, CENTER[1] + d, 0.0],
        [CENTER[0] - d, CENTER[1] + d, 0.0],
    ]]);
    // one pattern per atlas, filling it: a solid green one and a fully transparent one
    let atlas = |rgba: [u8; 4]| {
        Arc::new(FillPatternAtlas {
            image: BitmapImage::new(2, 2, rgba.repeat(4)),
            mapping: HashMap::from([(
                "pattern".to_string(),
                FillPattern {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                },
            )]),
        })
    };
    let green = atlas([0, 255, 0, 255]);
    let hole = atlas([0, 0, 0, 0]);
    let polygon = |atlas: &Arc<FillPatternAtlas>, pattern: &'static str, mask: bool| -> Box<dyn Layer> {
        let square = square.clone();
        Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
            base: LayerProps {
                extensions: Extensions::from_one(FillStyleExtension {
                    fill_pattern_mask: mask,
                    ..FillStyleExtension::pattern(atlas.clone(), Accessor::Constant(pattern.to_string()))
                }),
                ..LayerProps::new("patterned")
            },
            data: LayerData::with_length(1),
            get_polygon: Accessor::func(move |_| (*square).clone()),
            get_fill_color: Accessor::Constant([0, 0, 255, 255]),
            ..Default::default()
        }))
    };
    // the pattern's colours replace the fill
    let pixels = render(&ctx, vec![polygon(&green, "pattern", false)]);
    assert_pixel(&pixels, c, c, [0, 255, 0, 255], 0);
    assert_pixel(&pixels, c - 8, c + 8, [0, 255, 0, 255], 0);
    // as a mask the fill keeps its colour where the pattern is opaque
    let pixels = render(&ctx, vec![polygon(&green, "pattern", true)]);
    assert_pixel(&pixels, c, c, [0, 0, 255, 255], 0);
    // a transparent pattern hides the fill, an unknown one leaves it alone
    let pixels = render(&ctx, vec![polygon(&hole, "pattern", true)]);
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    let pixels = render(&ctx, vec![polygon(&green, "missing", true)]);
    assert_pixel(&pixels, c, c, [0, 0, 255, 255], 0);
}

#[test]
fn path_style_extension_dashes_and_offsets_paths() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    // 48 pixels of path, 4 pixels wide: dash lengths count half widths as in deck.gl, so
    // dashes of 8 are 16 pixels on, 16 off from the start of the path
    let half = 24.0 / 23301.0;
    let path = |extension: PathStyleExtension| -> Box<dyn Layer> {
        Box::new(PathLayer::new(PathLayerProps {
            base: LayerProps {
                extensions: Extensions::from_one(extension),
                ..LayerProps::new("styled")
            },
            data: LayerData::with_length(1),
            get_path: Accessor::func(move |_| {
                vec![
                    [CENTER[0] - half, CENTER[1], 0.0],
                    [CENTER[0] + half, CENTER[1], 0.0],
                ]
            }),
            get_width: Accessor::Constant(4.0),
            width_units: Unit::Pixels,
            get_color: Accessor::Constant([255, 0, 0, 255]),
            ..Default::default()
        }))
    };
    let pixels = render(
        &ctx,
        vec![path(PathStyleExtension::dashed(Accessor::Constant([8.0, 8.0])))],
    );
    assert_pixel(&pixels, c - 16, c, [255, 0, 0, 255], 0);
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    assert_pixel(&pixels, c + 16, c, [255, 0, 0, 255], 0);

    // an offset of one width moves the line four pixels to one side
    let pixels = render(
        &ctx,
        vec![path(PathStyleExtension::offset(Accessor::Constant(1.0)))],
    );
    assert_pixel(&pixels, c, c, [0, 0, 0, 0], 0);
    let above = pixel(&pixels, c, c - 4);
    let below = pixel(&pixels, c, c + 4);
    assert!(
        (above == [255, 0, 0, 255]) != (below == [255, 0, 0, 255]),
        "the line moved to one side: {above:?} {below:?}"
    );
}

#[test]
fn path_style_extension_dashes_scatterplot_strokes() {
    let Some(ctx) = context() else { return };
    let c = SIZE / 2;
    let circle = |dash: [f32; 2]| -> Box<dyn Layer> {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                extensions: Extensions::from_one(PathStyleExtension {
                    target: PathStyleTarget::Scatterplot,
                    ..PathStyleExtension::dashed(Accessor::Constant(dash))
                }),
                ..LayerProps::new("ring")
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_radius: Accessor::Constant(12.0),
            radius_units: Unit::Pixels,
            stroked: true,
            filled: false,
            get_line_width: Accessor::Constant(4.0),
            line_width_units: Unit::Pixels,
            get_line_color: Accessor::Constant([255, 0, 0, 255]),
            antialiasing: false,
            ..Default::default()
        }))
    };
    // one long dash keeps the whole ring, a tiny dash with a huge gap removes it
    let pixels = render(&ctx, vec![circle([1000.0, 0.0])]);
    assert_pixel(&pixels, c + 12, c, [255, 0, 0, 255], 0);
    let pixels = render(&ctx, vec![circle([0.1, 1000.0])]);
    assert_pixel(&pixels, c + 12, c, [0, 0, 0, 0], 0);
}

#[test]
fn text_layer_takes_extension_attributes_per_glyph() {
    let Some(ctx) = context() else { return };
    // two labels on the same spot, a red and a blue one
    let labels = |extensions: Extensions| -> Box<dyn Layer> {
        Box::new(TextLayer::new(TextLayerProps {
            base: LayerProps {
                extensions,
                ..LayerProps::new("labels")
            },
            data: LayerData::with_length(2),
            get_text: Accessor::Constant("H".to_string()),
            get_position: Accessor::Constant(CENTER),
            get_size: Accessor::Constant(40.0),
            get_color: Accessor::func(|i| {
                if i == 0 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                }
            }),
            ..Default::default()
        }))
    };
    // glyph pixels of a colour; the collision filter fades thin glyphs where its five pixel
    // window reaches past the ink, so any visible alpha counts
    let count = |pixels: &[u8], red: bool| {
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|(x, y)| {
                let p = pixel(pixels, *x, *y);
                p[3] > 30 && if red { p[0] > p[2] } else { p[2] > p[0] }
            })
            .count()
    };
    // the collision filter keeps the label with the higher priority
    let priorities = Accessor::func(|i| if i == 0 { 10.0 } else { 0.0 });
    let pixels = render(
        &ctx,
        vec![labels(Extensions::from_one(CollisionFilterExtension::new(
            priorities,
        )))],
    );
    assert!(
        count(&pixels, true) > 40,
        "red glyph pixels {}",
        count(&pixels, true)
    );
    assert_eq!(count(&pixels, false), 0, "the blue label is hidden");
    // the data filter drops the first label and leaves the blue one
    let filter = DataFilterExtension::new(Accessor::func(|i| i as f32), [0.5, 1.5]);
    let pixels = render(&ctx, vec![labels(Extensions::from_one(filter))]);
    assert_eq!(count(&pixels, true), 0, "the red label is filtered out");
    assert!(
        count(&pixels, false) > 40,
        "blue glyph pixels {}",
        count(&pixels, false)
    );
}

#[test]
fn layers_of_one_type_share_shader_modules_and_pipelines() {
    let Some(ctx) = context() else { return };
    let circle = |id: &str, pickable: bool| -> Box<dyn Layer> {
        Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps {
                pickable,
                ..LayerProps::new(id)
            },
            data: LayerData::with_length(1),
            get_position: Accessor::Constant(CENTER),
            get_radius: Accessor::Constant(6.0),
            radius_units: Unit::Pixels,
            ..Default::default()
        }))
    };
    let mut deck = make_deck(
        &ctx,
        vec![circle("a", false), circle("b", false), circle("c", true)],
    );
    deck.update().expect("update");
    let cache = &deck.context().pipelines;
    // one pipeline for the three layers plus the picking pipeline of the pickable one
    assert_eq!(cache.builds(), 2, "pipelines built");
    assert_eq!(cache.len(), 2);
}

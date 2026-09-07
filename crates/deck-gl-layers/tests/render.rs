//! GPU integration tests: render each layer headlessly and check pixels.
//! Skipped (with a message) when no GPU adapter is available.

use std::sync::Arc;

use deck_gl::luma_gl::device::{
    create_headless_context, create_render_texture, read_texture_rgba8, HeadlessContext,
};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::wgpu;
use deck_gl::{
    Accessor, Deck, DeckProps, Layer, LayerData, LayerProps, Material, Path, PickingInfo, RenderParameters,
    Unit, ViewState,
};
use deck_gl_layers::{
    AggregationOperation, AggregationProps, ArcLayer, ArcLayerProps, BitmapImage, BitmapLayer,
    BitmapLayerProps, ColumnLayer, ColumnLayerProps, GeoJsonLayer, GeoJsonLayerProps, GridLayer,
    GridLayerProps, HexagonLayer, HexagonLayerProps, IconAtlas, IconLayer, IconLayerProps, IconMapping,
    LineLayer, LineLayerProps, PathLayer, PathLayerProps, PointCloudLayer, PointCloudLayerProps,
    PolygonLayer, PolygonLayerProps, ScatterplotLayer, ScatterplotLayerProps, ScreenGridLayer,
    ScreenGridLayerProps, SolidPolygonLayer, SolidPolygonLayerProps, TextLayer, TextLayerProps, TripsLayer,
    TripsLayerProps,
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

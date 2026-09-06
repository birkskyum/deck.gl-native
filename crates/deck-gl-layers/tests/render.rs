//! GPU integration tests: render each layer headlessly and check pixels.
//! Skipped (with a message) when no GPU adapter is available.

use std::sync::Arc;

use deck_gl::luma_gl::device::{
    create_headless_context, create_render_texture, read_texture_rgba8, HeadlessContext,
};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Accessor, Deck, DeckProps, Layer, LayerData, LayerProps, PickingInfo, Unit, ViewState};
use deck_gl_layers::{
    ArcLayer, ArcLayerProps, BitmapImage, BitmapLayer, BitmapLayerProps, ColumnLayer, ColumnLayerProps,
    GeoJsonLayer, GeoJsonLayerProps, IconAtlas, IconLayer, IconLayerProps, IconMapping, LineLayer,
    LineLayerProps, PathLayer, PathLayerProps, PointCloudLayer, PointCloudLayerProps, PolygonLayer,
    PolygonLayerProps, ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer, SolidPolygonLayerProps,
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

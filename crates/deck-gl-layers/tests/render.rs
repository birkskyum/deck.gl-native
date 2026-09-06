//! GPU integration tests: render each layer headlessly and check pixels.
//! Skipped (with a message) when no GPU adapter is available.

use std::sync::Arc;

use deck_gl::luma_gl::device::{
    create_headless_context, create_render_texture, read_texture_rgba8, HeadlessContext,
};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Accessor, Deck, DeckProps, Layer, LayerData, LayerProps, Unit, ViewState};
use deck_gl_layers::{
    LineLayer, LineLayerProps, ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer,
    SolidPolygonLayerProps,
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

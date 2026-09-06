//! GPU test: a JSON description renders and picks like the equivalent Rust layers.
//! Skipped (with a message) when no GPU adapter is available.

use deck_gl::luma_gl::device::{create_headless_context, create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_json::JsonConverter;

const SIZE: u32 = 64;

#[test]
fn json_layers_render_and_pick() {
    let ctx = match create_headless_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            return;
        }
    };
    let spec = r#"{
      "initialViewState": {"longitude": -122.4, "latitude": 37.8, "zoom": 14},
      "layers": [
        {
          "@@type": "SolidPolygonLayer",
          "id": "fill",
          "data": [{
            "polygon": [[-122.41, 37.79], [-122.39, 37.79], [-122.39, 37.81], [-122.41, 37.81]],
            "color": [0, 128, 255]
          }],
          "getPolygon": "@@=polygon",
          "getFillColor": "@@=color"
        },
        {
          "@@type": "ScatterplotLayer",
          "id": "dot",
          "data": [{"position": [-122.4, 37.8]}],
          "pickable": true,
          "radiusUnits": "pixels",
          "antialiasing": false,
          "getRadius": 6,
          "getFillColor": [255, 0, 0]
        }
      ]
    }"#;
    let json = JsonConverter::new().parse(spec).unwrap();
    assert!(json.warnings.is_empty(), "{:?}", json.warnings);

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
            view_state: json.view_state.unwrap(),
            layers: json.layers,
            ..Default::default()
        },
    )
    .unwrap();
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(wgpu::Color::TRANSPARENT),
    )
    .unwrap();
    ctx.queue.submit([encoder.finish()]);
    let pixels = read_texture_rgba8(&ctx.device, &ctx.queue, &color).unwrap();
    let pixel = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };

    let c = SIZE / 2;
    assert_eq!(pixel(c, c), [255, 0, 0, 255], "the dot covers the center");
    let corner = pixel(4, 4);
    assert!(
        corner[0] == 0 && corner[1].abs_diff(128) <= 1 && corner[2] == 255 && corner[3] == 255,
        "the polygon fills the corner, got {corner:?}"
    );

    let hit = deck.pick(c as f64, c as f64).unwrap().expect("the dot is picked");
    assert_eq!((hit.layer_id.as_str(), hit.index), ("dot", 0));
}

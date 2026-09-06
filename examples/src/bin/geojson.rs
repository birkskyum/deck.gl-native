//! Render a GeoJSON file with GeoJsonLayer, extruded and colored by properties, to a PNG.
//!
//! Defaults reproduce deck.gl's classic Vancouver blocks example:
//!
//! ```sh
//! curl -L -o vancouver-blocks.json \
//!   https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json
//! cargo run --release --bin geojson -- vancouver-blocks.json
//! ```

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use deck_gl::luma_gl::device::{create_headless_context, create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Accessor, Deck, DeckProps, FeatureCollection, LayerProps, Unit, ViewState};
use deck_gl_layers::{GeoJsonLayer, GeoJsonLayerProps};

/// deck.gl's GeoJsonLayer example threshold scale for the `growth` property
fn color_scale(growth: f64) -> [u8; 4] {
    const DOMAIN: [f64; 13] = [
        -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2,
    ];
    const RANGE: [[u8; 3]; 14] = [
        [65, 182, 196],
        [127, 205, 187],
        [199, 233, 180],
        [237, 248, 177],
        // zero
        [255, 255, 204],
        [255, 237, 160],
        [254, 217, 118],
        [254, 178, 76],
        [253, 141, 60],
        [252, 78, 42],
        [227, 26, 28],
        [189, 0, 38],
        [128, 0, 38],
        [128, 0, 38],
    ];
    let bucket = DOMAIN
        .iter()
        .position(|&threshold| growth < threshold)
        .unwrap_or(DOMAIN.len());
    let c = RANGE[bucket];
    [c[0], c[1], c[2], 255]
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "vancouver-blocks.json".to_string());
    let output = args.next().unwrap_or_else(|| "target/geojson.png".to_string());
    let (width, height) = (1280u32, 960u32);

    let started = Instant::now();
    let data = Arc::new(FeatureCollection::from_file(&input)?);
    println!("parsed {} features in {:?}", data.len(), started.elapsed());

    let features = data.clone();
    let layer = GeoJsonLayer::new(GeoJsonLayerProps {
        base: LayerProps {
            opacity: 0.8,
            pickable: true,
            ..LayerProps::new("geojson")
        },
        data: data.clone(),
        stroked: false,
        filled: true,
        extruded: true,
        wireframe: true,
        line_width_min_pixels: 1.0,
        get_elevation: Accessor::func({
            let features = features.clone();
            move |i| (features.features[i].number("valuePerSqm").unwrap_or(0.0).sqrt() * 10.0) as f32
        }),
        get_fill_color: Accessor::func({
            let features = features.clone();
            move |i| color_scale(features.features[i].number("growth").unwrap_or(0.0))
        }),
        get_line_color: Accessor::Constant([255, 255, 255, 255]),
        get_point_radius: Accessor::Constant(4.0),
        point_radius_units: Unit::Pixels,
        ..Default::default()
    });

    let ctx = create_headless_context()?;
    let target = RenderTarget::default();
    let color = create_render_texture(&ctx.device, "color", width, height, target.color_format);
    let depth = create_render_texture(&ctx.device, "depth", width, height, target.depth_format.unwrap());
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width,
            height,
            view_state: ViewState {
                longitude: -123.1,
                latitude: 49.25,
                zoom: 11.0,
                pitch: 45.0,
                bearing: 0.0,
            },
            layers: vec![Box::new(layer)],
            ..Default::default()
        },
    )?;

    let started = Instant::now();
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(wgpu::Color {
            r: 0.07,
            g: 0.08,
            b: 0.11,
            a: 1.0,
        }),
    )?;
    ctx.queue.submit([encoder.finish()]);
    let pixels = read_texture_rgba8(&ctx.device, &ctx.queue, &color)?;
    println!("tesselated, uploaded and rendered in {:?}", started.elapsed());

    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    image::save_buffer(&output, &pixels, width, height, image::ColorType::Rgba8)?;
    println!("wrote {output}");
    Ok(())
}

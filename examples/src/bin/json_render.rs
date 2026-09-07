//! Headless render of a JSON description into a PNG.
//!
//! Run with `cargo run --release --bin json_render -- spec.json [output.png] [WIDTHxHEIGHT]`.

use std::error::Error;

use deck_gl::luma_gl::device::{create_headless_context, create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_examples::{scene, spec};
use deck_gl_json::JsonConverter;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(spec) = args.next() else {
        eprintln!("usage: json_render <spec.json> [output.png] [WIDTHxHEIGHT]");
        std::process::exit(2);
    };
    let output = args
        .next()
        .unwrap_or_else(|| "target/json-render.png".to_string());
    let (width, height) = match args.next() {
        Some(size) => {
            let (w, h) = size.split_once('x').ok_or("size must look like 1024x768")?;
            (w.parse::<u32>()?, h.parse::<u32>()?)
        }
        None => (1024, 768),
    };

    let json = JsonConverter::parse_file(&spec)?;
    for warning in &json.warnings {
        eprintln!("{warning}");
    }
    let view_state = json
        .view_state
        .ok_or("the description has no initialViewState, so there is nothing to look at")?;

    let ctx = create_headless_context()?;
    let target = RenderTarget {
        sample_count: spec::msaa_samples(),
        ..RenderTarget::default()
    };
    let color = create_render_texture(&ctx.device, "color", width, height, target.color_format);
    let depth = create_render_texture(&ctx.device, "depth", width, height, target.depth_format.unwrap());
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width,
            height,
            view_state,
            layers: json.layers,
            ..Default::default()
        },
    )?;
    if let Some(lighting) = json.lighting {
        deck.set_lighting(lighting);
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(scene::CLEAR_COLOR),
    )?;
    ctx.queue.submit([encoder.finish()]);

    let pixels = read_texture_rgba8(&ctx.device, &ctx.queue, &color)?;
    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    image::save_buffer(&output, &pixels, width, height, image::ColorType::Rgba8)?;
    println!("wrote {output}");
    Ok(())
}

//! Headless render of the example scene into a texture, saved as a PNG.
//!
//! Run with `cargo run --release --bin texture_render [output.png]`.

use std::error::Error;

use deck_gl::luma_gl::device::{create_headless_context, create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_examples::scene;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/texture-render.png".to_string());
    let (width, height) = (1024u32, 768u32);

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
            view_state: scene::view_state(-25.0),
            layers: scene::layers(),
            ..Default::default()
        },
    )?;

    let color_view = color.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
    deck.render(
        &mut encoder,
        &color_view,
        Some(&depth_view),
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

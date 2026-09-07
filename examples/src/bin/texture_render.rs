//! Headless render of the example scene into a texture, saved as a PNG.
//!
//! Run with `cargo run --release --bin texture_render [output.png]`. Set `DECKGL_JSON` to render
//! a JSON description instead of the built-in scene.

use std::error::Error;

use deck_gl::luma_gl::device::create_headless_context;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_examples::{scene, spec};

fn main() -> Result<(), Box<dyn Error>> {
    deck_gl_examples::init_logging();
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/texture-render.png".to_string());
    let (width, height) = (1024u32, 768u32);

    let loaded = spec::load(-25.0)?;
    let ctx = create_headless_context()?;
    let target = RenderTarget {
        sample_count: spec::msaa_samples(),
        ..RenderTarget::default()
    };
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width,
            height,
            view_state: loaded.view_state,
            layers: loaded.layers,
            ..Default::default()
        },
    )?;
    if let Some(lighting) = loaded.lighting {
        deck.set_lighting(lighting);
    }
    deck.set_repeat(loaded.repeat);
    deck.set_view(loaded.view);
    if let Some(camera) = loaded.camera {
        deck.set_any_view_state(camera);
    }
    spec::apply_views(&mut deck, &loaded.views, &loaded.cameras);

    deck.snapshot(Some(scene::CLEAR_COLOR))?.save_png(&output)?;
    println!("wrote {output}");
    Ok(())
}

//! Headless render of a JSON description into a PNG.
//!
//! Run with `cargo run --release --bin json_render -- spec.json [output.png] [WIDTHxHEIGHT]`.

use std::error::Error;

use deck_gl::luma_gl::device::create_headless_context;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_examples::{scene, spec};
use deck_gl_json::JsonConverter;

fn main() -> Result<(), Box<dyn Error>> {
    deck_gl_examples::init_logging();
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
    if json.camera.is_none() {
        return Err("the description has no initialViewState, so there is nothing to look at".into());
    }
    let view_state = json.view_state.unwrap_or_default();

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
            view_state,
            layers: json.layers,
            ..Default::default()
        },
    )?;
    if let Some(lighting) = json.lighting {
        deck.set_lighting(lighting);
    }
    deck.set_post_process(json.post_process);
    deck.set_repeat(json.repeat);
    deck.set_view(json.view);
    if let Some(camera) = json.camera {
        deck.set_any_view_state(camera);
    }
    spec::apply_views(&mut deck, &json.views, &json.cameras);

    deck.snapshot(Some(scene::CLEAR_COLOR))?.save_png(&output)?;
    let stats = deck.stats();
    println!(
        "{} layers, {} draw calls, {} instances, {} KB uploaded, update {:.1} ms, draw {:.1} ms",
        stats.layers,
        stats.draw_calls,
        stats.instances,
        stats.uploaded_bytes / 1024,
        stats.update_ms,
        stats.draw_ms
    );
    println!("wrote {output}");
    Ok(())
}

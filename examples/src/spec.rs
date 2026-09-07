//! Layers from a JSON description, chosen with the `DECKGL_JSON` environment variable.
//!
//! Every example calls [`load`]. With `DECKGL_JSON=path/to/spec.json` the description's layers
//! and `initialViewState` replace the built-in scene; see `docs/json.md` for the format.

use std::error::Error;

use deck_gl::{Layer, LightingEffect, ViewState};
use deck_gl_json::JsonConverter;

use crate::scene;

pub const ENV_VAR: &str = "DECKGL_JSON";
/// Multisample count for the examples' render targets (`DECKGL_MSAA`, default 4).
pub fn msaa_samples() -> u32 {
    std::env::var("DECKGL_MSAA")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| [1, 2, 4, 8].contains(n))
        .unwrap_or(4)
}

/// Comma separated layer ids to keep, for looking at one layer at a time.
pub const ONLY_VAR: &str = "DECKGL_ONLY";

fn keep_only(mut layers: Vec<Box<dyn Layer>>) -> Vec<Box<dyn Layer>> {
    if let Ok(only) = std::env::var(ONLY_VAR) {
        let ids: Vec<&str> = only.split(',').map(str::trim).collect();
        layers.retain(|layer| ids.contains(&layer.id()));
    }
    layers
}

pub struct Scene {
    pub view_state: ViewState,
    pub layers: Vec<Box<dyn Layer>>,
    /// The description's `LightingEffect`, when it has one
    pub lighting: Option<LightingEffect>,
}

/// The built-in scene, or the description named by `DECKGL_JSON`.
pub fn load(bearing: f64) -> Result<Scene, Box<dyn Error>> {
    match std::env::var(ENV_VAR) {
        Ok(path) if !path.is_empty() => {
            let json = JsonConverter::parse_file(&path)?;
            for warning in &json.warnings {
                eprintln!("{ENV_VAR}: {warning}");
            }
            println!("loaded {} layers from {path}", json.layers.len());
            Ok(Scene {
                view_state: json.view_state.unwrap_or_else(|| scene::view_state(bearing)),
                layers: keep_only(json.layers),
                lighting: json.lighting,
            })
        }
        _ => Ok(Scene {
            view_state: scene::view_state(bearing),
            layers: keep_only(scene::layers()),
            lighting: None,
        }),
    }
}

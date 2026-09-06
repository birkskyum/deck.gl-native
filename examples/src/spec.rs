//! Layers from a JSON description, chosen with the `DECKGL_JSON` environment variable.
//!
//! Every example calls [`load`]. With `DECKGL_JSON=path/to/spec.json` the description's layers
//! and `initialViewState` replace the built-in scene; see `docs/json.md` for the format.

use std::error::Error;

use deck_gl::{Layer, ViewState};
use deck_gl_json::JsonConverter;

use crate::scene;

pub const ENV_VAR: &str = "DECKGL_JSON";

pub struct Scene {
    pub view_state: ViewState,
    pub layers: Vec<Box<dyn Layer>>,
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
                layers: json.layers,
            })
        }
        _ => Ok(Scene {
            view_state: scene::view_state(bearing),
            layers: scene::layers(),
        }),
    }
}

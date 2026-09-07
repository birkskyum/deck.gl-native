//! Port of `@deck.gl/extensions/src/terrain`: put a layer on the ground that the terrain
//! layers of the deck draw.
//!
//! A layer whose operation is [`Operation::TERRAIN`](deck_gl::layer::Operation::TERRAIN)
//! writes its elevation into a height map covering the view, and every layer carrying this
//! extension is lifted onto it: the anchor of each object is looked up in the map and its
//! geometry moves up by the ground's height there. deck.gl's other terrain mode, draping a
//! layer's pixels onto the terrain, is not here yet.

use std::any::Any;

use deck_gl::luma_gl::Model;
use deck_gl::terrain::{write_terrain_uniforms, TerrainMode};
use deck_gl::{same_extension, ExtensionShaders, LayerContext, LayerExtension, LayerProps, Result, Viewport};

/// deck.gl's `TerrainExtension`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainExtension;

impl TerrainExtension {
    pub fn new() -> Self {
        Self
    }
}

impl LayerExtension for TerrainExtension {
    fn name(&self) -> &'static str {
        "TerrainExtension"
    }

    fn needs_terrain(&self) -> bool {
        true
    }

    /// The terrain module is on every layer of a deck that has ground, so the extension adds
    /// no shader code of its own; it only picks the mode.
    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders::default()
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        _viewport: &Viewport,
        _props: &LayerProps,
    ) -> Result<()> {
        // While the height map itself is being drawn, this layer is not part of it
        let mode = if ctx.terrain_pass {
            TerrainMode::None
        } else {
            TerrainMode::UseHeightMap
        };
        write_terrain_uniforms(model, ctx, mode)
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

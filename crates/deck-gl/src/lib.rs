//! Rust port of `@deck.gl/core` on top of [`wgpu`].
//!
//! The module layout follows the JavaScript package so that the two code bases can be read
//! side by side: `viewport`, `shaderlib`, `layer`, `deck`. Shader code is the WGSL that
//! deck.gl 9 ships, reused verbatim where possible.
//!
//! The rendering entry point is [`Deck::draw`], which encodes into a caller-owned render
//! pass. That is the contract a basemap renderer such as maplibre-native needs for
//! interleaved rendering: the map owns the device, the attachments and the camera, and deck
//! layers draw into the same pass with the same depth buffer.

pub mod constants;
pub mod data;
pub mod deck;
pub mod layer;
pub mod lighting;
pub mod shaderlib;
pub mod viewport;

pub use constants::{CoordinateSystem, ProjectionMode, Unit};
pub use data::{Accessor, Color, LayerData, Path, Polygon, Position};
pub use deck::{Deck, DeckProps, PickingInfo, ViewState};
pub use layer::{Layer, LayerContext, LayerProps, SubLayers};
pub use lighting::{AmbientLight, DirectionalLight, LightingEffect, PointLight};
pub use viewport::{Viewport, WebMercatorViewportOptions};

pub use glam;
pub use luma_gl;
pub use luma_gl::wgpu;
pub use math_gl;

/// Errors produced by deck-gl.
#[derive(Debug, thiserror::Error)]
pub enum DeckError {
    #[error(transparent)]
    Luma(#[from] luma_gl::LumaError),
    #[error("data error: {0}")]
    Data(String),
    #[error("layer `{layer}`: {message}")]
    Layer { layer: String, message: String },
}

pub type Result<T> = std::result::Result<T, DeckError>;

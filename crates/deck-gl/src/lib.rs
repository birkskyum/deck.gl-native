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

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unreachable)
)]
pub mod constants;
pub mod controller;
pub mod data;
pub mod deck;
pub mod geojson;
pub mod globe_controller;
pub mod layer;
pub mod lighting;
pub mod orbit_controller;
pub mod parameters;
pub mod shaderlib;
pub mod transition;
pub mod viewport;
pub mod views;

pub use constants::{ClipDepthRange, CoordinateSystem, ProjectionMode, Unit};
pub use controller::{Constraints, MapController};
pub use data::{Accessor, Color, LayerData, Path, Polygon, Position};
pub use deck::{Deck, DeckProps, FrameStats, PickingInfo, Snapshot, ViewState};
pub use geojson::{Feature, FeatureCollection, Geometry};
pub use globe_controller::{GlobeConstraints, GlobeController};
pub use layer::{initialized, ClickCallback, HoverCallback, Layer, LayerContext, LayerProps, SubLayers};
pub use lighting::{AmbientLight, DirectionalLight, LightingEffect, Material, PointLight};
pub use orbit_controller::{OrbitConstraints, OrbitController};
pub use parameters::{CullMode, RenderParameters};
pub use transition::{
    TransitionDuration, TransitionInterpolator, TransitionInterruption, TransitionProps, ViewStateTransition,
};
pub use viewport::{
    FirstPersonViewportOptions, GlobeViewportOptions, OrbitViewportOptions, OrthographicViewportOptions,
    Padding, Viewport, ViewportOptions, WebMercatorViewportOptions,
};
pub use views::{
    AnyViewState, DeckView, Extent, FirstPersonViewProps, FirstPersonViewState, GlobeViewProps, LayerFilter,
    OrbitAxis, OrbitViewProps, OrbitViewState, OrthographicViewProps, OrthographicViewState, View,
    ViewPadding, ViewRect,
};

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
    #[error("render error: {0}")]
    Render(String),
}

pub type Result<T> = std::result::Result<T, DeckError>;

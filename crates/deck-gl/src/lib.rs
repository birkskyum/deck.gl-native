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
pub mod attribute_manager;
pub mod attributes;
pub mod collision;
pub mod constants;
pub mod controller;
pub mod data;
pub mod deck;
pub mod extension;
pub mod geojson;
pub mod globe_controller;
pub mod layer;
pub mod lighting;
pub mod mask;
pub mod orbit_controller;
pub mod parameters;
pub mod post_process;
pub mod shaderlib;
pub mod shadow;
pub mod terrain;
pub mod transition;
pub mod viewport;
pub mod views;
pub mod wkb;

pub use attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field, Part};
pub use collision::CollisionMaps;
pub use constants::{ClipDepthRange, CoordinateSystem, ProjectionMode, Unit};
pub use controller::{Constraints, MapController};
pub use data::{explode_multi, Accessor, Color, LayerData, MultiParts, Path, Polygon, Position};
pub use deck::{Deck, DeckProps, FrameStats, PickingInfo, Snapshot, ViewState};
pub use extension::{
    default_shaders, same_extension, set_default_shaders, ExtensionAttribute, ExtensionShaders, Extensions,
    LayerExtension,
};
pub use geojson::{Feature, FeatureCollection, Geometry};
pub use globe_controller::{GlobeConstraints, GlobeController};
pub use layer::{
    initialized, ClickCallback, HoverCallback, Layer, LayerContext, LayerProps, Operation, SubLayers,
};
pub use lighting::{AmbientLight, DirectionalLight, LightingEffect, Material, PointLight};
pub use mask::{MaskChannel, MaskMaps};
pub use orbit_controller::{OrbitConstraints, OrbitController};
pub use parameters::{CullMode, RenderParameters};
pub use post_process::{
    builtin_module, PassKind, PassSpec, PostProcessEffect, ShaderPassModule, UniformValue,
};
pub use shadow::{light_matrices, shadow_shaders, shadows_enabled, ShadowState, ShadowTarget};
pub use terrain::{terrain_shaders, TerrainMap, TerrainMode};
pub use transition::{
    EasingKind, PropTransition, PropTransitions, TransitionDuration, TransitionInterpolator,
    TransitionInterruption, TransitionProps, ViewStateTransition,
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

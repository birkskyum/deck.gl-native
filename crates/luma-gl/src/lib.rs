//! Rust port of the parts of [luma.gl](https://luma.gl) that deck.gl-native needs,
//! implemented as a thin layer over [`wgpu`].
//!
//! - [`shader`]: assembles WGSL shader modules and discovers uniform block layouts with naga
//! - [`uniform`]: CPU-side uniform blocks written by field name, uploaded to GPU buffers
//! - [`model`]: a render pipeline plus its bind group and vertex buffers
//! - [`buffer`]: helpers for building vertex buffers (fp64 splitting, colors)
//! - [`device`]: headless device creation and texture readback

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unreachable)
)]
pub mod buffer;
pub mod device;
pub mod model;
pub mod shader;
pub mod stats;
pub mod uniform;

pub use model::{
    create_rgba8_texture, default_sampler, Model, ModelDescriptor, RenderTarget, VertexBufferLayout,
    PICKING_FORMAT,
};
pub use shader::{
    assemble, assemble_shader, AssembledShader, ShaderAssembly, ShaderField, ShaderHook, ShaderInjection,
    ShaderModuleSource,
};
pub use uniform::UniformBlock;
pub use wgpu;

/// Errors produced by luma-gl.
#[derive(Debug, thiserror::Error)]
pub enum LumaError {
    #[error("shader error: {0}")]
    Shader(String),
    #[error("uniform error: {0}")]
    Uniform(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("device error: {0}")]
    Device(String),
}

pub type Result<T> = std::result::Result<T, LumaError>;

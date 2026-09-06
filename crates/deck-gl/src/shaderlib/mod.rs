//! Port of `@deck.gl/core/src/shaderlib`: the WGSL shader modules every layer is assembled
//! from, plus the CPU code that fills their uniform blocks.

pub mod project;

use luma_gl::ShaderModuleSource;

pub const GEOMETRY: ShaderModuleSource = ShaderModuleSource {
    name: "geometry",
    source: include_str!("wgsl/geometry.wgsl"),
};
pub const PROJECT: ShaderModuleSource = ShaderModuleSource {
    name: "project",
    source: include_str!("wgsl/project.wgsl"),
};
pub const PROJECT32: ShaderModuleSource = ShaderModuleSource {
    name: "project32",
    source: include_str!("wgsl/project32.wgsl"),
};
pub const LAYER: ShaderModuleSource = ShaderModuleSource {
    name: "layer",
    source: include_str!("wgsl/layer.wgsl"),
};
pub const PICKING: ShaderModuleSource = ShaderModuleSource {
    name: "picking",
    source: include_str!("wgsl/picking.wgsl"),
};
pub const COLOR: ShaderModuleSource = ShaderModuleSource {
    name: "color",
    source: include_str!("wgsl/color.wgsl"),
};
pub const FLOAT_COLORS: ShaderModuleSource = ShaderModuleSource {
    name: "floatColors",
    source: include_str!("wgsl/float_colors.wgsl"),
};
pub const LIGHTING: ShaderModuleSource = ShaderModuleSource {
    name: "lighting",
    source: include_str!("wgsl/lighting.wgsl"),
};
pub const GOURAUD_MATERIAL: ShaderModuleSource = ShaderModuleSource {
    name: "gouraudMaterial",
    source: include_str!("wgsl/gouraud_material.wgsl"),
};

/// The modules every deck.gl layer shader depends on, in dependency order.
/// Equivalent to `modules: [project32, color, picking]` plus the default `geometry` and
/// `layer` modules in deck.gl.
pub const STANDARD_MODULES: [ShaderModuleSource; 6] = [GEOMETRY, PROJECT, PROJECT32, LAYER, PICKING, COLOR];

/// Modules for lit layers (the `gouraudMaterial` module and its dependencies).
pub const LIGHTING_MODULES: [ShaderModuleSource; 3] = [FLOAT_COLORS, LIGHTING, GOURAUD_MATERIAL];

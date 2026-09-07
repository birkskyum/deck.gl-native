//! Port of `@deck.gl/core/src/shaderlib`: the WGSL shader modules every layer is assembled
//! from, plus the CPU code that fills their uniform blocks.

pub mod project;

use luma_gl::{ShaderHook, ShaderModuleSource};

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

/// deck.gl's shader hooks in their WGSL form. Layer shaders call these functions; extensions
/// inject code into them under the `key` (see [`luma_gl::ShaderInjection`]). The value is
/// returned rather than passed `inout`, and the fragment stage colour hook has its own name
/// because WGSL has no overloading.
pub const HOOKS: [ShaderHook; 4] = [
    ShaderHook {
        key: "vs:DECKGL_FILTER_SIZE",
        function: "deckgl_filter_size",
        value: "size",
        value_type: "vec3<f32>",
        context: "geometry: Geometry",
    },
    ShaderHook {
        key: "vs:DECKGL_FILTER_GL_POSITION",
        function: "deckgl_filter_gl_position",
        value: "position",
        value_type: "vec4<f32>",
        context: "geometry: Geometry",
    },
    ShaderHook {
        key: "vs:DECKGL_FILTER_COLOR",
        function: "deckgl_filter_color",
        value: "color",
        value_type: "vec4<f32>",
        context: "geometry: Geometry",
    },
    ShaderHook {
        key: "fs:DECKGL_FILTER_COLOR",
        function: "deckgl_filter_fragment_color",
        value: "color",
        value_type: "vec4<f32>",
        context: "geometry: FragmentGeometry",
    },
];

/// The modules every deck.gl layer shader depends on, in dependency order.
/// Equivalent to `modules: [project32, color, picking]` plus the default `geometry` and
/// `layer` modules in deck.gl.
pub const STANDARD_MODULES: [ShaderModuleSource; 6] = [GEOMETRY, PROJECT, PROJECT32, LAYER, PICKING, COLOR];

/// Modules for lit layers (the `gouraudMaterial` module and its dependencies).
pub const LIGHTING_MODULES: [ShaderModuleSource; 3] = [FLOAT_COLORS, LIGHTING, GOURAUD_MATERIAL];

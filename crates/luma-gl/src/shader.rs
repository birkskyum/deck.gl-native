//! WGSL shader module assembly.
//!
//! deck.gl composes each layer shader from reusable shader modules (`project`, `picking`,
//! `lighting`...). The modules declare their uniform buffers with `@group(N) @binding(auto)`.
//! This assembler concatenates module sources, assigns concrete binding slots, moves every
//! uniform into bind group 0, and then parses the result with naga to discover the layout of
//! each uniform struct so that [`crate::UniformBlock`]s can be written by field name.

use std::collections::BTreeSet;

use naga::{AddressSpace, ScalarKind, TypeInner, VectorSize};
use regex::Regex;

use crate::{LumaError, Result};

/// A named WGSL source fragment. Mirrors luma.gl's `ShaderModule.source`.
#[derive(Clone, Copy, Debug)]
pub struct ShaderModuleSource {
    pub name: &'static str,
    pub source: &'static str,
}

/// Kind of a uniform struct field, as far as the writer needs to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformKind {
    F32,
    I32,
    U32,
    Bool,
    Vec2F,
    Vec3F,
    Vec4F,
    Vec2I,
    Vec3I,
    Vec4I,
    Mat3F,
    Mat4F,
    /// Nested structs, arrays and anything else: written as raw bytes.
    Opaque,
}

/// One field of a uniform struct.
#[derive(Clone, Debug)]
pub struct UniformField {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: UniformKind,
}

/// Memory layout of a uniform struct in the WGSL uniform address space.
#[derive(Clone, Debug)]
pub struct UniformLayout {
    pub struct_name: String,
    pub size: u32,
    pub fields: Vec<UniformField>,
}

impl UniformLayout {
    pub fn field(&self, name: &str) -> Option<&UniformField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// A `var<uniform>` declaration found in the assembled shader.
#[derive(Clone, Debug)]
pub struct UniformBinding {
    /// Variable name in WGSL (for example `project`).
    pub name: String,
    pub group: u32,
    pub binding: u32,
    pub layout: UniformLayout,
}

/// Kind of a non-buffer resource binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    /// `texture_2d<f32>`
    Texture2D,
    /// `sampler`
    Sampler,
}

/// A texture or sampler declaration found in the assembled shader.
#[derive(Clone, Debug)]
pub struct ResourceBinding {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    pub kind: ResourceKind,
}

/// Result of [`assemble_shader`].
#[derive(Clone, Debug)]
pub struct AssembledShader {
    pub label: String,
    pub wgsl: String,
    pub uniforms: Vec<UniformBinding>,
    pub resources: Vec<ResourceBinding>,
}

impl AssembledShader {
    pub fn uniform(&self, name: &str) -> Option<&UniformBinding> {
        self.uniforms.iter().find(|u| u.name == name)
    }

    pub fn resource(&self, name: &str) -> Option<&ResourceBinding> {
        self.resources.iter().find(|r| r.name == name)
    }
}

/// Concatenate shader modules and the main shader, resolve `@binding(auto)`, and extract
/// uniform block layouts.
pub fn assemble_shader(label: &str, modules: &[ShaderModuleSource], main: &str) -> Result<AssembledShader> {
    let mut wgsl = String::new();
    for module in modules {
        wgsl.push_str(&format!("// ---- module: {} ----\n", module.name));
        wgsl.push_str(module.source);
        wgsl.push('\n');
    }
    wgsl.push_str("// ---- main shader ----\n");
    wgsl.push_str(main);
    wgsl.push('\n');

    let wgsl = resolve_bindings(&wgsl);

    let module = naga::front::wgsl::parse_str(&wgsl)
        .map_err(|e| LumaError::Shader(format!("{label}: {}", e.emit_to_string(&wgsl))))?;

    let gctx = module.to_ctx();
    let mut uniforms = Vec::new();
    let mut resources = Vec::new();
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else { continue };
        let name = var.name.clone().unwrap_or_default();
        let ty = &module.types[var.ty];
        if var.space == AddressSpace::Handle {
            let kind = match &ty.inner {
                TypeInner::Image {
                    dim: naga::ImageDimension::D2,
                    arrayed: false,
                    class: naga::ImageClass::Sampled { multi: false, .. },
                } => ResourceKind::Texture2D,
                TypeInner::Sampler { comparison: false } => ResourceKind::Sampler,
                other => {
                    return Err(LumaError::Shader(format!(
                        "{label}: unsupported resource binding `{name}`: {other:?}"
                    )))
                }
            };
            resources.push(ResourceBinding {
                name,
                group: binding.group,
                binding: binding.binding,
                kind,
            });
            continue;
        }
        if var.space != AddressSpace::Uniform {
            continue;
        }
        let layout = match &ty.inner {
            TypeInner::Struct { members, span } => {
                let mut fields = Vec::with_capacity(members.len());
                for member in members {
                    let member_ty = &module.types[member.ty];
                    fields.push(UniformField {
                        name: member.name.clone().unwrap_or_default(),
                        offset: member.offset,
                        size: member_ty.inner.size(gctx),
                        kind: classify(&member_ty.inner),
                    });
                }
                UniformLayout {
                    struct_name: ty.name.clone().unwrap_or_default(),
                    size: *span,
                    fields,
                }
            }
            other => UniformLayout {
                struct_name: ty.name.clone().unwrap_or_default(),
                size: other.size(gctx),
                fields: vec![UniformField {
                    name: String::new(),
                    offset: 0,
                    size: other.size(gctx),
                    kind: classify(other),
                }],
            },
        };
        uniforms.push(UniformBinding {
            name,
            group: binding.group,
            binding: binding.binding,
            layout,
        });
    }
    uniforms.sort_by_key(|u| u.binding);
    resources.sort_by_key(|r| r.binding);

    Ok(AssembledShader {
        label: label.to_string(),
        wgsl,
        uniforms,
        resources,
    })
}

fn classify(inner: &TypeInner) -> UniformKind {
    match inner {
        TypeInner::Scalar(s) => match s.kind {
            ScalarKind::Float => UniformKind::F32,
            ScalarKind::Sint => UniformKind::I32,
            ScalarKind::Uint => UniformKind::U32,
            ScalarKind::Bool => UniformKind::Bool,
            _ => UniformKind::Opaque,
        },
        TypeInner::Vector { size, scalar } => match (scalar.kind, size) {
            (ScalarKind::Float, VectorSize::Bi) => UniformKind::Vec2F,
            (ScalarKind::Float, VectorSize::Tri) => UniformKind::Vec3F,
            (ScalarKind::Float, VectorSize::Quad) => UniformKind::Vec4F,
            (ScalarKind::Sint | ScalarKind::Uint, VectorSize::Bi) => UniformKind::Vec2I,
            (ScalarKind::Sint | ScalarKind::Uint, VectorSize::Tri) => UniformKind::Vec3I,
            (ScalarKind::Sint | ScalarKind::Uint, VectorSize::Quad) => UniformKind::Vec4I,
            _ => UniformKind::Opaque,
        },
        TypeInner::Matrix { columns, rows, .. } => match (columns, rows) {
            (VectorSize::Quad, VectorSize::Quad) => UniformKind::Mat4F,
            (VectorSize::Tri, VectorSize::Tri) => UniformKind::Mat3F,
            _ => UniformKind::Opaque,
        },
        _ => UniformKind::Opaque,
    }
}

/// Replace `@binding(auto)` with concrete slots and move all groups to 0.
///
/// Explicit `@binding(N)` slots that appear in the source are reserved; `auto` slots take the
/// remaining numbers in order of appearance.
fn resolve_bindings(wgsl: &str) -> String {
    let explicit = Regex::new(r"@binding\((\d+)\)").expect("regex");
    let mut used: BTreeSet<u32> = explicit
        .captures_iter(wgsl)
        .filter_map(|c| c[1].parse().ok())
        .collect();

    let auto = Regex::new(r"@binding\(auto\)").expect("regex");
    let mut next = 0u32;
    let resolved = auto.replace_all(wgsl, |_: &regex::Captures| {
        while used.contains(&next) {
            next += 1;
        }
        used.insert(next);
        format!("@binding({next})")
    });

    let group = Regex::new(r"@group\(\d+\)").expect("regex");
    group.replace_all(&resolved, "@group(0)").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULE_A: &str = r#"
struct AUniforms { scale: f32, offset: vec3<f32>, flag: i32, };
@group(2) @binding(auto) var<uniform> a: AUniforms;
"#;
    const MAIN: &str = r#"
struct BUniforms { m: mat4x4<f32>, v: vec2<f32>, };
@group(0) @binding(0) var<uniform> b: BUniforms;
@vertex fn vertexMain(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
  return b.m * vec4<f32>(p * a.scale + a.offset, 1.0) + vec4<f32>(b.v, 0.0, f32(a.flag));
}
@fragment fn fragmentMain() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
"#;

    #[test]
    fn assigns_bindings_and_layouts() {
        let shader = assemble_shader(
            "test",
            &[ShaderModuleSource {
                name: "a",
                source: MODULE_A,
            }],
            MAIN,
        )
        .unwrap();
        assert!(shader.wgsl.contains("@group(0) @binding(1) var<uniform> a"));
        let a = shader.uniform("a").unwrap();
        assert_eq!(a.binding, 1);
        assert_eq!(a.group, 0);
        assert_eq!(a.layout.field("scale").unwrap().offset, 0);
        assert_eq!(a.layout.field("offset").unwrap().offset, 16);
        assert_eq!(a.layout.field("flag").unwrap().offset, 28);
        assert_eq!(a.layout.size, 32);
        let b = shader.uniform("b").unwrap();
        assert_eq!(b.binding, 0);
        assert_eq!(b.layout.field("m").unwrap().kind, UniformKind::Mat4F);
        assert_eq!(b.layout.field("v").unwrap().offset, 64);
        assert_eq!(b.layout.size, 80);
    }

    #[test]
    fn finds_textures_and_samplers() {
        let main = r#"
@group(0) @binding(auto) var tex: texture_2d<f32>;
@group(0) @binding(auto) var texSampler: sampler;
@vertex fn vertexMain(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
@fragment fn fragmentMain(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return textureSample(tex, texSampler, p.xy); }
"#;
        let shader = assemble_shader("tex", &[], main).unwrap();
        assert_eq!(shader.resource("tex").unwrap().kind, ResourceKind::Texture2D);
        assert_eq!(shader.resource("tex").unwrap().binding, 0);
        assert_eq!(shader.resource("texSampler").unwrap().kind, ResourceKind::Sampler);
        assert_eq!(shader.resource("texSampler").unwrap().binding, 1);
    }
}

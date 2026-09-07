//! WGSL shader module assembly.
//!
//! deck.gl composes each layer shader from reusable shader modules (`project`, `picking`,
//! `lighting`...). The modules declare their uniform buffers with `@group(N) @binding(auto)`.
//! This assembler concatenates module sources, assigns concrete binding slots, moves every
//! uniform into bind group 0, and then parses the result with naga to discover the layout of
//! each uniform struct so that [`crate::UniformBlock`]s can be written by field name.
//!
//! It also generates the shader hooks that layer shaders call so that extensions can inject
//! code (luma.gl's `vs:DECKGL_FILTER_COLOR` and friends). WGSL has no preprocessor, no
//! function overloading and no `inout` parameters, so the convention differs from GLSL:
//!
//! - a [`ShaderHook`] is a plain function taking the filtered value and returning it, whose body
//!   is generated from the [`ShaderInjection`]s registered for it;
//! - extension vertex attributes and varyings are appended to the main shader's `Attributes`
//!   and `Varyings` structs with the next free `@location`, and mirrored in module scope
//!   `var<private>` variables of the same name so injected code can read and write them;
//! - the generated `deckgl_vertex_start(attributes)`, `deckgl_vertex_end(&varyings)` and
//!   `deckgl_fragment_start(varyings)` functions move those values in and out of the structs
//!   and host the `#main-start` and `#main-end` injections.

use std::collections::BTreeSet;

use naga::{AddressSpace, ScalarKind, TypeInner, VectorSize};
use regex::Regex;

use crate::{LumaError, Result};

/// A named WGSL source fragment. Mirrors luma.gl's `ShaderModule.source`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderModuleSource {
    pub name: &'static str,
    pub source: &'static str,
}

/// A function the layer shader calls so that extensions can modify a value on its way through
/// the shader. Mirrors luma.gl's shader hooks (`vs:DECKGL_FILTER_COLOR(inout vec4 color, ...)`):
/// the assembler generates the function from the injections registered under `key`. WGSL has
/// no `inout`, so the hook takes the value and returns the filtered one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderHook {
    /// Injection key, for example `vs:DECKGL_FILTER_COLOR`.
    pub key: &'static str,
    /// WGSL function name, for example `deckgl_filter_color`.
    pub function: &'static str,
    /// Name of the filtered value inside injected code, for example `color`.
    pub value: &'static str,
    /// WGSL type of the value.
    pub value_type: &'static str,
    /// Further parameters the hook takes, for example `geometry: Geometry`. May be empty.
    pub context: &'static str,
}

/// Code an extension inserts into a layer shader. `hook` is a [`ShaderHook::key`] or one of
/// the fixed points `vs:#decl`, `vs:#main-start`, `vs:#main-end`, `fs:#decl` and
/// `fs:#main-start`. Declarations land in module scope; the others inside the generated
/// `deckgl_vertex_start`, `deckgl_vertex_end` and `deckgl_fragment_start` functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderInjection {
    pub hook: &'static str,
    pub code: String,
    /// Injections at the same hook run in ascending order, equal orders in registration order.
    pub order: i32,
}

impl ShaderInjection {
    pub fn new(hook: &'static str, code: impl Into<String>) -> Self {
        Self {
            hook,
            code: code.into(),
            order: 0,
        }
    }

    pub fn with_order(mut self, order: i32) -> Self {
        self.order = order;
        self
    }
}

/// A field an extension appends to the layer shader's `Attributes` struct (a vertex attribute)
/// or `Varyings` struct (a value interpolated from the vertex to the fragment stage). The field
/// is also available as a module scope variable of the same name inside injected code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderField {
    pub name: &'static str,
    /// WGSL type, for example `f32` or `vec2<f32>`.
    pub ty: &'static str,
}

/// Everything [`assemble`] combines into one WGSL shader.
#[derive(Clone, Copy, Debug)]
pub struct ShaderAssembly<'a> {
    pub label: &'a str,
    pub modules: &'a [ShaderModuleSource],
    pub main: &'a str,
    pub hooks: &'a [ShaderHook],
    pub injections: &'a [ShaderInjection],
    pub attributes: &'a [ShaderField],
    pub varyings: &'a [ShaderField],
}

const FIXED_INJECTION_POINTS: [&str; 5] = [
    "vs:#decl",
    "vs:#main-start",
    "vs:#main-end",
    "fs:#decl",
    "fs:#main-start",
];

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

/// Result of [`assemble`].
#[derive(Clone, Debug)]
pub struct AssembledShader {
    pub label: String,
    pub wgsl: String,
    pub uniforms: Vec<UniformBinding>,
    pub resources: Vec<ResourceBinding>,
    /// Shader locations assigned to the extension attributes, in declaration order.
    pub attribute_locations: Vec<(String, u32)>,
}

impl AssembledShader {
    pub fn uniform(&self, name: &str) -> Option<&UniformBinding> {
        self.uniforms.iter().find(|u| u.name == name)
    }

    pub fn resource(&self, name: &str) -> Option<&ResourceBinding> {
        self.resources.iter().find(|r| r.name == name)
    }

    /// The `@location` an extension attribute was appended at.
    pub fn attribute_location(&self, name: &str) -> Option<u32> {
        self.attribute_locations
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, location)| *location)
    }
}

/// Concatenate shader modules and the main shader, resolve `@binding(auto)`, and extract
/// uniform block layouts. [`assemble`] without hooks or extensions.
pub fn assemble_shader(label: &str, modules: &[ShaderModuleSource], main: &str) -> Result<AssembledShader> {
    assemble(&ShaderAssembly {
        label,
        modules,
        main,
        hooks: &[],
        injections: &[],
        attributes: &[],
        varyings: &[],
    })
}

/// Concatenate shader modules, the generated hooks and the main shader, resolve
/// `@binding(auto)`, and extract uniform block layouts.
pub fn assemble(assembly: &ShaderAssembly<'_>) -> Result<AssembledShader> {
    let label = assembly.label;
    // Sources checked out with CRLF line endings must still match exact text anchors.
    let main = assembly.main.replace("\r\n", "\n");
    let (main, attribute_locations) = append_fields(&main, "Attributes", assembly.attributes, false, label)?;
    let (main, _) = append_fields(&main, "Varyings", assembly.varyings, true, label)?;
    let hooks = generate_hooks(assembly, &main)?;

    let mut wgsl = String::new();
    for module in assembly.modules {
        wgsl.push_str(&format!("// ---- module: {} ----\n", module.name));
        wgsl.push_str(module.source);
        wgsl.push('\n');
    }
    wgsl.push_str("// ---- shader hooks ----\n");
    wgsl.push_str(&hooks);
    wgsl.push_str("// ---- main shader ----\n");
    wgsl.push_str(&main);
    wgsl.push('\n');

    let wgsl = resolve_bindings(&wgsl.replace("\r\n", "\n"));
    let wgsl = pad_uniform_blocks(&wgsl, label)?;

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

    let shader = AssembledShader {
        label: label.to_string(),
        wgsl,
        uniforms,
        resources,
        attribute_locations,
    };
    if std::env::var_os("LUMA_GL_CHECK_GLSL").is_some() {
        check_webgl2(&shader)?;
    }
    Ok(shader)
}

/// Whether an assembled shader can run on WebGL2: its WGSL lowers to GLSL ES 3.00, and its
/// uniform blocks are sized the way a WebGL2 device wants them.
///
/// wgpu checks all of this itself when it builds a pipeline, so this is not needed to run.
/// It is here so that a shader that cannot reach WebGL2 fails a test on any machine rather
/// than only in a browser. `LUMA_GL_CHECK_GLSL=1` runs it on every shader [`assemble`]
/// produces, which the layer render tests then cover.
pub fn check_webgl2(shader: &AssembledShader) -> Result<()> {
    // WebGL2 has no `DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`: a uniform block's
    // type has to be a whole number of 16 byte rows.
    let unaligned: Vec<String> = shader
        .uniforms
        .iter()
        .filter(|u| !u.layout.size.is_multiple_of(16))
        .map(|u| format!("{} ({}, {} bytes)", u.name, u.layout.struct_name, u.layout.size))
        .collect();
    if !unaligned.is_empty() {
        return Err(LumaError::Shader(format!(
            "{}: uniform blocks are not a multiple of 16 bytes, which WebGL2 needs: {}",
            shader.label,
            unaligned.join(", ")
        )));
    }
    to_glsl(shader)?;
    Ok(())
}

/// Lower an assembled shader to GLSL ES 3.00, the dialect WebGL2 speaks, returning one source
/// per entry point.
///
/// wgpu's WebGL2 backend does this itself when it builds a pipeline, so this is not needed to
/// run. It is here so that a shader that cannot reach WebGL2 fails a test on any machine
/// rather than only in a browser. Setting `LUMA_GL_CHECK_GLSL` runs it on every shader
/// [`assemble`] produces.
pub fn to_glsl(shader: &AssembledShader) -> Result<Vec<(String, String)>> {
    let label = &shader.label;
    let module = naga::front::wgsl::parse_str(&shader.wgsl)
        .map_err(|e| LumaError::Shader(format!("{label}: {}", e.emit_to_string(&shader.wgsl))))?;
    // The capabilities WebGL2 offers: no more than the base set.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| LumaError::Shader(format!("{label}: {}", e.emit_to_string(&shader.wgsl))))?;

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    let mut sources = Vec::with_capacity(module.entry_points.len());
    for entry_point in &module.entry_points {
        let pipeline_options = naga::back::glsl::PipelineOptions {
            shader_stage: entry_point.stage,
            entry_point: entry_point.name.clone(),
            multiview: None,
        };
        let mut glsl = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .map_err(|e| LumaError::Shader(format!("{label}: {} is not WebGL2 ready: {e}", entry_point.name)))?;
        writer.write().map_err(|e| {
            LumaError::Shader(format!("{label}: {} is not WebGL2 ready: {e}", entry_point.name))
        })?;
        sources.push((entry_point.name.clone(), glsl));
    }
    Ok(sources)
}

/// Round every uniform block up to a whole number of 16 byte rows.
///
/// WebGL2 devices have no `DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`, so a uniform
/// block whose type is, say, 8 bytes is rejected when the pipeline is built. Rather than
/// leave that to each shader to remember, the assembler appends a padding member to any
/// uniform struct that needs one. It costs a few bytes per block and nothing else: the field
/// is never written, since [`crate::UniformBlock`] writes by name.
fn pad_uniform_blocks(wgsl: &str, label: &str) -> Result<String> {
    let module = naga::front::wgsl::parse_str(wgsl)
        .map_err(|e| LumaError::Shader(format!("{label}: {}", e.emit_to_string(wgsl))))?;
    let mut pads: Vec<(String, u32)> = Vec::new();
    for (_, var) in module.global_variables.iter() {
        if var.space != AddressSpace::Uniform {
            continue;
        }
        let ty = &module.types[var.ty];
        let TypeInner::Struct { span, .. } = &ty.inner else {
            continue;
        };
        if span.is_multiple_of(16) {
            continue;
        }
        let Some(name) = ty.name.clone() else {
            continue;
        };
        if pads.iter().any(|(n, _)| *n == name) {
            continue;
        }
        pads.push((name, 16 - span % 16));
    }
    if pads.is_empty() {
        return Ok(wgsl.to_string());
    }

    let mut out = wgsl.to_string();
    for (name, pad) in pads {
        let Some(opening) = struct_regex(&name).find(&out) else {
            continue;
        };
        let Some(offset) = out[opening.end()..].find('}') else {
            return Err(LumaError::Shader(format!(
                "{label}: struct {name} is never closed"
            )));
        };
        let close = opening.end() + offset;
        // A body whose last member has no trailing comma needs one before the padding
        let separator = match out[opening.end()..close].trim_end().chars().last() {
            Some(',') | None => "",
            _ => ",",
        };
        out.insert_str(
            close,
            &format!("{separator}\n  // WebGL2 wants a uniform block to be a whole number of 16 byte rows\n  @size({pad}) _uniformBlockPadding: f32,\n"),
        );
    }
    Ok(out)
}

fn has_struct(wgsl: &str, name: &str) -> bool {
    struct_regex(name).is_match(wgsl)
}

fn struct_regex(name: &str) -> Regex {
    #[allow(clippy::expect_used)] // literal pattern around an identifier
    Regex::new(&format!(r"\bstruct\s+{name}\s*\{{")).expect("regex")
}

fn is_integer_type(ty: &str) -> bool {
    let ty = ty.trim();
    ty == "u32" || ty == "i32" || ty.ends_with("<u32>") || ty.ends_with("<i32>")
}

/// Append `fields` to the `struct <name>` of `main` with the next free `@location`s. Varyings
/// of integer type get `@interpolate(flat)`, as WGSL requires. Returns the new source and the
/// assigned locations.
fn append_fields(
    main: &str,
    name: &str,
    fields: &[ShaderField],
    varying: bool,
    label: &str,
) -> Result<(String, Vec<(String, u32)>)> {
    if fields.is_empty() {
        return Ok((main.to_string(), Vec::new()));
    }
    let names: Vec<&str> = fields.iter().map(|f| f.name).collect();
    let Some(found) = struct_regex(name).find(main) else {
        return Err(LumaError::Shader(format!(
            "{label}: the shader has no `{name}` struct to add {} to",
            names.join(", ")
        )));
    };
    let body_start = found.end();
    let Some(body_len) = main[body_start..].find('}') else {
        return Err(LumaError::Shader(format!(
            "{label}: unterminated `{name}` struct"
        )));
    };
    let body = &main[body_start..body_start + body_len];
    #[allow(clippy::expect_used)] // literal pattern
    let location = Regex::new(r"@location\((\d+)\)").expect("regex");
    let first_free = location
        .captures_iter(body)
        .filter_map(|c| c[1].parse::<u32>().ok())
        .max()
        .map_or(0, |max| max + 1);

    let kept = body.trim_end();
    let mut insert = String::new();
    if !kept.is_empty() && !kept.ends_with(',') {
        insert.push(',');
    }
    insert.push('\n');
    let mut locations = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let slot = first_free + i as u32;
        let interpolate = if varying && is_integer_type(field.ty) {
            "@interpolate(flat) "
        } else {
            ""
        };
        insert.push_str(&format!(
            "  @location({slot}) {interpolate}{}: {},\n",
            field.name, field.ty
        ));
        locations.push((field.name.to_string(), slot));
    }
    let mut out = String::with_capacity(main.len() + insert.len());
    out.push_str(&main[..body_start + kept.len()]);
    out.push_str(&insert);
    out.push_str(&main[body_start + body_len..]);
    Ok((out, locations))
}

/// The generated hook section: private mirrors of the extension fields, declarations, the
/// stage entry helpers and one function per hook.
fn generate_hooks(assembly: &ShaderAssembly<'_>, main: &str) -> Result<String> {
    let label = assembly.label;
    let mut injections: Vec<&ShaderInjection> = assembly.injections.iter().collect();
    injections.sort_by_key(|i| i.order);
    for injection in &injections {
        let known = FIXED_INJECTION_POINTS.contains(&injection.hook)
            || assembly.hooks.iter().any(|h| h.key == injection.hook);
        if !known {
            return Err(LumaError::Shader(format!(
                "{label}: unknown shader hook `{}`",
                injection.hook
            )));
        }
    }
    let at = |hook: &str, indent: &str| -> String {
        injections
            .iter()
            .filter(|i| i.hook == hook)
            .map(|i| format!("{indent}{}\n", i.code.trim()))
            .collect()
    };
    let has_attributes = has_struct(main, "Attributes");
    let has_varyings = has_struct(main, "Varyings");

    let mut out = String::new();
    for field in assembly.attributes.iter().chain(assembly.varyings) {
        out.push_str(&format!("var<private> {}: {};\n", field.name, field.ty));
    }
    out.push_str(&at("vs:#decl", ""));
    out.push_str(&at("fs:#decl", ""));

    if has_attributes {
        out.push_str("fn deckgl_vertex_start(attributes: Attributes) {\n");
        for field in assembly.attributes {
            out.push_str(&format!("  {0} = attributes.{0};\n", field.name));
        }
        out.push_str(&at("vs:#main-start", "  "));
        out.push_str("}\n");
    } else if injections.iter().any(|i| i.hook == "vs:#main-start") {
        return Err(LumaError::Shader(format!(
            "{label}: `vs:#main-start` needs an `Attributes` struct in the shader"
        )));
    }
    if has_varyings {
        out.push_str("fn deckgl_vertex_end(varyings: ptr<function, Varyings>) {\n");
        out.push_str(&at("vs:#main-end", "  "));
        for field in assembly.varyings {
            out.push_str(&format!("  (*varyings).{0} = {0};\n", field.name));
        }
        out.push_str("}\n");
        out.push_str("fn deckgl_fragment_start(varyings: Varyings) {\n");
        for field in assembly.varyings {
            out.push_str(&format!("  {0} = varyings.{0};\n", field.name));
        }
        out.push_str(&at("fs:#main-start", "  "));
        out.push_str("}\n");
    } else if injections
        .iter()
        .any(|i| i.hook == "vs:#main-end" || i.hook == "fs:#main-start")
    {
        return Err(LumaError::Shader(format!(
            "{label}: `vs:#main-end` and `fs:#main-start` need a `Varyings` struct in the shader"
        )));
    }
    for hook in assembly.hooks {
        let context = if hook.context.trim().is_empty() {
            String::new()
        } else {
            format!(", {}", hook.context)
        };
        out.push_str(&format!(
            "fn {}({1}_in: {2}{context}) -> {2} {{\n  var {1} = {1}_in;\n",
            hook.function, hook.value, hook.value_type
        ));
        out.push_str(&at(hook.key, "  "));
        out.push_str(&format!("  return {};\n}}\n", hook.value));
    }
    Ok(out)
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
    #[allow(clippy::expect_used)] // literal patterns
    let explicit = Regex::new(r"@binding\((\d+)\)").expect("regex");
    let mut used: BTreeSet<u32> = explicit
        .captures_iter(wgsl)
        .filter_map(|c| c[1].parse().ok())
        .collect();

    #[allow(clippy::expect_used)]
    let auto = Regex::new(r"@binding\(auto\)").expect("regex");
    let mut next = 0u32;
    let resolved = auto.replace_all(wgsl, |_: &regex::Captures| {
        while used.contains(&next) {
            next += 1;
        }
        used.insert(next);
        format!("@binding({next})")
    });

    #[allow(clippy::expect_used)]
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

    const HOOKED_MAIN: &str = r#"
struct Attributes {
  @builtin(instance_index) instanceIndex: u32,
  @location(0) p: vec3<f32>,
  @location(3) c: vec4<f32>
};
struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};
@vertex fn vertexMain(attributes: Attributes) -> Varyings {
  var varyings: Varyings;
  deckgl_vertex_start(attributes);
  varyings.position = filter_position(vec4<f32>(attributes.p, 1.0));
  varyings.color = attributes.c;
  deckgl_vertex_end(&varyings);
  return varyings;
}
@fragment fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> {
  deckgl_fragment_start(varyings);
  return filter_color(varyings.color, 2.0) * f32(id);
}
"#;

    const HOOKS: [ShaderHook; 2] = [
        ShaderHook {
            key: "vs:POSITION",
            function: "filter_position",
            value: "position",
            value_type: "vec4<f32>",
            context: "",
        },
        ShaderHook {
            key: "fs:COLOR",
            function: "filter_color",
            value: "color",
            value_type: "vec4<f32>",
            context: "scale: f32",
        },
    ];

    #[test]
    fn generates_hooks_and_extension_fields() {
        let injections = [
            ShaderInjection::new(
                "vs:POSITION",
                "position = vec4<f32>(position.xy * tint, position.zw);",
            ),
            ShaderInjection::new("fs:COLOR", "color = color * scale * tint;").with_order(1),
            ShaderInjection::new("fs:COLOR", "color = color + extra.offset;").with_order(-1),
            ShaderInjection::new("vs:#main-start", "tint = weight * 2.0; id = 1u;"),
            ShaderInjection::new(
                "vs:#decl",
                "struct Extra { offset: vec4<f32> };\n@group(0) @binding(auto) var<uniform> extra: Extra;",
            ),
        ];
        let shader = assemble(&ShaderAssembly {
            label: "hooked",
            modules: &[],
            main: HOOKED_MAIN,
            hooks: &HOOKS,
            injections: &injections,
            attributes: &[ShaderField {
                name: "weight",
                ty: "f32",
            }],
            varyings: &[
                ShaderField {
                    name: "tint",
                    ty: "f32",
                },
                ShaderField {
                    name: "id",
                    ty: "u32",
                },
            ],
        })
        .unwrap();
        assert_eq!(shader.attribute_location("weight"), Some(4));
        assert!(shader.wgsl.contains("@location(4) weight: f32,"));
        assert!(shader.wgsl.contains("@location(1) tint: f32,"));
        assert!(shader.wgsl.contains("@location(2) @interpolate(flat) id: u32,"));
        assert!(shader.wgsl.contains("var<private> weight: f32;"));
        assert!(shader
            .wgsl
            .contains("  weight = attributes.weight;\n  tint = weight * 2.0; id = 1u;"));
        assert!(shader.wgsl.contains("(*varyings).tint = tint;"));
        assert!(shader
            .wgsl
            .contains("fn filter_position(position_in: vec4<f32>) -> vec4<f32> {"));
        assert!(shader
            .wgsl
            .contains("fn filter_color(color_in: vec4<f32>, scale: f32) -> vec4<f32> {"));
        let offset = shader.wgsl.find("color = color + extra.offset;").unwrap();
        let scale = shader.wgsl.find("color = color * scale * tint;").unwrap();
        assert!(offset < scale, "injections run in `order`");
        assert_eq!(shader.uniform("extra").unwrap().layout.size, 16);
    }

    #[test]
    fn rejects_unknown_hooks_and_missing_structs() {
        let bad_hook = assemble(&ShaderAssembly {
            label: "bad",
            modules: &[],
            main: HOOKED_MAIN,
            hooks: &HOOKS,
            injections: &[ShaderInjection::new("vs:NOPE", "")],
            attributes: &[],
            varyings: &[],
        });
        assert!(bad_hook
            .unwrap_err()
            .to_string()
            .contains("unknown shader hook `vs:NOPE`"));
        let no_struct = assemble(&ShaderAssembly {
            label: "bad",
            modules: &[],
            main: MAIN,
            hooks: &[],
            injections: &[],
            attributes: &[ShaderField {
                name: "weight",
                ty: "f32",
            }],
            varyings: &[],
        });
        assert!(no_struct
            .unwrap_err()
            .to_string()
            .contains("no `Attributes` struct"));
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

#[cfg(test)]
mod glsl_tests {
    use super::*;

    const MAIN: &str = r#"
struct Attributes { @location(0) position: vec2<f32> };
struct Varyings { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vertexMain(attributes: Attributes) -> Varyings {
  return Varyings(vec4<f32>(attributes.position, 0.0, 1.0), vec4<f32>(1.0));
}
@fragment fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> { return varyings.color; }
"#;

    #[test]
    fn lowers_to_webgl2_glsl() {
        let shader = assemble_shader("glsl", &[], MAIN).unwrap();
        let sources = to_glsl(&shader).unwrap();
        assert_eq!(sources.len(), 2);
        for (entry, source) in &sources {
            assert!(source.starts_with("#version 300 es"), "{entry}: {source}");
        }
    }

    #[test]
    fn reports_what_webgl2_cannot_do() {
        // Storage buffers are a WebGPU feature; GLSL ES 3.00 has no equivalent.
        let main = format!(
            "@group(0) @binding(auto) var<storage, read> values: array<f32>;\n{}",
            MAIN.replace("vec4<f32>(1.0)", "vec4<f32>(values[0])")
        );
        // With `LUMA_GL_CHECK_GLSL` set, `assemble` reports it itself; without, `to_glsl` does
        let error = match assemble_shader("storage", &[], &main) {
            Ok(shader) => to_glsl(&shader).unwrap_err().to_string(),
            Err(e) => e.to_string(),
        };
        assert!(error.contains("not WebGL2 ready"), "{error}");
    }
}

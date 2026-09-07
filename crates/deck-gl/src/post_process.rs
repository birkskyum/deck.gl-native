//! Port of deck.gl's `PostProcessEffect` and `ScreenPass`: shader passes over the rendered
//! frame, with luma.gl's post-processing modules (brightness and contrast, hue and
//! saturation, sepia, vibrance, vignette, noise, denoise, triangle blur, tilt shift, zoom
//! blur, colour halftone, dot screen, edge work, hexagonal pixelate, ink, magnify, bulge and
//! pinch, swirl) ported to WGSL.
//!
//! With effects set, the deck draws its layers into a texture of its own and each pass of
//! each effect reads the previous result and writes the next; the last pass blends onto the
//! caller's attachment with premultiplied alpha, so a host's basemap underneath is kept.

use std::sync::Arc;

use glam::{Vec2, Vec4};
use luma_gl::device::create_render_texture;
use luma_gl::model::premultiplied_alpha_blend;
use luma_gl::{assemble_shader, AssembledShader, Model, ModelDescriptor, PipelineCache, RenderTarget};

use crate::Result;

/// A uniform of a shader pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniformValue {
    F32(f32),
    I32(i32),
    Vec2([f32; 2]),
    Vec4([f32; 4]),
}

impl From<f32> for UniformValue {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}

impl From<i32> for UniformValue {
    fn from(v: i32) -> Self {
        Self::I32(v)
    }
}

impl From<[f32; 2]> for UniformValue {
    fn from(v: [f32; 2]) -> Self {
        Self::Vec2(v)
    }
}

impl From<[f32; 4]> for UniformValue {
    fn from(v: [f32; 4]) -> Self {
        Self::Vec4(v)
    }
}

/// What a pass of a module does with the previous result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// `<name>_filterColor_ext(color, texSize, texCoord)` maps each pixel's colour
    Filter,
    /// `<name>_sampleColor(source, sampler, texSize, texCoord)` samples the previous result
    /// however it likes
    Sampler,
}

/// One pass of a shader pass module, with the uniforms it overrides.
#[derive(Clone, Debug, PartialEq)]
pub struct PassSpec {
    pub kind: PassKind,
    pub uniforms: Vec<(String, UniformValue)>,
}

impl PassSpec {
    pub fn filter() -> Self {
        Self {
            kind: PassKind::Filter,
            uniforms: Vec::new(),
        }
    }

    pub fn sampler() -> Self {
        Self {
            kind: PassKind::Sampler,
            uniforms: Vec::new(),
        }
    }

    pub fn with_uniform(mut self, name: &str, value: impl Into<UniformValue>) -> Self {
        self.uniforms.push((name.to_string(), value.into()));
        self
    }
}

/// A luma.gl style shader pass module: WGSL declaring `struct <name>Uniforms` bound as
/// `var<uniform> <name>` and the filter or sampler functions of its passes.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderPassModule {
    pub name: String,
    pub wgsl: String,
    /// Values of every uniform of the module; props and pass uniforms override them
    pub defaults: Vec<(String, UniformValue)>,
    pub passes: Vec<PassSpec>,
}

impl ShaderPassModule {
    pub fn new(name: &str, wgsl: &str) -> Self {
        Self {
            name: name.to_string(),
            wgsl: wgsl.to_string(),
            defaults: Vec::new(),
            passes: Vec::new(),
        }
    }

    pub fn with_default(mut self, name: &str, value: impl Into<UniformValue>) -> Self {
        self.defaults.push((name.to_string(), value.into()));
        self
    }

    pub fn with_pass(mut self, pass: PassSpec) -> Self {
        self.passes.push(pass);
        self
    }
}

/// A post-processing effect: a module and the props that override its defaults, deck.gl's
/// `new PostProcessEffect(module, props)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PostProcessEffect {
    pub module: Arc<ShaderPassModule>,
    pub props: Vec<(String, UniformValue)>,
}

impl PostProcessEffect {
    /// An effect from a built-in module by its luma.gl name, `None` for an unknown name.
    pub fn new(module: &str) -> Option<Self> {
        builtin_module(module).map(|module| Self {
            module: Arc::new(module),
            props: Vec::new(),
        })
    }

    pub fn with_module(module: ShaderPassModule) -> Self {
        Self {
            module: Arc::new(module),
            props: Vec::new(),
        }
    }

    pub fn with_prop(mut self, name: &str, value: impl Into<UniformValue>) -> Self {
        self.set_prop(name, value);
        self
    }

    pub fn set_prop(&mut self, name: &str, value: impl Into<UniformValue>) {
        let value = value.into();
        match self.props.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = value,
            None => self.props.push((name.to_string(), value)),
        }
    }

    pub fn prop(&self, name: &str) -> Option<UniformValue> {
        self.props
            .iter()
            .find(|(n, _)| n == name)
            .or_else(|| self.module.defaults.iter().find(|(n, _)| n == name))
            .map(|(_, v)| *v)
    }
}

/// The names of the built-in modules, luma.gl's `@luma.gl/shadertools` post-processing
/// passes.
pub const BUILTIN_MODULES: [&str; 18] = [
    "brightnessContrast",
    "hueSaturation",
    "sepia",
    "vibrance",
    "vignette",
    "noise",
    "denoise",
    "triangleBlur",
    "tiltShift",
    "zoomBlur",
    "colorHalftone",
    "dotScreen",
    "edgeWork",
    "hexagonalPixelate",
    "ink",
    "magnify",
    "bulgePinch",
    "swirl",
];

const SCREEN: &str = include_str!("shaderlib/wgsl/post/screen.wgsl");
const WARP: &str = include_str!("shaderlib/wgsl/post/warp.wgsl");

/// A built-in module by name, with luma.gl's defaults.
pub fn builtin_module(name: &str) -> Option<ShaderPassModule> {
    let center = [0.5f32, 0.5];
    Some(match name {
        "brightnessContrast" => {
            ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/brightness_contrast.wgsl"))
                .with_default("brightness", 0.0)
                .with_default("contrast", 0.0)
                .with_pass(PassSpec::filter())
        }
        "hueSaturation" => {
            ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/hue_saturation.wgsl"))
                .with_default("hue", 0.0)
                .with_default("saturation", 0.0)
                .with_pass(PassSpec::filter())
        }
        "sepia" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/sepia.wgsl"))
            .with_default("amount", 0.5)
            .with_pass(PassSpec::filter()),
        "vibrance" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/vibrance.wgsl"))
            .with_default("amount", 0.0)
            .with_pass(PassSpec::filter()),
        "vignette" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/vignette.wgsl"))
            .with_default("radius", 0.5)
            .with_default("amount", 0.5)
            .with_pass(PassSpec::filter()),
        "noise" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/noise.wgsl"))
            .with_default("amount", 0.5)
            .with_pass(PassSpec::filter()),
        "denoise" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/denoise.wgsl"))
            .with_default("strength", 0.5)
            .with_pass(PassSpec::sampler())
            .with_pass(PassSpec::sampler()),
        "triangleBlur" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/triangle_blur.wgsl"))
            .with_default("radius", 20.0)
            .with_default("delta", [1.0, 0.0])
            .with_pass(PassSpec::sampler().with_uniform("delta", [1.0, 0.0]))
            .with_pass(PassSpec::sampler().with_uniform("delta", [0.0, 1.0])),
        "tiltShift" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/tilt_shift.wgsl"))
            .with_default("blurRadius", 15.0)
            .with_default("gradientRadius", 200.0)
            .with_default("start", [0.0, 0.0])
            .with_default("end", [1.0, 1.0])
            .with_default("invert", 0.0)
            .with_pass(PassSpec::sampler().with_uniform("invert", 0.0))
            .with_pass(PassSpec::sampler().with_uniform("invert", 1.0)),
        "zoomBlur" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/zoom_blur.wgsl"))
            .with_default("center", center)
            .with_default("strength", 0.3)
            .with_pass(PassSpec::sampler()),
        "colorHalftone" => {
            ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/color_halftone.wgsl"))
                .with_default("center", center)
                .with_default("angle", 1.1)
                .with_default("size", 4.0)
                .with_pass(PassSpec::filter())
        }
        "dotScreen" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/dot_screen.wgsl"))
            .with_default("center", center)
            .with_default("angle", 1.1)
            .with_default("size", 3.0)
            .with_pass(PassSpec::filter()),
        "edgeWork" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/edge_work.wgsl"))
            .with_default("radius", 2.0)
            .with_default("mode", 0i32)
            .with_pass(PassSpec::sampler().with_uniform("mode", 0i32))
            .with_pass(PassSpec::sampler().with_uniform("mode", 1i32)),
        "hexagonalPixelate" => {
            ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/hexagonal_pixelate.wgsl"))
                .with_default("center", center)
                .with_default("scale", 10.0)
                .with_pass(PassSpec::sampler())
        }
        "ink" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/ink.wgsl"))
            .with_default("strength", 0.25)
            .with_pass(PassSpec::sampler()),
        "magnify" => ShaderPassModule::new(name, include_str!("shaderlib/wgsl/post/magnify.wgsl"))
            .with_default("screenXY", [0.0, 0.0])
            .with_default("radiusPixels", 200.0)
            .with_default("zoom", 2.0)
            .with_default("borderWidthPixels", 0.0)
            .with_default("borderColor", [1.0, 1.0, 1.0, 1.0])
            .with_pass(PassSpec::sampler()),
        "bulgePinch" => ShaderPassModule::new(
            name,
            &format!("{WARP}\n{}", include_str!("shaderlib/wgsl/post/bulge_pinch.wgsl")),
        )
        .with_default("radius", 200.0)
        .with_default("strength", 0.5)
        .with_default("center", center)
        .with_pass(PassSpec::sampler()),
        "swirl" => ShaderPassModule::new(
            name,
            &format!("{WARP}\n{}", include_str!("shaderlib/wgsl/post/swirl.wgsl")),
        )
        .with_default("radius", 200.0)
        .with_default("angle", 3.0)
        .with_default("center", center)
        .with_pass(PassSpec::sampler()),
        _ => return None,
    })
}

/// The WGSL of one screen pass: the screen module, the effect's module and a full screen
/// triangle calling its filter or sampler.
pub fn screen_pass_wgsl(module: &ShaderPassModule, kind: PassKind) -> String {
    let name = &module.name;
    let body = match kind {
        PassKind::Filter => format!(
            "  var color = screen_texture(texSrc, texSrcSampler, varyings.coordinate);\n  color = {name}_filterColor_ext(color, screen.texSize, varyings.coordinate);\n  return color;"
        ),
        PassKind::Sampler => {
            format!("  return {name}_sampleColor(texSrc, texSrcSampler, screen.texSize, varyings.coordinate);")
        }
    };
    format!(
        "{SCREEN}\n{}\n
struct Varyings {{
  @builtin(position) position: vec4<f32>,
  @location(0) coordinate: vec2<f32>,
}};

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> Varyings {{
  // One triangle covering the frame; coordinates run from the bottom left as in luma.gl
  let x = f32((index << 1u) & 2u);
  let y = f32(index & 2u);
  var varyings: Varyings;
  varyings.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  varyings.coordinate = vec2<f32>(x, y);
  return varyings;
}}

@fragment
fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> {{
  screen_fragCoord = varyings.position.xyz;
{body}
}}
",
        module.wgsl
    )
}

/// Assemble the shader of one pass, checking the module's WGSL.
pub fn screen_pass_shader(label: &str, module: &ShaderPassModule, kind: PassKind) -> Result<AssembledShader> {
    Ok(assemble_shader(label, &[], &screen_pass_wgsl(module, kind))?)
}

struct ScreenPass {
    model: Model,
    effect: usize,
    pass: usize,
}

struct PostTextures {
    scene: wgpu::Texture,
    swap: wgpu::Texture,
}

/// The screen passes and textures of a deck's effects.
#[derive(Default)]
pub(crate) struct PostProcessor {
    passes: Vec<ScreenPass>,
    /// The effects the passes were built for
    built_for: Vec<PostProcessEffect>,
    textures: Option<PostTextures>,
}

impl PostProcessor {
    /// Build the passes for `effects` when they changed since the last frame.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        target: RenderTarget,
        cache: &PipelineCache,
        effects: &[PostProcessEffect],
    ) -> Result<()> {
        let same_modules = self.built_for.len() == effects.len()
            && self
                .built_for
                .iter()
                .zip(effects)
                .all(|(a, b)| a.module == b.module);
        if same_modules && !self.passes.is_empty() {
            return Ok(());
        }
        self.passes.clear();
        let pass_target = RenderTarget {
            color_format: target.color_format,
            depth_format: None,
            sample_count: 1,
        };
        for (effect_index, effect) in effects.iter().enumerate() {
            for (pass_index, spec) in effect.module.passes.iter().enumerate() {
                let label = format!("{}-pass-{pass_index}", effect.module.name);
                let shader = screen_pass_shader(&label, &effect.module, spec.kind)?;
                let mut desc = ModelDescriptor::new(
                    &label,
                    &shader,
                    &[],
                    wgpu::PrimitiveTopology::TriangleList,
                    pass_target,
                );
                desc.blend = Some(premultiplied_alpha_blend());
                desc.depth_write_enabled = false;
                desc.depth_compare = wgpu::CompareFunction::Always;
                desc.cull_mode = None;
                desc.cache = Some(cache.clone());
                let mut model = Model::new(device, &desc)?;
                model.set_vertex_count(3);
                self.passes.push(ScreenPass {
                    model,
                    effect: effect_index,
                    pass: pass_index,
                });
            }
        }
        self.built_for = effects.to_vec();
        Ok(())
    }

    pub(crate) fn has_passes(&self) -> bool {
        !self.passes.is_empty()
    }

    /// The texture the layers render into, sized like the attachment.
    pub(crate) fn scene_texture(
        &mut self,
        device: &wgpu::Device,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
    ) -> &wgpu::Texture {
        let fresh = self
            .textures
            .as_ref()
            .is_some_and(|t| t.scene.size() == size && t.scene.format() == format);
        if !fresh {
            self.textures = None;
        }
        let textures = self.textures.get_or_insert_with(|| PostTextures {
            scene: create_render_texture(device, "deck.gl post scene", size.width, size.height, format),
            swap: create_render_texture(device, "deck.gl post swap", size.width, size.height, format),
        });
        &textures.scene
    }

    /// Run every pass, reading the scene texture and ending on `output` with `output_load`.
    pub(crate) fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        effects: &[PostProcessEffect],
        output: &wgpu::TextureView,
        output_load: wgpu::LoadOp<wgpu::Color>,
    ) -> Result<()> {
        let Some(textures) = self.textures.as_ref() else {
            return Ok(());
        };
        let size = textures.scene.size();
        let tex_size = Vec2::new(size.width as f32, size.height as f32);
        let views = [
            textures.scene.create_view(&Default::default()),
            textures.swap.create_view(&Default::default()),
        ];
        let count = self.passes.len();
        let mut input = 0;
        for (index, pass) in self.passes.iter_mut().enumerate() {
            let Some(effect) = effects.get(pass.effect) else {
                continue;
            };
            let Some(spec) = effect.module.passes.get(pass.pass) else {
                continue;
            };
            let model = &mut pass.model;
            model.set_texture("texSrc", views[input].clone())?;
            model.uniforms("screen")?.set_vec2("texSize", tex_size)?;
            {
                let block = model.uniforms(&effect.module.name)?;
                for (name, value) in effect
                    .module
                    .defaults
                    .iter()
                    .chain(effect.props.iter())
                    .chain(spec.uniforms.iter())
                {
                    match *value {
                        UniformValue::F32(v) => block.set_f32(name, v)?,
                        UniformValue::I32(v) => block.set_i32(name, v)?,
                        UniformValue::Vec2(v) => block.set_vec2(name, Vec2::from(v))?,
                        UniformValue::Vec4(v) => block.set_vec4(name, Vec4::from(v))?,
                    }
                }
            }
            model.upload_uniforms(queue);
            let last = index + 1 == count;
            let (view, load) = if last {
                (output, output_load)
            } else {
                (&views[1 - input], wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT))
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deck.gl post-process"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            model.draw(&mut render_pass)?;
            drop(render_pass);
            input = 1 - input;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_module_assembles_for_each_of_its_passes() {
        for name in BUILTIN_MODULES {
            let module = builtin_module(name).unwrap_or_else(|| panic!("{name} missing"));
            assert!(!module.passes.is_empty(), "{name} has passes");
            for (i, pass) in module.passes.iter().enumerate() {
                screen_pass_shader(&format!("{name}-{i}"), &module, pass.kind)
                    .unwrap_or_else(|e| panic!("{name} pass {i}: {e}"));
            }
            // Every default names a uniform the WGSL declares
            for (uniform, _) in &module.defaults {
                assert!(
                    module.wgsl.contains(&format!("{uniform}:")),
                    "{name}: {uniform} is not a uniform"
                );
            }
        }
        assert!(builtin_module("sparkle").is_none());
    }

    #[test]
    fn effects_override_module_defaults() {
        let effect = PostProcessEffect::new("vignette")
            .unwrap()
            .with_prop("radius", 0.9);
        assert_eq!(effect.prop("radius"), Some(UniformValue::F32(0.9)));
        assert_eq!(effect.prop("amount"), Some(UniformValue::F32(0.5)));
        assert_eq!(effect.prop("missing"), None);
        let mut other = effect.clone();
        other.set_prop("radius", 0.1);
        assert_ne!(effect, other);
        assert_eq!(effect.module, other.module);
    }
}

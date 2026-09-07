//! Port of `@deck.gl/extensions/src/fill-style`: tile filled areas with a pattern from an
//! atlas image.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::data::resolve_strings;
use deck_gl::glam::Vec2;
use deck_gl::layer::project_props;
use deck_gl::luma_gl::model::{create_rgba8_texture, default_sampler};
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::shaderlib::project::common_origin;
use deck_gl::{
    same_extension, Accessor, ExtensionAttribute, ExtensionShaders, LayerContext, LayerData, LayerExtension,
    LayerProps, Result, Viewport,
};
use wgpu::VertexFormat;

use crate::bitmap_layer::BitmapImage;

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "fill",
    source: include_str!("../wgsl/fill_style.wgsl"),
};

/// Where a pattern sits in the atlas, in pixels from the top left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillPattern {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// deck.gl's `fillPatternAtlas` and `fillPatternMapping`: one image with every pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct FillPatternAtlas {
    pub image: BitmapImage,
    pub mapping: HashMap<String, FillPattern>,
}

/// The atlas on the GPU, created on first use.
#[derive(Clone, Debug)]
struct PatternTexture {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    size: Vec2,
}

/// The extension's GPU side cache of the atlas; leave it at its default.
#[derive(Clone, Debug, Default)]
pub struct PatternTextureCache(OnceLock<PatternTexture>);

/// deck.gl's `FillStyleExtension` with the `pattern` option: the fill of every object is
/// masked (or coloured, with `fill_pattern_mask` off) by a pattern from the atlas, tiled in
/// meters at `get_fill_pattern_scale` times its pixel size and shifted by
/// `get_fill_pattern_offset` in pattern units. Objects whose pattern is not in the atlas are
/// drawn without one.
#[derive(Clone, Debug)]
pub struct FillStyleExtension {
    /// The `pattern` option: without it the extension does nothing.
    pub pattern: bool,
    pub fill_pattern_enabled: bool,
    pub fill_pattern_atlas: Option<Arc<FillPatternAtlas>>,
    /// Treat the pattern as a transparency mask over the fill colour (`true`), or draw its colours
    pub fill_pattern_mask: bool,
    pub get_fill_pattern: Accessor<String>,
    pub get_fill_pattern_scale: Accessor<f32>,
    pub get_fill_pattern_offset: Accessor<[f32; 2]>,
    pub gpu: PatternTextureCache,
}

impl PartialEq for FillStyleExtension {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.fill_pattern_enabled == other.fill_pattern_enabled
            && self.fill_pattern_atlas == other.fill_pattern_atlas
            && self.fill_pattern_mask == other.fill_pattern_mask
            && self.get_fill_pattern == other.get_fill_pattern
            && self.get_fill_pattern_scale == other.get_fill_pattern_scale
            && self.get_fill_pattern_offset == other.get_fill_pattern_offset
    }
}

impl Default for FillStyleExtension {
    fn default() -> Self {
        Self {
            pattern: true,
            fill_pattern_enabled: true,
            fill_pattern_atlas: None,
            fill_pattern_mask: true,
            get_fill_pattern: Accessor::column("pattern"),
            get_fill_pattern_scale: Accessor::Constant(1.0),
            get_fill_pattern_offset: Accessor::Constant([0.0, 0.0]),
            gpu: PatternTextureCache::default(),
        }
    }
}

impl FillStyleExtension {
    /// Pattern fills from `atlas`, the pattern of each object named by `get_fill_pattern`.
    pub fn pattern(atlas: Arc<FillPatternAtlas>, get_fill_pattern: Accessor<String>) -> Self {
        Self {
            fill_pattern_atlas: Some(atlas),
            get_fill_pattern,
            ..Default::default()
        }
    }

    fn texture(&self, ctx: &LayerContext) -> &PatternTexture {
        self.gpu.0.get_or_init(|| {
            let (width, height, rgba): (u32, u32, &[u8]) = match &self.fill_pattern_atlas {
                Some(atlas) => (atlas.image.width, atlas.image.height, &atlas.image.rgba),
                None => (1, 1, &[0, 0, 0, 0]),
            };
            let texture =
                create_rgba8_texture(&ctx.device, &ctx.queue, "fill pattern atlas", width, height, rgba);
            PatternTexture {
                view: texture.create_view(&Default::default()),
                sampler: default_sampler(&ctx.device),
                size: Vec2::new(width as f32, height as f32),
            }
        })
    }
}

impl LayerExtension for FillStyleExtension {
    fn name(&self) -> &'static str {
        "FillStyleExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        if !self.pattern {
            return ExtensionShaders::default();
        }
        ExtensionShaders {
            modules: vec![MODULE],
            injections: vec![
                ShaderInjection::new("vs:DECKGL_FILTER_GL_POSITION", "fill_uv = geometry.position.xy;"),
                ShaderInjection::new(
                    "vs:DECKGL_FILTER_COLOR",
                    "if (fill.patternEnabled != 0) {\n    fill_patternBounds = fillPatternFrames / vec4<f32>(fill.patternTextureSize, fill.patternTextureSize);\n    fill_patternPlacement = vec4<f32>(fillPatternOffsets, fillPatternScales * fillPatternFrames.zw);\n  }",
                ),
                ShaderInjection::new(
                    "fs:DECKGL_FILTER_COLOR",
                    "if (fill.patternEnabled != 0 && fill_patternBounds.z > 0.0) {\n    let scale = FILL_UV_SCALE * fill_patternPlacement.zw;\n    var patternUV = fill_mod2(fill_mod2(fill.uvCoordinateOrigin, scale) + fill.uvCoordinateOrigin64Low + fill_uv, scale) / scale;\n    patternUV = fract(fill_patternPlacement.xy + patternUV);\n    let texCoords = fill_patternBounds.xy + fill_patternBounds.zw * patternUV;\n    let patternColor = textureSampleLevel(fill_patternTexture, fill_patternSampler, texCoords, 0.0);\n    color.a *= patternColor.a;\n    if (fill.patternMask == 0) {\n      color = vec4<f32>(patternColor.rgb, color.a);\n    }\n  }",
                ),
            ],
            attributes: vec![
                ExtensionAttribute {
                    name: "fillPatternFrames",
                    format: VertexFormat::Float32x4,
                },
                ExtensionAttribute {
                    name: "fillPatternScales",
                    format: VertexFormat::Float32,
                },
                ExtensionAttribute {
                    name: "fillPatternOffsets",
                    format: VertexFormat::Float32x2,
                },
            ],
            varyings: vec![
                ShaderField {
                    name: "fill_uv",
                    ty: "vec2<f32>",
                },
                ShaderField {
                    name: "fill_patternBounds",
                    ty: "vec4<f32>",
                },
                ShaderField {
                    name: "fill_patternPlacement",
                    ty: "vec4<f32>",
                },
            ],
        }
    }

    fn attributes(&self, data: &LayerData) -> Result<Vec<(&'static str, AttributeSource)>> {
        if !self.pattern {
            return Ok(Vec::new());
        }
        // The frame of each object's pattern in the atlas; unknown patterns get an empty frame
        let names = resolve_strings(data, &self.get_fill_pattern)?;
        let frames: Arc<Vec<[f32; 4]>> = Arc::new(
            names
                .iter()
                .map(|name| {
                    self.fill_pattern_atlas
                        .as_ref()
                        .and_then(|atlas| atlas.mapping.get(name))
                        .map_or([0.0; 4], |p| {
                            [p.x as f32, p.y as f32, p.width as f32, p.height as f32]
                        })
                })
                .collect(),
        );
        let last = frames.len().saturating_sub(1);
        Ok(vec![
            (
                "fillPatternFrames",
                AttributeSource::Vec4(Accessor::func(move |i| frames[i.min(last)])),
            ),
            (
                "fillPatternScales",
                AttributeSource::Floats(self.get_fill_pattern_scale.clone()),
            ),
            (
                "fillPatternOffsets",
                AttributeSource::Vec2(self.get_fill_pattern_offset.clone()),
            ),
        ])
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        viewport: &Viewport,
        props: &LayerProps,
    ) -> Result<()> {
        if !self.pattern {
            return Ok(());
        }
        let texture = self.texture(ctx);
        model.set_texture("fill_patternTexture", texture.view.clone())?;
        model.set_sampler("fill_patternSampler", texture.sampler.clone())?;
        // the pattern tiles in absolute common space; the shader position is relative to the origin
        let origin = common_origin(&project_props(ctx, viewport, props));
        let high = Vec2::new(origin.x as f32, origin.y as f32);
        let low = Vec2::new(
            (origin.x - high.x as f64) as f32,
            (origin.y - high.y as f64) as f32,
        );
        let u = model.uniforms("fill")?;
        u.set_vec2("patternTextureSize", texture.size)?;
        u.set_vec2("uvCoordinateOrigin", high)?;
        u.set_vec2("uvCoordinateOrigin64Low", low)?;
        u.set_i32(
            "patternEnabled",
            (self.fill_pattern_enabled && self.fill_pattern_atlas.is_some()) as i32,
        )?;
        u.set_i32("patternMask", self.fill_pattern_mask as i32)?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

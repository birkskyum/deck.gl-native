//! Port of `@deck.gl/extensions/src/clip`: clip what a layer draws to rectangular bounds.

use std::any::Any;

use deck_gl::glam::{DVec3, Vec4};
use deck_gl::layer::project_props;
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::shaderlib::project::project_position;
use deck_gl::{same_extension, ExtensionShaders, LayerContext, LayerExtension, LayerProps, Result, Viewport};

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "clip",
    source: include_str!("../wgsl/clip.wgsl"),
};

/// deck.gl's `ClipExtension`: only what lies inside `clip_bounds` is drawn. With
/// `clip_by_instance` an object is shown or hidden whole by its anchor position (points,
/// icons, columns); without it the geometry is trimmed at the bounds per fragment (paths,
/// polygons, bitmaps). deck.gl deduces the mode from the layer; here the default is by
/// instance, and the JSON layers pick the mode from the layer type.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipExtension {
    /// `[left, bottom, right, top]` in the layer's coordinates
    pub clip_bounds: [f64; 4],
    pub clip_by_instance: bool,
}

impl Default for ClipExtension {
    fn default() -> Self {
        Self {
            clip_bounds: [0.0, 0.0, 1.0, 1.0],
            clip_by_instance: true,
        }
    }
}

impl ClipExtension {
    pub fn new(clip_bounds: [f64; 4], clip_by_instance: bool) -> Self {
        Self {
            clip_bounds,
            clip_by_instance,
        }
    }
}

impl LayerExtension for ClipExtension {
    fn name(&self) -> &'static str {
        "ClipExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        if self.clip_by_instance {
            ExtensionShaders {
                modules: vec![MODULE],
                injections: vec![
                    ShaderInjection::new(
                        "vs:DECKGL_FILTER_GL_POSITION",
                        "clip_isVisible = f32(clip_isInBounds(geometry.worldPosition.xy));",
                    ),
                    ShaderInjection::new(
                        "fs:DECKGL_FILTER_COLOR",
                        "if (clip_isVisible < 0.5) {\n    discard;\n  }",
                    ),
                ],
                attributes: Vec::new(),
                varyings: vec![ShaderField {
                    name: "clip_isVisible",
                    ty: "f32",
                }],
            }
        } else {
            ExtensionShaders {
                modules: vec![MODULE],
                injections: vec![
                    ShaderInjection::new(
                        "vs:DECKGL_FILTER_GL_POSITION",
                        "clip_commonPosition = geometry.position.xy;",
                    ),
                    ShaderInjection::new(
                        "fs:DECKGL_FILTER_COLOR",
                        "if (!clip_isInBounds(clip_commonPosition)) {\n    discard;\n  }",
                    ),
                ],
                attributes: Vec::new(),
                varyings: vec![ShaderField {
                    name: "clip_commonPosition",
                    ty: "vec2<f32>",
                }],
            }
        }
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        viewport: &Viewport,
        props: &LayerProps,
    ) -> Result<()> {
        let [left, bottom, right, top] = self.clip_bounds;
        let bounds = if self.clip_by_instance {
            [left as f32, bottom as f32, right as f32, top as f32]
        } else {
            // the bounds in the common space the fragment shader compares against
            let project = project_props(ctx, viewport, props);
            let a = project_position(&project, DVec3::new(left, bottom, 0.0));
            let b = project_position(&project, DVec3::new(right, top, 0.0));
            [
                a.x.min(b.x) as f32,
                a.y.min(b.y) as f32,
                a.x.max(b.x) as f32,
                a.y.max(b.y) as f32,
            ]
        };
        model.uniforms("clip")?.set_vec4("bounds", Vec4::from(bounds))?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

//! Port of `@deck.gl/extensions/src/mask`: show only what lies inside (or outside) the
//! geometry of another layer, the mask layer, whose operation is `mask`.

use std::any::Any;

use deck_gl::glam::{DVec3, Vec4};
use deck_gl::layer::project_props;
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::shaderlib::project::project_position;
use deck_gl::{
    same_extension, DeckError, ExtensionShaders, LayerContext, LayerExtension, LayerProps, Result, Viewport,
};

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "mask",
    source: include_str!("../wgsl/mask.wgsl"),
};

/// deck.gl's `MaskExtension`: fragments outside the mask drawn by the layer with id
/// `mask_id` (a layer with [`Operation::MASK`](deck_gl::Operation::MASK)) are discarded, or
/// those inside with `mask_inverted`. With `mask_by_instance` whole objects are kept or
/// dropped by their anchor position; without it the geometry is trimmed per fragment.
/// The mask and the reading layer are expected to share a coordinate system.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskExtension {
    pub mask_id: String,
    pub mask_by_instance: bool,
    pub mask_inverted: bool,
}

impl Default for MaskExtension {
    fn default() -> Self {
        Self {
            mask_id: String::new(),
            mask_by_instance: true,
            mask_inverted: false,
        }
    }
}

impl MaskExtension {
    pub fn new(mask_id: impl Into<String>) -> Self {
        Self {
            mask_id: mask_id.into(),
            ..Default::default()
        }
    }
}

impl LayerExtension for MaskExtension {
    fn name(&self) -> &'static str {
        "MaskExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders {
            modules: vec![MODULE],
            injections: vec![
                ShaderInjection::new(
                    "vs:#main-end",
                    "var mask_common_position: vec4<f32>;\n  if (mask.maskByInstance != 0) {\n    mask_common_position = project_position_vec4_f32(vec4<f32>(geometry.worldPosition, 1.0));\n  } else {\n    mask_common_position = geometry.position;\n  }\n  mask_texCoords = mask_getCoords(mask_common_position);",
                ),
                ShaderInjection::new(
                    "fs:#main-start",
                    "if (mask.enabled != 0 && !mask_isInBounds(mask_texCoords)) {\n    discard;\n  }",
                ),
            ],
            attributes: Vec::new(),
            varyings: vec![ShaderField {
                name: "mask_texCoords",
                ty: "vec2<f32>",
            }],
        }
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        viewport: &Viewport,
        props: &LayerProps,
    ) -> Result<()> {
        let Some(maps) = ctx.masks.as_ref() else {
            return Err(DeckError::Layer {
                layer: props.id.clone(),
                message: "the mask extension needs the mask pass of a Deck".to_string(),
            });
        };
        let channel = maps.channels.get(&self.mask_id);
        let view = channel.map_or_else(|| maps.dummy.clone(), |c| c.view.clone());
        model.set_texture("mask_texture", view)?;
        model.set_sampler("mask_sampler", maps.sampler.clone())?;
        let bounds = match channel {
            Some(channel) => {
                // the texture bounds in the common space this layer's shader works in
                let project = project_props(ctx, viewport, props);
                let [x0, y0, x1, y1] = channel.bounds_common;
                let bl = project_position(&project, viewport.unproject_position(DVec3::new(x0, y0, 0.0)));
                let tr = project_position(&project, viewport.unproject_position(DVec3::new(x1, y1, 0.0)));
                [bl.x as f32, bl.y as f32, tr.x as f32, tr.y as f32]
            }
            None => {
                if !self.mask_id.is_empty() {
                    tracing::warn!(layer = props.id, mask = self.mask_id, "mask layer not found");
                }
                [0.0, 0.0, 1.0, 1.0]
            }
        };
        let u = model.uniforms("mask")?;
        u.set_vec4("bounds", Vec4::from(bounds))?;
        u.set_i32("enabled", channel.is_some() as i32)?;
        u.set_i32("inverted", self.mask_inverted as i32)?;
        u.set_i32("maskByInstance", self.mask_by_instance as i32)?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

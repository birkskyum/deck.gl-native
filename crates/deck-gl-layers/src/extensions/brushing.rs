//! Port of `@deck.gl/extensions/src/brushing`: show only the objects within a radius of the
//! pointer.

use std::any::Any;

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::glam::{DVec2, Vec2};
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::{
    same_extension, Accessor, ExtensionAttribute, ExtensionShaders, LayerContext, LayerExtension, LayerProps,
    Result, Viewport,
};
use wgpu::VertexFormat;

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "brushing",
    source: include_str!("../wgsl/brushing.wgsl"),
};

/// Which position of an object the brushing distance is measured to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushingTarget {
    /// The object's position (`geometry.worldPosition`)
    #[default]
    Source,
    /// The second position of lines and arcs (`geometry.worldPositionAlt`)
    Target,
    /// Either end of a line or arc
    SourceTarget,
    /// The position from `get_brushing_target`
    Custom,
}

impl BrushingTarget {
    fn shader_value(self) -> i32 {
        match self {
            Self::Source => 0,
            Self::Target => 1,
            Self::Custom => 2,
            Self::SourceTarget => 3,
        }
    }
}

/// deck.gl's `BrushingExtension`: objects farther than `brushing_radius` meters from the
/// pointer are hidden. The pointer comes from [`LayerContext::pointer`], which
/// `Deck::pointer_move` and `Deck::pointer_leave` maintain; with no pointer over the
/// viewport everything is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct BrushingExtension {
    /// The position of each object when `brushing_target` is `Custom`
    pub get_brushing_target: Accessor<[f32; 2]>,
    pub brushing_target: BrushingTarget,
    pub brushing_enabled: bool,
    /// In meters
    pub brushing_radius: f32,
}

impl Default for BrushingExtension {
    fn default() -> Self {
        Self {
            get_brushing_target: Accessor::Constant([0.0, 0.0]),
            brushing_target: BrushingTarget::Source,
            brushing_enabled: true,
            brushing_radius: 10_000.0,
        }
    }
}

impl BrushingExtension {
    pub fn new(brushing_radius: f32) -> Self {
        Self {
            brushing_radius,
            ..Default::default()
        }
    }
}

impl LayerExtension for BrushingExtension {
    fn name(&self) -> &'static str {
        "BrushingExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders {
            modules: vec![MODULE],
            injections: vec![
                ShaderInjection::new(
                    "vs:DECKGL_FILTER_GL_POSITION",
                    "var brushingTarget: vec2<f32>;\n  var brushingSource: vec2<f32>;\n  if (brushing.targetMode == 3) {\n    brushingTarget = geometry.worldPositionAlt.xy;\n    brushingSource = geometry.worldPosition.xy;\n  } else if (brushing.targetMode == 0) {\n    brushingTarget = geometry.worldPosition.xy;\n  } else if (brushing.targetMode == 1) {\n    brushingTarget = geometry.worldPositionAlt.xy;\n  } else {\n    brushingTarget = brushingTargets;\n  }\n  var visible: bool;\n  if (brushing.targetMode == 3) {\n    visible = brushing_arePointsInRange(brushingSource, brushingTarget);\n  } else {\n    visible = brushing_isPointInRange(brushingTarget);\n  }\n  brushing_isVisible = f32(visible);",
                ),
                ShaderInjection::new(
                    "fs:DECKGL_FILTER_COLOR",
                    "if (brushing.enabled != 0 && brushing_isVisible < 0.5) {\n    discard;\n  }",
                ),
            ],
            attributes: vec![ExtensionAttribute {
                name: "brushingTargets",
                format: VertexFormat::Float32x2,
            }],
            varyings: vec![ShaderField {
                name: "brushing_isVisible",
                ty: "f32",
            }],
        }
    }

    fn attributes(&self) -> Vec<(&'static str, AttributeSource)> {
        vec![(
            "brushingTargets",
            AttributeSource::Vec2(self.get_brushing_target.clone()),
        )]
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        viewport: &Viewport,
        _props: &LayerProps,
    ) -> Result<()> {
        let pointer = ctx.pointer.filter(|[x, y]| {
            *x >= viewport.x
                && *x < viewport.x + viewport.width
                && *y >= viewport.y
                && *y < viewport.y + viewport.height
        });
        let mouse = pointer
            .map(|[x, y]| viewport.unproject(DVec2::new(x - viewport.x, y - viewport.y), None, true, None))
            .unwrap_or_default();
        let u = model.uniforms("brushing")?;
        u.set_vec2("mousePos", Vec2::new(mouse.x as f32, mouse.y as f32))?;
        u.set_f32("radius", self.brushing_radius)?;
        u.set_i32("enabled", (self.brushing_enabled && pointer.is_some()) as i32)?;
        u.set_i32("targetMode", self.brushing_target.shader_value())?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

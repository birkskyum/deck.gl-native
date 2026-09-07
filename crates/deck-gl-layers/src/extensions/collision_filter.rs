//! Port of `@deck.gl/extensions/src/collision-filter`: hide objects that overlap objects of
//! a higher collision priority, for labels and icons.

use std::any::Any;

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::luma_gl::{Model, ShaderInjection, ShaderModuleSource};
use deck_gl::{
    same_extension, Accessor, DeckError, ExtensionAttribute, ExtensionShaders, LayerContext, LayerData,
    LayerExtension, LayerProps, Result, Viewport,
};
use wgpu::VertexFormat;

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "collision",
    source: include_str!("../wgsl/collision.wgsl"),
};

/// deck.gl's `CollisionFilterExtension`. Before every frame the deck draws the layers of each
/// `collision_group` into a half resolution map with their picking colours, the objects with
/// the highest `get_collision_priority` in front; an object is then only drawn where the map
/// still shows it at its anchor. Priorities range from -1000 to 1000. Objects fade over a
/// five pixel window as they appear and disappear.
#[derive(Clone, Debug, PartialEq)]
pub struct CollisionFilterExtension {
    pub get_collision_priority: Accessor<f32>,
    pub collision_enabled: bool,
    pub collision_group: String,
}

impl Default for CollisionFilterExtension {
    fn default() -> Self {
        Self {
            get_collision_priority: Accessor::Constant(0.0),
            collision_enabled: true,
            collision_group: "default".to_string(),
        }
    }
}

impl CollisionFilterExtension {
    pub fn new(get_collision_priority: Accessor<f32>) -> Self {
        Self {
            get_collision_priority,
            ..Default::default()
        }
    }
}

impl LayerExtension for CollisionFilterExtension {
    fn name(&self) -> &'static str {
        "CollisionFilterExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders {
            modules: vec![MODULE],
            injections: vec![
                ShaderInjection::new("vs:#decl", "var<private> collision_fade: f32 = 1.0;"),
                ShaderInjection::new(
                    "vs:DECKGL_FILTER_GL_POSITION",
                    "if (collision.sort != 0) {\n    // depth from the priority (-1000 to 1000), higher in front\n    position.z = (0.5 - 0.0005 * collisionPriorities) * position.w;\n  }\n  if (collision.enabled != 0) {\n    let collision_common_position = project_position_vec4_f32(vec4<f32>(geometry.worldPosition, 1.0));\n    let collision_texCoords = collision_getCoords(collision_common_position);\n    collision_fade = collision_isVisible(collision_texCoords, geometry.pickingColor);\n    if (collision_fade < 0.0001) {\n      // outside the clip space, so the object is dropped\n      position = vec4<f32>(0.0, 0.0, 2.0, 1.0);\n    }\n  }",
                ),
                ShaderInjection::new("vs:DECKGL_FILTER_COLOR", "color.a *= collision_fade;"),
            ],
            attributes: vec![ExtensionAttribute {
                name: "collisionPriorities",
                format: VertexFormat::Float32,
            }],
            varyings: Vec::new(),
        }
    }

    fn attributes(&self, _data: &LayerData) -> Result<Vec<(&'static str, AttributeSource)>> {
        Ok(vec![(
            "collisionPriorities",
            AttributeSource::Floats(self.get_collision_priority.clone()),
        )])
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        _viewport: &Viewport,
        props: &LayerProps,
    ) -> Result<()> {
        let Some(maps) = ctx.collisions.as_ref() else {
            return Err(DeckError::Layer {
                layer: props.id.clone(),
                message: "the collision filter extension needs the collision pass of a Deck".to_string(),
            });
        };
        let drawing = maps.drawing_to_map;
        let map = if drawing {
            None
        } else {
            maps.groups.get(&self.collision_group)
        };
        model.set_texture(
            "collision_texture",
            map.map_or_else(|| maps.dummy.clone(), |view| view.clone()),
        )?;
        model.set_sampler("collision_sampler", maps.sampler.clone())?;
        let u = model.uniforms("collision")?;
        u.set_i32("sort", drawing as i32)?;
        u.set_i32("enabled", (self.collision_enabled && map.is_some()) as i32)?;
        Ok(())
    }

    fn collision_group(&self) -> Option<String> {
        self.collision_enabled.then(|| self.collision_group.clone())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

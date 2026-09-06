//! Port of the parts of `@deck.gl/core/src/lib/layer.ts` that a native layer needs.

use glam::DMat4;
use luma_gl::{Model, RenderTarget};

use crate::constants::CoordinateSystem;
use crate::lighting::LightingEffect;
use crate::shaderlib::project::{get_uniforms_from_viewport, ProjectProps};
use crate::viewport::Viewport;
use crate::Result;

/// Properties shared by all layers. Mirrors deck.gl's `LayerProps`.
#[derive(Clone, Debug)]
pub struct LayerProps {
    pub id: String,
    pub visible: bool,
    /// Opacity of the layer, between 0 and 1
    pub opacity: f64,
    pub pickable: bool,
    pub coordinate_system: CoordinateSystem,
    pub coordinate_origin: [f64; 3],
    pub model_matrix: Option<DMat4>,
    pub wrap_longitude: bool,
}

impl Default for LayerProps {
    fn default() -> Self {
        Self {
            id: "layer".to_string(),
            visible: true,
            opacity: 1.0,
            pickable: false,
            coordinate_system: CoordinateSystem::Default,
            coordinate_origin: [0.0; 3],
            model_matrix: None,
            wrap_longitude: false,
        }
    }
}

impl LayerProps {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }
}

/// What a layer needs from the `Deck` to create and draw its resources.
#[derive(Clone, Debug)]
pub struct LayerContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub target: RenderTarget,
    pub device_pixel_ratio: f32,
    pub lighting: LightingEffect,
    /// Position of the layer being initialized, updated or drawn in the deck's layer list.
    pub layer_index: u32,
}

/// deck.gl's default `getPolygonOffset`: `[0, -layerIndex * 100]`, so that a layer drawn later
/// wins the depth test against earlier layers on the same surface instead of z-fighting.
pub fn depth_bias_for_layer(layer_index: u32) -> wgpu::DepthBiasState {
    wgpu::DepthBiasState {
        constant: -(layer_index as i32) * 100,
        slope_scale: 0.0,
        clamp: 0.0,
    }
}

/// A deck.gl layer. Implementations own their GPU resources.
///
/// Lifecycle: `initialize` once, then `update` followed by `draw` every frame. `update` runs
/// outside of any render pass and is where buffers and uniforms are written. `draw` only
/// encodes commands into the pass.
pub trait Layer {
    fn props(&self) -> &LayerProps;

    fn id(&self) -> &str {
        &self.props().id
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()>;

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()>;

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()>;
}

/// Fill the uniform blocks every deck.gl layer shader has: `project`, `layer` and `picking`,
/// plus `lighting`, `gouraudMaterial` and `floatColors` when the model uses them.
pub fn update_standard_uniforms(
    model: &mut Model,
    ctx: &LayerContext,
    viewport: &Viewport,
    props: &LayerProps,
) -> Result<()> {
    let project = get_uniforms_from_viewport(&ProjectProps {
        viewport,
        device_pixel_ratio: ctx.device_pixel_ratio,
        model_matrix: props.model_matrix,
        coordinate_system: props.coordinate_system,
        coordinate_origin: glam::DVec3::from(props.coordinate_origin),
        auto_wrap_longitude: props.wrap_longitude,
    });
    project.write(model.uniforms("project")?)?;

    // apply gamma to opacity to make it visually "linear"
    let opacity = props.opacity.clamp(0.0, 1.0).powf(1.0 / 2.2) as f32;
    model.uniforms("layer")?.set_f32("opacity", opacity)?;

    let picking = model.uniforms("picking")?;
    picking.set_f32("isActive", 0.0)?;
    picking.set_f32("isAttribute", 0.0)?;
    picking.set_f32("isHighlightActive", 0.0)?;
    picking.set_f32("useByteColors", 1.0)?;
    picking.set_vec3("highlightedObjectColor", glam::Vec3::ZERO)?;
    picking.set_vec4("highlightColor", glam::Vec4::new(0.0, 1.0, 1.0, 1.0))?;
    picking.set_f32("disabledPickingIndexCount", 0.0)?;

    if model.has_uniforms("lighting") {
        ctx.lighting.write(model.uniforms("lighting")?)?;
    }
    if model.has_uniforms("gouraudMaterial") {
        LightingEffect::write_default_material(model.uniforms("gouraudMaterial")?)?;
    }
    if model.has_uniforms("floatColors") {
        model.uniforms("floatColors")?.set_f32("useByteColors", 1.0)?;
    }

    model.upload_uniforms(&ctx.queue);
    Ok(())
}

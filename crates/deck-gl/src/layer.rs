//! Port of the parts of `@deck.gl/core/src/lib/layer.ts` that a native layer needs.

use glam::DMat4;
use luma_gl::{Model, RenderTarget};

use crate::constants::CoordinateSystem;
use crate::data::Color;
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
    /// Object (data row) drawn with `highlight_color`, typically the hovered one.
    pub highlighted_object_index: Option<u32>,
    /// RGBA in 0..255, blended over the highlighted object.
    pub highlight_color: Color,
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
            highlighted_object_index: None,
            highlight_color: [0, 0, 128, 128],
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

/// Top-level layers are spaced this far apart in `layer_index`, leaving room for the sub
/// layers of composite layers to get their own depth bias slots.
pub const LAYER_INDEX_STRIDE: u32 = 8;

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

    /// Switch the layer's shaders between normal and picking color output. Called outside of
    /// any render pass, before and after [`Layer::draw_picking`].
    fn set_picking_active(&mut self, _ctx: &LayerContext, _active: bool) -> Result<()> {
        Ok(())
    }

    /// Draw into the picking target. Only called for layers whose props say `pickable`.
    fn draw_picking(&mut self, _ctx: &LayerContext, _pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        Ok(())
    }

    /// Change which object is drawn highlighted. Takes effect on the next update.
    fn set_highlighted_object(&mut self, _index: Option<u32>) {}
}

/// Encode an object index the way the `picking` shader module does: `index + 1` as three
/// bytes, little endian. Zero means "not pickable".
pub fn encode_picking_color(index: u32) -> [u8; 3] {
    let value = index + 1;
    [
        (value & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        ((value >> 16) & 0xff) as u8,
    ]
}

/// Inverse of [`encode_picking_color`]. `None` for the zero color.
pub fn decode_picking_color(color: [u8; 3]) -> Option<u32> {
    let value = color[0] as u32 + ((color[1] as u32) << 8) + ((color[2] as u32) << 16);
    value.checked_sub(1)
}

/// The sub layers of a composite layer. Mirrors what deck.gl's `CompositeLayer` does with the
/// result of `renderLayers`: initialize new sub layers, update and draw them in order, and give
/// each one its own depth bias slot after the parent's.
#[derive(Default)]
pub struct SubLayers {
    layers: Vec<(Box<dyn Layer>, bool)>,
}

impl SubLayers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Replace the sub layers. GPU resources of the old ones are dropped.
    pub fn replace(&mut self, layers: Vec<Box<dyn Layer>>) {
        self.layers = layers.into_iter().map(|layer| (layer, false)).collect();
    }

    pub fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        let mut sub_ctx = ctx.clone();
        for (index, (layer, initialized)) in self.layers.iter_mut().enumerate() {
            sub_ctx.layer_index = ctx.layer_index + index as u32;
            if !*initialized {
                layer.initialize(&sub_ctx)?;
                *initialized = true;
            }
            if layer.props().visible {
                layer.update(&sub_ctx, viewport)?;
            }
        }
        Ok(())
    }

    pub fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for (layer, initialized) in &mut self.layers {
            if *initialized && layer.props().visible {
                layer.draw(ctx, pass)?;
            }
        }
        Ok(())
    }

    pub fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for (layer, initialized) in &mut self.layers {
            if *initialized {
                layer.set_picking_active(ctx, active)?;
            }
        }
        Ok(())
    }

    pub fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for (layer, initialized) in &mut self.layers {
            if *initialized && layer.props().visible && layer.props().pickable {
                layer.draw_picking(ctx, pass)?;
            }
        }
        Ok(())
    }

    pub fn set_highlighted_object(&mut self, index: Option<u32>) {
        for (layer, _) in &mut self.layers {
            layer.set_highlighted_object(index);
        }
    }
}

/// Write `picking.isActive` on a model and upload it. Used by layers' `set_picking_active`.
pub fn set_model_picking_active(model: &mut Model, queue: &wgpu::Queue, active: bool) -> Result<()> {
    model
        .uniforms("picking")?
        .set_f32("isActive", if active { 1.0 } else { 0.0 })?;
    model.upload_uniforms(queue);
    Ok(())
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
    picking.set_f32("useByteColors", 1.0)?;
    picking.set_f32("disabledPickingIndexCount", 0.0)?;
    match props.highlighted_object_index {
        Some(index) => {
            let color = encode_picking_color(index);
            picking.set_f32("isHighlightActive", 1.0)?;
            picking.set_vec3(
                "highlightedObjectColor",
                glam::Vec3::new(color[0] as f32, color[1] as f32, color[2] as f32),
            )?;
        }
        None => {
            picking.set_f32("isHighlightActive", 0.0)?;
            picking.set_vec3("highlightedObjectColor", glam::Vec3::ZERO)?;
        }
    }
    let h = props.highlight_color;
    picking.set_vec4(
        "highlightColor",
        glam::Vec4::new(h[0] as f32, h[1] as f32, h[2] as f32, h[3] as f32) / 255.0,
    )?;

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

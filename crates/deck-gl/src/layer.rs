//! Port of the parts of `@deck.gl/core/src/lib/layer.ts` that a native layer needs.

use std::sync::Arc;

use glam::DMat4;
use luma_gl::{Model, ModelDescriptor, RenderTarget};

use crate::constants::{ClipDepthRange, CoordinateSystem};
use crate::data::{Color, Position};
use crate::deck::PickingInfo;
use crate::extension::Extensions;
use crate::lighting::{LightingEffect, Material};
use crate::mask::MaskMaps;
use crate::parameters::RenderParameters;
use crate::shaderlib::project::{get_uniforms_from_viewport, ProjectProps};
use crate::viewport::Viewport;
use crate::Result;

/// The function behind a [`HoverCallback`].
pub type HoverFn = dyn Fn(Option<&PickingInfo>) + Send + Sync;
/// The function behind a [`ClickCallback`].
pub type ClickFn = dyn Fn(&PickingInfo) + Send + Sync;

/// deck.gl's `onHover`: called with the picked object when the pointer moves onto an object of
/// the layer, and with `None` when it leaves the layer's objects. Compared by identity, so
/// keep one instance around rather than wrapping a new closure on every `set_layers`.
#[derive(Clone)]
pub struct HoverCallback(pub Arc<HoverFn>);

impl HoverCallback {
    pub fn new(f: impl Fn(Option<&PickingInfo>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn call(&self, info: Option<&PickingInfo>) {
        (self.0)(info)
    }
}

impl PartialEq for HoverCallback {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for HoverCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HoverCallback")
    }
}

/// deck.gl's `onClick`: called with the object under a click. Compared by identity.
#[derive(Clone)]
pub struct ClickCallback(pub Arc<ClickFn>);

impl ClickCallback {
    pub fn new(f: impl Fn(&PickingInfo) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn call(&self, info: &PickingInfo) {
        (self.0)(info)
    }
}

impl PartialEq for ClickCallback {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ClickCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ClickCallback")
    }
}

/// What a layer renders into, deck.gl's `operation` prop. A layer with `mask` is drawn into
/// a mask texture that layers with the `MaskExtension` sample, and not on screen (deck.gl's
/// `mask+draw` is not supported yet: add a second layer to draw the same geometry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operation {
    pub draw: bool,
    pub mask: bool,
}

impl Operation {
    pub const DRAW: Self = Self {
        draw: true,
        mask: false,
    };
    pub const MASK: Self = Self {
        draw: false,
        mask: true,
    };
}

impl Default for Operation {
    fn default() -> Self {
        Self::DRAW
    }
}

/// Properties shared by all layers. Mirrors deck.gl's `LayerProps`.
#[derive(Clone, Debug, PartialEq)]
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
    /// Reflectance of lit layers (extruded polygons, columns, point clouds)
    pub material: Material,
    /// Pipeline state overrides (blending, depth test, culling), deck.gl's `parameters`
    pub parameters: RenderParameters,
    /// Highlight the object under the pointer (see [`crate::Deck::pointer_move`])
    pub auto_highlight: bool,
    pub on_hover: Option<HoverCallback>,
    pub on_click: Option<ClickCallback>,
    /// Extensions that add shader code and attributes to the layer, see
    /// [`LayerExtension`](crate::LayerExtension).
    pub extensions: Extensions,
    /// Whether the layer draws on screen or into a mask, see [`Operation`].
    pub operation: Operation,
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
            material: Material::default(),
            parameters: RenderParameters::default(),
            auto_highlight: false,
            on_hover: None,
            on_click: None,
            extensions: Extensions::default(),
            operation: Operation::DRAW,
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

    /// True when replacing these props with `next` needs a new model: the pipeline state,
    /// the picking variant or the shader code the extensions contribute changed. Extension
    /// options that only feed attributes and uniforms do not.
    pub fn needs_new_model(&self, next: &LayerProps) -> bool {
        self.pickable != next.pickable
            || self.parameters != next.parameters
            || self.operation != next.operation
            || self.extensions.shaders() != next.extensions.shaders()
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
    /// Constant depth bias added to every layer, in depth buffer units. A host that shares its
    /// depth buffer sets this so deck's ground level geometry wins against the host's ground.
    pub depth_bias_base: i32,
    /// Depth convention of the depth buffer, see [`ClipDepthRange`].
    pub clip_depth_range: ClipDepthRange,
    /// Uniform slot models write and draw with (see `Model::set_uniform_slot`): 0 for the main
    /// viewport, one more for each repeated world copy the deck draws.
    pub uniform_slot: usize,
    /// Where the pointer is over the deck, in logical pixels from the top left, or `None`
    /// when it is outside (deck.gl's `mousePosition`). Maintained by `Deck::pointer_move`
    /// and `Deck::pointer_leave`; the brushing extension reads it.
    pub pointer: Option<[f64; 2]>,
    /// The masks rendered this frame, for the mask extension. `None` outside a `Deck`.
    pub masks: Option<Arc<MaskMaps>>,
}

/// Attachments of the mask pass: one red channel texture and no depth buffer.
pub const MASK_TARGET: RenderTarget = RenderTarget {
    color_format: wgpu::TextureFormat::R8Unorm,
    depth_format: None,
    sample_count: 1,
};

/// deck.gl's mask pass blending: every drawn fragment sets the channel to zero, whatever the
/// layer's colour (`zero * source - one * destination`, clamped).
pub fn mask_blend() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Subtract,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

impl LayerContext {
    /// Depth bias for the current layer: deck.gl's per layer polygon offset plus the base.
    pub fn depth_bias(&self) -> wgpu::DepthBiasState {
        let mut bias = depth_bias_for_layer(self.layer_index);
        bias.constant += self.depth_bias_base;
        bias
    }

    /// Set a model descriptor up the way every layer does: the layer's depth bias, its
    /// picking variant and render parameters, and the mask pass state when its operation is
    /// `mask`.
    pub fn configure(&self, desc: &mut ModelDescriptor<'_>, props: &LayerProps) {
        desc.depth_bias = self.depth_bias();
        desc.pickable = props.pickable;
        props.parameters.apply(desc);
        if props.operation.mask {
            desc.blend = Some(mask_blend());
            desc.depth_compare = wgpu::CompareFunction::Always;
            desc.depth_write_enabled = false;
            desc.pickable = false;
        }
    }
}

/// The bounding box of positions, `[min x, min y, max x, max y]`; `None` without positions.
pub fn position_bounds<'a>(positions: impl IntoIterator<Item = &'a Position>) -> Option<[f64; 4]> {
    positions.into_iter().fold(None, |bounds, p| {
        Some(match bounds {
            None => [p[0], p[1], p[0], p[1]],
            Some([x0, y0, x1, y1]) => [x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1])],
        })
    })
}

/// The bounding box of two bounding boxes.
pub fn union_bounds(a: Option<[f64; 4]>, b: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (a, None) => a,
        (None, b) => b,
    }
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

    /// The extent of the layer's positions in its coordinate system, `[min x, min y, max x,
    /// max y]`, when known after an update. The mask pass fits its texture to it.
    fn bounds(&self) -> Option<[f64; 4]> {
        None
    }

    /// For downcasting in [`Layer::update_from`].
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Take over the props of `incoming`, a layer with the same id that replaces this one in
    /// [`Deck::set_layers`](crate::Deck::set_layers), keeping this layer's GPU resources.
    /// Returns false when `incoming` is a different layer type, in which case it is used as is.
    /// Layers compare the props and only rebuild attributes when something changed.
    fn update_from(&mut self, _incoming: &mut dyn Layer) -> bool {
        false
    }
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

    /// The bounds of all sub layers together.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        self.layers
            .iter()
            .fold(None, |bounds, (layer, _)| union_bounds(bounds, layer.bounds()))
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

/// The projection inputs of a layer: its coordinate system, origin and model matrix with the
/// viewport it is drawn in.
pub fn project_props<'a>(ctx: &LayerContext, viewport: &'a Viewport, props: &LayerProps) -> ProjectProps<'a> {
    ProjectProps {
        viewport,
        device_pixel_ratio: ctx.device_pixel_ratio,
        model_matrix: props.model_matrix,
        coordinate_system: props.coordinate_system,
        coordinate_origin: glam::DVec3::from(props.coordinate_origin),
        auto_wrap_longitude: props.wrap_longitude,
        clip_depth_range: ctx.clip_depth_range,
    }
}

/// Fill the uniform blocks every deck.gl layer shader has: `project`, `layer` and `picking`,
/// plus `lighting`, `gouraudMaterial` and `floatColors` when the model uses them.
pub fn update_standard_uniforms(
    model: &mut Model,
    ctx: &LayerContext,
    viewport: &Viewport,
    props: &LayerProps,
) -> Result<()> {
    model.set_uniform_slot(ctx.uniform_slot);
    let project = get_uniforms_from_viewport(&project_props(ctx, viewport, props));
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
        props.material.write(model.uniforms("gouraudMaterial")?)?;
    }
    if model.has_uniforms("floatColors") {
        model.uniforms("floatColors")?.set_f32("useByteColors", 1.0)?;
    }
    props.extensions.update_uniforms(model, ctx, viewport, props)?;

    model.upload_uniforms(&ctx.queue);
    Ok(())
}

/// The GPU resources of a layer, or the error a layer returns when it is used before
/// `initialize`.
pub fn initialized<'a, T>(resource: Option<&'a mut T>, layer: &str) -> Result<&'a mut T> {
    resource.ok_or_else(|| crate::DeckError::Layer {
        layer: layer.to_string(),
        message: "used before initialize".to_string(),
    })
}

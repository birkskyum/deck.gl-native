//! Port of `@deck.gl/core/src/lib/deck.ts`, reduced to what a native host needs.

use luma_gl::{RenderTarget, PICKING_FORMAT};

use luma_gl::device::{create_render_texture, read_texture_rgba8};

use crate::constants::ClipDepthRange;
use crate::layer::{
    decode_picking_color, ClickCallback, HoverCallback, Layer, LayerContext, LayerProps, LAYER_INDEX_STRIDE,
};
use crate::lighting::LightingEffect;
use crate::transition::{TransitionProps, ViewStateTransition};
use crate::viewport::Viewport;
use crate::views::{AnyViewState, View};
use crate::{DeckError, Result};

/// Camera state for the default map view. Mirrors deck.gl's `MapViewState`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
    pub pitch: f64,
    pub bearing: f64,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            longitude: 0.0,
            latitude: 0.0,
            zoom: 0.0,
            pitch: 0.0,
            bearing: 0.0,
        }
    }
}

/// Initial properties of a [`Deck`].
pub struct DeckProps {
    /// Size of the render target in CSS pixels
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f32,
    pub view_state: ViewState,
    pub layers: Vec<Box<dyn Layer>>,
    pub lighting: LightingEffect,
    /// Constant depth bias for all layers, see [`LayerContext::depth_bias_base`].
    pub depth_bias_base: i32,
    /// Depth convention of the depth buffer, see [`ClipDepthRange`].
    pub clip_depth_range: ClipDepthRange,
    /// Draw extra copies of the world when the view spans the antimeridian, deck.gl's
    /// `MapView({repeat: true})`.
    pub repeat: bool,
    /// The kind of camera: a map by default, or an orthographic, orbit or first person view.
    /// With a non map view, set the camera with [`Deck::set_any_view_state`].
    pub view: View,
}

impl Default for DeckProps {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            device_pixel_ratio: 1.0,
            view_state: ViewState::default(),
            layers: Vec::new(),
            lighting: LightingEffect::default(),
            depth_bias_base: 0,
            clip_depth_range: ClipDepthRange::default(),
            repeat: false,
            view: View::Map,
        }
    }
}

struct LayerEntry {
    layer: Box<dyn Layer>,
    initialized: bool,
}

/// What [`Deck::pick`] found under a pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct PickingInfo {
    /// Id of the top-level layer
    pub layer_id: String,
    /// Data row of the picked object
    pub index: u32,
    /// The queried pixel in logical coordinates
    pub pixel: [f64; 2],
    /// The pixel unprojected onto the ground plane, as lng, lat, 0
    pub coordinate: [f64; 3],
}

/// Offscreen attachments for the picking pass.
struct PickingTarget {
    color: wgpu::Texture,
    depth: Option<wgpu::Texture>,
    readback: wgpu::Buffer,
}

/// Takes layer instances and a camera, and draws the layers into a render pass.
///
/// A `Deck` does not own the render target. Either call [`Deck::draw`] from inside a render
/// pass you own (interleaved with a basemap), or use [`Deck::render`] to have the deck begin
/// its own pass on textures you provide.
/// Multisampled attachments the deck renders into before resolving to the caller's texture.
struct MsaaTextures {
    color: wgpu::Texture,
    depth: Option<wgpu::Texture>,
}

pub struct Deck {
    ctx: LayerContext,
    layers: Vec<LayerEntry>,
    msaa: Option<MsaaTextures>,
    width: u32,
    height: u32,
    view: View,
    view_state: ViewState,
    /// The view state of the current view; mirrors `view_state` for a map view
    camera: AnyViewState,
    viewport: Viewport,
    external_viewport: bool,
    picking: Option<PickingTarget>,
    repeat: bool,
    /// A view state transition started with [`Deck::transition_to`]
    transition: Option<ViewStateTransition>,
    /// The object under the pointer after the last `pointer_move`
    hovered: Option<PickingInfo>,
    on_hover: Option<HoverCallback>,
    on_click: Option<ClickCallback>,
}

impl Deck {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: RenderTarget,
        props: DeckProps,
    ) -> Result<Self> {
        let ctx = LayerContext {
            device: device.clone(),
            queue: queue.clone(),
            target,
            device_pixel_ratio: props.device_pixel_ratio,
            lighting: props.lighting,
            layer_index: 0,
            depth_bias_base: props.depth_bias_base,
            clip_depth_range: props.clip_depth_range,
            uniform_slot: 0,
        };
        let camera = AnyViewState::Map(props.view_state);
        let viewport = props
            .view
            .make_viewport(&camera, props.width as f64, props.height as f64);
        let mut deck = Self {
            ctx,
            layers: Vec::new(),
            msaa: None,
            width: props.width,
            height: props.height,
            view: props.view,
            view_state: props.view_state,
            camera,
            viewport,
            external_viewport: false,
            picking: None,
            repeat: props.repeat,
            transition: None,
            hovered: None,
            on_hover: None,
            on_click: None,
        };
        deck.set_layers(props.layers);
        Ok(deck)
    }

    pub fn context(&self) -> &LayerContext {
        &self.ctx
    }

    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    pub fn view_state(&self) -> &ViewState {
        &self.view_state
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resize the render target (CSS pixels).
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        if !self.external_viewport {
            self.viewport = self.view.make_viewport(&self.camera, width as f64, height as f64);
        }
    }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.ctx.device_pixel_ratio = ratio;
    }

    /// Move the map camera, ending any transition. With a non map view this only records the
    /// state; see [`Deck::set_any_view_state`].
    pub fn set_view_state(&mut self, view_state: ViewState) {
        self.transition = None;
        self.view_state = view_state;
        if matches!(self.view, View::Map) {
            self.camera = AnyViewState::Map(view_state);
        }
        self.external_viewport = false;
        self.viewport = self
            .view
            .make_viewport(&self.camera, self.width as f64, self.height as f64);
    }

    /// Switch the kind of camera (map, orthographic, orbit or first person). The camera keeps
    /// its state when it matches the new view, otherwise the view's default state is used.
    pub fn set_view(&mut self, view: View) {
        self.view = view;
        let matches = matches!(
            (&view, &self.camera),
            (View::Map, AnyViewState::Map(_))
                | (View::Orthographic(_), AnyViewState::Orthographic(_))
                | (View::Orbit(_), AnyViewState::Orbit(_))
                | (View::FirstPerson(_), AnyViewState::FirstPerson(_))
        );
        if !matches {
            self.camera = view.default_view_state();
        }
        self.external_viewport = false;
        self.viewport = self
            .view
            .make_viewport(&self.camera, self.width as f64, self.height as f64);
    }

    pub fn view(&self) -> View {
        self.view
    }

    /// Move the camera of any view kind; a map state also updates [`Deck::view_state`].
    pub fn set_any_view_state(&mut self, state: AnyViewState) {
        self.transition = None;
        if let AnyViewState::Map(view_state) = state {
            self.view_state = view_state;
        }
        self.camera = state;
        self.external_viewport = false;
        self.viewport = self
            .view
            .make_viewport(&self.camera, self.width as f64, self.height as f64);
    }

    /// The camera state of the current view.
    pub fn any_view_state(&self) -> AnyViewState {
        self.camera
    }

    /// Animate the map camera to `end` with deck.gl's transition props, for decks driven
    /// without a [`crate::MapController`] (which has its own transitions). Call
    /// [`Deck::tick`] every frame with the same millisecond clock as `now`.
    pub fn transition_to(&mut self, end: ViewState, props: TransitionProps, now: f64) {
        let (current, proceed) =
            crate::transition::interrupt(self.transition.as_ref(), self.view_state, props.interruption);
        if !proceed {
            return;
        }
        self.set_view_state(current);
        self.transition = ViewStateTransition::new(
            self.view_state,
            end,
            self.width as f64,
            self.height as f64,
            props,
            now,
        );
        if self.transition.is_none() {
            self.set_view_state(end);
        }
    }

    /// Fly to `end` along the van Wijk and Nuij path with an automatic duration.
    pub fn fly_to(&mut self, end: ViewState, now: f64) {
        self.transition_to(end, TransitionProps::fly_to(), now);
    }

    /// Advance a transition started with [`Deck::transition_to`]; returns true while the
    /// camera is still moving.
    pub fn tick(&mut self, now: f64) -> bool {
        let Some(transition) = self.transition else {
            return false;
        };
        let view = transition.at(now);
        self.view_state = view;
        self.camera = AnyViewState::Map(view);
        self.external_viewport = false;
        self.viewport = self
            .view
            .make_viewport(&self.camera, self.width as f64, self.height as f64);
        if transition.is_done(now) {
            self.transition = None;
        }
        true
    }

    /// The view state transition in flight, if any.
    pub fn transition(&self) -> Option<&ViewStateTransition> {
        self.transition.as_ref()
    }

    /// Use a caller-built viewport instead of the internal view state. This is how a host
    /// such as maplibre-native drives the deck camera from its own.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
        self.external_viewport = true;
    }

    pub fn set_lighting(&mut self, lighting: LightingEffect) {
        self.ctx.lighting = lighting;
    }

    /// Draw extra copies of the world across the antimeridian, see [`DeckProps::repeat`].
    pub fn set_repeat(&mut self, repeat: bool) {
        self.repeat = repeat;
    }

    /// Replace the layer list. Layers whose id matches an existing layer of the same type keep
    /// that layer's GPU resources and only take over the new props (see
    /// [`Layer::update_from`]); everything else is initialized on the next update.
    pub fn set_layers(&mut self, layers: Vec<Box<dyn Layer>>) {
        let hovered = self.hovered.clone();
        let mut previous: Vec<Option<LayerEntry>> = self.layers.drain(..).map(Some).collect();
        self.layers = layers
            .into_iter()
            .map(|mut layer| {
                let existing = previous
                    .iter_mut()
                    .find(|slot| slot.as_ref().is_some_and(|e| e.layer.id() == layer.id()))
                    .and_then(Option::take);
                match existing {
                    Some(mut entry)
                        if entry.initialized && same_pipelines(entry.layer.props(), layer.props()) =>
                    {
                        // An auto highlight lives in the layer's props; take it out before the
                        // diff so it does not count as a change, and put it back after.
                        let auto_highlight = hovered
                            .as_ref()
                            .filter(|h| h.layer_id == entry.layer.id() && entry.layer.props().auto_highlight)
                            .map(|h| h.index);
                        if auto_highlight.is_some() {
                            entry.layer.set_highlighted_object(None);
                        }
                        if entry.layer.update_from(layer.as_mut()) {
                            if auto_highlight.is_some() && entry.layer.props().auto_highlight {
                                entry.layer.set_highlighted_object(auto_highlight);
                            }
                            entry
                        } else {
                            LayerEntry {
                                layer,
                                initialized: false,
                            }
                        }
                    }
                    _ => LayerEntry {
                        layer,
                        initialized: false,
                    },
                }
            })
            .collect();
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(LayerEntry {
            layer,
            initialized: false,
        });
    }

    /// A layer by id, to change it in place (for instance a `TripsLayer`'s current time).
    pub fn layer_mut(&mut self, id: &str) -> Option<&mut (dyn Layer + 'static)> {
        self.layers
            .iter_mut()
            .find(|e| e.layer.id() == id)
            .map(|e| e.layer.as_mut())
    }

    pub fn layers(&self) -> impl Iterator<Item = &dyn Layer> {
        self.layers.iter().map(|e| e.layer.as_ref())
    }

    /// Highlight one object of a layer (by top-level layer id), or clear the highlight.
    pub fn set_highlighted_object(&mut self, layer_id: &str, index: Option<u32>) {
        for entry in &mut self.layers {
            if entry.layer.id() == layer_id {
                entry.layer.set_highlighted_object(index);
            }
        }
    }

    /// Clear highlights on every layer.
    pub fn clear_highlights(&mut self) {
        for entry in &mut self.layers {
            entry.layer.set_highlighted_object(None);
        }
    }

    /// A deck level `onHover`, called after the layer callbacks whenever the hovered object
    /// changes (`None` when the pointer is over nothing).
    pub fn set_on_hover(&mut self, callback: Option<HoverCallback>) {
        self.on_hover = callback;
    }

    /// A deck level `onClick`, called after the layer callback for every click on an object.
    pub fn set_on_click(&mut self, callback: Option<ClickCallback>) {
        self.on_click = callback;
    }

    /// The object under the pointer after the last [`Deck::pointer_move`].
    pub fn hovered(&self) -> Option<&PickingInfo> {
        self.hovered.as_ref()
    }

    /// Pick under the pointer. When the hovered object changed, the layer the pointer left
    /// gets `on_hover(None)`, the layer it entered gets `on_hover(Some(info))`, layers with
    /// `auto_highlight` highlight the hovered object, and the deck's own hover callback runs.
    /// Returns the object under the pointer. Like [`Deck::pick`], this waits for the GPU.
    pub fn pointer_move(&mut self, x: f64, y: f64) -> Result<Option<PickingInfo>> {
        let hit = self.pick(x, y)?;
        self.set_hovered(hit.clone());
        Ok(hit)
    }

    /// The pointer left the deck: hover callbacks and auto highlights are cleared.
    pub fn pointer_leave(&mut self) {
        self.set_hovered(None);
    }

    /// Pick under a click and run the picked layer's `on_click`, then the deck's.
    pub fn click(&mut self, x: f64, y: f64) -> Result<Option<PickingInfo>> {
        let hit = self.pick(x, y)?;
        if let Some(info) = &hit {
            if let Some(callback) = self.layer_props(&info.layer_id).and_then(|p| p.on_click.clone()) {
                callback.call(info);
            }
            if let Some(callback) = self.on_click.clone() {
                callback.call(info);
            }
        }
        Ok(hit)
    }

    fn set_hovered(&mut self, hit: Option<PickingInfo>) {
        let same = match (&self.hovered, &hit) {
            (Some(a), Some(b)) => a.layer_id == b.layer_id && a.index == b.index,
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        if let Some(previous) = self.hovered.take() {
            let left_layer = hit.as_ref().is_none_or(|h| h.layer_id != previous.layer_id);
            if left_layer {
                if let Some(callback) = self
                    .layer_props(&previous.layer_id)
                    .and_then(|p| p.on_hover.clone())
                {
                    callback.call(None);
                }
            }
            if self
                .layer_props(&previous.layer_id)
                .is_some_and(|p| p.auto_highlight)
            {
                self.set_highlighted_object(&previous.layer_id, None);
            }
        }
        if let Some(info) = &hit {
            if let Some(callback) = self.layer_props(&info.layer_id).and_then(|p| p.on_hover.clone()) {
                callback.call(Some(info));
            }
            if self.layer_props(&info.layer_id).is_some_and(|p| p.auto_highlight) {
                self.set_highlighted_object(&info.layer_id, Some(info.index));
            }
        }
        if let Some(callback) = self.on_hover.clone() {
            callback.call(hit.as_ref());
        }
        self.hovered = hit;
    }

    fn layer_props(&self, id: &str) -> Option<&LayerProps> {
        self.layers
            .iter()
            .find(|entry| entry.layer.id() == id)
            .map(|entry| entry.layer.props())
    }

    /// Find the object under a pixel (logical coordinates, origin top left).
    ///
    /// Renders every visible, pickable layer into an offscreen picking target with object
    /// indices encoded as colors and the layer encoded in alpha, then reads the pixel back.
    /// This waits for the GPU, so call it at most once per frame.
    pub fn pick(&mut self, x: f64, y: f64) -> Result<Option<PickingInfo>> {
        let dpr = self.ctx.device_pixel_ratio as f64;
        let width = ((self.width as f64 * dpr).round() as u32).max(1);
        let height = ((self.height as f64 * dpr).round() as u32).max(1);
        let px = (x * dpr).floor();
        let py = (y * dpr).floor();
        if px < 0.0 || py < 0.0 || px >= width as f64 || py >= height as f64 {
            return Ok(None);
        }
        self.update()?;
        self.ensure_picking_target(width, height);

        let pickable: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, e)| e.initialized && e.layer.props().visible && e.layer.props().pickable)
            .map(|(i, _)| i)
            .collect();
        if pickable.is_empty() {
            return Ok(None);
        }
        for &i in &pickable {
            self.layers[i].layer.set_picking_active(&self.ctx, true)?;
        }

        let target = self.picking.as_ref().expect("picking target");
        let color_view = target.color.create_view(&Default::default());
        let depth_view = target.depth.as_ref().map(|d| d.create_view(&Default::default()));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deck.gl picking"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deck.gl picking"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Alpha carries the layer: slot 1 for the first pickable layer, and so on.
            for (slot, &i) in pickable.iter().enumerate() {
                let alpha = (slot + 1) as f64 / 255.0;
                pass.set_blend_constant(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: alpha,
                });
                self.ctx.layer_index = i as u32 * LAYER_INDEX_STRIDE;
                self.layers[i].layer.draw_picking(&self.ctx, &mut pass)?;
            }
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.color,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: px as u32,
                    y: py as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.ctx.queue.submit([encoder.finish()]);

        for &i in &pickable {
            self.layers[i].layer.set_picking_active(&self.ctx, false)?;
        }

        let slice = target.readback.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| luma_gl::LumaError::Device(format!("poll failed: {e:?}")))?;
        rx.recv()
            .map_err(|_| luma_gl::LumaError::Device("map_async callback dropped".into()))?
            .map_err(|e| luma_gl::LumaError::Device(format!("map failed: {e:?}")))?;
        let pixel = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| luma_gl::LumaError::Device(format!("mapped range: {e:?}")))?;
            [view[0], view[1], view[2], view[3]]
        };
        target.readback.unmap();

        let slot = pixel[3] as usize;
        if slot == 0 {
            return Ok(None);
        }
        let Some(&layer_index) = pickable.get(slot - 1) else {
            return Ok(None);
        };
        let Some(index) = decode_picking_color([pixel[0], pixel[1], pixel[2]]) else {
            return Ok(None);
        };
        let coordinate = self.viewport.unproject(glam::DVec2::new(x, y), None, true, None);
        Ok(Some(PickingInfo {
            layer_id: self.layers[layer_index].layer.id().to_string(),
            index,
            pixel: [x, y],
            coordinate: [coordinate.x, coordinate.y, coordinate.z],
        }))
    }

    fn ensure_picking_target(&mut self, width: u32, height: u32) {
        let fresh = match &self.picking {
            Some(target) => target.color.size().width != width || target.color.size().height != height,
            None => true,
        };
        if !fresh {
            return;
        }
        let device = &self.ctx.device;
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deck.gl picking color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICKING_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = self.ctx.target.depth_format.map(|format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("deck.gl picking depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deck.gl picking readback"),
            size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.picking = Some(PickingTarget {
            color,
            depth,
            readback,
        });
    }

    /// Initialize new layers and update all layers for the current viewport.
    /// Must be called before [`Deck::draw`], outside of any render pass.
    pub fn update(&mut self) -> Result<()> {
        self.ctx.uniform_slot = 0;
        for (index, entry) in self.layers.iter_mut().enumerate() {
            self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
            if !entry.initialized {
                entry.layer.initialize(&self.ctx)?;
                entry.initialized = true;
            }
            if entry.layer.props().visible {
                entry.layer.update(&self.ctx, &self.viewport)?;
            }
        }
        Ok(())
    }

    /// Encode all visible layers into a render pass whose attachments match the deck's
    /// [`RenderTarget`]. The pass viewport is expected to cover the full deck size.
    pub fn draw(&mut self, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.ctx.uniform_slot = 0;
        for (index, entry) in self.layers.iter_mut().enumerate() {
            self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
            if entry.initialized && entry.layer.props().visible {
                entry.layer.draw(&self.ctx, pass)?;
            }
        }
        if self.repeat {
            // Extra world copies: every layer updates its uniforms for the shifted viewport
            // in its own uniform slot, so the copies do not overwrite each other's uniforms.
            let copies = self.viewport.sub_viewports();
            for (slot, viewport) in copies.iter().filter(|v| v.world_offset != 0).enumerate() {
                self.ctx.uniform_slot = slot + 1;
                for (index, entry) in self.layers.iter_mut().enumerate() {
                    self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
                    if entry.initialized && entry.layer.props().visible {
                        entry.layer.update(&self.ctx, viewport)?;
                        entry.layer.draw(&self.ctx, pass)?;
                    }
                }
            }
            self.ctx.uniform_slot = 0;
        }
        Ok(())
    }

    /// Render one frame into deck owned textures and read the pixels back. The image is the
    /// deck's size times its device pixel ratio; `clear_color` fills the background and `None`
    /// leaves it transparent. Blocks until the GPU is done, so this is for tools and tests
    /// rather than interactive frames.
    pub fn snapshot(&mut self, clear_color: Option<wgpu::Color>) -> Result<Snapshot> {
        let ratio = self.ctx.device_pixel_ratio.max(f32::EPSILON);
        let width = ((self.width as f32 * ratio).round() as u32).max(1);
        let height = ((self.height as f32 * ratio).round() as u32).max(1);
        let target = self.ctx.target;
        let color = create_render_texture(
            &self.ctx.device,
            "snapshot color",
            width,
            height,
            target.color_format,
        );
        let depth = target
            .depth_format
            .map(|format| create_render_texture(&self.ctx.device, "snapshot depth", width, height, format));
        let color_view = color.create_view(&Default::default());
        let depth_view = depth.as_ref().map(|t| t.create_view(&Default::default()));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("snapshot"),
            });
        self.render(
            &mut encoder,
            &color_view,
            depth_view.as_ref(),
            Some(clear_color.unwrap_or(wgpu::Color::TRANSPARENT)),
        )?;
        self.ctx.queue.submit([encoder.finish()]);
        let mut rgba = read_texture_rgba8(&self.ctx.device, &self.ctx.queue, &color)?;
        if matches!(
            target.color_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Ok(Snapshot { width, height, rgba })
    }

    /// Convenience: update, then begin a render pass on the given views and draw.
    ///
    /// `clear_color` clears the color attachment first; `None` loads the existing contents so
    /// the deck composites over whatever was drawn before (a basemap, for instance). Depth is
    /// cleared when the color is cleared and loaded otherwise.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        clear_color: Option<wgpu::Color>,
    ) -> Result<()> {
        let color_load = match clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };
        let depth_load = if clear_color.is_some() {
            wgpu::LoadOp::Clear(1.0)
        } else {
            wgpu::LoadOp::Load
        };
        self.render_with(encoder, color_view, depth_view, color_load, depth_load)
    }

    /// Like [`Deck::render`] with explicit load operations for the color and depth attachments.
    /// With a multisampled [`RenderTarget`] (`sample_count > 1`) the layers are drawn into
    /// deck owned multisampled attachments and resolved into `color_view`; `depth_view` is
    /// then unused, and `color_load` must clear, since a host's single sample contents cannot
    /// be loaded into the multisampled buffer. Sharing a host's depth buffer therefore needs
    /// `sample_count: 1`.
    pub fn render_with(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        color_load: wgpu::LoadOp<wgpu::Color>,
        depth_load: wgpu::LoadOp<f32>,
    ) -> Result<()> {
        self.update()?;
        let sample_count = self.ctx.target.sample_count;
        let (msaa_color_view, msaa_depth_view) = if sample_count > 1 {
            if matches!(color_load, wgpu::LoadOp::Load) {
                return Err(DeckError::Render(
                    "a multisampled deck cannot load the target's contents; clear it or use sample_count 1"
                        .into(),
                ));
            }
            let msaa = self.msaa_textures(color_view.texture().size());
            (
                Some(msaa.color.create_view(&Default::default())),
                msaa.depth.as_ref().map(|t| t.create_view(&Default::default())),
            )
        } else {
            (None, None)
        };
        let color_attachment = match &msaa_color_view {
            Some(view) => wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: Some(color_view),
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Discard,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            },
        };
        let depth_view = if msaa_color_view.is_some() {
            msaa_depth_view.as_ref()
        } else {
            depth_view
        };
        let depth_attachment = depth_view.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: depth_load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("deck.gl"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.draw(&mut pass)
    }

    /// Multisampled attachments matching the target's size and formats.
    fn msaa_textures(&mut self, size: wgpu::Extent3d) -> &MsaaTextures {
        let fresh = self
            .msaa
            .as_ref()
            .is_some_and(|m| m.color.size() == size && m.color.format() == self.ctx.target.color_format);
        if !fresh {
            let target = self.ctx.target;
            let make = |label: &str, format: wgpu::TextureFormat| {
                self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: target.sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
            };
            self.msaa = Some(MsaaTextures {
                color: make("deck.gl msaa color", target.color_format),
                depth: target.depth_format.map(|f| make("deck.gl msaa depth", f)),
            });
        }
        self.msaa.as_ref().expect("msaa textures")
    }
}

/// Pixels read back by [`Deck::snapshot`]: RGBA, 8 bits per channel, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Snapshot {
    /// The pixel at `x`, `y` (origin top left).
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// Write the image as a PNG, creating parent directories. Needs the `png` feature.
    #[cfg(feature = "png")]
    pub fn save_png(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| DeckError::Render(format!("cannot create {}: {e}", parent.display())))?;
            }
        }
        image::save_buffer(path, &self.rgba, self.width, self.height, image::ColorType::Rgba8)
            .map_err(|e| DeckError::Render(format!("cannot write {}: {e}", path.display())))
    }
}

/// Whether an initialized layer can keep its models when it takes over these props. Picking
/// pipelines and render parameters are baked into the pipelines, so a change there means the
/// layer is initialized again.
fn same_pipelines(current: &LayerProps, incoming: &LayerProps) -> bool {
    current.pickable == incoming.pickable && !current.parameters.differs(&incoming.parameters)
}

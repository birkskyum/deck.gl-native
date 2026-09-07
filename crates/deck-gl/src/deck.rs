//! Port of `@deck.gl/core/src/lib/deck.ts`, reduced to what a native host needs.

use luma_gl::{create_rgba8_texture, PipelineCache, RenderTarget, PICKING_FORMAT};

use luma_gl::device::{create_render_texture, read_texture_rgba8};

use crate::collision::{CollisionMaps, CollisionTarget, COLLISION_DOWNSCALE, COLLISION_PADDING};
use crate::constants::{ClipDepthRange, CoordinateSystem, ProjectionMode};
use crate::layer::{
    decode_picking_color, ClickCallback, HoverCallback, Layer, LayerContext, LayerProps, LAYER_INDEX_STRIDE,
    MASK_TARGET,
};
use crate::lighting::LightingEffect;
use crate::mask::{
    create_mask_texture, mask_viewport, render_bounds, MaskChannel, MaskMaps, MASK_BORDER, MASK_MAP_SIZE,
    MAX_MASKS,
};
use crate::post_process::{PostProcessEffect, PostProcessor};
use crate::shadow::{
    light_matrices, shadow_map_size, shadow_shaders, shadows_enabled, ShadowState, ShadowTarget,
};
use crate::transition::{TransitionProps, ViewStateTransition};
use crate::viewport::Viewport;
use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;

use crate::views::{AnyViewState, DeckView, LayerFilter, View, ViewRect};
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
    /// Post-processing effects applied to the rendered frame, in order.
    pub post_process: Vec<PostProcessEffect>,
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
            post_process: Vec::new(),
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
    /// Id of the view the pixel is in (`"default"` with a single view)
    pub view_id: String,
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
    /// Several views in sub rectangles of the canvas; empty for the single full size view
    views: Vec<DeckView>,
    /// Cameras of `views` by view id
    cameras: HashMap<String, AnyViewState>,
    layer_filter: Option<LayerFilter>,
    /// Size of the attachments of the last render, to clamp view rectangles
    attachment_size: Option<(u32, u32)>,
    /// Viewport of the view picked last, for the unprojection of the hit
    pick_viewport: Option<Viewport>,
    stats: FrameStats,
    external_viewport: bool,
    picking: Option<PickingTarget>,
    repeat: bool,
    /// A view state transition started with [`Deck::transition_to`]
    transition: Option<ViewStateTransition>,
    /// The object under the pointer after the last `pointer_move`
    hovered: Option<PickingInfo>,
    on_hover: Option<HoverCallback>,
    on_click: Option<ClickCallback>,
    /// The masks of the current frame, shared with the layer context
    masks: Arc<MaskMaps>,
    /// One texture per mask layer id, kept across frames
    mask_textures: HashMap<String, wgpu::Texture>,
    /// The collision maps of the current frame, shared with the layer context
    collisions: Arc<CollisionMaps>,
    /// One colour and depth target per collision group, kept across frames
    collision_targets: HashMap<String, CollisionTarget>,
    /// When the deck was created, the origin of its own clock
    created: std::time::Instant,
    /// The time of the last `tick`, which replaces the deck's own clock once used
    now: Option<f64>,
    /// Post-processing effects applied after the layers, in order
    post_process: Vec<PostProcessEffect>,
    post_processor: PostProcessor,
    /// One shadow map per light that casts shadows, kept across frames
    shadow_targets: Vec<ShadowTarget>,
    /// Bound where a light has no map yet
    shadow_dummy: wgpu::TextureView,
}

impl Deck {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: RenderTarget,
        props: DeckProps,
    ) -> Result<Self> {
        let masks = Arc::new(MaskMaps::new(device, queue));
        let collisions = Arc::new(CollisionMaps::new(device, queue));
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
            pointer: None,
            masks: Some(masks.clone()),
            collisions: Some(collisions.clone()),
            pipelines: PipelineCache::new(),
            time: 0.0,
            shadow_enabled: false,
            shadow: None,
            shadow_pass: None,
        };
        let camera = match props.view {
            View::Globe(_) => AnyViewState::Globe(props.view_state),
            _ => AnyViewState::Map(props.view_state),
        };
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
            views: Vec::new(),
            cameras: HashMap::new(),
            layer_filter: None,
            attachment_size: None,
            pick_viewport: None,
            stats: FrameStats::default(),
            external_viewport: false,
            picking: None,
            repeat: props.repeat,
            transition: None,
            hovered: None,
            on_hover: None,
            on_click: None,
            masks,
            mask_textures: HashMap::new(),
            collisions,
            collision_targets: HashMap::new(),
            created: std::time::Instant::now(),
            now: None,
            post_process: props.post_process,
            post_processor: PostProcessor::default(),
            shadow_targets: Vec::new(),
            shadow_dummy: create_rgba8_texture(device, queue, "shadow fallback", 1, 1, &[255, 255, 255, 255])
                .create_view(&Default::default()),
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
            self.refresh_viewport();
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
        match self.view {
            View::Map => self.camera = AnyViewState::Map(view_state),
            View::Globe(_) => self.camera = AnyViewState::Globe(view_state),
            _ => {}
        }
        self.sync_first_view_camera();
        self.external_viewport = false;
        self.refresh_viewport();
    }

    /// Switch the kind of camera (map, orthographic, orbit or first person). The camera keeps
    /// its state when it matches the new view, otherwise the view's default state is used.
    pub fn set_view(&mut self, view: View) {
        self.view = view;
        if !same_kind(&view, &self.camera) {
            self.camera = view.default_view_state();
        }
        self.external_viewport = false;
        self.refresh_viewport();
    }

    pub fn view(&self) -> View {
        self.view
    }

    /// Move the camera of any view kind; a map state also updates [`Deck::view_state`].
    pub fn set_any_view_state(&mut self, state: AnyViewState) {
        self.transition = None;
        if let AnyViewState::Map(view_state) | AnyViewState::Globe(view_state) = state {
            self.view_state = view_state;
        }
        self.camera = state;
        self.sync_first_view_camera();
        self.external_viewport = false;
        self.refresh_viewport();
    }

    /// The camera state of the current view.
    pub fn any_view_state(&self) -> AnyViewState {
        self.camera
    }

    /// Render into several views, each in its own rectangle of the canvas (deck.gl's `views`).
    /// Views draw in order, later ones over earlier ones. The first view becomes the deck's
    /// main view, so [`Deck::set_view_state`] and picking without a view keep working. An
    /// empty list returns to the single full size view.
    pub fn set_views(&mut self, views: Vec<DeckView>) {
        for view in &views {
            let camera = self
                .cameras
                .entry(view.id.clone())
                .or_insert_with(|| view.view.default_view_state());
            let matches = same_kind(&view.view, camera);
            if !matches {
                *camera = view.view.default_view_state();
            }
        }
        self.cameras.retain(|id, _| views.iter().any(|v| &v.id == id));
        self.views = views;
        if let Some(first) = self.views.first() {
            self.view = first.view;
            self.camera = self.cameras[&first.id];
            if let AnyViewState::Map(vs) | AnyViewState::Globe(vs) = self.camera {
                self.view_state = vs;
            }
        }
        self.external_viewport = false;
        self.refresh_viewport();
    }

    pub fn views(&self) -> &[DeckView] {
        &self.views
    }

    /// Move the camera of the view with `id`; ignored for unknown views or states of another
    /// kind than the view.
    pub fn set_view_state_for(&mut self, id: &str, state: AnyViewState) {
        let Some(view) = self.views.iter().find(|v| v.id == id) else {
            return;
        };
        if !same_kind(&view.view, &state) {
            return;
        }
        self.cameras.insert(id.to_string(), state);
        if self.views.first().is_some_and(|v| v.id == id) {
            self.camera = state;
            if let AnyViewState::Map(vs) | AnyViewState::Globe(vs) = state {
                self.view_state = vs;
            }
            self.transition = None;
        }
        self.external_viewport = false;
        self.refresh_viewport();
    }

    fn sync_first_view_camera(&mut self) {
        if let Some(first) = self.views.first() {
            if same_kind(&first.view, &self.camera) {
                self.cameras.insert(first.id.clone(), self.camera);
            }
        }
    }

    pub fn view_state_for(&self, id: &str) -> Option<AnyViewState> {
        self.cameras.get(id).copied()
    }

    /// Restrict which layers each view draws, deck.gl's `layerFilter`.
    pub fn set_layer_filter(&mut self, filter: Option<LayerFilter>) {
        self.layer_filter = filter;
    }

    /// Every view's rectangle and viewport, in drawing order. A single entry for the plain
    /// deck; views whose rectangle has no area are left out.
    pub fn viewports(&self) -> Vec<(ViewRect, Viewport)> {
        if self.views.is_empty() {
            let rect = ViewRect {
                x: 0.0,
                y: 0.0,
                width: self.width as f64,
                height: self.height as f64,
                padding: None,
            };
            return vec![(rect, self.viewport.clone())];
        }
        let (w, h) = (self.width as f64, self.height as f64);
        self.views
            .iter()
            .filter_map(|view| {
                let camera = self.cameras.get(&view.id)?;
                view.make_viewport(camera, w, h)
                    .map(|viewport| (view.rect(w, h), viewport))
            })
            .collect()
    }

    /// The view containing a canvas pixel, topmost first.
    fn view_at(&self, x: f64, y: f64) -> Option<(ViewRect, Viewport)> {
        self.viewports()
            .into_iter()
            .rev()
            .find(|(rect, _)| rect.contains(x, y))
    }

    /// Recompute the main viewport from the view kind, camera and size (or the first view).
    fn refresh_viewport(&mut self) {
        let (w, h) = (self.width as f64, self.height as f64);
        self.viewport = match self.views.first() {
            Some(first) => first
                .make_viewport(&self.camera, w, h)
                .unwrap_or_else(|| self.view.make_viewport(&self.camera, w, h)),
            None => self.view.make_viewport(&self.camera, w, h),
        };
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
        self.now = Some(now);
        let Some(transition) = self.transition else {
            return self.animating();
        };
        let view = transition.at(now);
        self.view_state = view;
        self.camera = match self.view {
            View::Globe(_) => AnyViewState::Globe(view),
            _ => AnyViewState::Map(view),
        };
        self.sync_first_view_camera();
        self.external_viewport = false;
        self.refresh_viewport();
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

    /// Replace the post-processing effects. With any, the layers render into a texture of
    /// the deck and each effect's passes run over it in order, the last one blending onto
    /// the render target; see [`PostProcessEffect`].
    pub fn set_post_process(&mut self, effects: Vec<PostProcessEffect>) {
        self.post_process = effects;
    }

    pub fn post_process(&self) -> &[PostProcessEffect] {
        &self.post_process
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
        self.ctx.pointer = Some([x, y]);
        let hit = self.pick(x, y)?;
        self.set_hovered(hit.clone());
        Ok(hit)
    }

    /// The pointer left the deck: hover callbacks and auto highlights are cleared, and
    /// layers brushed by the pointer draw everything again.
    pub fn pointer_leave(&mut self) {
        self.ctx.pointer = None;
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
        // With several views, pick in the topmost view under the pixel with its own viewport
        let (view_rect, view_id) = match (self.views.is_empty(), self.view_at(x, y)) {
            (false, Some((rect, viewport))) => {
                let id = viewport.id.clone();
                self.ctx.uniform_slot = 0;
                for (index, entry) in self.layers.iter_mut().enumerate() {
                    self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
                    if entry.initialized && entry.layer.props().visible && !entry.layer.props().operation.mask
                    {
                        entry.layer.update(&self.ctx, &viewport)?;
                    }
                }
                self.pick_viewport = Some(viewport);
                (Some(rect), id)
            }
            (false, None) => return Ok(None),
            (true, _) => (None, "default".to_string()),
        };
        self.ensure_picking_target(width, height);

        let pickable: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                let props = e.layer.props();
                e.initialized && props.visible && props.pickable && !props.operation.mask
            })
            .filter(|(_, e)| {
                self.layer_filter
                    .as_ref()
                    .is_none_or(|f| f.allows(e.layer.id(), &view_id))
            })
            .map(|(i, _)| i)
            .collect();
        if pickable.is_empty() {
            return Ok(None);
        }
        for &i in &pickable {
            self.layers[i].layer.set_picking_active(&self.ctx, true)?;
        }

        let Some(target) = self.picking.as_ref() else {
            return Err(DeckError::Render("picking target missing".into()));
        };
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
            if let Some(rect) = view_rect {
                let vx = ((rect.x * dpr).round() as u32).min(width);
                let vy = ((rect.y * dpr).round() as u32).min(height);
                let vw = ((rect.width * dpr).round() as u32).min(width - vx);
                let vh = ((rect.height * dpr).round() as u32).min(height - vy);
                pass.set_viewport(vx as f32, vy as f32, vw.max(1) as f32, vh.max(1) as f32, 0.0, 1.0);
                pass.set_scissor_rect(vx, vy, vw.max(1), vh.max(1));
            }
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
        let (viewport, local) = match (&self.pick_viewport, view_rect) {
            (Some(viewport), Some(rect)) => (viewport, glam::DVec2::new(x - rect.x, y - rect.y)),
            _ => (&self.viewport, glam::DVec2::new(x, y)),
        };
        let coordinate = viewport.unproject(local, None, true, None);
        Ok(Some(PickingInfo {
            layer_id: self.layers[layer_index].layer.id().to_string(),
            index,
            pixel: [x, y],
            coordinate: [coordinate.x, coordinate.y, coordinate.z],
            view_id,
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
    /// Must be called before [`Deck::draw`], outside of any render pass. Layers whose
    /// operation is `mask` are updated first and rendered into the mask textures, so that the
    /// layers sampling them see this frame's masks.
    pub fn update(&mut self) -> Result<()> {
        self.ctx.time = self.time();
        self.prepare_shadows();
        let result = (|| {
            self.update_layers(true)?;
            self.update_masks()?;
            self.update_layers(false)?;
            self.update_shadows()?;
            self.update_collisions()
        })();
        // The default shaders are per thread: another deck on this thread must not get them
        crate::extension::set_default_shaders(None);
        result
    }

    /// Turn the shadow module on or off before the layers build their models, so every model
    /// of a frame with shadows has the module and a shadow pipeline.
    fn prepare_shadows(&mut self) {
        let enabled = shadows_enabled(&self.ctx.lighting) && self.viewport.is_geospatial;
        if enabled != self.ctx.shadow_enabled {
            // The module changes the shaders: every model has to be built again
            for entry in &mut self.layers {
                entry.initialized = false;
            }
            self.ctx.shadow_enabled = enabled;
            self.shadow_targets.clear();
        }
        crate::extension::set_default_shaders(enabled.then(shadow_shaders));
        self.ctx.shadow = enabled.then(|| {
            Arc::new(ShadowState {
                light_matrices: light_matrices(&self.ctx.lighting, &self.viewport),
                viewport_center: self.viewport.center,
                maps: Vec::new(),
                dummy: self.shadow_dummy.clone(),
                color: self.ctx.lighting.shadow_color,
            })
        });
    }

    /// deck.gl's `ShadowPass`: draw every layer that casts shadows into one map per light,
    /// seen from the light, then publish the maps so the layers sample them.
    fn update_shadows(&mut self) -> Result<()> {
        let Some(state) = self.ctx.shadow.clone() else {
            return Ok(());
        };
        let lights = state.light_matrices.len();
        if lights == 0 {
            return Ok(());
        }
        let (width, height) = shadow_map_size(self.width, self.height, self.ctx.device_pixel_ratio);
        if self.shadow_targets.len() != lights
            || self
                .shadow_targets
                .first()
                .is_some_and(|t| t.size() != (width, height))
        {
            self.shadow_targets = (0..lights)
                .map(|i| ShadowTarget::new(&self.ctx.device, i, width, height))
                .collect();
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deck.gl shadows"),
            });
        let mut maps = Vec::with_capacity(lights);
        for light in 0..lights {
            self.ctx.shadow_pass = Some(light);
            self.ctx.uniform_slot = 0;
            let view = self.shadow_targets[light].color.create_view(&Default::default());
            let depth = self.shadow_targets[light].depth.create_view(&Default::default());
            for (index, entry) in self.layers.iter_mut().enumerate() {
                let props = entry.layer.props();
                if !entry.initialized || !props.visible || props.operation.mask || !props.shadow_enabled {
                    continue;
                }
                self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
                entry.layer.update(&self.ctx, &self.viewport)?;
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("deck.gl shadow map"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        // White is the farthest packed depth: nothing shadows by default
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                for (index, entry) in self.layers.iter_mut().enumerate() {
                    let props = entry.layer.props();
                    if !entry.initialized || !props.visible || props.operation.mask || !props.shadow_enabled {
                        continue;
                    }
                    self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
                    entry.layer.draw(&self.ctx, &mut pass)?;
                }
            }
            maps.push(view);
        }
        self.ctx.queue.submit([encoder.finish()]);
        self.ctx.shadow_pass = None;
        self.ctx.shadow = Some(Arc::new(ShadowState {
            light_matrices: state.light_matrices.clone(),
            viewport_center: state.viewport_center,
            maps,
            dummy: self.shadow_dummy.clone(),
            color: state.color,
        }));
        // The layers wrote shadow pass uniforms; write the frame's again
        self.update_layers(false)
    }

    /// deck.gl's `CollisionFilterEffect`: draw the layers of every collision group into a
    /// half resolution map with their picking colours, sorted by collision priority, that the
    /// collision filter extension samples to hide overlapping objects.
    fn update_collisions(&mut self) -> Result<()> {
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, entry) in self.layers.iter().enumerate() {
            let props = entry.layer.props();
            if !entry.initialized || !props.visible || props.operation.mask {
                continue;
            }
            let Some(group) = props.extensions.collision_group() else {
                continue;
            };
            match groups.iter_mut().find(|(name, _)| *name == group) {
                Some((_, layers)) => layers.push(i),
                None => groups.push((group, vec![i])),
            }
        }
        if groups.is_empty() {
            if !self.collisions.groups.is_empty() {
                self.publish_collisions(HashMap::new(), false);
            }
            return Ok(());
        }
        let dpr = self.ctx.device_pixel_ratio;
        let scale = dpr as f64 / COLLISION_DOWNSCALE as f64;
        let width = ((self.width as f64 * scale).round() as u32).max(3);
        let height = ((self.height as f64 * scale).round() as u32).max(3);
        self.collision_targets
            .retain(|group, _| groups.iter().any(|(name, _)| name == group));
        for (group, _) in &groups {
            let fresh = self
                .collision_targets
                .get(group)
                .is_none_or(|target| target.size() != (width, height));
            if fresh {
                let target = CollisionTarget::new(
                    &self.ctx.device,
                    group,
                    width,
                    height,
                    self.ctx.target.depth_format,
                );
                self.collision_targets.insert(group.clone(), target);
            }
        }
        let views: HashMap<String, wgpu::TextureView> = self
            .collision_targets
            .iter()
            .map(|(group, target)| (group.clone(), target.color.create_view(&Default::default())))
            .collect();

        // Draw the maps: picking colours at half resolution, depth from the priorities
        self.publish_collisions(views.clone(), true);
        self.ctx.device_pixel_ratio = dpr / COLLISION_DOWNSCALE as f32;
        self.ctx.uniform_slot = 0;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deck.gl collisions"),
            });
        for (group, layers) in &groups {
            for &i in layers {
                self.ctx.layer_index = i as u32 * LAYER_INDEX_STRIDE;
                self.layers[i].layer.update(&self.ctx, &self.viewport)?;
                self.layers[i].layer.set_picking_active(&self.ctx, true)?;
            }
            let Some(target) = self.collision_targets.get(group) else {
                continue;
            };
            let color_view = target.color.create_view(&Default::default());
            let depth_view = target.depth.as_ref().map(|d| d.create_view(&Default::default()));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("deck.gl collision map"),
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
                pass.set_scissor_rect(
                    COLLISION_PADDING,
                    COLLISION_PADDING,
                    width - 2 * COLLISION_PADDING,
                    height - 2 * COLLISION_PADDING,
                );
                for (slot, &i) in layers.iter().enumerate() {
                    pass.set_blend_constant(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: (slot + 1) as f64 / 255.0,
                    });
                    self.ctx.layer_index = i as u32 * LAYER_INDEX_STRIDE;
                    self.layers[i].layer.draw_picking(&self.ctx, &mut pass)?;
                }
            }
        }
        // Uniform writes land before later submissions, so picking stays on until the maps
        // were submitted
        self.ctx.queue.submit([encoder.finish()]);
        for (_, layers) in &groups {
            for &i in layers {
                self.layers[i].layer.set_picking_active(&self.ctx, false)?;
            }
        }
        self.ctx.device_pixel_ratio = dpr;

        // Uniforms for the frame, with the fresh maps bound
        self.publish_collisions(views, false);
        for (_, layers) in &groups {
            for &i in layers {
                self.ctx.layer_index = i as u32 * LAYER_INDEX_STRIDE;
                self.layers[i].layer.update(&self.ctx, &self.viewport)?;
            }
        }
        Ok(())
    }

    /// The collision map of a group as RGBA pixels, for debugging. Waits for the GPU.
    #[doc(hidden)]
    pub fn collision_map(&self, group: &str) -> Option<Vec<u8>> {
        let target = self.collision_targets.get(group)?;
        read_texture_rgba8(&self.ctx.device, &self.ctx.queue, &target.color).ok()
    }

    fn publish_collisions(&mut self, groups: HashMap<String, wgpu::TextureView>, drawing_to_map: bool) {
        self.collisions = Arc::new(CollisionMaps {
            sampler: self.collisions.sampler.clone(),
            dummy: self.collisions.dummy.clone(),
            groups,
            drawing_to_map,
        });
        self.ctx.collisions = Some(self.collisions.clone());
    }

    /// The time transitions run on, in seconds: the last [`Deck::tick`], or the deck's own
    /// clock since its creation when the host never ticks.
    pub fn time(&self) -> f64 {
        self.now.unwrap_or_else(|| self.created.elapsed().as_secs_f64())
    }

    /// Whether a view state, prop or attribute transition is still running, so the host
    /// should keep drawing frames (and ticking).
    pub fn animating(&self) -> bool {
        self.transition.is_some()
            || self
                .layers
                .iter()
                .any(|e| e.initialized && e.layer.in_transition())
    }

    /// Initialize and update the layers whose operation is (`masks`) or is not `mask`.
    fn update_layers(&mut self, masks: bool) -> Result<()> {
        let main_target = self.ctx.target;
        self.ctx.uniform_slot = 0;
        for (index, entry) in self.layers.iter_mut().enumerate() {
            if entry.layer.props().operation.mask != masks {
                continue;
            }
            self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
            self.ctx.target = if masks { MASK_TARGET } else { main_target };
            if !entry.initialized {
                entry.layer.initialize(&self.ctx)?;
                entry.initialized = true;
            }
            if entry.layer.props().visible {
                entry.layer.update(&self.ctx, &self.viewport)?;
            }
        }
        self.ctx.target = main_target;
        Ok(())
    }

    /// deck.gl's `MaskEffect`: render every visible mask layer into its texture through a
    /// viewport fitted to the layer's bounds, and publish the textures in the layer context.
    fn update_masks(&mut self) -> Result<()> {
        let mask_layers: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, e)| e.initialized && e.layer.props().visible && e.layer.props().operation.mask)
            .map(|(i, _)| i)
            .collect();
        let mut maps = MaskMaps {
            sampler: self.masks.sampler.clone(),
            dummy: self.masks.dummy.clone(),
            channels: HashMap::new(),
        };
        if mask_layers.is_empty() {
            if !self.masks.channels.is_empty() {
                self.publish_masks(maps);
            }
            return Ok(());
        }
        if !self.viewport.is_geospatial || self.viewport.projection_mode() == ProjectionMode::Globe {
            tracing::warn!("mask layers are only supported with the map view for now");
            self.publish_masks(maps);
            return Ok(());
        }
        let viewport_bounds = {
            let b = self.viewport.get_bounds(0.0);
            let bl = self.viewport.project_position(DVec3::new(b[0], b[1], 0.0));
            let tr = self.viewport.project_position(DVec3::new(b[2], b[3], 0.0));
            [bl.x.min(tr.x), bl.y.min(tr.y), bl.x.max(tr.x), bl.y.max(tr.y)]
        };
        let main_target = self.ctx.target;
        let dpr = self.ctx.device_pixel_ratio;
        self.ctx.target = MASK_TARGET;
        self.ctx.device_pixel_ratio = 1.0;
        self.ctx.uniform_slot = 0;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deck.gl masks"),
            });
        for (count, &i) in mask_layers.iter().enumerate() {
            if count >= MAX_MASKS {
                tracing::warn!("too many mask layers, the most supported is {MAX_MASKS}");
                break;
            }
            let (id, layer_bounds, coordinate_system, coordinate_origin) = {
                let layer = &self.layers[i].layer;
                let props = layer.props();
                // bounds in absolute common space; other coordinate systems fall back to the view
                let bounds = match props.coordinate_system {
                    CoordinateSystem::Default | CoordinateSystem::LngLat => layer.bounds().map(|b| {
                        let bl = self.viewport.project_position(DVec3::new(b[0], b[1], 0.0));
                        let tr = self.viewport.project_position(DVec3::new(b[2], b[3], 0.0));
                        [bl.x.min(tr.x), bl.y.min(tr.y), bl.x.max(tr.x), bl.y.max(tr.y)]
                    }),
                    _ => None,
                };
                (
                    layer.id().to_string(),
                    bounds,
                    props.coordinate_system,
                    props.coordinate_origin,
                )
            };
            let bounds = render_bounds(layer_bounds, viewport_bounds);
            let Some((mask_viewport, bounds_common)) =
                mask_viewport(bounds, &self.viewport, MASK_MAP_SIZE, MASK_BORDER)
            else {
                continue;
            };
            let device = &self.ctx.device;
            let texture = self
                .mask_textures
                .entry(id.clone())
                .or_insert_with(|| create_mask_texture(device, &id));
            let view = texture.create_view(&Default::default());
            self.ctx.layer_index = i as u32 * LAYER_INDEX_STRIDE;
            self.layers[i].layer.update(&self.ctx, &mask_viewport)?;
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("deck.gl mask"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let inner = MASK_MAP_SIZE - 2 * MASK_BORDER;
                pass.set_viewport(
                    MASK_BORDER as f32,
                    MASK_BORDER as f32,
                    inner as f32,
                    inner as f32,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(MASK_BORDER, MASK_BORDER, inner, inner);
                self.layers[i].layer.draw(&self.ctx, &mut pass)?;
            }
            maps.channels.insert(
                id,
                MaskChannel {
                    view,
                    bounds_common,
                    coordinate_system,
                    coordinate_origin,
                },
            );
        }
        self.ctx.queue.submit([encoder.finish()]);
        self.ctx.target = main_target;
        self.ctx.device_pixel_ratio = dpr;
        self.publish_masks(maps);
        Ok(())
    }

    fn publish_masks(&mut self, maps: MaskMaps) {
        self.masks = Arc::new(maps);
        self.ctx.masks = Some(self.masks.clone());
    }

    /// Encode all visible layers into a render pass whose attachments match the deck's
    /// [`RenderTarget`]. The pass viewport is expected to cover the full deck size.
    pub fn draw(&mut self, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        let dpr = self.ctx.device_pixel_ratio as f64;
        let (target_w, target_h) = self.attachment_size.unwrap_or((
            ((self.width as f64 * dpr).round() as u32).max(1),
            ((self.height as f64 * dpr).round() as u32).max(1),
        ));
        let views = self.viewports();
        let multi_view = !self.views.is_empty();
        let mut slot = 0usize;
        for (view_index, (rect, viewport)) in views.iter().enumerate() {
            let view_id = if multi_view {
                self.views[view_index].id.clone()
            } else {
                "default".to_string()
            };
            if multi_view {
                // Restrict drawing to the view's rectangle, in physical pixels
                let x = ((rect.x * dpr).round() as u32).min(target_w);
                let y = ((rect.y * dpr).round() as u32).min(target_h);
                let w = ((rect.width * dpr).round() as u32).min(target_w - x);
                let h = ((rect.height * dpr).round() as u32).min(target_h - y);
                if w == 0 || h == 0 {
                    continue;
                }
                pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, w, h);
            }
            // The first view was updated by `update`; every other view (and world copy) gets
            // its own uniform slot so the draws do not overwrite each other's uniforms
            let mut copies = vec![viewport.clone()];
            if self.repeat {
                copies.extend(
                    viewport
                        .sub_viewports()
                        .into_iter()
                        .filter(|v| v.world_offset != 0),
                );
            }
            for copy in &copies {
                let needs_update = slot != 0;
                self.ctx.uniform_slot = slot;
                for (index, entry) in self.layers.iter_mut().enumerate() {
                    self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
                    if !entry.initialized
                        || !entry.layer.props().visible
                        || entry.layer.props().operation.mask
                    {
                        continue;
                    }
                    if let Some(filter) = &self.layer_filter {
                        if !filter.allows(entry.layer.id(), &view_id) {
                            continue;
                        }
                    }
                    if needs_update {
                        entry.layer.update(&self.ctx, copy)?;
                    }
                    entry.layer.draw(&self.ctx, pass)?;
                }
                slot += 1;
            }
        }
        self.ctx.uniform_slot = 0;
        if multi_view {
            pass.set_viewport(0.0, 0.0, target_w as f32, target_h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, target_w, target_h);
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
        luma_gl::stats::reset();
        let started = std::time::Instant::now();
        self.update()?;
        let update_ms = started.elapsed().as_secs_f64() * 1000.0;
        let size = color_view.texture().size();
        self.attachment_size = Some((size.width, size.height));
        let sample_count = self.ctx.target.sample_count;
        // With effects, the layers draw into the deck's scene texture and the passes end on
        // the caller's attachment
        self.post_processor.prepare(
            &self.ctx.device,
            self.ctx.target,
            &self.ctx.pipelines,
            &self.post_process,
        )?;
        let post = !self.post_process.is_empty() && self.post_processor.has_passes();
        let scene_view = post.then(|| {
            self.post_processor
                .scene_texture(&self.ctx.device, size, self.ctx.target.color_format)
                .create_view(&Default::default())
        });
        let (layer_view, layer_load) = match &scene_view {
            Some(view) => (view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            None => (color_view, color_load),
        };
        let (msaa_color_view, msaa_depth_view) = if sample_count > 1 {
            if matches!(layer_load, wgpu::LoadOp::Load) {
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
                resolve_target: Some(layer_view),
                ops: wgpu::Operations {
                    load: layer_load,
                    store: wgpu::StoreOp::Discard,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view: layer_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: layer_load,
                    store: wgpu::StoreOp::Store,
                },
            },
        };
        // The multisampled depth buffer is the deck's own: with effects the layers start on
        // a cleared scene, so it is cleared too rather than loaded from a host that never
        // wrote it
        let depth_load = if post && msaa_color_view.is_some() {
            wgpu::LoadOp::Clear(1.0)
        } else {
            depth_load
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
        let draw_started = std::time::Instant::now();
        let mut result = self.draw(&mut pass);
        drop(pass);
        if post && result.is_ok() {
            result = self.post_processor.render(
                encoder,
                &self.ctx.queue,
                &self.post_process,
                color_view,
                color_load,
            );
        }
        let counters = luma_gl::stats::snapshot();
        self.stats = FrameStats {
            frame: self.stats.frame + 1,
            layers: self
                .layers
                .iter()
                .filter(|e| e.initialized && e.layer.props().visible)
                .count(),
            draw_calls: counters.draw_calls,
            instances: counters.instances,
            uploaded_bytes: counters.uploaded_bytes,
            update_ms,
            draw_ms: draw_started.elapsed().as_secs_f64() * 1000.0,
        };
        tracing::trace!(
            frame = self.stats.frame,
            layers = self.stats.layers,
            draw_calls = self.stats.draw_calls,
            instances = self.stats.instances,
            uploaded_bytes = self.stats.uploaded_bytes,
            "frame"
        );
        result
    }

    /// Counters of the last frame rendered with [`Deck::render`] or [`Deck::render_with`].
    pub fn stats(&self) -> FrameStats {
        self.stats
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
        // Created above when missing or stale; the borrow rules keep this two step
        match self.msaa.as_ref() {
            Some(msaa) => msaa,
            None => unreachable_after_creation(),
        }
    }
}

/// What the last frame cost, deck.gl's stats: filled by [`Deck::render_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    /// Frames rendered so far
    pub frame: u64,
    /// Layers that were initialized and visible (including sub layers of composites is not
    /// counted; those show up in draw calls)
    pub layers: usize,
    pub draw_calls: u64,
    /// Instances drawn (vertices for non instanced models)
    pub instances: u64,
    /// Bytes uploaded to vertex and index buffers during the frame
    pub uploaded_bytes: u64,
    /// CPU time of the layer updates in milliseconds
    pub update_ms: f64,
    /// CPU time of encoding the draws in milliseconds
    pub draw_ms: f64,
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

/// Whether a view kind and a camera state belong together.
fn same_kind(view: &View, state: &AnyViewState) -> bool {
    matches!(
        (view, state),
        (View::Map, AnyViewState::Map(_))
            | (View::Globe(_), AnyViewState::Globe(_))
            | (View::Orthographic(_), AnyViewState::Orthographic(_))
            | (View::Orbit(_), AnyViewState::Orbit(_))
            | (View::FirstPerson(_), AnyViewState::FirstPerson(_))
    )
}

/// Whether an initialized layer can keep its models when it takes over these props. Picking
/// pipelines and render parameters are baked into the pipelines, so a change there means the
/// layer is initialized again.
fn same_pipelines(current: &LayerProps, incoming: &LayerProps) -> bool {
    current.pickable == incoming.pickable && !current.parameters.differs(&incoming.parameters)
}

/// `msaa_textures` fills the slot just before reading it; this only documents the invariant.
#[cold]
fn unreachable_after_creation() -> &'static MsaaTextures {
    // Cannot happen: the textures were stored a few lines earlier. Returning an error would
    // need a signature change on every caller for a case that cannot occur.
    #[allow(clippy::panic)]
    {
        panic!("msaa textures were not created")
    }
}

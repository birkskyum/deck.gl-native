//! Port of `@deck.gl/core/src/lib/deck.ts`, reduced to what a native host needs.

use luma_gl::{RenderTarget, PICKING_FORMAT};

use crate::constants::ClipDepthRange;
use crate::layer::{decode_picking_color, Layer, LayerContext, LayerProps, LAYER_INDEX_STRIDE};
use crate::lighting::LightingEffect;
use crate::viewport::{Viewport, WebMercatorViewportOptions};
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
    view_state: ViewState,
    viewport: Viewport,
    external_viewport: bool,
    picking: Option<PickingTarget>,
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
        };
        let viewport = make_viewport(props.width, props.height, &props.view_state);
        let mut deck = Self {
            ctx,
            layers: Vec::new(),
            msaa: None,
            width: props.width,
            height: props.height,
            view_state: props.view_state,
            viewport,
            external_viewport: false,
            picking: None,
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
            self.viewport = make_viewport(width, height, &self.view_state);
        }
    }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.ctx.device_pixel_ratio = ratio;
    }

    /// Move the default map camera.
    pub fn set_view_state(&mut self, view_state: ViewState) {
        self.view_state = view_state;
        self.external_viewport = false;
        self.viewport = make_viewport(self.width, self.height, &view_state);
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

    /// Replace the layer list. Layers whose id matches an existing layer of the same type keep
    /// that layer's GPU resources and only take over the new props (see
    /// [`Layer::update_from`]); everything else is initialized on the next update.
    pub fn set_layers(&mut self, layers: Vec<Box<dyn Layer>>) {
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
                        if entry.layer.update_from(layer.as_mut()) {
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
        for (index, entry) in self.layers.iter_mut().enumerate() {
            self.ctx.layer_index = index as u32 * LAYER_INDEX_STRIDE;
            if entry.initialized && entry.layer.props().visible {
                entry.layer.draw(&self.ctx, pass)?;
            }
        }
        Ok(())
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

fn make_viewport(width: u32, height: u32, view_state: &ViewState) -> Viewport {
    Viewport::web_mercator(&WebMercatorViewportOptions {
        width: width as f64,
        height: height as f64,
        longitude: view_state.longitude,
        latitude: view_state.latitude,
        zoom: view_state.zoom,
        pitch: view_state.pitch,
        bearing: view_state.bearing,
        ..Default::default()
    })
}

/// Whether an initialized layer can keep its models when it takes over these props. Picking
/// pipelines and render parameters are baked into the pipelines, so a change there means the
/// layer is initialized again.
fn same_pipelines(current: &LayerProps, incoming: &LayerProps) -> bool {
    current.pickable == incoming.pickable && !current.parameters.differs(&incoming.parameters)
}

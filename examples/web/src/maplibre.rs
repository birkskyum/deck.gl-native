//! A maplibre-gl-js custom layer.
//!
//! maplibre-gl-js hands a custom layer its live WebGL2 context with its own framebuffer and
//! depth buffer bound. wgpu can be told to use that context rather than making one of its
//! own (`wgpu_hal::gles::Adapter::new_external`), and to draw into that framebuffer rather
//! than a texture of its own (`TextureInner::DefaultRenderbuffer`). Both sides then write
//! colour and depth into the same attachments in the same frame, so an arc is hidden where
//! it passes behind a building and visible where it clears one.
//!
//! The camera is not taken from maplibre's matrix but rebuilt from its centre, zoom, pitch
//! and bearing by deck.gl's own projection, the same way the maplibre-native host works.

use deck_gl::luma_gl::RenderTarget;
use deck_gl::viewport::{Viewport, WebMercatorViewportOptions};
use deck_gl::{Deck, DeckProps};
use deck_gl_json::JsonConverter;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::WebGl2RenderingContext;

use crate::err;

/// deck.gl-native drawing inside a maplibre-gl-js map, in the map's own WebGL2 context.
#[wasm_bindgen]
pub struct DeckGlOverlay {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gl: WebGl2RenderingContext,
    deck: Deck,
    /// Never read: the layers draw into the framebuffer maplibre has bound, which brings its
    /// own depth buffer with it. wgpu still wants a depth attachment for the pass, since the
    /// pipelines depth test.
    depth: wgpu::Texture,
    size: (u32, u32),
    spec: Option<serde_json::Value>,
    converted_at: u64,
    warnings: Vec<String>,
}

/// Take over a maplibre-gl-js map's WebGL2 context and start a deck in it. Call this from the
/// custom layer's `onAdd(map, gl)`.
#[wasm_bindgen]
pub async fn create_overlay(gl: WebGl2RenderingContext) -> Result<DeckGlOverlay, JsValue> {
    console_error_panic_hook::set_once();

    let exposed =
        unsafe { wgpu::hal::gles::Adapter::new_external(gl.clone(), wgpu::wgt::GlBackendOptions::default()) }
            .ok_or_else(|| err("wgpu could not adopt the map's WebGL2 context"))?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // Safety: the adapter was made from this instance's backend, and the context outlives the
    // overlay because the map holds both.
    let adapter = unsafe { instance.create_adapter_from_hal(exposed) };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("deck.gl-native in maplibre-gl-js"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(err)?;

    let size = drawing_buffer_size(&gl);
    let target = RenderTarget {
        // What a canvas' default framebuffer holds
        color_format: wgpu::TextureFormat::Rgba8Unorm,
        depth_format: Some(wgpu::TextureFormat::Depth24Plus),
        sample_count: 1,
    };
    let ratio = crate::device_pixel_ratio();
    let deck = Deck::new(
        &device,
        &queue,
        target,
        DeckProps {
            width: (size.0 as f32 / ratio).round().max(1.0) as u32,
            height: (size.1 as f32 / ratio).round().max(1.0) as u32,
            device_pixel_ratio: ratio,
            // WebGL2 reads a zero vertex stride as "tightly packed", see `DeckProps`
            constant_attributes: false,
            // A canvas' default framebuffer starts at the bottom left, so the layers go in
            // the other way up from how wgpu would draw them into a texture of its own
            clip_origin: deck_gl::ClipOrigin::BottomLeft,
            ..Default::default()
        },
    )
    .map_err(err)?;
    let depth = crate::depth_texture(&device, &target, size.0, size.1);

    Ok(DeckGlOverlay {
        device,
        queue,
        gl,
        deck,
        depth,
        size,
        spec: None,
        converted_at: 0,
        warnings: Vec::new(),
    })
}

#[wasm_bindgen]
impl DeckGlOverlay {
    /// The layers to draw, as a deck.gl JSON description. Its `initialViewState` is ignored:
    /// the map drives the camera.
    pub fn set_spec(&mut self, json: &str) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(err)?;
        self.spec = Some(value);
        self.convert()
    }

    #[wasm_bindgen(getter)]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Draw the layers into whatever maplibre has bound, with the camera the map is at. Call
    /// this from the custom layer's `render(gl, args)`.
    /// `near_z` and `far_z` are the map's own depth planes, from the custom layer's render
    /// parameters. Sharing a depth buffer means agreeing on what a depth value stands for,
    /// and deck and maplibre place their planes differently, so an arc well above a tower can
    /// still lose to it at some angles.
    ///
    /// Passing them through as they are makes the layers disappear, so they are not used yet
    /// and deck keeps its own planes: the remaining disagreement is a known gap rather than a
    /// solved one. `@deck.gl/maplibre` divides them by the viewport height, which is the same
    /// unit deck's planes are in, so the difference is somewhere else.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        longitude: f64,
        latitude: f64,
        zoom: f64,
        pitch: f64,
        bearing: f64,
        near_z: f64,
        far_z: f64,
    ) -> Result<(), JsValue> {
        let generation = deck_gl_layers::fetch::Fetcher::global().generation();
        if generation != self.converted_at && self.spec.is_some() {
            self.convert()?;
        }

        let size = drawing_buffer_size(&self.gl);
        if size != self.size {
            self.size = size;
            let ratio = crate::device_pixel_ratio();
            self.deck.set_device_pixel_ratio(ratio);
            self.deck.set_size(
                (size.0 as f32 / ratio).round().max(1.0) as u32,
                (size.1 as f32 / ratio).round().max(1.0) as u32,
            );
            self.depth = crate::depth_texture(&self.device, &self.deck.context().target, size.0, size.1);
        }
        let _ = (near_z, far_z);
        let ratio = crate::device_pixel_ratio() as f64;
        let width = (self.size.0 as f64 / ratio).max(1.0);
        let height = (self.size.1 as f64 / ratio).max(1.0);
        // maplibre measures its planes in CSS pixels, deck in viewport heights: its camera
        // sits at an altitude of 1.5, not at `cameraToCenterDistance` pixels.
        self.deck
            .set_viewport(Viewport::web_mercator(&WebMercatorViewportOptions {
                width,
                height,
                longitude,
                latitude,
                zoom,
                pitch,
                bearing,
                near_z: None,
                far_z: None,
                ..Default::default()
            }));

        let color = self.map_framebuffer()?;
        let color_view = color.create_view(&Default::default());
        let depth_view = self.depth.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deck.gl in maplibre"),
            });
        // Load, never clear: the map has already drawn its basemap and its buildings into
        // these attachments, and the layers go in among them.
        self.deck
            .render(&mut encoder, &color_view, Some(&depth_view), None)
            .map_err(err)?;
        self.queue.submit([encoder.finish()]);
        self.restore_pixel_store();
        Ok(())
    }

    /// Put the pixel store parameters back to what a GL context starts with.
    ///
    /// wgpu sets these while it uploads a texture and leaves them set. maplibre never touches
    /// them, so it never sets them back either, and every texture it uploads afterwards, the
    /// glyph and icon atlases a label needs, fails with `invalid unpack params combination`
    /// and draws as a solid block. Only the symbols already uploaded when the layer first
    /// drew survive, which is what makes it look like panning breaks them.
    fn restore_pixel_store(&self) {
        let gl = &self.gl;
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_ROW_LENGTH, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_SKIP_ROWS, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_SKIP_PIXELS, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_IMAGE_HEIGHT, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_SKIP_IMAGES, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_ALIGNMENT, 4);
    }

    /// The framebuffer maplibre has bound, as a texture the layers can be drawn into.
    ///
    /// A canvas' default framebuffer is the interesting case: wgpu binds it and uses the
    /// depth buffer that comes with it, which is what makes the layers interleave with the
    /// map's own geometry. A map rendering into a framebuffer of its own (with terrain on,
    /// for instance) would need wgpu to attach depth to it, which would take that depth
    /// buffer away from the map, so this says so rather than doing it.
    fn map_framebuffer(&self) -> Result<wgpu::Texture, JsValue> {
        let bound = self
            .gl
            .get_parameter(WebGl2RenderingContext::FRAMEBUFFER_BINDING)
            .ok()
            .filter(|value| !value.is_null());
        if let Some(bound) = bound {
            let _ = bound.dyn_into::<web_sys::WebGlFramebuffer>();
            return Err(err(
                "the map is drawing into a framebuffer of its own, which this layer cannot \
                 share a depth buffer with; turn off terrain or globe",
            ));
        }
        let hal_texture = wgpu::hal::gles::Texture::default_framebuffer(wgpu::TextureFormat::Rgba8Unorm);
        let descriptor = wgpu::TextureDescriptor {
            label: Some("maplibre framebuffer"),
            size: wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        // Safety: the texture stands for the context's own default framebuffer, which is
        // valid for as long as the map is.
        Ok(unsafe {
            self.device.create_texture_from_hal::<wgpu::hal::api::Gles>(
                hal_texture,
                &descriptor,
                // The map has already drawn into it this frame
                wgpu::wgt::TextureUses::COLOR_TARGET,
            )
        })
    }

    fn convert(&mut self) -> Result<(), JsValue> {
        let Some(value) = &self.spec else {
            return Ok(());
        };
        self.converted_at = deck_gl_layers::fetch::Fetcher::global().generation();
        let converted = JsonConverter::new().convert(value).map_err(err)?;
        self.warnings = converted.warnings;
        if let Some(lighting) = converted.lighting {
            self.deck.set_lighting(lighting);
        }
        self.deck.set_layers(converted.layers);
        Ok(())
    }
}

fn drawing_buffer_size(gl: &WebGl2RenderingContext) -> (u32, u32) {
    (
        gl.drawing_buffer_width().max(1) as u32,
        gl.drawing_buffer_height().max(1) as u32,
    )
}

//! deck.gl-native in a browser.
//!
//! The same layers, shaders and projection math as the native build, compiled to wasm32 and
//! drawing through wgpu. On WebGPU it owns a canvas; with the `webgl` feature it draws
//! through WebGL2 instead, which is what the maplibre-gl-js custom layer needs, and naga
//! lowers the WGSL to GLSL ES 3.00 on the way.
//!
//! The API is the JSON one: hand it a deck.gl JSON description and it draws it.

use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps, MapController, ViewState};
use deck_gl_json::JsonConverter;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// Whether this build draws through WebGL2 rather than WebGPU.
const WEBGL: bool = cfg!(feature = "webgl");

fn err(message: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&message.to_string())
}

/// A deck drawing into a canvas of its own.
#[wasm_bindgen]
pub struct DeckGl {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::Texture,
    deck: Deck,
    /// deck.gl's `MapController`: the same drag, wheel and inertia behaviour as the native
    /// window, rather than a second implementation in JavaScript.
    controller: MapController,
    /// The description last given to [`DeckGl::set_spec`], reconverted when data arrives.
    spec: Option<serde_json::Value>,
    /// The fetcher generation the description was last converted at, so that layers whose
    /// data was still loading are built once it lands.
    converted_at: u64,
    warnings: Vec<String>,
}

/// Start a deck on `canvas`. Asynchronous because asking the browser for a GPU device is.
#[wasm_bindgen]
pub async fn create(canvas: HtmlCanvasElement) -> Result<DeckGl, JsValue> {
    console_error_panic_hook::set_once();

    // The canvas' drawing buffer is in physical pixels; a deck's size is in CSS pixels.
    let ratio = device_pixel_ratio();
    let width = canvas.width().max(1);
    let height = canvas.height().max(1);
    let logical_width = (width as f32 / ratio).round().max(1.0) as u32;
    let logical_height = (height as f32 / ratio).round().max(1.0) as u32;
    let backends = if WEBGL {
        wgpu::Backends::GL
    } else {
        wgpu::Backends::BROWSER_WEBGPU
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(err)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| err("no GPU adapter: this browser has neither WebGPU nor WebGL2"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("deck.gl-native"),
            // The browser's own limits, so a WebGL2 device is not asked for more than it has
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(err)?;

    let capabilities = surface.get_capabilities(&adapter);
    let format = capabilities
        .formats
        .iter()
        .copied()
        .find(|format| !format.is_srgb())
        .unwrap_or(capabilities.formats[0]);
    let mut config = surface
        .get_default_config(&adapter, width, height)
        .ok_or_else(|| err("the canvas cannot be drawn into by this adapter"))?;
    config.format = format;
    surface.configure(&device, &config);

    let target = RenderTarget {
        color_format: format,
        depth_format: Some(wgpu::TextureFormat::Depth24Plus),
        sample_count: 1,
    };
    let depth = depth_texture(&device, &target, width, height);
    let deck = Deck::new(
        &device,
        &queue,
        target,
        DeckProps {
            width: logical_width,
            height: logical_height,
            device_pixel_ratio: ratio,
            // A zero vertex stride means "tightly packed" in OpenGL rather than "one element
            // every instance reads", so constant attributes cannot be shared on WebGL2.
            constant_attributes: !WEBGL,
            ..Default::default()
        },
    )
    .map_err(err)?;

    let controller = MapController::new(
        ViewState::default(),
        logical_width as f64,
        logical_height as f64,
    );
    Ok(DeckGl {
        device,
        queue,
        surface,
        config,
        depth,
        deck,
        controller,
        spec: None,
        converted_at: 0,
        warnings: Vec::new(),
    })
}

#[wasm_bindgen]
impl DeckGl {
    /// Draw the layers of a deck.gl JSON description, and look at it from its
    /// `initialViewState`.
    pub fn set_spec(&mut self, json: &str) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(err)?;
        self.spec = Some(value);
        self.convert(true)
    }

    /// The layer types and props the description asked for that this build does not have,
    /// deck.gl's console warnings.
    #[wasm_bindgen(getter)]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Move the camera. Longitude and latitude in degrees, pitch and bearing in degrees.
    pub fn set_view_state(&mut self, longitude: f64, latitude: f64, zoom: f64, pitch: f64, bearing: f64) {
        self.controller.set_view_state(ViewState {
            longitude,
            latitude,
            zoom,
            pitch,
            bearing,
        });
    }

    /// The camera, as `[longitude, latitude, zoom, pitch, bearing]`.
    #[wasm_bindgen(getter)]
    pub fn view_state(&self) -> Vec<f64> {
        let view = self.controller.view_state();
        vec![view.longitude, view.latitude, view.zoom, view.pitch, view.bearing]
    }

    /// Drag to pan. Pixels are logical, from the top left of the canvas.
    pub fn pan_start(&mut self, x: f64, y: f64, now: f64) {
        self.controller.pan_start([x, y], now);
    }

    pub fn pan(&mut self, x: f64, y: f64, now: f64) {
        self.controller.pan([x, y], now);
    }

    pub fn pan_end(&mut self, now: f64) {
        self.controller.pan_end(now);
    }

    /// Drag with the right button or a modifier held to turn and tilt.
    pub fn rotate_start(&mut self, x: f64, y: f64) {
        self.controller.rotate_start([x, y]);
    }

    pub fn rotate(&mut self, x: f64, y: f64) {
        self.controller.rotate([x, y]);
    }

    pub fn rotate_end(&mut self) {
        self.controller.rotate_end();
    }

    /// Wheel to zoom around the pointer. `delta` is in zoom levels.
    pub fn zoom_by(&mut self, x: f64, y: f64, delta: f64) {
        self.controller.zoom_by([x, y], delta);
    }

    /// Advance the camera's inertia and transitions. Returns whether it is still moving.
    pub fn tick(&mut self, now: f64) -> bool {
        self.controller.tick(now)
    }

    /// Resize the canvas' drawing buffer, in physical pixels.
    pub fn set_size(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = depth_texture(&self.device, &self.deck.context().target, width, height);
        let ratio = device_pixel_ratio();
        self.deck.set_device_pixel_ratio(ratio);
        let (logical_width, logical_height) = (
            (width as f32 / ratio).round().max(1.0) as u32,
            (height as f32 / ratio).round().max(1.0) as u32,
        );
        self.deck.set_size(logical_width, logical_height);
        self.controller
            .set_size(logical_width as f64, logical_height as f64);
    }

    /// Draw a frame. Returns whether anything is still loading, so the caller knows to keep
    /// asking for frames.
    pub fn render(&mut self) -> Result<bool, JsValue> {
        // Layers whose data or tiles were still loading are rebuilt once something arrived
        let generation = deck_gl_layers::fetch::Fetcher::global().generation();
        if generation != self.converted_at && self.spec.is_some() {
            self.convert(false)?;
        }
        self.deck.set_view_state(self.controller.view_state());
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(true);
            }
            other => return Err(err(format!("no frame to draw into: {other:?}"))),
        };
        let color = frame.texture.create_view(&Default::default());
        let depth = self.depth.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("deck.gl") });
        self.deck
            .render(
                &mut encoder,
                &color,
                Some(&depth),
                Some(wgpu::Color::TRANSPARENT),
            )
            .map_err(err)?;
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        Ok(!deck_gl_layers::fetch::Fetcher::global().is_idle())
    }

    /// Which backend the frames go through, `"webgpu"` or `"webgl2"`.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        if WEBGL { "webgl2".into() } else { "webgpu".into() }
    }

    /// Build the layers from the description. `camera` places the camera at the
    /// `initialViewState`, which only the first conversion of a description does: the later
    /// ones, when data arrives, must not throw away where the reader has moved to.
    fn convert(&mut self, camera: bool) -> Result<(), JsValue> {
        let Some(value) = &self.spec else {
            return Ok(());
        };
        self.converted_at = deck_gl_layers::fetch::Fetcher::global().generation();
        let converted = JsonConverter::new().convert(value).map_err(err)?;
        self.warnings = converted.warnings;
        if camera {
            if let Some(view_state) = converted.view_state {
                self.controller.set_view_state(view_state);
            }
        }
        if let Some(lighting) = converted.lighting {
            self.deck.set_lighting(lighting);
        }
        self.deck.set_post_process(converted.post_process);
        self.deck.set_layers(converted.layers);
        Ok(())
    }
}

fn depth_texture(
    device: &wgpu::Device,
    target: &RenderTarget,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deck.gl depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: target.sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: target.depth_format.unwrap_or(wgpu::TextureFormat::Depth24Plus),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn device_pixel_ratio() -> f32 {
    web_sys::window()
        .map(|window| window.device_pixel_ratio() as f32)
        .unwrap_or(1.0)
        .max(1.0)
}

//! deck.gl-native on a maplibre-native map, all in Rust.
//!
//! wgpu owns the window and a color texture. Each frame maplibre-native renders the map into
//! that texture through maplibre-native-ffi's caller-owned Metal texture target, deck draws its
//! layers into the same texture, and wgpu copies the result to the swapchain.
//!
//! The runtime and the map live on their own thread, as in maplibre-native-ffi's own
//! `rust-map` example: pumping the runtime on the winit thread would spin the Cocoa run loop
//! from inside a winit callback. The render session is attached and driven on the winit thread.
//!
//! Run with `cargo run --release --manifest-path examples/maplibre-ffi/Cargo.toml`.
//! Set `DECKGL_SCREENSHOT=frame.png` to save a frame after ten seconds.

#![cfg(target_os = "macos")]

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use deck_gl::luma_gl::device::{create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_examples::scene;
use deckgl::{maplibre_near_far_pixels, viewport_from_camera, DeckglCamera};
use maplibre_native_ffi as mln;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;
use wgpu::hal::api::Metal;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const STYLE_URL: &str = "https://tiles.openfreemap.org/styles/liberty";
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const DEFAULT_FOV_DEGREES: f64 = 36.86989764584402;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewportSize {
    logical_width: u32,
    logical_height: u32,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
}

impl ViewportSize {
    fn of(window: &Window) -> Self {
        let physical = window.inner_size();
        let scale = window.scale_factor().max(0.01);
        Self {
            logical_width: ((physical.width as f64 / scale).ceil() as u32).max(1),
            logical_height: ((physical.height as f64 / scale).ceil() as u32).max(1),
            physical_width: physical.width.max(1),
            physical_height: physical.height.max(1),
            scale_factor: scale,
        }
    }

    fn extent(&self) -> mln::RenderTargetExtent {
        mln::RenderTargetExtent::new(self.logical_width, self.logical_height, self.scale_factor)
    }
}

/// State shared between the map thread and the render thread.
#[derive(Default)]
struct Shared {
    camera: Mutex<Option<mln::CameraOptions>>,
    shutdown: AtomicBool,
    failure: Mutex<Option<String>>,
}

/// What the map thread hands the render thread once the map exists.
struct MapHandles {
    attach_ref: mln::MapAttachRef,
}

/// Runs the runtime and the map until shutdown. Owns both for their whole lifetime.
fn map_thread(size: ViewportSize, handles: mpsc::Sender<MapHandles>, shared: Arc<Shared>) {
    let result = (|| -> Result<(), Box<dyn Error>> {
        let mut runtime_options = mln::RuntimeOptions::default();
        runtime_options.cache_path = Some(
            std::env::temp_dir()
                .join("deckgl-maplibre-cache.db")
                .display()
                .to_string(),
        );
        let mut runtime = mln::RuntimeHandle::with_options(&runtime_options)?;
        let mut map_options =
            mln::MapOptions::new(size.logical_width, size.logical_height, size.scale_factor);
        map_options.mode = mln::MapMode::Continuous;
        let map = mln::MapHandle::with_options(&runtime, &map_options)?;
        map.set_event_mask(
            mln::RuntimeEventMask::MAP_RENDER_UPDATE_AVAILABLE
                | mln::RuntimeEventMask::MAP_RENDER_FRAME_FINISHED,
        )?;
        map.set_style_url(STYLE_URL)?;
        let view = scene::view_state(-25.0);
        let mut camera = mln::CameraOptions::default();
        camera.center = Some(mln::LatLng::new(view.latitude, view.longitude));
        camera.zoom = Some(view.zoom);
        camera.pitch = Some(view.pitch);
        camera.bearing = Some(view.bearing);
        map.jump_to(&camera)?;
        map.request_repaint()?;

        if handles
            .send(MapHandles {
                attach_ref: map.attach_ref()?,
            })
            .is_err()
        {
            return Ok(());
        }

        let start = Instant::now();
        while !shared.shutdown.load(Ordering::Relaxed) {
            // Slow orbit so the demo moves on its own
            let mut orbit = mln::CameraOptions::default();
            orbit.bearing = Some(view.bearing + start.elapsed().as_secs_f64() * 6.0);
            map.jump_to(&orbit)?;

            runtime.pump(Some(Duration::from_millis(4)), None)?;
            let _ = runtime.drain_events(0)?;
            *shared.camera.lock().unwrap() = Some(map.camera()?);
        }
        // The session is closed by the render thread before shutdown is requested.
        map.close().map_err(|e| e.to_string())?;
        runtime.close().map_err(|e| e.to_string())?;
        Ok(())
    })();
    if let Err(e) = result {
        *shared.failure.lock().unwrap() = Some(e.to_string());
    }
}

/// The wgpu texture maplibre renders into, plus deck's depth buffer.
struct Attachments {
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

impl Attachments {
    fn new(device: &wgpu::Device, size: ViewportSize) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("map + deck color"),
            size: wgpu::Extent3d {
                width: size.physical_width,
                height: size.physical_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = create_render_texture(
            device,
            "deck depth",
            size.physical_width,
            size.physical_height,
            DEPTH_FORMAT,
        );
        Self { color, depth }
    }

    /// The `id<MTLTexture>` behind the color texture, for maplibre to borrow.
    fn borrowed_descriptor(
        &self,
        size: ViewportSize,
    ) -> Result<mln::MetalBorrowedTextureDescriptor, Box<dyn Error>> {
        // SAFETY: the texture was created by the Metal backend.
        let hal = unsafe { self.color.as_hal::<Metal>() }.ok_or("color texture is not a Metal texture")?;
        let raw: &ProtocolObject<dyn MTLTexture> = hal.raw_handle();
        // SAFETY: the texture stays alive as long as `Attachments` does, and maplibre is told
        // about a replacement before the old one is dropped.
        let pointer = unsafe { mln::NativePointer::from_address(raw as *const _ as usize) };
        Ok(mln::MetalBorrowedTextureDescriptor::new(
            size.extent(),
            size.physical_width,
            size.physical_height,
            pointer,
        ))
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: ViewportSize,
    attachments: Attachments,
    shared: Arc<Shared>,
    map_thread: Option<JoinHandle<()>>,
    session: Option<mln::RenderSessionHandle>,
    deck: Deck,
    start: Instant,
    screenshot_done: bool,
}

impl State {
    fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let size = ViewportSize::of(&window);

        // wgpu: window surface and device
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("deck.gl-native maplibre demo"),
            ..Default::default()
        }))?;
        let mut config = surface
            .get_default_config(&adapter, size.physical_width, size.physical_height)
            .ok_or("surface is not supported by the adapter")?;
        config.format = COLOR_FORMAT;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let attachments = Attachments::new(&device, size);

        // maplibre-native on its own thread; we get a reference to attach a session to
        let shared = Arc::new(Shared::default());
        let (handles_tx, handles_rx) = mpsc::channel();
        let map_thread = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("maplibre".into())
                .spawn(move || map_thread(size, handles_tx, shared))?
        };
        let handles = match handles_rx.recv() {
            Ok(handles) => handles,
            Err(_) => {
                let failure = shared.failure.lock().unwrap().clone();
                return Err(failure.unwrap_or_else(|| "map thread exited".to_string()).into());
            }
        };
        let session = handles
            .attach_ref
            .attach_metal_borrowed_texture(&attachments.borrowed_descriptor(size)?)?;

        // deck.gl-native on the same wgpu device
        let deck = Deck::new(
            &device,
            &queue,
            RenderTarget {
                color_format: COLOR_FORMAT,
                depth_format: Some(DEPTH_FORMAT),
                sample_count: 1,
            },
            DeckProps {
                width: size.logical_width,
                height: size.logical_height,
                device_pixel_ratio: size.scale_factor as f32,
                view_state: scene::view_state(-25.0),
                layers: scene::layers(),
                ..Default::default()
            },
        )?;

        println!(
            "window {}x{} logical, {}x{} physical, scale {:.2}",
            size.logical_width,
            size.logical_height,
            size.physical_width,
            size.physical_height,
            size.scale_factor
        );

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            attachments,
            shared,
            map_thread: Some(map_thread),
            session: Some(session),
            deck,
            start: Instant::now(),
            screenshot_done: false,
        })
    }

    fn resize(&mut self) -> Result<(), Box<dyn Error>> {
        let size = ViewportSize::of(&self.window);
        if size == self.size {
            return Ok(());
        }
        self.size = size;
        self.config.width = size.physical_width;
        self.config.height = size.physical_height;
        self.surface.configure(&self.device, &self.config);
        // Make sure nothing still reads the old texture before maplibre lets go of it.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let replacement = Attachments::new(&self.device, size);
        if let Some(session) = &self.session {
            session.set_metal_borrowed_texture_target(&replacement.borrowed_descriptor(size)?)?;
        }
        self.attachments = replacement;
        self.deck.set_device_pixel_ratio(size.scale_factor as f32);
        self.deck.set_size(size.logical_width, size.logical_height);
        Ok(())
    }

    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(failure) = self.shared.failure.lock().unwrap().clone() {
            return Err(format!("map thread failed: {failure}").into());
        }
        let Some(session) = &self.session else {
            return Ok(());
        };

        // 1. maplibre renders the map into our texture and waits for the GPU
        let _update = session.render_update()?;

        // 2. deck draws its layers into the same texture, with the map's camera
        let Some(map_camera) = self.shared.camera.lock().unwrap().clone() else {
            return Ok(());
        };
        let fov = map_camera.field_of_view.unwrap_or(DEFAULT_FOV_DEGREES);
        let pitch = map_camera.pitch.unwrap_or(0.0);
        // deck's depth buffer is its own and cleared each frame, so deck keeps its default near
        // and far planes, which have far better depth precision than maplibre's 1 pixel near
        // plane. Matching maplibre's planes only matters when sharing the map's depth buffer.
        let (near, far) = if std::env::var("DECKGL_MATCH_MAP_PLANES").is_ok() {
            maplibre_near_far_pixels(fov, pitch, self.size.logical_height as f64)
        } else {
            (0.0, 0.0)
        };
        let center = map_camera.center.unwrap_or(mln::LatLng::new(0.0, 0.0));
        let deck_camera = DeckglCamera {
            longitude: center.longitude,
            latitude: center.latitude,
            zoom: map_camera.zoom.unwrap_or(0.0),
            bearing: map_camera.bearing.unwrap_or(0.0),
            pitch,
            fov_degrees: fov,
            near_z_pixels: near,
            far_z_pixels: far,
            width: self.size.logical_width,
            height: self.size.logical_height,
            pixel_ratio: self.size.scale_factor as f32,
        };
        self.deck.set_viewport(viewport_from_camera(&deck_camera));

        let color_view = self.attachments.color.create_view(&Default::default());
        let depth_view = self.attachments.depth.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        self.deck.render_with(
            &mut encoder,
            &color_view,
            Some(&depth_view),
            wgpu::LoadOp::Load,
            wgpu::LoadOp::Clear(1.0),
        )?;

        // 3. copy the composited texture to the swapchain and present
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Validation => return Err("surface validation error".into()),
        };
        encoder.copy_texture_to_texture(
            self.attachments.color.as_image_copy(),
            frame.texture.as_image_copy(),
            self.attachments.color.size(),
        );
        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
        // maplibre writes the texture on its own queue next frame, so let our copy finish first.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        self.maybe_screenshot()?;
        Ok(())
    }

    fn maybe_screenshot(&mut self) -> Result<(), Box<dyn Error>> {
        if self.screenshot_done {
            return Ok(());
        }
        let Ok(path) = std::env::var("DECKGL_SCREENSHOT") else {
            return Ok(());
        };
        let after_ms: u64 = std::env::var("DECKGL_SCREENSHOT_AFTER_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10000);
        if self.start.elapsed() < Duration::from_millis(after_ms) {
            return Ok(());
        }
        self.screenshot_done = true;
        let mut pixels = read_texture_rgba8(&self.device, &self.queue, &self.attachments.color)?;
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let size = self.attachments.color.size();
        image::save_buffer(&path, &pixels, size.width, size.height, image::ColorType::Rgba8)?;
        println!("wrote {path}");
        Ok(())
    }

    /// Close the session, then let the map thread close the map and runtime.
    fn shutdown(&mut self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(session) = self.session.take() {
            if let Err(e) = session.close() {
                eprintln!("session close failed: {e}");
            }
        }
        self.shared.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.map_thread.take() {
            let _ = thread.join();
        }
        if let Some(failure) = self.shared.failure.lock().unwrap().clone() {
            eprintln!("map thread: {failure}");
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
    failed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() || self.failed {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("deck.gl-native + maplibre-native-ffi")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                eprintln!("window creation failed: {e}");
                self.failed = true;
                event_loop.exit();
                return;
            }
        };
        match State::new(window) {
            Ok(state) => self.state = Some(state),
            Err(e) => {
                eprintln!("startup failed: {e}");
                self.failed = true;
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        let result = match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                Ok(())
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => state.resize(),
            WindowEvent::RedrawRequested => {
                let result = state.render();
                state.window.request_redraw();
                result
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            eprintln!("error: {e}");
            self.failed = true;
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.shutdown();
        }
        self.state = None;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    mln::set_log_callback(|record| {
        if !matches!(record.severity, mln::LogSeverity::Info) {
            eprintln!("maplibre {:?}: {}", record.severity, record.message);
        }
        true
    })?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    if app.failed {
        return Err("demo failed".into());
    }
    Ok(())
}

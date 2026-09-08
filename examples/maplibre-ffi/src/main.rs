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
use deck_gl::{Deck, DeckProps, ViewState};
use deck_gl_examples::spec;
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
    /// Held while the map's camera must not move. The map thread takes it to apply commands
    /// and publish where the camera ended up; the render thread takes it to render the map and
    /// read that same value. Without it the two are sampling independently, the map thread
    /// republishing every few milliseconds and the render thread reading once a frame, so the
    /// camera deck draws with is never quite the one maplibre just drew with and the layers
    /// slide against the basemap whenever it moves.
    frame: Mutex<()>,
    shutdown: AtomicBool,
    failure: Mutex<Option<String>>,
    /// Set on the first pointer or key interaction; stops the automatic orbit.
    interacted: AtomicBool,
}

/// A camera change decoded on the winit thread and applied on the map's owner thread.
#[derive(Clone, Copy, Debug)]
enum CameraCommand {
    GestureStart,
    GestureEnd,
    MoveBy { dx: f64, dy: f64 },
    ScaleBy { scale: f64, anchor: mln::ScreenPoint },
    BearingBy { delta: f64 },
    PitchBy { delta: f64 },
    Reset,
}

const DRAG_ROTATE_FACTOR: f64 = 0.5;
const DRAG_PITCH_FACTOR: f64 = 0.5;

/// Decodes winit input into camera commands, in logical pixels.
#[derive(Default)]
struct Input {
    left_down: bool,
    right_down: bool,
    control: bool,
    cursor: (f64, f64),
    last: (f64, f64),
}

impl Input {
    fn handle(
        &mut self,
        event: &WindowEvent,
        scale_factor: f64,
        commands: &mpsc::Sender<CameraCommand>,
    ) -> bool {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta};
        let send = |command: CameraCommand| {
            let _ = commands.send(command);
        };
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x / scale_factor, position.y / scale_factor);
                let (dx, dy) = (x - self.last.0, y - self.last.1);
                self.last = (x, y);
                self.cursor = (x, y);
                if self.right_down || (self.left_down && self.control) {
                    if dx != 0.0 {
                        send(CameraCommand::BearingBy {
                            delta: dx * DRAG_ROTATE_FACTOR,
                        });
                    }
                    if dy != 0.0 {
                        send(CameraCommand::PitchBy {
                            delta: -dy * DRAG_PITCH_FACTOR,
                        });
                    }
                    true
                } else if self.left_down {
                    send(CameraCommand::MoveBy { dx, dy });
                    true
                } else {
                    false
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.left_down = down,
                    MouseButton::Right => self.right_down = down,
                    _ => return false,
                }
                self.last = self.cursor;
                send(if down {
                    CameraCommand::GestureStart
                } else {
                    CameraCommand::GestureEnd
                });
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                if lines == 0.0 {
                    return false;
                }
                send(CameraCommand::ScaleBy {
                    scale: 2f64.powf(lines * 0.25),
                    anchor: mln::ScreenPoint::new(self.cursor.0, self.cursor.1),
                });
                true
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.control = modifiers.state().control_key();
                false
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state != ElementState::Pressed {
                    return false;
                }
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Digit0) => send(CameraCommand::Reset),
                    PhysicalKey::Code(KeyCode::KeyQ) => send(CameraCommand::BearingBy { delta: -10.0 }),
                    PhysicalKey::Code(KeyCode::KeyE) => send(CameraCommand::BearingBy { delta: 10.0 }),
                    PhysicalKey::Code(KeyCode::Equal) => send(CameraCommand::ScaleBy {
                        scale: 1.25,
                        anchor: mln::ScreenPoint::new(self.cursor.0, self.cursor.1),
                    }),
                    PhysicalKey::Code(KeyCode::Minus) => send(CameraCommand::ScaleBy {
                        scale: 0.8,
                        anchor: mln::ScreenPoint::new(self.cursor.0, self.cursor.1),
                    }),
                    _ => return false,
                }
                true
            }
            _ => false,
        }
    }
}

/// What the map thread hands the render thread once the map exists.
struct MapHandles {
    attach_ref: mln::MapAttachRef,
}

/// Runs the runtime and the map until shutdown. Owns both for their whole lifetime.
fn map_thread(
    size: ViewportSize,
    view: ViewState,
    handles: mpsc::Sender<MapHandles>,
    commands: mpsc::Receiver<CameraCommand>,
    shared: Arc<Shared>,
) {
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
            {
                let _frame = shared.frame.lock().unwrap();
                for command in commands.try_iter() {
                    apply_command(&map, command, view)?;
                }
                if !shared.interacted.load(Ordering::Relaxed) {
                    // Slow orbit until the user takes over
                    let mut orbit = mln::CameraOptions::default();
                    orbit.bearing = Some(view.bearing + start.elapsed().as_secs_f64() * 6.0);
                    map.jump_to(&orbit)?;
                }

                runtime.pump(Some(Duration::from_millis(4)), None)?;
                let _ = runtime.drain_events(0)?;
                *shared.camera.lock().unwrap() = Some(map.camera()?);
            }
            // Let the render thread in: this loop is far faster than the display, so without
            // a yield it can hold the lock again before the render thread is ever scheduled.
            std::thread::yield_now();
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

fn apply_command(map: &mln::MapHandle, command: CameraCommand, home: ViewState) -> mln::Result<()> {
    match command {
        CameraCommand::GestureStart => map.set_gesture_in_progress(true),
        CameraCommand::GestureEnd => map.set_gesture_in_progress(false),
        CameraCommand::MoveBy { dx, dy } => map.move_by(dx, dy),
        CameraCommand::ScaleBy { scale, anchor } => map.scale_by(scale, Some(anchor)),
        CameraCommand::BearingBy { delta } => {
            let mut camera = mln::CameraOptions::default();
            camera.bearing = Some(map.camera()?.bearing.unwrap_or(0.0) + delta);
            map.jump_to(&camera)
        }
        CameraCommand::PitchBy { delta } => {
            let mut camera = mln::CameraOptions::default();
            camera.pitch = Some((map.camera()?.pitch.unwrap_or(0.0) + delta).clamp(0.0, 85.0));
            map.jump_to(&camera)
        }
        CameraCommand::Reset => {
            let mut camera = mln::CameraOptions::default();
            camera.center = Some(mln::LatLng::new(home.latitude, home.longitude));
            camera.zoom = Some(home.zoom);
            camera.pitch = Some(home.pitch);
            camera.bearing = Some(home.bearing);
            map.jump_to(&camera)
        }
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
    commands: mpsc::Sender<CameraCommand>,
    input: Input,
    session: Option<mln::RenderSessionHandle>,
    deck: Deck,
    start: Instant,
    screenshot_done: bool,
    hover_cursor: Option<(f64, f64)>,
    hovered: Option<(String, u32)>,
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
        let loaded = spec::load(-25.0)?;
        let view = loaded.view_state;

        // maplibre-native on its own thread; we get a reference to attach a session to
        let shared = Arc::new(Shared::default());
        let (handles_tx, handles_rx) = mpsc::channel();
        let (commands, commands_rx) = mpsc::channel();
        let map_thread = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("maplibre".into())
                .spawn(move || map_thread(size, view, handles_tx, commands_rx, shared))?
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
                view_state: view,
                layers: loaded.layers,
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
            commands,
            input: Input::default(),
            session: Some(session),
            deck,
            start: Instant::now(),
            screenshot_done: false,
            hover_cursor: None,
            hovered: None,
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

        // 1. maplibre renders the map into our texture and waits for the GPU, and we take the
        //    camera it rendered with. Both under the frame lock, so the map thread cannot move
        //    the camera in between: `render_update` has no way to report the camera it used, so
        //    holding it still is what makes the published one the right answer.
        let Some(map_camera) = ({
            let _frame = self.shared.frame.lock().unwrap();
            let _update = session.render_update()?;
            self.shared.camera.lock().unwrap().clone()
        }) else {
            return Ok(());
        };

        // 2. deck draws its layers into the same texture, with the map's camera
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
        // Padding, roll and centre altitude all move where the map puts its centre, so deck
        // has to be told about them or the layers drift away from the basemap as soon as any
        // of the three is used.
        let padding = map_camera.padding.unwrap_or_default();
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
            padding_left: padding.left,
            padding_right: padding.right,
            padding_top: padding.top,
            padding_bottom: padding.bottom,
            roll_degrees: map_camera.roll.unwrap_or(0.0),
            center_elevation_meters: map_camera.center_altitude.unwrap_or(0.0),
            // maplibre-native does not hand out its projection matrix, so deck builds its own
            // from the field of view and the planes above
            ..Default::default()
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
        self.update_hover()?;
        Ok(())
    }

    /// Pick under the cursor and highlight the hit. Prints when the hit changes.
    fn update_hover(&mut self) -> Result<(), Box<dyn Error>> {
        let Some((x, y)) = self.hover_cursor.take() else {
            return Ok(());
        };
        let hit = self.deck.pick(x, y)?;
        let key = hit.as_ref().map(|h| (h.layer_id.clone(), h.index));
        if key != self.hovered {
            self.deck.clear_highlights();
            if let Some(hit) = &hit {
                self.deck.set_highlighted_object(&hit.layer_id, Some(hit.index));
                println!(
                    "hover: layer {} object {} at {:.5}, {:.5}",
                    hit.layer_id, hit.index, hit.coordinate[0], hit.coordinate[1]
                );
            }
            self.hovered = key;
        }
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
            ref event => {
                if state
                    .input
                    .handle(event, state.size.scale_factor, &state.commands)
                {
                    state.shared.interacted.store(true, Ordering::Relaxed);
                }
                Ok(())
            }
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

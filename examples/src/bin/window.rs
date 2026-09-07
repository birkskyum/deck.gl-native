//! The example scene in a window, with the camera orbiting the scene.
//!
//! Run with `cargo run --release --bin window`. Set `DECKGL_JSON` to show a JSON description
//! instead of the built-in scene. The camera orbits until you touch it: drag to pan, right
//! drag (or Ctrl or Cmd drag) to rotate and pitch, scroll to zoom, arrows to move, `+`/`-` to
//! zoom, `q`/`e` to rotate, `r`/`f` to pitch and `0` to return to the start.

use std::sync::Arc;
use std::time::Instant;

use deck_gl::luma_gl::device::create_render_texture;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{ClickCallback, Deck, DeckProps, HoverCallback, MapController, ViewState};
use deck_gl_examples::{scene, spec};
use deck_gl_layers::TripsLayer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::Texture,
    deck: Deck,
    base_view: ViewState,
    controller: MapController,
    start: Instant,
    cursor: Option<(f64, f64)>,
    last_cursor: (f64, f64),
    dragging: Option<MouseButton>,
    /// Where the pressed button went down, to tell clicks from drags
    press_pixel: [f64; 2],
    modifiers: winit::keyboard::ModifiersState,
}

impl State {
    fn new(window: Arc<Window>) -> State {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("deck.gl-native window"),
            ..Default::default()
        }))
        .expect("device");

        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        // deck.gl renders in a non-sRGB color space, as the JavaScript version does on a canvas.
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(capabilities.formats[0]);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface config");
        config.format = format;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let target = RenderTarget {
            color_format: format,
            depth_format: Some(wgpu::TextureFormat::Depth24Plus),
            sample_count: spec::msaa_samples(),
        };
        let depth = create_render_texture(
            &device,
            "depth",
            config.width,
            config.height,
            wgpu::TextureFormat::Depth24Plus,
        );
        let scale = window.scale_factor() as f32;
        let loaded = spec::load(-25.0).expect("scene");
        let base_view = loaded.view_state;
        let controller = MapController::new(
            base_view,
            config.width as f64 / scale as f64,
            config.height as f64 / scale as f64,
        );
        let mut deck = Deck::new(
            &device,
            &queue,
            target,
            DeckProps {
                width: (config.width as f32 / scale) as u32,
                height: (config.height as f32 / scale) as u32,
                device_pixel_ratio: scale,
                view_state: base_view,
                layers: loaded.layers,
                ..Default::default()
            },
        )
        .expect("deck");
        if let Some(lighting) = loaded.lighting {
            deck.set_lighting(lighting);
        }
        deck.set_on_hover(Some(HoverCallback::new(|info| {
            if let Some(hit) = info {
                println!(
                    "hover: layer {} object {} at {:.5}, {:.5}",
                    hit.layer_id, hit.index, hit.coordinate[0], hit.coordinate[1]
                );
            }
        })));
        deck.set_on_click(Some(ClickCallback::new(|hit| {
            println!("click: layer {} object {}", hit.layer_id, hit.index);
        })));

        State {
            window,
            surface,
            device,
            queue,
            config,
            depth,
            deck,
            base_view,
            controller,
            start: Instant::now(),
            cursor: None,
            last_cursor: (0.0, 0.0),
            dragging: None,
            press_pixel: [0.0, 0.0],
            modifiers: Default::default(),
        }
    }

    /// Pick under the cursor: layers with `auto_highlight` tint the hit, and the deck's hover
    /// callback (set below) prints it.
    fn update_hover(&mut self) {
        let Some((x, y)) = self.cursor.take() else { return };
        if let Err(e) = self.deck.pointer_move(x, y) {
            eprintln!("pick error: {e}");
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth = create_render_texture(
            &self.device,
            "depth",
            self.config.width,
            self.config.height,
            wgpu::TextureFormat::Depth24Plus,
        );
        let scale = self.window.scale_factor() as f32;
        self.deck.set_device_pixel_ratio(scale);
        self.deck.set_size(
            (self.config.width as f32 / scale) as u32,
            (self.config.height as f32 / scale) as u32,
        );
        self.controller.set_size(
            self.config.width as f64 / scale as f64,
            self.config.height as f64 / scale as f64,
        );
    }

    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Pointer buttons: left drags pan, right (or a modifier with left) rotates.
    fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        let pixel = [self.last_cursor.0, self.last_cursor.1];
        let rotate = button == MouseButton::Right
            || (button == MouseButton::Left && (self.modifiers.control_key() || self.modifiers.super_key()));
        if pressed {
            if self.dragging.is_some() {
                return;
            }
            self.dragging = Some(button);
            self.press_pixel = pixel;
            if rotate {
                self.controller.rotate_start(pixel);
            } else if button == MouseButton::Left {
                self.controller.pan_start(pixel, self.now_ms());
            }
        } else if self.dragging == Some(button) {
            self.dragging = None;
            self.controller.rotate_end();
            self.controller.pan_end(self.now_ms());
            // A left button released where it was pressed is a click
            let moved = (pixel[0] - self.press_pixel[0]).abs() + (pixel[1] - self.press_pixel[1]).abs();
            if button == MouseButton::Left && !rotate && moved < 3.0 {
                if let Err(e) = self.deck.click(pixel[0], pixel[1]) {
                    eprintln!("pick error: {e}");
                }
            }
        }
    }

    fn key(&mut self, key: &Key) {
        let step = 60.0;
        match key {
            Key::Named(NamedKey::ArrowLeft) => self.controller.move_by([step, 0.0]),
            Key::Named(NamedKey::ArrowRight) => self.controller.move_by([-step, 0.0]),
            Key::Named(NamedKey::ArrowUp) => self.controller.move_by([0.0, step]),
            Key::Named(NamedKey::ArrowDown) => self.controller.move_by([0.0, -step]),
            Key::Character(c) => match c.as_str() {
                "+" | "=" => self.controller.zoom_in(),
                "-" => self.controller.zoom_out(),
                "q" => self.controller.rotate_by(-15.0, 0.0),
                "e" => self.controller.rotate_by(15.0, 0.0),
                "r" => self.controller.rotate_by(0.0, 10.0),
                "f" => self.controller.rotate_by(0.0, -10.0),
                "0" => self.controller.set_view_state(self.base_view),
                _ => {}
            },
            _ => {}
        }
    }

    fn render(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let now = elapsed * 1000.0;
        self.controller.tick(now);
        if self.controller.interacted() {
            self.deck.set_view_state(self.controller.view_state());
        } else {
            let mut view = self.base_view;
            view.bearing += elapsed * 8.0;
            self.controller.set_view_state(view);
            self.deck.set_view_state(view);
        }
        if let Some(trips) = self
            .deck
            .layer_mut("trips")
            .and_then(|layer| layer.as_any_mut().downcast_mut::<TripsLayer>())
        {
            trips.set_current_time((elapsed as f32) % scene::TRIP_LOOP_SECONDS);
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.config.width, self.config.height);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("surface validation error");
                return;
            }
        };
        let color_view = frame.texture.create_view(&Default::default());
        let depth_view = self.depth.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        if let Err(e) = self.deck.render(
            &mut encoder,
            &color_view,
            Some(&depth_view),
            Some(scene::CLEAR_COLOR),
        ) {
            eprintln!("render error: {e}");
        }
        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
        self.update_hover();
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("deck.gl-native")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        self.state = Some(State::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = state.window.inner_size();
                state.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = state.window.scale_factor();
                let pixel = (position.x / scale, position.y / scale);
                state.last_cursor = pixel;
                if state.dragging.is_some() {
                    let now = state.now_ms();
                    state.controller.pan([pixel.0, pixel.1], now);
                    state.controller.rotate([pixel.0, pixel.1]);
                } else {
                    state.cursor = Some(pixel);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                state.cursor = None;
                state.deck.pointer_leave();
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                state.mouse_button(button, button_state == ElementState::Pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                let pixel = [state.last_cursor.0, state.last_cursor.1];
                state.controller.zoom_by(pixel, lines * 0.25);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                state.key(&event.logical_key);
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop");
}

//! The example scene in a window, with the camera orbiting the scene.
//!
//! Run with `cargo run --release --bin window`. Set `DECKGL_JSON` to show a JSON description
//! instead of the built-in scene.

use std::sync::Arc;
use std::time::Instant;

use deck_gl::luma_gl::device::create_render_texture;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps, ViewState};
use deck_gl_examples::{scene, spec};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
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
    start: Instant,
    cursor: Option<(f64, f64)>,
    hovered: Option<(String, u32)>,
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
            sample_count: 1,
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
        let deck = Deck::new(
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

        State {
            window,
            surface,
            device,
            queue,
            config,
            depth,
            deck,
            base_view,
            start: Instant::now(),
            cursor: None,
            hovered: None,
        }
    }

    /// Pick under the cursor and highlight the hit. Prints when the hit changes.
    fn update_hover(&mut self) {
        let Some((x, y)) = self.cursor.take() else { return };
        let hit = match self.deck.pick(x, y) {
            Ok(hit) => hit,
            Err(e) => {
                eprintln!("pick error: {e}");
                return;
            }
        };
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
    }

    fn render(&mut self) {
        let mut view = self.base_view;
        view.bearing += self.start.elapsed().as_secs_f64() * 8.0;
        self.deck.set_view_state(view);

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
                state.cursor = Some((position.x / scale, position.y / scale));
            }
            WindowEvent::CursorLeft { .. } => {
                state.cursor = None;
                state.hovered = None;
                state.deck.clear_highlights();
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

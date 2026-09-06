//! C ABI for embedding deck.gl-native in a host renderer.
//!
//! See `include/deckgl.h` for the API. The Metal entry points adopt the host's `MTLDevice` and
//! `MTLCommandQueue` through wgpu's hal layer, so deck's command buffers are committed on the
//! host's queue and execute after whatever the host committed before them.

// Only the Metal host exists so far, so the shared handle machinery is unused on other
// platforms until Vulkan host interop lands.
#![cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]

use std::ffi::{c_char, c_void, CString};

use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps, Viewport, WebMercatorViewportOptions};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
mod screenshot;

/// Camera of the host map. Mirrors `DeckglCamera` in `deckgl.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeckglCamera {
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
    pub bearing: f64,
    pub pitch: f64,
    pub fov_degrees: f64,
    pub near_z_pixels: f64,
    pub far_z_pixels: f64,
    pub width: u32,
    pub height: u32,
    pub pixel_ratio: f32,
}

/// Opaque handle returned to the host.
pub struct DeckglHandle {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) deck: Option<Deck>,
    pub(crate) target: Option<RenderTarget>,
    pub(crate) camera: Option<DeckglCamera>,
    pub(crate) pending_layers: Option<Vec<Box<dyn deck_gl::Layer>>>,
    pub(crate) last_error: CString,
    /// Number of frames rendered so far
    pub(crate) frame: u64,
}

impl DeckglHandle {
    pub(crate) fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            deck: None,
            target: None,
            camera: None,
            pending_layers: None,
            last_error: CString::default(),
            frame: 0,
        }
    }

    pub(crate) fn set_error(&mut self, message: impl Into<String>) -> i32 {
        let message = message.into();
        eprintln!("deck.gl-native: {message}");
        self.last_error = CString::new(message).unwrap_or_default();
        1
    }

    /// Create the deck once the attachment formats are known.
    pub(crate) fn ensure_deck(&mut self, target: RenderTarget) -> Result<(), String> {
        if self.target != Some(target) {
            self.deck = None;
        }
        if let Some(deck) = self.deck.as_mut() {
            if let Some(layers) = self.pending_layers.take() {
                deck.set_layers(layers);
            }
        } else {
            let camera = self.camera.unwrap_or(DeckglCamera {
                longitude: 0.0,
                latitude: 0.0,
                zoom: 0.0,
                bearing: 0.0,
                pitch: 0.0,
                fov_degrees: 0.0,
                near_z_pixels: 0.0,
                far_z_pixels: 0.0,
                width: 1,
                height: 1,
                pixel_ratio: 1.0,
            });
            let mut deck = Deck::new(
                &self.device,
                &self.queue,
                target,
                DeckProps {
                    width: camera.width.max(1),
                    height: camera.height.max(1),
                    device_pixel_ratio: camera.pixel_ratio,
                    layers: self.pending_layers.take().unwrap_or_default(),
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
            deck.set_viewport(viewport_from_camera(&camera));
            self.deck = Some(deck);
            self.target = Some(target);
        }
        Ok(())
    }

    pub(crate) fn apply_camera(&mut self) {
        if let (Some(deck), Some(camera)) = (self.deck.as_mut(), self.camera) {
            deck.set_device_pixel_ratio(camera.pixel_ratio);
            deck.set_size(camera.width.max(1), camera.height.max(1));
            deck.set_viewport(viewport_from_camera(&camera));
        }
    }
}

/// Near and far plane distances, in pixels, that maplibre-native uses for its projection
/// matrix (`TransformState::getProjMatrix` with no padding, roll or center altitude).
/// `height` is the map height in logical pixels.
pub fn maplibre_near_far_pixels(fov_degrees: f64, pitch_degrees: f64, height: f64) -> (f64, f64) {
    let fov = fov_degrees.to_radians();
    let pitch = pitch_degrees.clamp(0.0, 89.0).to_radians();
    let camera_to_center = 0.5 * height / (fov / 2.0).tan();
    let tan_fov_above_center = (fov / 2.0).tan();
    let tan_multiple = (tan_fov_above_center * pitch.tan()).clamp(0.0, 0.99);
    let furthest = camera_to_center / (1.0 - tan_multiple);
    (1.0, furthest * 1.01)
}

/// Build a deck viewport that matches the host's projection. Port of deck.gl's
/// `getViewport` in `@deck.gl/mapbox`: the host's near and far planes are converted from
/// pixels to deck's units by dividing by the viewport height.
pub fn viewport_from_camera(camera: &DeckglCamera) -> Viewport {
    let height = camera.height.max(1) as f64;
    let mut opts = WebMercatorViewportOptions {
        width: camera.width.max(1) as f64,
        height,
        longitude: ((camera.longitude + 540.0) % 360.0) - 180.0,
        latitude: camera.latitude,
        zoom: camera.zoom,
        bearing: camera.bearing,
        pitch: camera.pitch,
        ..Default::default()
    };
    if camera.fov_degrees > 0.0 {
        opts.fovy = Some(camera.fov_degrees);
    }
    if camera.near_z_pixels > 0.0 && camera.far_z_pixels > camera.near_z_pixels {
        opts.near_z = Some(camera.near_z_pixels / height);
        opts.far_z = Some(camera.far_z_pixels / height);
    }
    Viewport::web_mercator(&opts)
}

/// # Safety
/// `deck` must be a handle returned by a `deckgl_*_create` function, or null.
#[no_mangle]
pub unsafe extern "C" fn deckgl_destroy(deck: *mut DeckglHandle) {
    if !deck.is_null() {
        drop(unsafe { Box::from_raw(deck) });
    }
}

/// # Safety
/// `deck` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn deckgl_load_demo_scene(deck: *mut DeckglHandle) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let layers = deck_gl_examples::scene::layers();
    match handle.deck.as_mut() {
        Some(deck) => deck.set_layers(layers),
        None => handle.pending_layers = Some(layers),
    }
    0
}

/// # Safety
/// `deck` must be a valid handle and `camera` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn deckgl_set_camera(deck: *mut DeckglHandle, camera: *const DeckglCamera) {
    let (Some(handle), Some(camera)) = (unsafe { deck.as_mut() }, unsafe { camera.as_ref() }) else {
        return;
    };
    handle.camera = Some(*camera);
    handle.apply_camera();
}

/// # Safety
/// `deck` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn deckgl_last_error(deck: *mut DeckglHandle) -> *const c_char {
    match unsafe { deck.as_ref() } {
        Some(handle) => handle.last_error.as_ptr(),
        None => c"".as_ptr(),
    }
}

// Silence unused warnings on platforms without a backend implementation.
#[allow(dead_code)]
fn _unused(_: *mut c_void) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_uses_host_near_far_planes() {
        let camera = DeckglCamera {
            longitude: -122.42,
            latitude: 37.775,
            zoom: 12.6,
            bearing: -25.0,
            pitch: 50.0,
            fov_degrees: 36.86989764584402,
            near_z_pixels: 1.0,
            far_z_pixels: 1500.0,
            width: 1024,
            height: 768,
            pixel_ratio: 2.0,
        };
        let viewport = viewport_from_camera(&camera);
        assert!((viewport.near - 1.0 / 768.0).abs() < 1e-12);
        assert!((viewport.far - 1500.0 / 768.0).abs() < 1e-12);
        assert!(
            (viewport.altitude - 1.5).abs() < 1e-9,
            "default maplibre fov is altitude 1.5"
        );
        assert_eq!(viewport.width, 1024.0);
        assert_eq!(viewport.bearing, -25.0);
    }

    #[test]
    fn near_far_match_maplibre_formula() {
        // fov 36.87 degrees, no pitch: camera at 1.5 * height, far = 1.5 * height * 1.01
        let (near, far) = maplibre_near_far_pixels(36.86989764584402, 0.0, 600.0);
        assert!((near - 1.0).abs() < 1e-12);
        assert!((far - 900.0 * 1.01).abs() < 1e-6, "{far}");
        let (_, far_pitched) = maplibre_near_far_pixels(36.86989764584402, 50.0, 600.0);
        assert!(far_pitched > far);
    }

    #[test]
    fn viewport_falls_back_to_deck_planes() {
        let camera = DeckglCamera {
            longitude: 540.0,
            latitude: 0.0,
            zoom: 3.0,
            bearing: 0.0,
            pitch: 0.0,
            fov_degrees: 0.0,
            near_z_pixels: 0.0,
            far_z_pixels: 0.0,
            width: 100,
            height: 100,
            pixel_ratio: 1.0,
        };
        let viewport = viewport_from_camera(&camera);
        assert_eq!(
            viewport.longitude, -180.0,
            "longitude is wrapped into [-180, 180)"
        );
        assert!((viewport.near - 0.1).abs() < 1e-12);
    }
}

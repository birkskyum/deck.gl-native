//! C ABI for embedding deck.gl-native in a host renderer.
//!
//! See `include/deckgl.h` for the API. The Metal entry points adopt the host's `MTLDevice` and
//! `MTLCommandQueue` through wgpu's hal layer, so deck's command buffers are committed on the
//! host's queue and execute after whatever the host committed before them.

// Only the Metal host renders so far; the screenshot helpers are unused on other platforms
// until Vulkan host interop lands.
#![cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};

use arrow_array::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Array, RecordBatch, StructArray};

use deck_gl::luma_gl::RenderTarget;
use deck_gl::{ClipDepthRange, Deck, DeckProps, Layer, ViewState, Viewport, WebMercatorViewportOptions};
use deck_gl_json::JsonConverter;

mod debug;
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
    /// Padding in logical pixels: the map centre sits in the middle of the unpadded area
    pub padding_left: f64,
    pub padding_right: f64,
    pub padding_top: f64,
    pub padding_bottom: f64,
    /// Rotation around the view axis in degrees
    pub roll_degrees: f64,
    /// Elevation of the map centre in meters (terrain), the camera looks at this height
    pub center_elevation_meters: f64,
    /// When nonzero, `projection_matrix` replaces the projection built from the field of view
    /// and the planes
    pub has_projection_matrix: i32,
    /// Column major, OpenGL clip conventions (as maplibre's), used with `has_projection_matrix`
    pub projection_matrix: [f64; 16],
}

impl Default for DeckglCamera {
    fn default() -> Self {
        Self {
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
            padding_left: 0.0,
            padding_right: 0.0,
            padding_top: 0.0,
            padding_bottom: 0.0,
            roll_degrees: 0.0,
            center_elevation_meters: 0.0,
            has_projection_matrix: 0,
            projection_matrix: [0.0; 16],
        }
    }
}

/// Opaque handle returned to the host.
pub struct DeckglHandle {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) deck: Option<Deck>,
    /// Arrow tables registered with `deckgl_set_arrow_table`, by name
    pub(crate) tables: HashMap<String, RecordBatch>,
    pub(crate) target: Option<RenderTarget>,
    pub(crate) camera: Option<DeckglCamera>,
    pub(crate) pending_layers: Option<Vec<Box<dyn deck_gl::Layer>>>,
    /// Lighting from the last JSON description, applied to every deck
    pub(crate) lighting: Option<deck_gl::LightingEffect>,
    /// `initialViewState` of the last JSON description, the camera of headless snapshots
    pub(crate) view_state: Option<ViewState>,
    /// Layer id of the last `deckgl_pick` hit, so its pointer stays valid
    pub(crate) picked_layer: CString,
    /// `MapView.repeat` of the last JSON description
    pub(crate) repeat: bool,
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
            tables: HashMap::new(),
            target: None,
            camera: None,
            pending_layers: None,
            lighting: None,
            view_state: None,
            picked_layer: CString::default(),
            repeat: false,
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
            let camera = self.camera.unwrap_or_default();
            let mut deck = Deck::new(
                &self.device,
                &self.queue,
                target,
                DeckProps {
                    width: camera.width.max(1),
                    height: camera.height.max(1),
                    device_pixel_ratio: camera.pixel_ratio,
                    layers: self.pending_layers.take().unwrap_or_default(),
                    // The host's background and opaque fills write the ground plane into the
                    // shared depth buffer; bias deck above it so ground layers do not z-fight.
                    depth_bias_base: -100,
                    // maplibre-native writes OpenGL style depth on every backend.
                    clip_depth_range: ClipDepthRange::NegativeOneToOne,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
            deck.set_viewport(viewport_from_camera(&camera));
            if let Some(lighting) = self.lighting.clone() {
                deck.set_lighting(lighting);
            }
            deck.set_repeat(self.repeat);
            self.deck = Some(deck);
            self.target = Some(target);
        }
        Ok(())
    }

    /// Replace the layers now, or once the deck exists.
    pub(crate) fn set_layers(&mut self, layers: Vec<Box<dyn Layer>>) {
        match self.deck.as_mut() {
            Some(deck) => deck.set_layers(layers),
            None => self.pending_layers = Some(layers),
        }
    }

    pub(crate) fn apply_camera(&mut self) {
        if let (Some(deck), Some(camera)) = (self.deck.as_mut(), self.camera) {
            deck.set_device_pixel_ratio(camera.pixel_ratio);
            deck.set_size(camera.width.max(1), camera.height.max(1));
            deck.set_viewport(viewport_from_camera(&camera));
        }
    }
}

/// Near and far plane distances, in pixels, of the projection maplibre-native uses for its 3D
/// layers: `PaintParameters::nearClippedProjMatrix`, whose near plane is a tenth of the camera
/// distance truncated to whole pixels, with the far plane of `TransformState::getProjMatrix`
/// (no padding, roll or center altitude). Deck must use these planes for its depth values to
/// be comparable with the map's buildings. `height` is the map height in logical pixels.
pub fn maplibre_near_far_pixels(fov_degrees: f64, pitch_degrees: f64, height: f64) -> (f64, f64) {
    let fov = fov_degrees.to_radians();
    let pitch = pitch_degrees.clamp(0.0, 89.0).to_radians();
    let camera_to_center = 0.5 * height / (fov / 2.0).tan();
    let tan_fov_above_center = (fov / 2.0).tan();
    let tan_multiple = (tan_fov_above_center * pitch.tan()).clamp(0.0, 0.99);
    let furthest = camera_to_center / (1.0 - tan_multiple);
    ((0.1 * camera_to_center).floor().max(1.0), furthest * 1.01)
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
    let padding = [
        camera.padding_left,
        camera.padding_right,
        camera.padding_top,
        camera.padding_bottom,
    ];
    if padding.iter().any(|p| *p != 0.0) {
        opts.padding = Some(deck_gl::Padding {
            left: padding[0],
            right: padding[1],
            top: padding[2],
            bottom: padding[3],
        });
    }
    opts.roll = camera.roll_degrees;
    if camera.center_elevation_meters != 0.0 {
        opts.position = Some(deck_gl::glam::DVec3::new(
            0.0,
            0.0,
            camera.center_elevation_meters,
        ));
    }
    if camera.has_projection_matrix != 0 {
        opts.projection_matrix = Some(deck_gl::glam::DMat4::from_cols_array(&camera.projection_matrix));
    }
    let viewport = Viewport::web_mercator(&opts);
    if std::env::var_os("DECKGL_DEBUG").is_some() {
        eprintln!(
            "deck.gl-native: viewport {}x{} zoom {:.2} pitch {:.1} fovy {:.3} altitude {:.3} near {} far {}",
            viewport.width,
            viewport.height,
            viewport.zoom,
            viewport.pitch,
            viewport.fovy,
            viewport.altitude,
            viewport.near,
            viewport.far
        );
    }
    viewport
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
    handle.set_layers(deck_gl_examples::scene::layers());
    0
}

/// # Safety
/// `ptr` must be null or a valid C string.
unsafe fn c_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }
}

fn apply_json(handle: &mut DeckglHandle, result: deck_gl_json::Result<deck_gl_json::JsonDeck>) -> i32 {
    match result {
        Ok(json) => {
            for warning in &json.warnings {
                eprintln!("deck.gl-native: {warning}");
            }
            handle.set_layers(json.layers);
            if let Some(lighting) = json.lighting {
                if let Some(deck) = handle.deck.as_mut() {
                    deck.set_lighting(lighting.clone());
                }
                handle.lighting = Some(lighting);
            }
            if json.view_state.is_some() {
                handle.view_state = json.view_state;
            }
            handle.repeat = json.repeat;
            if let Some(deck) = handle.deck.as_mut() {
                deck.set_repeat(json.repeat);
            }
            0
        }
        Err(e) => handle.set_error(e.to_string()),
    }
}

/// # Safety
/// `deck` must be a valid handle; `json` must be a valid C string and `base_dir` null or one.
#[no_mangle]
pub unsafe extern "C" fn deckgl_set_layers_json(
    deck: *mut DeckglHandle,
    json: *const c_char,
    base_dir: *const c_char,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(json) = (unsafe { c_string(json) }) else {
        return handle.set_error("deckgl_set_layers_json: json is null");
    };
    let mut converter = match unsafe { c_string(base_dir) } {
        Some(dir) if !dir.is_empty() => JsonConverter::with_base_dir(dir),
        _ => JsonConverter::new(),
    };
    converter.options.tables = handle.tables.clone();
    apply_json(handle, converter.parse(&json))
}

/// # Safety
/// `deck` must be a valid handle and `path` a valid C string.
#[no_mangle]
pub unsafe extern "C" fn deckgl_load_json_file(deck: *mut DeckglHandle, path: *const c_char) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(path) = (unsafe { c_string(path) }) else {
        return handle.set_error("deckgl_load_json_file: path is null");
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => return handle.set_error(format!("could not read {path}: {e}")),
    };
    let mut converter = match std::path::Path::new(&path).parent() {
        Some(dir) if !dir.as_os_str().is_empty() => JsonConverter::with_base_dir(dir),
        _ => JsonConverter::new(),
    };
    converter.options.tables = handle.tables.clone();
    apply_json(handle, converter.parse(&text))
}

/// Create a deck on a headless wgpu device (any platform). Rendering needs a host texture
/// through a backend specific entry point; this is for tooling and tests that only convert
/// layers and data.
///
/// # Safety
/// Always safe to call; returns null when no GPU adapter is available.
#[no_mangle]
pub unsafe extern "C" fn deckgl_headless_create() -> *mut DeckglHandle {
    match deck_gl::luma_gl::device::create_headless_context() {
        Ok(ctx) => Box::into_raw(Box::new(DeckglHandle::new(ctx.device, ctx.queue))),
        Err(e) => {
            eprintln!("deck.gl-native: {e}");
            std::ptr::null_mut()
        }
    }
}

impl DeckglHandle {
    /// Render a frame of `width` x `height` pixels and read it back. With a host camera the
    /// deck keeps that camera and the size must match it; otherwise the last JSON description's
    /// `initialViewState` (or a default view) is used at the given size.
    pub(crate) fn snapshot(&mut self, width: u32, height: u32) -> Result<deck_gl::Snapshot, String> {
        let target = self.target.unwrap_or_default();
        self.ensure_deck(target)?;
        match self.camera {
            Some(camera) => {
                let expected = (camera.width.max(1), camera.height.max(1));
                if expected != (width, height) {
                    return Err(format!(
                        "deckgl_snapshot: size {width}x{height} does not match the camera's {}x{}",
                        expected.0, expected.1
                    ));
                }
                self.apply_camera();
            }
            None => {
                let deck = self.deck.as_mut().expect("ensured");
                deck.set_device_pixel_ratio(1.0);
                deck.set_size(width, height);
                deck.set_view_state(self.view_state.unwrap_or_default());
            }
        }
        let deck = self.deck.as_mut().expect("ensured");
        deck.snapshot(None).map_err(|e| e.to_string())
    }
}

/// What `deckgl_pick` found under a pixel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeckglPickingInfo {
    /// 1 when an object was hit, 0 otherwise (the other fields are then zero)
    pub picked: i32,
    /// Data row of the picked object
    pub index: u32,
    /// The queried pixel in logical coordinates
    pub x: f64,
    pub y: f64,
    /// The pixel unprojected onto the ground plane
    pub longitude: f64,
    pub latitude: f64,
    /// Id of the picked layer; valid until the next `deckgl_pick` on this deck
    pub layer_id: *const c_char,
}

impl Default for DeckglPickingInfo {
    fn default() -> Self {
        Self {
            picked: 0,
            index: 0,
            x: 0.0,
            y: 0.0,
            longitude: 0.0,
            latitude: 0.0,
            layer_id: std::ptr::null(),
        }
    }
}

/// Find the object under a pixel (logical coordinates, origin top left) with the current
/// camera, deck.gl's `pickObject`. Waits for the GPU. Returns 0 on success and fills `info`;
/// `info.picked` says whether anything was hit.
///
/// # Safety
/// `deck` must be a valid handle and `info` writable.
#[no_mangle]
pub unsafe extern "C" fn deckgl_pick(
    deck: *mut DeckglHandle,
    x: f64,
    y: f64,
    info: *mut DeckglPickingInfo,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(info) = (unsafe { info.as_mut() }) else {
        return handle.set_error("deckgl_pick: info is null");
    };
    *info = DeckglPickingInfo::default();
    let target = handle.target.unwrap_or_default();
    if let Err(e) = handle.ensure_deck(target) {
        return handle.set_error(e);
    }
    if handle.camera.is_some() {
        handle.apply_camera();
    } else if let Some(view_state) = handle.view_state {
        handle.deck.as_mut().expect("ensured").set_view_state(view_state);
    }
    let deck = handle.deck.as_mut().expect("ensured");
    match deck.pick(x, y) {
        Ok(Some(hit)) => {
            handle.picked_layer = CString::new(hit.layer_id).unwrap_or_default();
            *info = DeckglPickingInfo {
                picked: 1,
                index: hit.index,
                x: hit.pixel[0],
                y: hit.pixel[1],
                longitude: hit.coordinate[0],
                latitude: hit.coordinate[1],
                layer_id: handle.picked_layer.as_ptr(),
            };
            0
        }
        Ok(None) => 0,
        Err(e) => handle.set_error(e.to_string()),
    }
}

/// Render the layers headlessly into `rgba`, which must hold `width * height * 4` bytes
/// (rows top to bottom, transparent background). The camera is the one from
/// `deckgl_set_camera`, whose size must then match, or else the last JSON description's
/// `initialViewState` at the given size. Returns 0 on success.
///
/// # Safety
/// `deck` must be a valid handle and `rgba` writable for `width * height * 4` bytes.
#[no_mangle]
pub unsafe extern "C" fn deckgl_snapshot(
    deck: *mut DeckglHandle,
    width: u32,
    height: u32,
    rgba: *mut u8,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    if rgba.is_null() || width == 0 || height == 0 {
        return handle.set_error("deckgl_snapshot: rgba is null or the size is zero");
    }
    match handle.snapshot(width, height) {
        Ok(snapshot) => {
            let out = unsafe { std::slice::from_raw_parts_mut(rgba, (width * height * 4) as usize) };
            out.copy_from_slice(&snapshot.rgba);
            0
        }
        Err(e) => handle.set_error(e),
    }
}

/// Like `deckgl_snapshot`, written as a PNG file. Returns 0 on success.
///
/// # Safety
/// `deck` must be a valid handle and `path` a C string.
#[no_mangle]
pub unsafe extern "C" fn deckgl_snapshot_png(
    deck: *mut DeckglHandle,
    width: u32,
    height: u32,
    path: *const c_char,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(path) = (unsafe { c_string(path) }) else {
        return handle.set_error("deckgl_snapshot_png: path is null");
    };
    match handle
        .snapshot(width, height)
        .and_then(|s| s.save_png(&path).map_err(|e| e.to_string()))
    {
        Ok(()) => 0,
        Err(e) => handle.set_error(e),
    }
}

/// Register an Arrow table for JSON layers (`"data": "@@table:<name>"`). The array must be a
/// struct array whose fields are the table's columns, given through the Arrow C Data
/// Interface. Ownership of `array` moves to the deck: its release callback runs when the table
/// is replaced or the deck is destroyed, and the caller's struct is marked released. The
/// schema is only read. Buffers are not copied.
///
/// # Safety
/// `deck` must be a valid handle, `name` a C string, and `schema` and `array` valid, non
/// released Arrow C Data Interface structs.
#[no_mangle]
pub unsafe extern "C" fn deckgl_set_arrow_table(
    deck: *mut DeckglHandle,
    name: *const c_char,
    schema: *const FFI_ArrowSchema,
    array: *mut FFI_ArrowArray,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(name) = (unsafe { c_string(name) }) else {
        return handle.set_error("deckgl_set_arrow_table: name is null");
    };
    if schema.is_null() || array.is_null() {
        return handle.set_error("deckgl_set_arrow_table: schema or array is null");
    }
    let array = unsafe { FFI_ArrowArray::from_raw(array) };
    let data = match unsafe { from_ffi(array, &*schema) } {
        Ok(data) => data,
        Err(e) => return handle.set_error(format!("invalid Arrow array for table `{name}`: {e}")),
    };
    let array = arrow_array::make_array(data);
    let Some(columns) = array.as_any().downcast_ref::<StructArray>() else {
        return handle.set_error(format!(
            "table `{name}` must be a struct array of columns, got {}",
            array.data_type()
        ));
    };
    handle.tables.insert(name, RecordBatch::from(columns.clone()));
    0
}

/// Forget a table registered with `deckgl_set_arrow_table`. Layers already built keep their
/// data.
///
/// # Safety
/// `deck` must be a valid handle and `name` a C string.
#[no_mangle]
pub unsafe extern "C" fn deckgl_remove_arrow_table(deck: *mut DeckglHandle, name: *const c_char) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    match unsafe { c_string(name) } {
        Some(name) => {
            handle.tables.remove(&name);
            0
        }
        None => handle.set_error("deckgl_remove_arrow_table: name is null"),
    }
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
            ..Default::default()
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
    fn elevated_points_project_above_the_ground() {
        let (near, far) = maplibre_near_far_pixels(36.86989764584402, 60.0, 1000.0);
        let camera = DeckglCamera {
            longitude: -122.42,
            latitude: 37.775,
            zoom: 14.5,
            bearing: -25.0,
            pitch: 60.0,
            fov_degrees: 36.86989764584402,
            near_z_pixels: near,
            far_z_pixels: far,
            width: 1400,
            height: 1000,
            pixel_ratio: 2.0,
            ..Default::default()
        };
        let host = viewport_from_camera(&camera);
        let plain = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 1400.0,
            height: 1000.0,
            longitude: -122.42,
            latitude: 37.775,
            zoom: 14.5,
            pitch: 60.0,
            bearing: -25.0,
            ..Default::default()
        });
        for (name, viewport) in [("host", &host), ("plain", &plain)] {
            let ground = viewport.project(deck_gl::glam::DVec3::new(-122.42, 37.775, 0.0), true);
            let roof = viewport.project(deck_gl::glam::DVec3::new(-122.42, 37.775, 400.0), true);
            eprintln!(
                "{name}: ground {ground:?} roof {roof:?} near {} far {}",
                viewport.near, viewport.far
            );
            assert!(
                ground.y - roof.y > 50.0,
                "{name}: a 400 m roof must be well above the ground on screen"
            );
        }
    }

    #[test]
    fn near_far_match_maplibre_formula() {
        // fov 36.87 degrees, no pitch: camera at 1.5 * height, near a tenth of that in whole
        // pixels, far = 1.5 * height * 1.01
        let (near, far) = maplibre_near_far_pixels(36.86989764584402, 0.0, 600.0);
        assert!((near - 90.0).abs() < 1e-12, "{near}");
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
            ..Default::default()
        };
        let viewport = viewport_from_camera(&camera);
        assert_eq!(
            viewport.longitude, -180.0,
            "longitude is wrapped into [-180, 180)"
        );
        assert!((viewport.near - 0.1).abs() < 1e-12);
    }
}

#[cfg(test)]
mod camera_tests {
    use super::*;

    fn camera() -> DeckglCamera {
        DeckglCamera {
            longitude: 10.0,
            latitude: 50.0,
            zoom: 12.0,
            bearing: 0.0,
            pitch: 0.0,
            fov_degrees: 36.87,
            near_z_pixels: 0.0,
            far_z_pixels: 0.0,
            width: 400,
            height: 300,
            pixel_ratio: 1.0,
            padding_left: 0.0,
            padding_right: 0.0,
            padding_top: 0.0,
            padding_bottom: 0.0,
            roll_degrees: 0.0,
            center_elevation_meters: 0.0,
            has_projection_matrix: 0,
            projection_matrix: [0.0; 16],
        }
    }

    #[test]
    fn padding_elevation_and_host_projection_reach_the_viewport() {
        let plain = viewport_from_camera(&camera());
        let padded = viewport_from_camera(&DeckglCamera {
            padding_left: 200.0,
            ..camera()
        });
        let c = padded.project(deck_gl::glam::DVec3::new(10.0, 50.0, 0.0), true);
        assert!((c.x - 300.0).abs() < 1e-6 && (c.y - 150.0).abs() < 1e-6, "{c:?}");
        // A centre on terrain: the ground at the centre sits below the screen centre
        let raised = viewport_from_camera(&DeckglCamera {
            center_elevation_meters: 100.0,
            pitch: 45.0,
            ..camera()
        });
        let ground = raised.project(deck_gl::glam::DVec3::new(10.0, 50.0, 0.0), true);
        let top = raised.project(deck_gl::glam::DVec3::new(10.0, 50.0, 100.0), true);
        assert!(
            (top.y - 150.0).abs() < 1e-6 && ground.y > 150.0,
            "{ground:?} {top:?}"
        );
        // The host's projection matrix is used as is
        let mut host = camera();
        host.has_projection_matrix = 1;
        host.projection_matrix = plain.projection_matrix.to_cols_array();
        let from_host = viewport_from_camera(&host);
        assert_eq!(from_host.projection_matrix, plain.projection_matrix);
        assert!((from_host.fovy - plain.fovy).abs() < 1e-9);
    }
}

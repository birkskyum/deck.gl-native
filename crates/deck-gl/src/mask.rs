//! The mask pass behind deck.gl's `MaskExtension` (`MaskEffect` and `MaskPass` in
//! `@deck.gl/extensions`): layers whose [`Operation`](crate::Operation) is `mask` are drawn
//! into a texture through a viewport fitted to their bounds, and layers with the extension
//! sample that texture to discard what lies outside (or inside) the mask.

use std::collections::HashMap;

use glam::DVec3;
use luma_gl::model::{create_rgba8_texture, default_sampler};

use crate::constants::CoordinateSystem;
use crate::viewport::{Viewport, WebMercatorViewportOptions};

/// Size of every mask texture in pixels, deck.gl's `mapSize`.
pub const MASK_MAP_SIZE: u32 = 2048;
/// Pixels left clear around the mask so that clamped samples outside it read as unmasked.
pub const MASK_BORDER: u32 = 1;
/// deck.gl renders at most four masks (one per channel of one texture).
pub const MAX_MASKS: usize = 4;

/// One rendered mask.
#[derive(Clone, Debug)]
pub struct MaskChannel {
    pub view: wgpu::TextureView,
    /// Bounds of the whole texture in absolute common (Web Mercator world) space
    pub bounds_common: [f64; 4],
    /// Coordinate system of the mask layer
    pub coordinate_system: CoordinateSystem,
    pub coordinate_origin: [f64; 3],
}

/// The masks of the current frame by mask layer id, with the sampler and a white fallback
/// texture for readers whose mask does not exist.
#[derive(Debug)]
pub struct MaskMaps {
    pub sampler: wgpu::Sampler,
    pub dummy: wgpu::TextureView,
    pub channels: HashMap<String, MaskChannel>,
}

impl MaskMaps {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let dummy = create_rgba8_texture(device, queue, "mask fallback", 1, 1, &[255, 255, 255, 255]);
        Self {
            sampler: default_sampler(device),
            dummy: dummy.create_view(&Default::default()),
            channels: HashMap::new(),
        }
    }
}

/// A texture for one mask: the red channel, cleared to 1 and drawn to 0.
pub fn create_mask_texture(device: &wgpu::Device, id: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("mask {id}")),
        size: wgpu::Extent3d {
            width: MASK_MAP_SIZE,
            height: MASK_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// deck.gl's `getRenderBounds`: the mask layer's bounds, cut down to twice the viewport when
/// the mask is larger than that, so the texture resolution follows the view. Without layer
/// bounds the doubled viewport is used. Everything in absolute common space.
pub fn render_bounds(layer_bounds: Option<[f64; 4]>, viewport_bounds: [f64; 4]) -> [f64; 4] {
    let [x0, y0, x1, y1] = viewport_bounds;
    let (dx, dy) = (x1 - x0, y1 - y0);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let padded = [cx - dx, cy - dy, cx + dx, cy + dy];
    let Some(layer) = layer_bounds else {
        return padded;
    };
    if layer[2] - layer[0] <= padded[2] - padded[0] && layer[3] - layer[1] <= padded[3] - padded[1] {
        return layer;
    }
    [
        layer[0].max(padded[0]),
        layer[1].max(padded[1]),
        layer[2].min(padded[2]),
        layer[3].min(padded[3]),
    ]
}

/// deck.gl's `makeViewport`: an unpitched Web Mercator viewport that fits `bounds` (absolute
/// common space) into a `size` pixel texture inside `border` pixels, and the common space
/// bounds of the whole texture. `None` for empty bounds.
pub fn mask_viewport(
    bounds: [f64; 4],
    reference: &Viewport,
    size: u32,
    border: u32,
) -> Option<(Viewport, [f64; 4])> {
    let [x0, y0, x1, y1] = bounds;
    if x1 <= x0 || y1 <= y0 || !bounds.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let center = reference.unproject_position(DVec3::new(cx, cy, 0.0));
    let inner = size.saturating_sub(2 * border).max(1) as f64;
    let scale = (inner / (x1 - x0)).min(inner / (y1 - y0));
    let zoom = scale.log2().min(20.0);
    let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
        id: format!("{}-mask", reference.id),
        x: border as f64,
        y: border as f64,
        width: inner,
        height: inner,
        longitude: center.x,
        latitude: center.y,
        zoom,
        pitch: 0.0,
        bearing: 0.0,
        ..Default::default()
    });
    let half = size as f64 / 2f64.powf(zoom) / 2.0;
    Some((viewport, [cx - half, cy - half, cx + half, cy + half]))
}

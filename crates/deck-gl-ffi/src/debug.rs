//! Diagnostics: dump the host's depth buffer and compare it with deck's projection.
//! Enabled with `DECKGL_DUMP_DEPTH=/path/prefix`; writes `<prefix>.f32` (row major
//! `f32` depth values) once and prints sampled values against deck's expected ground depth.

use std::sync::atomic::{AtomicBool, Ordering};

use deck_gl::glam::{DVec2, DVec3};
use deck_gl::{Viewport, WebMercatorViewportOptions};

static DONE: AtomicBool = AtomicBool::new(false);

pub fn maybe_dump(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    depth: &wgpu::Texture,
    viewport: &Viewport,
    frame: u64,
    after_deck: bool,
) {
    let Ok(mut prefix) = std::env::var("DECKGL_DUMP_DEPTH") else {
        return;
    };
    if after_deck != std::env::var_os("DECKGL_DUMP_DEPTH_AFTER").is_some() {
        return;
    }
    if after_deck {
        prefix.push_str("-after");
    }
    let at_frame: u64 = std::env::var("DECKGL_DUMP_DEPTH_FRAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    if frame < at_frame || DONE.swap(true, Ordering::SeqCst) {
        return;
    }
    let size = depth.size();
    let (width, height) = (size.width, size.height);
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("depth readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: depth,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = match slice.get_mapped_range() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("deck.gl-native: depth readback failed: {e}");
            return;
        }
    };
    let mut values = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        let bytes = &data[start..start + unpadded as usize];
        values.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = format!("{prefix}.f32");
    if let Err(e) = std::fs::write(&path, raw) {
        eprintln!("deck.gl-native: could not write {path}: {e}");
    }
    eprintln!("deck.gl-native: wrote {path} ({width}x{height} f32)");

    // deck's ground depth at sampled pixels on its own planes, and on maplibre's 1 px planes
    // that flat map layers use. Logical pixel coordinates.
    let ratio = width as f64 / viewport.width;
    let flat = Viewport::web_mercator(&WebMercatorViewportOptions {
        width: viewport.width,
        height: viewport.height,
        longitude: viewport.longitude,
        latitude: viewport.latitude,
        zoom: viewport.zoom,
        pitch: viewport.pitch,
        bearing: viewport.bearing,
        fovy: Some(viewport.fovy),
        near_z: Some(1.0 / viewport.height),
        far_z: Some(viewport.far),
        ..Default::default()
    });
    eprintln!(
        "deck.gl-native: viewport near {} far {} (flat layers near {})",
        viewport.near, viewport.far, flat.near
    );
    let ground_depths = |col: u32, row: u32| {
        let px = DVec2::new(col as f64 / ratio, row as f64 / ratio);
        let ground = viewport.unproject(px, None, true, None);
        let deck_z = viewport.project(DVec3::new(ground.x, ground.y, 0.0), true).z;
        let flat_z = flat.project(DVec3::new(ground.x, ground.y, 0.0), true).z;
        (deck_z, flat_z)
    };
    let cleared = values.iter().filter(|v| **v >= 1.0).count();
    let flat_like = values.iter().filter(|v| **v < 1.0 && **v > 0.997).count();
    let near_like = values.iter().filter(|v| **v <= 0.997).count();
    eprintln!("  pixels: cleared {cleared}, flat curve {flat_like}, near clipped curve {near_like}");
    eprintln!("  px(x,y)       map depth   deck ground   flat ground");
    let step = height / 12;
    for row in (step / 2..height).step_by(step as usize) {
        for col in (width / 8..width).step_by((width / 4) as usize) {
            let d = values[(row * width + col) as usize];
            let (deck_z, flat_z) = ground_depths(col, row);
            eprintln!("  ({col:5},{row:5})  {d:.6}    {deck_z:.6}    {flat_z:.6}");
        }
    }
    // Scanning up each column, the first fragment on the near clipped curve is the foot of a
    // building; its depth should equal deck's ground depth at that pixel.
    eprintln!("  building feet: px(x,y)  map depth  deck ground  (rows above with same curve)");
    for col in (width / 16..width).step_by((width / 8) as usize) {
        let mut row = height - 1;
        while row > 0 && values[(row * width + col) as usize] > 0.997 {
            row -= 1;
        }
        if row == 0 {
            continue;
        }
        let d = values[(row * width + col) as usize];
        let (deck_z, _) = ground_depths(col, row);
        let mut top = row;
        while top > 0 && values[(top * width + col) as usize] <= 0.997 {
            top -= 1;
        }
        eprintln!("  ({col:5},{row:5})  {d:.6}   {deck_z:.6}   ({} rows)", row - top);
    }
}

//! Device creation and readback helpers, mostly for headless rendering and tests.

use crate::{LumaError, Result};

/// A headless GPU context: instance, adapter, device and queue.
pub struct HeadlessContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Create a device without a window. Uses the platform's default backend
/// (Metal on macOS and iOS, Vulkan on Linux, Android and Windows).
pub fn create_headless_context() -> Result<HeadlessContext> {
    pollster::block_on(create_headless_context_async())
}

pub async fn create_headless_context_async() -> Result<HeadlessContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| LumaError::Device(format!("no adapter: {e}")))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("deck.gl-native"),
            ..Default::default()
        })
        .await
        .map_err(|e| LumaError::Device(format!("no device: {e}")))?;
    Ok(HeadlessContext {
        instance,
        adapter,
        device,
        queue,
    })
}

/// Create a 2D texture usable as a render attachment and as a copy source.
pub fn create_render_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let is_depth = matches!(
        format,
        wgpu::TextureFormat::Depth24Plus
            | wgpu::TextureFormat::Depth32Float
            | wgpu::TextureFormat::Depth24PlusStencil8
            | wgpu::TextureFormat::Depth32FloatStencil8
    );
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if !is_depth {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// Read back an RGBA8 texture into tightly packed bytes (4 bytes per pixel, row major).
pub fn read_texture_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<u8>> {
    let size = texture.size();
    let width = size.width;
    let height = size.height;
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        size,
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| LumaError::Device(format!("poll failed: {e:?}")))?;
    rx.recv()
        .map_err(|_| LumaError::Device("map_async callback dropped".into()))?
        .map_err(|e| LumaError::Device(format!("map failed: {e:?}")))?;

    let mut out = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    {
        let view = slice
            .get_mapped_range()
            .map_err(|e| LumaError::Device(format!("mapped range: {e:?}")))?;
        for row in 0..height as usize {
            let start = row * padded_bytes_per_row as usize;
            out.extend_from_slice(&view[start..start + unpadded_bytes_per_row as usize]);
        }
    }
    buffer.unmap();
    Ok(out)
}

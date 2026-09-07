//! Vertex buffer helpers.

use wgpu::util::DeviceExt;

/// Split f64 values into high and low f32 parts, as deck.gl does for `fp64` attributes.
///
/// The shader reconstructs `hi + lo` in a way that preserves precision relative to an offset.
pub fn split_f64(values: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let split = |v: f64| {
        let h = v as f32;
        (h, (v - h as f64) as f32)
    };
    #[cfg(feature = "parallel")]
    if values.len() >= 65_536 {
        use rayon::prelude::*;
        return values.par_iter().map(|&v| split(v)).unzip();
    }
    values.iter().map(|&v| split(v)).unzip()
}

/// Create a vertex buffer from bytes.
pub fn create_vertex_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    crate::stats::count_upload(contents.len());
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &padded(contents),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

/// Upload vertex data into `existing` when it is large enough, and create a buffer
/// otherwise. Writing into a buffer in use by an earlier frame is safe: the queue orders the
/// write after that work.
pub fn write_or_create_vertex_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    existing: Option<&wgpu::Buffer>,
    label: &str,
    contents: &[u8],
) -> wgpu::Buffer {
    write_or_grow_vertex_buffer(device, queue, existing, label, contents, 0)
}

/// Like [`write_or_create_vertex_buffer`], but a buffer created here gets at least
/// `capacity` bytes, so later writes of more rows can go into the same buffer.
pub fn write_or_grow_vertex_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    existing: Option<&wgpu::Buffer>,
    label: &str,
    contents: &[u8],
    capacity: u64,
) -> wgpu::Buffer {
    let bytes = padded(contents);
    if let Some(buffer) = existing {
        if buffer.size() >= bytes.len() as u64 {
            crate::stats::count_upload(contents.len());
            queue.write_buffer(buffer, 0, &bytes);
            return buffer.clone();
        }
    }
    if capacity <= bytes.len() as u64 {
        return create_vertex_buffer(device, label, contents);
    }
    crate::stats::count_upload(contents.len());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: capacity.div_ceil(4) * 4,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

/// Write `contents` at a byte `offset` of a vertex buffer. The offset and the length must be
/// multiples of 4, as wgpu requires; nothing is written otherwise.
pub fn write_vertex_buffer_range(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, contents: &[u8]) {
    if contents.is_empty()
        || !offset.is_multiple_of(4)
        || !contents.len().is_multiple_of(4)
        || offset + contents.len() as u64 > buffer.size()
    {
        return;
    }
    crate::stats::count_upload(contents.len());
    queue.write_buffer(buffer, offset, contents);
}

/// Contents padded to what wgpu accepts: at least 16 bytes and a multiple of 4.
fn padded(contents: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if contents.is_empty() {
        std::borrow::Cow::Owned(vec![0u8; 16])
    } else if !contents.len().is_multiple_of(4) {
        let mut padded = contents.to_vec();
        padded.resize(contents.len().div_ceil(4) * 4, 0);
        std::borrow::Cow::Owned(padded)
    } else {
        std::borrow::Cow::Borrowed(contents)
    }
}

/// Create a vertex buffer from a slice of plain data.
pub fn create_vertex_buffer_from<T: bytemuck::NoUninit>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    create_vertex_buffer(device, label, bytemuck::cast_slice(data))
}

/// Create a 32-bit index buffer.
pub fn create_index_buffer(device: &wgpu::Device, label: &str, indices: &[u32]) -> wgpu::Buffer {
    crate::stats::count_upload(indices.len() * 4);
    let contents: &[u8] = if indices.is_empty() {
        &[0u8; 4]
    } else {
        bytemuck::cast_slice(indices)
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    })
}

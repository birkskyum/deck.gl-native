//! Vertex buffer helpers.

use wgpu::util::DeviceExt;

/// Split f64 values into high and low f32 parts, as deck.gl does for `fp64` attributes.
///
/// The shader reconstructs `hi + lo` in a way that preserves precision relative to an offset.
pub fn split_f64(values: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let mut hi = Vec::with_capacity(values.len());
    let mut lo = Vec::with_capacity(values.len());
    for &v in values {
        let h = v as f32;
        hi.push(h);
        lo.push((v - h as f64) as f32);
    }
    (hi, lo)
}

/// Create a vertex buffer from bytes.
pub fn create_vertex_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    // wgpu rejects zero sized buffers; keep a minimal placeholder so bindings stay valid.
    let mut padded;
    let contents = if contents.is_empty() {
        padded = vec![0u8; 16];
        padded.as_slice()
    } else if contents.len() % 4 != 0 {
        padded = contents.to_vec();
        padded.resize(contents.len().div_ceil(4) * 4, 0);
        padded.as_slice()
    } else {
        contents
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
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

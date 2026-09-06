//! Uniform blocks written by field name. Mirrors luma.gl's `UniformBlock` / `ShaderInputs`.

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::shader::{UniformKind, UniformLayout};
use crate::{LumaError, Result};

/// CPU shadow of a WGSL uniform struct plus the GPU buffer it is uploaded to.
#[derive(Debug)]
pub struct UniformBlock {
    layout: UniformLayout,
    data: Vec<u8>,
    buffer: wgpu::Buffer,
    dirty: bool,
}

fn round_up(n: u32, align: u32) -> u32 {
    n.div_ceil(align) * align
}

impl UniformBlock {
    pub fn new(device: &wgpu::Device, label: &str, layout: &UniformLayout) -> Self {
        let size = round_up(layout.size.max(16), 16);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            layout: layout.clone(),
            data: vec![0; size as usize],
            buffer,
            // Upload zeros on first use so the buffer is always initialized.
            dirty: true,
        }
    }

    pub fn layout(&self) -> &UniformLayout {
        &self.layout
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Write pending changes to the GPU. Must be called before the buffer is used in a pass.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            queue.write_buffer(&self.buffer, 0, &self.data);
            self.dirty = false;
        }
    }

    fn write(&mut self, name: &str, bytes: &[u8], kinds: &[UniformKind]) -> Result<()> {
        let field = self.layout.field(name).ok_or_else(|| {
            LumaError::Uniform(format!(
                "no field `{name}` in uniform struct `{}`",
                self.layout.struct_name
            ))
        })?;
        if !kinds.is_empty() && !kinds.contains(&field.kind) {
            return Err(LumaError::Uniform(format!(
                "field `{name}` in `{}` has kind {:?}, cannot write {:?}",
                self.layout.struct_name, field.kind, kinds
            )));
        }
        if bytes.len() > field.size as usize {
            return Err(LumaError::Uniform(format!(
                "field `{name}` in `{}` is {} bytes, cannot write {} bytes",
                self.layout.struct_name,
                field.size,
                bytes.len()
            )));
        }
        let start = field.offset as usize;
        self.data[start..start + bytes.len()].copy_from_slice(bytes);
        self.dirty = true;
        Ok(())
    }

    pub fn set_f32(&mut self, name: &str, value: f32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::F32])
    }

    pub fn set_i32(&mut self, name: &str, value: i32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::I32])
    }

    pub fn set_u32(&mut self, name: &str, value: u32) -> Result<()> {
        self.write(name, &value.to_le_bytes(), &[UniformKind::U32])
    }

    /// Write a boolean into an `f32`, `i32`, `u32` or `bool` field as 1 or 0.
    pub fn set_bool(&mut self, name: &str, value: bool) -> Result<()> {
        let kind = self
            .layout
            .field(name)
            .map(|f| f.kind)
            .ok_or_else(|| LumaError::Uniform(format!("no field `{name}`")))?;
        match kind {
            UniformKind::F32 => self.set_f32(name, if value { 1.0 } else { 0.0 }),
            UniformKind::I32 => self.set_i32(name, value as i32),
            UniformKind::U32 | UniformKind::Bool => self.write(
                name,
                &(value as u32).to_le_bytes(),
                &[UniformKind::U32, UniformKind::Bool],
            ),
            other => Err(LumaError::Uniform(format!(
                "field `{name}` has kind {other:?}, cannot write bool"
            ))),
        }
    }

    pub fn set_vec2(&mut self, name: &str, value: Vec2) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec2F])
    }

    pub fn set_vec3(&mut self, name: &str, value: Vec3) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec3F])
    }

    pub fn set_vec4(&mut self, name: &str, value: Vec4) -> Result<()> {
        self.write(name, bytemuck::bytes_of(&value.to_array()), &[UniformKind::Vec4F])
    }

    /// Column-major 4x4 matrix.
    pub fn set_mat4(&mut self, name: &str, value: Mat4) -> Result<()> {
        self.write(
            name,
            bytemuck::bytes_of(&value.to_cols_array()),
            &[UniformKind::Mat4F],
        )
    }

    /// Column-major 3x3 matrix, padded to WGSL's 16-byte column stride.
    pub fn set_mat3(&mut self, name: &str, value: Mat3) -> Result<()> {
        let mut padded = [0f32; 12];
        let cols = value.to_cols_array();
        for c in 0..3 {
            padded[c * 4..c * 4 + 3].copy_from_slice(&cols[c * 3..c * 3 + 3]);
        }
        self.write(name, bytemuck::bytes_of(&padded), &[UniformKind::Mat3F])
    }

    /// Raw write for nested structs and arrays.
    pub fn set_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.write(name, bytes, &[])
    }
}

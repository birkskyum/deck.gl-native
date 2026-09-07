//! Port of deck.gl's `AttributeManager`: a layer declares its vertex buffers once (which
//! attribute feeds which field, at which shader location and byte offset) and the manager
//! builds the buffers, uploads only those whose accessor or data changed, packs interleaved
//! buffers and uploads Arrow columns that already have the GPU layout without conversion.

use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt8Type};
use arrow_array::Array;
use luma_gl::buffer::create_vertex_buffer;
use luma_gl::{Model, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::data::{
    resolve_colors, resolve_f32, resolve_positions, resolve_vec2, resolve_with, Accessor, Color, LayerData,
    Position,
};
use crate::{DeckError, Result};

/// Where an attribute's values come from, typed.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeSource {
    /// Longitude, latitude and z (or x, y, z): high and low parts are available as fields
    Positions(Accessor<Position>),
    Colors(Accessor<Color>),
    Floats(Accessor<f32>),
    Vec2(Accessor<[f32; 2]>),
    Vec3(Accessor<[f32; 3]>),
    Vec4(Accessor<[f32; 4]>),
    /// The row of the layer's data each object comes from (picking)
    RowIndex,
}

/// Which part of a position a field holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The value itself (`f32` rounded for positions)
    High,
    /// The remainder of a position after rounding to `f32`
    Low,
}

/// One attribute inside a buffer.
#[derive(Clone, Debug)]
pub struct Field {
    /// Name of the source given to [`AttributeManager::update`]
    pub attribute: &'static str,
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u64,
    pub part: Part,
}

impl Field {
    pub fn new(attribute: &'static str, location: u32, format: VertexFormat, offset: u64) -> Self {
        Self {
            attribute,
            location,
            format,
            offset,
            part: Part::High,
        }
    }

    /// The low part of a position attribute.
    pub fn low(attribute: &'static str, location: u32, offset: u64) -> Self {
        Self {
            attribute,
            location,
            format: VertexFormat::Float32x3,
            offset,
            part: Part::Low,
        }
    }
}

/// A vertex buffer: one field, or several interleaved.
#[derive(Clone, Debug)]
pub struct BufferSpec {
    pub name: &'static str,
    pub step_mode: wgpu::VertexStepMode,
    pub stride: u64,
    pub fields: Vec<Field>,
}

fn format_size(format: VertexFormat) -> u64 {
    match format {
        VertexFormat::Float32 | VertexFormat::Uint32 | VertexFormat::Unorm8x4 | VertexFormat::Uint8x4 => 4,
        VertexFormat::Float32x2 => 8,
        VertexFormat::Float32x3 => 12,
        VertexFormat::Float32x4 => 16,
        other => other.size(),
    }
}

impl BufferSpec {
    /// A tightly packed buffer of one instance attribute.
    pub fn instance(
        name: &'static str,
        attribute: &'static str,
        location: u32,
        format: VertexFormat,
    ) -> Self {
        Self {
            name,
            step_mode: wgpu::VertexStepMode::Instance,
            stride: format_size(format),
            fields: vec![Field::new(attribute, location, format, 0)],
        }
    }

    /// The low part of a position attribute in its own buffer.
    pub fn instance_low(name: &'static str, attribute: &'static str, location: u32) -> Self {
        Self {
            name,
            step_mode: wgpu::VertexStepMode::Instance,
            stride: 12,
            fields: vec![Field::low(attribute, location, 0)],
        }
    }

    /// Several instance attributes interleaved with the given stride.
    pub fn interleaved(name: &'static str, stride: u64, fields: Vec<Field>) -> Self {
        Self {
            name,
            step_mode: wgpu::VertexStepMode::Instance,
            stride,
            fields,
        }
    }

    pub fn layout(&self) -> VertexBufferLayout {
        let attributes: Vec<(u32, VertexFormat, u64)> = self
            .fields
            .iter()
            .map(|f| (f.location, f.format, f.offset))
            .collect();
        VertexBufferLayout::interleaved(self.name, self.stride, self.step_mode, &attributes)
    }
}

/// Resolved values of one attribute, shared by the fields that read it.
enum Resolved {
    Positions(Vec<Position>),
    Colors(Vec<Color>),
    Floats(Vec<f32>),
    Vec2(Vec<[f32; 2]>),
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
    RowIndex,
}

/// Builds and updates a model's vertex buffers from attribute sources.
#[derive(Debug)]
pub struct AttributeManager {
    buffers: Vec<BufferSpec>,
    previous: HashMap<&'static str, AttributeSource>,
    previous_data: Option<LayerData>,
    force: bool,
}

impl AttributeManager {
    pub fn new(buffers: Vec<BufferSpec>) -> Self {
        Self {
            buffers,
            previous: HashMap::new(),
            previous_data: None,
            force: true,
        }
    }

    /// The vertex buffer layouts for the model, in declaration order.
    pub fn layouts(&self) -> Vec<VertexBufferLayout> {
        self.buffers.iter().map(BufferSpec::layout).collect()
    }

    /// Upload every buffer on the next update (after the model was recreated).
    pub fn invalidate_all(&mut self) {
        self.force = true;
    }

    /// Upload the buffers whose sources or data changed; returns how many were uploaded.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        model: &mut Model,
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
    ) -> Result<usize> {
        self.update_many(device, &mut [model], data, sources)
    }

    /// Like [`AttributeManager::update`] for several models sharing the buffers (a layer with
    /// a filled and a wireframe model, for instance).
    pub fn update_many(
        &mut self,
        device: &wgpu::Device,
        models: &mut [&mut Model],
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
    ) -> Result<usize> {
        let data_changed = self.force || self.previous_data.as_ref() != Some(data);
        let changed = |attribute: &str| -> bool {
            let now = sources
                .iter()
                .find(|(name, _)| *name == attribute)
                .map(|(_, s)| s);
            self.previous.get(attribute) != now
        };
        let mut resolved: HashMap<&'static str, Resolved> = HashMap::new();
        let mut uploaded = 0;
        for buffer in &self.buffers {
            if !data_changed && !buffer.fields.iter().any(|f| changed(f.attribute)) {
                continue;
            }
            let gpu_buffer = build_buffer(device, buffer, data, sources, &mut resolved)?;
            for model in models.iter_mut() {
                model.set_vertex_buffer(buffer.name, gpu_buffer.clone())?;
            }
            uploaded += 1;
        }
        self.previous = sources
            .iter()
            .map(|(name, source)| (*name, source.clone()))
            .collect();
        self.previous_data = Some(data.clone());
        self.force = false;
        Ok(uploaded)
    }
}

fn source_of<'a>(
    sources: &'a [(&'static str, AttributeSource)],
    attribute: &str,
) -> Result<&'a AttributeSource> {
    sources
        .iter()
        .find(|(name, _)| *name == attribute)
        .map(|(_, s)| s)
        .ok_or_else(|| DeckError::Data(format!("no source for attribute `{attribute}`")))
}

/// A `FixedSizeList<Float32, 3>` column's values, when the accessor reads one.
fn f32x3_column<'a>(data: &'a LayerData, accessor: &Accessor<Position>) -> Option<&'a [f32]> {
    let Accessor::Column(name) = accessor else {
        return None;
    };
    let column = data.column(name).ok()?;
    let list = column.as_fixed_size_list_opt()?;
    if list.value_length() != 3 || list.len() != data.len() {
        return None;
    }
    let values = list.values().as_primitive_opt::<Float32Type>()?;
    let start = list.offset() * 3;
    Some(&values.values()[start..start + list.len() * 3])
}

/// A `FixedSizeList<UInt8, 4>` column's bytes, when the accessor reads one.
fn u8x4_column<'a>(data: &'a LayerData, accessor: &Accessor<Color>) -> Option<&'a [u8]> {
    let Accessor::Column(name) = accessor else {
        return None;
    };
    let column = data.column(name).ok()?;
    let list = column.as_fixed_size_list_opt()?;
    if list.value_length() != 4 || list.len() != data.len() {
        return None;
    }
    let values = list.values().as_primitive_opt::<UInt8Type>()?;
    let start = list.offset() * 4;
    Some(&values.values()[start..start + list.len() * 4])
}

fn build_buffer(
    device: &wgpu::Device,
    spec: &BufferSpec,
    data: &LayerData,
    sources: &[(&'static str, AttributeSource)],
    resolved: &mut HashMap<&'static str, Resolved>,
) -> Result<wgpu::Buffer> {
    // Single field buffers can take an Arrow column's bytes as they are
    if let [field] = spec.fields.as_slice() {
        if field.offset == 0 && spec.stride == format_size(field.format) {
            match (source_of(sources, field.attribute)?, field.part) {
                (AttributeSource::Positions(accessor), Part::High) => {
                    if let Some(values) = f32x3_column(data, accessor) {
                        return Ok(create_vertex_buffer(
                            device,
                            spec.name,
                            bytemuck::cast_slice(values),
                        ));
                    }
                }
                (AttributeSource::Positions(accessor), Part::Low) => {
                    if f32x3_column(data, accessor).is_some() {
                        return Ok(create_vertex_buffer(
                            device,
                            spec.name,
                            &vec![0u8; data.len() * 12],
                        ));
                    }
                }
                (AttributeSource::Colors(accessor), _) => {
                    if let Some(bytes) = u8x4_column(data, accessor) {
                        return Ok(create_vertex_buffer(device, spec.name, bytes));
                    }
                }
                _ => {}
            }
        }
    }
    let rows = data.len();
    for field in &spec.fields {
        if !resolved.contains_key(field.attribute) {
            let values = match source_of(sources, field.attribute)? {
                AttributeSource::Positions(a) => Resolved::Positions(resolve_positions(data, a)?),
                AttributeSource::Colors(a) => Resolved::Colors(resolve_colors(data, a)?),
                AttributeSource::Floats(a) => Resolved::Floats(resolve_f32(data, a)?),
                AttributeSource::Vec2(a) => Resolved::Vec2(resolve_vec2(data, a)?),
                AttributeSource::Vec3(a) => Resolved::Vec3(resolve_with(data, a, |_| {
                    Err(DeckError::Data("vec3 columns are not supported yet".into()))
                })?),
                AttributeSource::Vec4(a) => Resolved::Vec4(resolve_with(data, a, |_| {
                    Err(DeckError::Data("vec4 columns are not supported yet".into()))
                })?),
                AttributeSource::RowIndex => Resolved::RowIndex,
            };
            resolved.insert(field.attribute, values);
        }
    }
    // One field: cast the typed values straight into the buffer
    if let [field] = spec.fields.as_slice() {
        if field.offset == 0 && spec.stride == format_size(field.format) {
            let values = &resolved[field.attribute];
            let buffer = |bytes: &[u8]| create_vertex_buffer(device, spec.name, bytes);
            return Ok(match (values, field.part) {
                (Resolved::Positions(p), Part::High) => buffer(bytemuck::cast_slice(&map_rows(p, high_part))),
                (Resolved::Positions(p), Part::Low) => buffer(bytemuck::cast_slice(&map_rows(p, low_part))),
                (Resolved::Colors(c), _) => buffer(bytemuck::cast_slice(c)),
                (Resolved::Floats(f), _) => buffer(bytemuck::cast_slice(f)),
                (Resolved::Vec2(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::Vec3(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::Vec4(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::RowIndex, _) => {
                    let indices: Vec<u32> = (0..rows).map(|i| data.source_row(i)).collect();
                    buffer(bytemuck::cast_slice(&indices))
                }
            });
        }
    }
    // Interleaved: write every field of a row into its stride, rows in parallel when large
    let stride = spec.stride as usize;
    let mut bytes = vec![0u8; rows * stride];
    let fields: Vec<(&Field, &Resolved, usize)> = spec
        .fields
        .iter()
        .map(|f| (f, &resolved[f.attribute], format_size(f.format) as usize))
        .collect();
    let write_row = |row: usize, chunk: &mut [u8]| {
        for (field, values, size) in &fields {
            let out = &mut chunk[field.offset as usize..field.offset as usize + size];
            match (values, field.part) {
                (Resolved::Positions(p), Part::High) => {
                    out.copy_from_slice(bytemuck::bytes_of(&high_part(p[row])))
                }
                (Resolved::Positions(p), Part::Low) => {
                    out.copy_from_slice(bytemuck::bytes_of(&low_part(p[row])))
                }
                (Resolved::Colors(c), _) => out.copy_from_slice(&c[row]),
                (Resolved::Floats(f), _) => out.copy_from_slice(&f[row].to_ne_bytes()),
                (Resolved::Vec2(v), _) => out.copy_from_slice(bytemuck::bytes_of(&v[row])),
                (Resolved::Vec3(v), _) => out.copy_from_slice(bytemuck::bytes_of(&v[row])),
                (Resolved::Vec4(v), _) => out.copy_from_slice(bytemuck::bytes_of(&v[row])),
                (Resolved::RowIndex, _) => out.copy_from_slice(&data.source_row(row).to_ne_bytes()),
            }
        }
    };
    if rows >= PARALLEL_ROWS {
        use rayon::prelude::*;
        bytes
            .par_chunks_exact_mut(stride)
            .enumerate()
            .for_each(|(row, chunk)| write_row(row, chunk));
    } else {
        for (row, chunk) in bytes.chunks_exact_mut(stride).enumerate() {
            write_row(row, chunk);
        }
    }
    Ok(create_vertex_buffer(device, spec.name, &bytes))
}

/// Rows above which packing runs on all cores.
const PARALLEL_ROWS: usize = 16_384;

fn high_part(p: Position) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

fn low_part(p: Position) -> [f32; 3] {
    [
        (p[0] - p[0] as f32 as f64) as f32,
        (p[1] - p[1] as f32 as f64) as f32,
        (p[2] - p[2] as f32 as f64) as f32,
    ]
}

fn map_rows(positions: &[Position], f: fn(Position) -> [f32; 3]) -> Vec<[f32; 3]> {
    if positions.len() >= PARALLEL_ROWS {
        use rayon::prelude::*;
        positions.par_iter().map(|p| f(*p)).collect()
    } else {
        positions.iter().map(|p| f(*p)).collect()
    }
}

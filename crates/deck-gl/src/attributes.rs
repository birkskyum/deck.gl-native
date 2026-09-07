//! Attribute buffers from accessors, with zero copy uploads for Arrow columns that already have
//! the GPU layout (deck.gl's binary attributes): `FixedSizeList<Float32, 3>` positions and
//! `FixedSizeList<UInt8, 4>` colours go straight from the column's buffer to the GPU, and
//! `Float64` positions are split into high and low parts without an intermediate vector.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, UInt8Type};
use arrow_array::Array;
use luma_gl::buffer::{create_vertex_buffer, create_vertex_buffer_from, split_f64};

use crate::data::{resolve_colors, resolve_positions, Accessor, Color, LayerData, Position};
use crate::Result;

/// High and low f32 position buffers for an accessor (three components per row).
pub fn position_buffers(
    device: &wgpu::Device,
    data: &LayerData,
    accessor: &Accessor<Position>,
    label: &str,
) -> Result<(wgpu::Buffer, wgpu::Buffer)> {
    let low_label = format!("{label}64Low");
    if let Accessor::Column(name) = accessor {
        let column = data.column(name)?;
        if let Some(list) = column.as_fixed_size_list_opt() {
            let width = list.value_length() as usize;
            let start = list.offset() * width;
            let end = start + list.len() * width;
            if width == 3 && list.len() == data.len() {
                if let Some(values) = list.values().as_primitive_opt::<Float32Type>() {
                    // Already the GPU layout: upload the column's bytes as they are
                    let slice = &values.values()[start..end];
                    let zeros = vec![0u8; slice.len() * 4];
                    return Ok((
                        create_vertex_buffer_from(device, label, slice),
                        create_vertex_buffer(device, &low_label, &zeros),
                    ));
                }
                if let Some(values) = list.values().as_primitive_opt::<Float64Type>() {
                    let (hi, lo) = split_f64(&values.values()[start..end]);
                    return Ok((
                        create_vertex_buffer_from(device, label, &hi),
                        create_vertex_buffer_from(device, &low_label, &lo),
                    ));
                }
            }
        }
    }
    let positions = resolve_positions(data, accessor)?;
    let flat: Vec<f64> = positions.iter().flatten().copied().collect();
    let (hi, lo) = split_f64(&flat);
    Ok((
        create_vertex_buffer_from(device, label, &hi),
        create_vertex_buffer_from(device, &low_label, &lo),
    ))
}

/// An RGBA8 colour buffer for an accessor.
pub fn color_buffer(
    device: &wgpu::Device,
    data: &LayerData,
    accessor: &Accessor<Color>,
    label: &str,
) -> Result<wgpu::Buffer> {
    if let Accessor::Column(name) = accessor {
        let column = data.column(name)?;
        if let Some(list) = column.as_fixed_size_list_opt() {
            if list.value_length() == 4 && list.len() == data.len() {
                if let Some(values) = list.values().as_primitive_opt::<UInt8Type>() {
                    let start = list.offset() * 4;
                    let end = start + list.len() * 4;
                    return Ok(create_vertex_buffer(device, label, &values.values()[start..end]));
                }
            }
        }
    }
    let colors = resolve_colors(data, accessor)?;
    Ok(create_vertex_buffer_from(device, label, &colors))
}

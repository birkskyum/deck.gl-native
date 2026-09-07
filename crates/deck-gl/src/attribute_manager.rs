//! Port of deck.gl's `AttributeManager`: a layer declares its vertex buffers once (which
//! attribute feeds which field, at which shader location and byte offset) and the manager
//! builds the buffers, uploads only those whose accessor or data changed, packs interleaved
//! buffers and uploads Arrow columns that already have the GPU layout without conversion.
//! Data that names its changed rows ([`LayerData::changed_rows`]) is written in place for
//! those rows only, and buffers grow with headroom when rows are appended.

use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt8Type};
use arrow_array::Array;
use luma_gl::buffer::{
    write_or_create_vertex_buffer, write_or_grow_vertex_buffer, write_vertex_buffer_range,
};
use luma_gl::{Model, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::data::{
    resolve_colors, resolve_f32, resolve_positions, resolve_vec2, resolve_with, Accessor, Color, LayerData,
    Position,
};
use crate::layer::{position_bounds, union_bounds};
use crate::transition::{PropTransition, PropTransitions};
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

impl AttributeSource {
    /// Whether every row reads the same value, so one element can stand for all of them.
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Positions(a) => matches!(a, Accessor::Constant(_)),
            Self::Colors(a) => matches!(a, Accessor::Constant(_)),
            Self::Floats(a) => matches!(a, Accessor::Constant(_)),
            Self::Vec2(a) => matches!(a, Accessor::Constant(_)),
            Self::Vec3(a) => matches!(a, Accessor::Constant(_)),
            Self::Vec4(a) => matches!(a, Accessor::Constant(_)),
            Self::RowIndex => false,
        }
    }
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

    /// One attribute per vertex in its own buffer (for layers whose attributes are expanded per
    /// tessellated vertex).
    pub fn vertex(name: &'static str, attribute: &'static str, location: u32, format: VertexFormat) -> Self {
        Self {
            name,
            step_mode: wgpu::VertexStepMode::Vertex,
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
#[derive(Clone, Debug)]
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
    /// Bounds of every position attribute resolved so far
    position_bounds: HashMap<&'static str, Option<[f64; 4]>>,
    /// The GPU buffer of every spec, written into again when its size does not change
    gpu: HashMap<&'static str, wgpu::Buffer>,
    /// deck.gl's `transitions` for the attributes
    transitions: PropTransitions,
    /// The values of every attribute as last uploaded, kept while transitions are configured
    last_values: HashMap<&'static str, Resolved>,
    animations: HashMap<&'static str, AttributeAnimation>,
    time: f64,
    /// Buffers whose fields all read constant accessors: deck.gl's constant attributes. They
    /// hold one element and their layout has a zero stride, so every instance reads it.
    constant: HashSet<&'static str>,
}

/// A running attribute transition: values move from `from` to `to`.
#[derive(Debug)]
struct AttributeAnimation {
    transition: PropTransition,
    shape: Shape,
    from: Vec<f64>,
    to: Vec<f64>,
    current: Vec<f64>,
    velocity: Vec<f64>,
    start: f64,
    last_step: f64,
}

/// The typed layout of a flattened attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Positions,
    Colors,
    Floats,
    Vec2,
    Vec3,
    Vec4,
}

impl AttributeAnimation {
    /// The values at `time`; `false` once settled on the target.
    fn advance(&mut self, time: f64) -> bool {
        match self.transition {
            PropTransition::Interpolation { duration_ms, easing } => {
                let t = if duration_ms > 0.0 {
                    ((time - self.start) * 1000.0 / duration_ms).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let eased = easing.function()(t).clamp(0.0, 1.0);
                lerp_into(&mut self.current, &self.from, &self.to, eased);
                t < 1.0
            }
            PropTransition::Spring { stiffness, damping } => {
                if time == self.last_step {
                    return true;
                }
                self.last_step = time;
                let mut settled = true;
                for ((current, velocity), to) in self.current.iter_mut().zip(&mut self.velocity).zip(&self.to)
                {
                    *velocity = *velocity * damping + (to - *current) * stiffness;
                    *current += *velocity;
                    if (to - *current).abs() > 1e-6 || velocity.abs() > 1e-6 {
                        settled = false;
                    }
                }
                if settled {
                    self.current.clone_from(&self.to);
                }
                !settled
            }
        }
    }
}

fn lerp_into(out: &mut [f64], from: &[f64], to: &[f64], t: f64) {
    if out.len() >= PARALLEL_ROWS {
        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(from.par_iter())
            .zip(to.par_iter())
            .for_each(|((o, a), b)| *o = a + (b - a) * t);
    } else {
        for ((o, a), b) in out.iter_mut().zip(from).zip(to) {
            *o = a + (b - a) * t;
        }
    }
}

impl Resolved {
    /// The values flattened to numbers, for interpolation.
    fn components(&self) -> Option<(Shape, Vec<f64>)> {
        Some(match self {
            Self::Positions(v) => (Shape::Positions, v.iter().flatten().copied().collect()),
            Self::Colors(v) => (Shape::Colors, v.iter().flatten().map(|c| *c as f64).collect()),
            Self::Floats(v) => (Shape::Floats, v.iter().map(|f| *f as f64).collect()),
            Self::Vec2(v) => (Shape::Vec2, v.iter().flatten().map(|f| *f as f64).collect()),
            Self::Vec3(v) => (Shape::Vec3, v.iter().flatten().map(|f| *f as f64).collect()),
            Self::Vec4(v) => (Shape::Vec4, v.iter().flatten().map(|f| *f as f64).collect()),
            Self::RowIndex => return None,
        })
    }

    /// Values back from their flattened numbers.
    fn from_components(shape: Shape, values: &[f64]) -> Self {
        match shape {
            Shape::Positions => Self::Positions(values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()),
            Shape::Colors => Self::Colors(
                values
                    .chunks_exact(4)
                    .map(|c| [c[0], c[1], c[2], c[3]].map(|v| v.round().clamp(0.0, 255.0) as u8))
                    .collect(),
            ),
            Shape::Floats => Self::Floats(values.iter().map(|v| *v as f32).collect()),
            Shape::Vec2 => Self::Vec2(
                values
                    .chunks_exact(2)
                    .map(|c| [c[0] as f32, c[1] as f32])
                    .collect(),
            ),
            Shape::Vec3 => Self::Vec3(
                values
                    .chunks_exact(3)
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                    .collect(),
            ),
            Shape::Vec4 => Self::Vec4(
                values
                    .chunks_exact(4)
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
                    .collect(),
            ),
        }
    }
}

impl AttributeManager {
    pub fn new(buffers: Vec<BufferSpec>) -> Self {
        Self {
            buffers,
            previous: HashMap::new(),
            previous_data: None,
            force: true,
            position_bounds: HashMap::new(),
            gpu: HashMap::new(),
            transitions: PropTransitions::default(),
            last_values: HashMap::new(),
            animations: HashMap::new(),
            time: 0.0,
            constant: HashSet::new(),
        }
    }

    /// deck.gl's `transitions` prop: the attributes named in it (by accessor name, `getRadius`)
    /// animate from their old to their new values when their source changes.
    pub fn set_transitions(&mut self, transitions: &PropTransitions) {
        if self.transitions != *transitions {
            self.transitions = transitions.clone();
            if transitions.is_empty() {
                self.last_values.clear();
                self.animations.clear();
            }
        }
    }

    /// Whether an attribute is still moving towards its values.
    pub fn in_transition(&self) -> bool {
        !self.animations.is_empty()
    }

    /// Advance the running attribute transitions to `time` and upload the buffers they touch.
    /// Returns whether any is still running.
    pub fn animate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        models: &mut [&mut Model],
        data: &LayerData,
        time: f64,
    ) -> Result<bool> {
        if self.animations.is_empty() {
            return Ok(false);
        }
        let mut resolved: HashMap<&'static str, Resolved> = HashMap::new();
        let mut finished = Vec::new();
        for (name, animation) in &mut self.animations {
            let running = animation.advance(time);
            let values = Resolved::from_components(animation.shape, &animation.current);
            resolved.insert(name, values);
            if !running {
                finished.push(*name);
            }
        }
        let animated: Vec<&'static str> = resolved.keys().copied().collect();
        for buffer in &self.buffers {
            if !buffer.fields.iter().any(|f| animated.contains(&f.attribute)) {
                continue;
            }
            // the other attributes of the buffer come from the last upload
            for field in &buffer.fields {
                if let Some(values) = self.last_values.get(field.attribute) {
                    resolved.entry(field.attribute).or_insert_with(|| values.clone());
                }
            }
            let sources: Vec<(&'static str, AttributeSource)> = self
                .previous
                .iter()
                .map(|(name, source)| (*name, source.clone()))
                .collect();
            let existing = self.gpu.get(buffer.name);
            let upload =
                |bytes: &[u8]| write_or_create_vertex_buffer(device, queue, existing, buffer.name, bytes);
            let gpu_buffer = build_buffer(&upload, buffer, data, &sources, &mut resolved, None)?;
            for model in models.iter_mut() {
                model.set_vertex_buffer(buffer.name, gpu_buffer.clone())?;
            }
            self.gpu.insert(buffer.name, gpu_buffer);
        }
        for name in &animated {
            if let Some(values) = resolved.remove(name) {
                self.last_values.insert(name, values);
            }
        }
        for name in finished {
            self.animations.remove(name);
        }
        Ok(!self.animations.is_empty())
    }

    /// The bounds of the position attributes, `[min x, min y, max x, max y]`, once they were
    /// resolved (Arrow columns uploaded without conversion are not scanned).
    pub fn bounds(&self) -> Option<[f64; 4]> {
        self.position_bounds
            .values()
            .fold(None, |bounds, b| union_bounds(bounds, *b))
    }

    /// Add buffers (extension attributes, typically) after construction.
    pub fn extend(&mut self, buffers: Vec<BufferSpec>) {
        self.buffers.extend(buffers);
        self.force = true;
    }

    /// The vertex buffer layouts for the model, in declaration order.
    /// Which buffers can hold one element instead of one per row: deck.gl's constant
    /// attributes. A buffer qualifies when every field of it reads a constant accessor and
    /// no transition animates one of them.
    fn constant_buffers(&self, sources: &[(&'static str, AttributeSource)]) -> HashSet<&'static str> {
        let mut constant = HashSet::new();
        for buffer in &self.buffers {
            let all_constant = buffer.fields.iter().all(|field| {
                self.transitions.for_attribute(field.attribute).is_none()
                    && sources
                        .iter()
                        .find(|(name, _)| *name == field.attribute)
                        .is_some_and(|(_, source)| source.is_constant())
            });
            if all_constant && !buffer.fields.is_empty() {
                constant.insert(buffer.name);
            }
        }
        constant
    }

    /// Work out which buffers are constant for `sources` and keep it; the layouts and the
    /// model must be built after this. Returns whether the answer changed.
    pub fn plan(&mut self, sources: &[(&'static str, AttributeSource)]) -> bool {
        let constant = self.constant_buffers(sources);
        let changed = constant != self.constant;
        if changed {
            self.constant = constant;
            self.force = true;
        }
        changed
    }

    /// Whether [`plan`](Self::plan) would change the layouts, so the layer needs a new model.
    pub fn plan_changed(&self, sources: &[(&'static str, AttributeSource)]) -> bool {
        self.constant_buffers(sources) != self.constant
    }

    /// Whether a buffer holds a single element read by every instance.
    pub fn is_constant(&self, buffer: &str) -> bool {
        self.constant.contains(buffer)
    }

    pub fn layouts(&self) -> Vec<VertexBufferLayout> {
        self.buffers
            .iter()
            .map(|buffer| {
                let mut layout = buffer.layout();
                // A constant buffer holds one element every instance reads
                if self.constant.contains(buffer.name) {
                    layout.stride = 0;
                }
                layout
            })
            .collect()
    }

    /// Upload every buffer on the next update (after the model was recreated).
    pub fn invalidate_all(&mut self) {
        self.force = true;
    }

    /// The time of the frame, what a transition that starts in the next update begins at.
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Upload the buffers whose sources or data changed; returns how many were uploaded.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &mut Model,
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
    ) -> Result<usize> {
        self.update_many(device, queue, &mut [model], data, sources)
    }

    /// Like [`AttributeManager::update`] for several models sharing the buffers (a layer with
    /// a filled and a wireframe model, for instance).
    pub fn update_many(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        models: &mut [&mut Model],
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
    ) -> Result<usize> {
        self.update_impl(device, queue, models, data, sources, None)
    }

    /// Build every buffer with one entry per element of `expand`, the data row of each
    /// tessellated vertex or segment (deck.gl's `startIndices` expansion), for layers whose
    /// geometry is built by hand. Always rebuilds, so call it when the geometry was.
    pub fn update_expanded(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        models: &mut [&mut Model],
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
        expand: &[u32],
    ) -> Result<usize> {
        self.force = true;
        self.update_impl(device, queue, models, data, sources, Some(expand))
    }

    /// The rows to write in place, when the data names its changed rows and everything else
    /// allows it: the same row mapping as before, no expansion, no transitions and a length
    /// that did not shrink below the start of the range.
    fn incremental_rows(&self, data: &LayerData, expand: Option<&[u32]>) -> Option<std::ops::Range<usize>> {
        let rows = data.changed_rows.clone()?;
        let previous = self.previous_data.as_ref()?;
        if self.force || expand.is_some() || !self.transitions.is_empty() || self.gpu.is_empty() {
            return None;
        }
        let same_mapping = match (&previous.source_rows, &data.source_rows) {
            (None, None) => true,
            (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
            _ => false,
        };
        if !same_mapping || previous.row_offset != data.row_offset {
            return None;
        }
        // Rows before the range must exist in both, appended rows must be in the range
        if rows.start > previous.len() || rows.start > data.len() {
            return None;
        }
        if data.len() > previous.len() && rows.end < data.len() {
            return None;
        }
        Some(rows.start..rows.end.min(data.len()))
    }

    fn update_impl(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        models: &mut [&mut Model],
        data: &LayerData,
        sources: &[(&'static str, AttributeSource)],
        expand: Option<&[u32]>,
    ) -> Result<usize> {
        let incremental = self.incremental_rows(data, expand);
        let data_changed = self.force || incremental.is_none() && self.previous_data.as_ref() != Some(data);
        let changed = |attribute: &str| -> bool {
            let now = sources
                .iter()
                .find(|(name, _)| *name == attribute)
                .map(|(_, s)| s);
            self.previous.get(attribute) != now
        };
        let mut resolved: HashMap<&'static str, Resolved> = HashMap::new();
        let mut uploaded = 0;
        let same_rows = self.previous_data.as_ref().is_some_and(|d| d.len() == data.len());
        // Attributes with a transition start moving from their last values instead of jumping
        if !self.transitions.is_empty() && same_rows && expand.is_none() {
            let mut starting: Vec<(&'static str, AttributeAnimation)> = Vec::new();
            for buffer in &self.buffers {
                for field in &buffer.fields {
                    let name = field.attribute;
                    if starting.iter().any(|(n, _)| *n == name) || !changed(name) && !data_changed {
                        continue;
                    }
                    let (Some(transition), Some(last)) =
                        (self.transitions.for_attribute(name), self.last_values.get(name))
                    else {
                        continue;
                    };
                    let Some((shape, from)) = self
                        .animations
                        .get(name)
                        .map(|a| (a.shape, a.current.clone()))
                        .or_else(|| last.components())
                    else {
                        continue;
                    };
                    let source = source_of(sources, name)?;
                    let target = resolve_source(data, source)?;
                    let Some((target_shape, to)) = target.components() else {
                        continue;
                    };
                    if target_shape != shape || to.len() != from.len() {
                        continue;
                    }
                    let velocity = self
                        .animations
                        .get(name)
                        .map(|a| a.velocity.clone())
                        .unwrap_or_else(|| vec![0.0; from.len()]);
                    resolved.insert(name, Resolved::from_components(shape, &from));
                    starting.push((
                        name,
                        AttributeAnimation {
                            transition,
                            shape,
                            current: from.clone(),
                            from,
                            to,
                            velocity,
                            start: self.time,
                            last_step: f64::NAN,
                        },
                    ));
                }
            }
            for (name, animation) in starting {
                self.animations.insert(name, animation);
            }
        }
        let mut partial: HashMap<&'static str, Resolved> = HashMap::new();
        // Constant buffers resolve their single value from the first row
        let mut constant_resolved: HashMap<&'static str, Resolved> = HashMap::new();
        let first_row = if self.constant.is_empty() {
            LayerData::default()
        } else {
            data.slice(0..1)
        };
        for buffer in &self.buffers {
            let is_constant = self.constant.contains(buffer.name);
            let fields_changed = buffer.fields.iter().any(|f| changed(f.attribute));
            if is_constant {
                if !data_changed && !fields_changed {
                    continue;
                }
                let existing = self.gpu.get(buffer.name);
                let upload =
                    |bytes: &[u8]| write_or_create_vertex_buffer(device, queue, existing, buffer.name, bytes);
                let gpu_buffer =
                    build_buffer(&upload, buffer, &first_row, sources, &mut constant_resolved, None)?;
                for model in models.iter_mut() {
                    model.set_vertex_buffer(buffer.name, gpu_buffer.clone())?;
                }
                self.gpu.insert(buffer.name, gpu_buffer);
                uploaded += 1;
                continue;
            }
            let existing = self.gpu.get(buffer.name);
            let needed = data.len() as u64 * buffer.stride;
            // Changed rows go into the buffer the model already has when they fit
            if let (Some(rows), Some(existing), false) = (&incremental, existing, fields_changed) {
                if existing.size() < needed {
                    // Appends past the capacity: rebuild below with room for more
                } else {
                    if !rows.is_empty() {
                        let slice = data.slice(rows.clone());
                        let offset = rows.start as u64 * buffer.stride;
                        let upload = |bytes: &[u8]| {
                            write_vertex_buffer_range(queue, existing, offset, bytes);
                            existing.clone()
                        };
                        build_buffer(&upload, buffer, &slice, sources, &mut partial, None)?;
                        uploaded += 1;
                    }
                    continue;
                }
            }
            if !data_changed && !fields_changed && incremental.is_none() {
                continue;
            }
            let capacity = match (&incremental, existing) {
                (Some(_), Some(existing)) => needed.max(existing.size() * 2),
                _ => 0,
            };
            let upload = |bytes: &[u8]| {
                write_or_grow_vertex_buffer(device, queue, existing, buffer.name, bytes, capacity)
            };
            let gpu_buffer = build_buffer(&upload, buffer, data, sources, &mut resolved, expand)?;
            for model in models.iter_mut() {
                model.set_vertex_buffer(buffer.name, gpu_buffer.clone())?;
            }
            self.gpu.insert(buffer.name, gpu_buffer);
            uploaded += 1;
        }
        // Rows written in place widen the bounds; a full resolve replaces them below
        for (name, values) in &partial {
            if let Resolved::Positions(positions) = values {
                let bounds = union_bounds(
                    self.position_bounds.get(name).copied().flatten(),
                    parallel_bounds(positions),
                );
                self.position_bounds.insert(name, bounds);
            }
        }
        for (name, values) in &resolved {
            if let Resolved::Positions(positions) = values {
                self.position_bounds.insert(name, parallel_bounds(positions));
            }
        }
        if self.transitions.is_empty() {
            self.last_values.clear();
        } else {
            if !same_rows {
                self.last_values.clear();
                self.animations.clear();
            }
            for (name, values) in resolved {
                self.last_values.insert(name, values);
            }
        }
        self.previous = sources
            .iter()
            .map(|(name, source)| (*name, source.clone()))
            .collect();
        // Without the hint, so the same data sent again without one changes nothing
        self.previous_data = Some(LayerData {
            changed_rows: None,
            ..data.clone()
        });
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
    upload: &dyn Fn(&[u8]) -> wgpu::Buffer,
    spec: &BufferSpec,
    data: &LayerData,
    sources: &[(&'static str, AttributeSource)],
    resolved: &mut HashMap<&'static str, Resolved>,
    expand: Option<&[u32]>,
) -> Result<wgpu::Buffer> {
    // Single field buffers can take an Arrow column's bytes as they are
    if let [field] = spec.fields.as_slice() {
        if expand.is_none()
            && !resolved.contains_key(field.attribute)
            && field.offset == 0
            && spec.stride == format_size(field.format)
        {
            match (source_of(sources, field.attribute)?, field.part) {
                (AttributeSource::Positions(accessor), Part::High) => {
                    if let Some(values) = f32x3_column(data, accessor) {
                        return Ok(upload(bytemuck::cast_slice(values)));
                    }
                }
                (AttributeSource::Positions(accessor), Part::Low) => {
                    if f32x3_column(data, accessor).is_some() {
                        return Ok(upload(&vec![0u8; data.len() * 12]));
                    }
                }
                (AttributeSource::Colors(accessor), _) => {
                    if let Some(bytes) = u8x4_column(data, accessor) {
                        return Ok(upload(bytes));
                    }
                }
                _ => {}
            }
        }
    }
    let rows = expand.map_or(data.len(), <[u32]>::len);
    let source_row = |row: usize| data.source_row(expand.map_or(row, |e| e[row] as usize));
    for field in &spec.fields {
        if !resolved.contains_key(field.attribute) {
            let values = resolve_source(data, source_of(sources, field.attribute)?)?;
            let values = match expand {
                Some(indices) => values.gather(indices),
                None => values,
            };
            resolved.insert(field.attribute, values);
        }
    }
    // One field: cast the typed values straight into the buffer
    if let [field] = spec.fields.as_slice() {
        if field.offset == 0 && spec.stride == format_size(field.format) {
            let values = &resolved[field.attribute];
            let buffer = |bytes: &[u8]| upload(bytes);
            return Ok(match (values, field.part) {
                (Resolved::Positions(p), Part::High) => buffer(bytemuck::cast_slice(&map_rows(p, high_part))),
                (Resolved::Positions(p), Part::Low) => buffer(bytemuck::cast_slice(&map_rows(p, low_part))),
                (Resolved::Colors(c), _) => buffer(bytemuck::cast_slice(c)),
                (Resolved::Floats(f), _) => buffer(bytemuck::cast_slice(f)),
                (Resolved::Vec2(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::Vec3(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::Vec4(v), _) => buffer(bytemuck::cast_slice(v)),
                (Resolved::RowIndex, _) => {
                    let indices: Vec<u32> = (0..rows).map(source_row).collect();
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
                (Resolved::RowIndex, _) => out.copy_from_slice(&source_row(row).to_ne_bytes()),
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
    Ok(upload(&bytes))
}

/// The values of a source for every row of the data.
fn resolve_source(data: &LayerData, source: &AttributeSource) -> Result<Resolved> {
    Ok(match source {
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
    })
}

/// Rows above which packing runs on all cores.
const PARALLEL_ROWS: usize = 16_384;

/// The bounds of many positions, folded on all cores when there are many.
fn parallel_bounds(positions: &[Position]) -> Option<[f64; 4]> {
    if positions.len() < PARALLEL_ROWS {
        return position_bounds(positions.iter());
    }
    use rayon::prelude::*;
    positions
        .par_chunks(PARALLEL_ROWS)
        .map(|chunk| position_bounds(chunk.iter()))
        .reduce(|| None, union_bounds)
}

impl Resolved {
    /// The values of the rows in `indices`, in that order (out of range rows take the last).
    fn gather(self, indices: &[u32]) -> Self {
        fn pick<T: Copy>(values: &[T], indices: &[u32]) -> Vec<T> {
            let Some(last) = values.len().checked_sub(1) else {
                return Vec::new();
            };
            indices.iter().map(|&i| values[(i as usize).min(last)]).collect()
        }
        match self {
            Self::Positions(v) => Self::Positions(pick(&v, indices)),
            Self::Colors(v) => Self::Colors(pick(&v, indices)),
            Self::Floats(v) => Self::Floats(pick(&v, indices)),
            Self::Vec2(v) => Self::Vec2(pick(&v, indices)),
            Self::Vec3(v) => Self::Vec3(pick(&v, indices)),
            Self::Vec4(v) => Self::Vec4(pick(&v, indices)),
            Self::RowIndex => Self::RowIndex,
        }
    }
}

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

//! `Model`: a render pipeline together with its uniform blocks, bind group and vertex buffers.
//! Mirrors luma.gl's `Model` at a much smaller scale.

use std::collections::HashMap;

use crate::shader::AssembledShader;
use crate::uniform::UniformBlock;
use crate::{LumaError, Result};

/// Format of the attachments a model will render into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderTarget {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub sample_count: u32,
}

impl Default for RenderTarget {
    fn default() -> Self {
        Self {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            depth_format: Some(wgpu::TextureFormat::Depth24Plus),
            sample_count: 1,
        }
    }
}

/// One vertex buffer slot.
#[derive(Clone, Debug)]
pub struct VertexBufferLayout {
    pub name: &'static str,
    pub stride: u64,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl VertexBufferLayout {
    /// A single attribute occupying the whole buffer slot.
    pub fn single(
        name: &'static str,
        location: u32,
        format: wgpu::VertexFormat,
        step_mode: wgpu::VertexStepMode,
    ) -> Self {
        Self {
            name,
            stride: format.size(),
            step_mode,
            attributes: vec![wgpu::VertexAttribute {
                format,
                offset: 0,
                shader_location: location,
            }],
        }
    }

    pub fn vertex(name: &'static str, location: u32, format: wgpu::VertexFormat) -> Self {
        Self::single(name, location, format, wgpu::VertexStepMode::Vertex)
    }

    pub fn instance(name: &'static str, location: u32, format: wgpu::VertexFormat) -> Self {
        Self::single(name, location, format, wgpu::VertexStepMode::Instance)
    }

    /// Several attributes interleaved in one buffer. `attributes` are (location, format, byte offset).
    pub fn interleaved(
        name: &'static str,
        stride: u64,
        step_mode: wgpu::VertexStepMode,
        attributes: &[(u32, wgpu::VertexFormat, u64)],
    ) -> Self {
        Self {
            name,
            stride,
            step_mode,
            attributes: attributes
                .iter()
                .map(|(location, format, offset)| wgpu::VertexAttribute {
                    format: *format,
                    offset: *offset,
                    shader_location: *location,
                })
                .collect(),
        }
    }
}

/// Everything needed to build a [`Model`].
pub struct ModelDescriptor<'a> {
    pub label: &'a str,
    pub shader: &'a AssembledShader,
    pub vertex_layouts: &'a [VertexBufferLayout],
    pub topology: wgpu::PrimitiveTopology,
    pub target: RenderTarget,
    pub blend: Option<wgpu::BlendState>,
    pub depth_write_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
    /// Constant and slope-scaled depth bias, in the units of `wgpu::DepthBiasState`.
    pub depth_bias: wgpu::DepthBiasState,
    pub cull_mode: Option<wgpu::Face>,
    /// Also build a picking pipeline: same shader, an RGBA8 target, and alpha taken from the
    /// blend constant so a pass can tag every layer with an id. See [`Model::draw_picking`].
    pub pickable: bool,
}

impl<'a> ModelDescriptor<'a> {
    /// deck.gl's default layer parameters: premultiplied alpha blending, depth test less-equal.
    pub fn new(
        label: &'a str,
        shader: &'a AssembledShader,
        vertex_layouts: &'a [VertexBufferLayout],
        topology: wgpu::PrimitiveTopology,
        target: RenderTarget,
    ) -> Self {
        Self {
            label,
            shader,
            vertex_layouts,
            topology,
            target,
            blend: Some(premultiplied_alpha_blend()),
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            depth_bias: wgpu::DepthBiasState::default(),
            cull_mode: None,
            pickable: false,
        }
    }
}

/// Format of the picking target every picking pipeline renders into.
pub const PICKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Blend state of deck.gl's picking pass: color written as is, alpha replaced by the blend
/// constant, which the pass sets to the layer's id.
pub fn picking_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// `one, one-minus-src-alpha` for both color and alpha, matching deck.gl's layers pass.
pub fn premultiplied_alpha_blend() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

#[derive(Debug)]
struct IndexBuffer {
    buffer: wgpu::Buffer,
    format: wgpu::IndexFormat,
    count: u32,
}

/// A drawable: pipeline, uniform blocks, bind group and vertex buffers.
#[derive(Debug)]
pub struct Model {
    label: String,
    pipeline: wgpu::RenderPipeline,
    picking_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    uniforms: HashMap<String, UniformBlock>,
    vertex_slots: Vec<Option<wgpu::Buffer>>,
    slot_names: Vec<&'static str>,
    index_buffer: Option<IndexBuffer>,
    vertex_count: u32,
    instance_count: u32,
}

impl Model {
    pub fn new(device: &wgpu::Device, desc: &ModelDescriptor<'_>) -> Result<Self> {
        let shader = desc.shader;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(desc.label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&shader.wgsl)),
        });

        // Uniform blocks and the bind group layout are derived from the shader.
        let mut uniforms = HashMap::new();
        let mut layout_entries = Vec::new();
        let mut group_entries = Vec::new();
        for binding in &shader.uniforms {
            let block = UniformBlock::new(
                device,
                &format!("{}:{}", desc.label, binding.name),
                &binding.layout,
            );
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            uniforms.insert(binding.name.clone(), block);
        }
        for binding in &shader.uniforms {
            let block = &uniforms[&binding.name];
            group_entries.push(wgpu::BindGroupEntry {
                binding: binding.binding,
                resource: block.buffer().as_entire_binding(),
            });
        }
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(desc.label),
            entries: &layout_entries,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.label),
            layout: &bind_group_layout,
            entries: &group_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(desc.label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = desc
            .vertex_layouts
            .iter()
            .map(|l| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: l.stride,
                    step_mode: l.step_mode,
                    attributes: &l.attributes,
                })
            })
            .collect();

        let make_pipeline = |label: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            let color_target = wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vertexMain"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: desc.topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: desc.cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: desc.target.depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(desc.depth_write_enabled),
                    depth_compare: Some(desc.depth_compare),
                    stencil: wgpu::StencilState::default(),
                    bias: desc.depth_bias,
                }),
                multisample: wgpu::MultisampleState {
                    count: desc.target.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fragmentMain"),
                    compilation_options: Default::default(),
                    targets: &[Some(color_target)],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline(desc.label, desc.target.color_format, desc.blend);
        let picking_pipeline = desc.pickable.then(|| {
            make_pipeline(
                &format!("{}:picking", desc.label),
                PICKING_FORMAT,
                Some(picking_blend()),
            )
        });

        Ok(Self {
            label: desc.label.to_string(),
            pipeline,
            picking_pipeline,
            bind_group,
            uniforms,
            vertex_slots: vec![None; desc.vertex_layouts.len()],
            slot_names: desc.vertex_layouts.iter().map(|l| l.name).collect(),
            index_buffer: None,
            vertex_count: 0,
            instance_count: 1,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Mutable access to a uniform block by its WGSL variable name (for example `project`).
    pub fn uniforms(&mut self, name: &str) -> Result<&mut UniformBlock> {
        self.uniforms
            .get_mut(name)
            .ok_or_else(|| LumaError::Uniform(format!("{}: no uniform block `{name}`", self.label)))
    }

    pub fn has_uniforms(&self, name: &str) -> bool {
        self.uniforms.contains_key(name)
    }

    /// Upload every dirty uniform block. Call before encoding the render pass.
    pub fn upload_uniforms(&mut self, queue: &wgpu::Queue) {
        for block in self.uniforms.values_mut() {
            block.upload(queue);
        }
    }

    /// Bind a vertex buffer to the slot with the given layout name.
    pub fn set_vertex_buffer(&mut self, name: &str, buffer: wgpu::Buffer) -> Result<()> {
        let slot = self
            .slot_names
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| LumaError::Model(format!("{}: no vertex slot `{name}`", self.label)))?;
        self.vertex_slots[slot] = Some(buffer);
        Ok(())
    }

    pub fn set_index_buffer(&mut self, buffer: wgpu::Buffer, format: wgpu::IndexFormat, count: u32) {
        self.index_buffer = Some(IndexBuffer {
            buffer,
            format,
            count,
        });
    }

    pub fn set_vertex_count(&mut self, count: u32) {
        self.vertex_count = count;
    }

    pub fn set_instance_count(&mut self, count: u32) {
        self.instance_count = count;
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Encode this model's draw call into the pass.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.draw_with(&self.pipeline, pass)
    }

    /// Encode a draw with the picking pipeline. The pass must render into a
    /// [`PICKING_FORMAT`] target and have its blend constant set to the layer id.
    pub fn draw_picking(&self, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        match &self.picking_pipeline {
            Some(pipeline) => self.draw_with(pipeline, pass),
            None => Ok(()),
        }
    }

    pub fn is_pickable(&self) -> bool {
        self.picking_pipeline.is_some()
    }

    fn draw_with(&self, pipeline: &wgpu::RenderPipeline, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if self.instance_count == 0 {
            return Ok(());
        }
        for (slot, buffer) in self.vertex_slots.iter().enumerate() {
            if buffer.is_none() {
                return Err(LumaError::Model(format!(
                    "{}: vertex slot `{}` has no buffer",
                    self.label, self.slot_names[slot]
                )));
            }
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for (slot, buffer) in self.vertex_slots.iter().enumerate() {
            pass.set_vertex_buffer(slot as u32, buffer.as_ref().unwrap().slice(..));
        }
        match &self.index_buffer {
            Some(index) => {
                if index.count == 0 {
                    return Ok(());
                }
                pass.set_index_buffer(index.buffer.slice(..), index.format);
                pass.draw_indexed(0..index.count, 0, 0..self.instance_count);
            }
            None => {
                if self.vertex_count == 0 {
                    return Ok(());
                }
                pass.draw(0..self.vertex_count, 0..self.instance_count);
            }
        }
        Ok(())
    }
}

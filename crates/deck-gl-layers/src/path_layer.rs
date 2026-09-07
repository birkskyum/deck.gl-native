//! Port of `@deck.gl/layers/src/path-layer/path-layer.ts`.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_paths};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Path, Result, Unit, Viewport};
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from};
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::path::tesselate;

const SHADER: &str = include_str!("wgsl/path_layer.wgsl");

/// Per-instance stroke data, interleaved like deck.gl's `path-instance-data` buffer group.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    width: f32,
    color: [u8; 4],
    row_index: u32,
}

/// Properties of a [`PathLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct PathLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub width_units: Unit,
    pub width_scale: f32,
    pub width_min_pixels: f32,
    pub width_max_pixels: f32,
    /// Round joints instead of mitered
    pub joint_rounded: bool,
    /// Round caps instead of butt
    pub cap_rounded: bool,
    pub miter_limit: f32,
    /// Keep a constant screen-space width regardless of pitch
    pub billboard: bool,
    pub get_path: Accessor<Path>,
    pub get_color: Accessor<Color>,
    pub get_width: Accessor<f32>,
}

impl Default for PathLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("PathLayer"),
            data: LayerData::default(),
            width_units: Unit::Meters,
            width_scale: 1.0,
            width_min_pixels: 0.0,
            width_max_pixels: f32::MAX,
            joint_rounded: false,
            cap_rounded: false,
            miter_limit: 4.0,
            billboard: false,
            get_path: Accessor::column("path"),
            get_color: Accessor::Constant([0, 0, 0, 255]),
            get_width: Accessor::Constant(1.0),
        }
    }
}

/// Renders polylines with joints and caps.
pub struct PathLayer {
    props: PathLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl PathLayer {
    pub fn new(props: PathLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &PathLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: PathLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;
        let model = self.model.as_mut().expect("initialized");

        let paths = resolve_paths(data, &props.get_path)?;
        let widths = resolve_f32(data, &props.get_width)?;
        let colors = resolve_colors(data, &props.get_color)?;
        let tesselated = tesselate(&paths);

        let packed = tesselated.packed_neighbor_positions();
        let instance_data: Vec<InstanceData> = tesselated
            .row_index
            .iter()
            .map(|&r| InstanceData {
                width: widths[r as usize],
                color: colors[r as usize],
                row_index: data.source_row(r as usize),
            })
            .collect();

        model.set_vertex_buffer(
            "instanceTypes",
            create_vertex_buffer_from(device, "instanceTypes", &tesselated.segment_types),
        )?;
        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(device, "instancePositions", &packed),
        )?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(device, "instanceData", &instance_data),
        )?;
        model.set_instance_count(tesselated.instance_count() as u32);
        Ok(())
    }
}

impl Layer for PathLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let position_stride = 24 * 4;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x2),
            VertexBufferLayout::instance("instanceTypes", 1, VertexFormat::Float32),
            VertexBufferLayout::interleaved(
                "instancePositions",
                position_stride,
                wgpu::VertexStepMode::Instance,
                &[
                    (2, VertexFormat::Float32x3, 0),
                    (3, VertexFormat::Float32x3, 12),
                    (4, VertexFormat::Float32x3, 24),
                    (5, VertexFormat::Float32x3, 36),
                    (6, VertexFormat::Float32x3, 48),
                    (7, VertexFormat::Float32x3, 60),
                    (8, VertexFormat::Float32x3, 72),
                    (9, VertexFormat::Float32x3, 84),
                ],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<InstanceData>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (10, VertexFormat::Float32, 0),
                    (11, VertexFormat::Unorm8x4, 4),
                    (12, VertexFormat::Uint32, 8),
                ],
            ),
        ];
        let mut desc = ModelDescriptor::new(
            &self.props.base.id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleList,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        let mut model = Model::new(&ctx.device, &desc)?;

        // [0] position on segment - 0: start, 1: end
        // [1] side of path - -1: left, 0: center (joint), 1: right
        let positions: [f32; 12] = [
            0.0, 0.0, // bevel start corner
            0.0, -1.0, // start inner corner
            0.0, 1.0, // start outer corner
            1.0, -1.0, // end inner corner
            1.0, 1.0, // end outer corner
            1.0, 0.0, // bevel end corner
        ];
        let indices: [u32; 12] = [
            0, 1, 2, // start corner
            1, 4, 2, // body
            1, 3, 4, //
            3, 5, 4, // end corner
        ];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_index_buffer(
            create_index_buffer(&ctx.device, "indices", &indices),
            wgpu::IndexFormat::Uint32,
            12,
        );
        self.model = Some(model);
        self.data_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = &self.props;
        let model = self.model.as_mut().expect("initialized");
        update_standard_uniforms(model, ctx, viewport, &props.base)?;

        let u = model.uniforms("path")?;
        u.set_f32("widthScale", props.width_scale)?;
        u.set_f32("widthMinPixels", props.width_min_pixels)?;
        u.set_f32("widthMaxPixels", props.width_max_pixels)?;
        u.set_f32("jointType", if props.joint_rounded { 1.0 } else { 0.0 })?;
        u.set_f32("capType", if props.cap_rounded { 1.0 } else { 0.0 })?;
        u.set_f32("miterLimit", props.miter_limit)?;
        u.set_f32("billboard", if props.billboard { 1.0 } else { 0.0 })?;
        u.set_i32("widthUnits", props.width_units.shader_value())?;
        model.upload_uniforms(&ctx.queue);
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(model) = &self.model {
            model.draw(pass)?;
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        if let Some(model) = &mut self.model {
            set_model_picking_active(model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(model) = &self.model {
            model.draw_picking(pass)?;
        }
        Ok(())
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn update_from(&mut self, incoming: &mut dyn Layer) -> bool {
        match incoming.as_any_mut().downcast_mut::<Self>() {
            Some(other) => {
                self.set_props(std::mem::take(&mut other.props));
                true
            }
            None => false,
        }
    }
}

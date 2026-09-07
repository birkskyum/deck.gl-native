//! Port of `@deck.gl/layers/src/point-cloud-layer/point-cloud-layer.ts`: lit points with
//! normals.

use deck_gl::data::{resolve_colors, resolve_positions};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/point_cloud_layer.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstancePositions {
    position: [f32; 3],
    position_low: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    normal: [f32; 3],
    color: [u8; 4],
}

/// Properties of a [`PointCloudLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloudLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub size_units: Unit,
    /// Radius of all points in `size_units`
    pub point_size: f32,
    pub get_position: Accessor<Position>,
    pub get_normal: Accessor<[f32; 3]>,
    pub get_color: Accessor<Color>,
}

impl Default for PointCloudLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("PointCloudLayer"),
            data: LayerData::default(),
            size_units: Unit::Pixels,
            point_size: 10.0,
            get_position: Accessor::column("position"),
            get_normal: Accessor::Constant([0.0, 0.0, 1.0]),
            get_color: Accessor::Constant([0, 0, 0, 255]),
        }
    }
}

/// Renders a point cloud with 3D positions, normals and colors.
pub struct PointCloudLayer {
    props: PointCloudLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl PointCloudLayer {
    pub fn new(props: PointCloudLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &PointCloudLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: PointCloudLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let model = self.model.as_mut().expect("initialized");

        let positions = resolve_positions(data, &props.get_position)?;
        let normals = deck_gl::data::resolve_with(data, &props.get_normal, |_| {
            Err(deck_gl::DeckError::Data(
                "normal columns are not supported yet".into(),
            ))
        })?;
        let colors = resolve_colors(data, &props.get_color)?;

        let instance_positions: Vec<InstancePositions> = positions
            .iter()
            .map(|p| {
                let hi = [p[0] as f32, p[1] as f32, p[2] as f32];
                InstancePositions {
                    position: hi,
                    position_low: [
                        (p[0] - hi[0] as f64) as f32,
                        (p[1] - hi[1] as f64) as f32,
                        (p[2] - hi[2] as f64) as f32,
                    ],
                }
            })
            .collect();
        let instance_data: Vec<InstanceData> = (0..data.len())
            .map(|i| InstanceData {
                normal: normals[i],
                color: colors[i],
            })
            .collect();
        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(&ctx.device, "instancePositions", &instance_positions),
        )?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(&ctx.device, "instanceData", &instance_data),
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl Layer for PointCloudLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let modules: Vec<ShaderModuleSource> = STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect();
        let shader = assemble_shader(&self.props.base.id, &modules, SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::interleaved(
                "instancePositions",
                std::mem::size_of::<InstancePositions>() as u64,
                wgpu::VertexStepMode::Instance,
                &[(1, VertexFormat::Float32x3, 0), (2, VertexFormat::Float32x3, 12)],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<InstanceData>() as u64,
                wgpu::VertexStepMode::Instance,
                &[(3, VertexFormat::Float32x3, 0), (4, VertexFormat::Unorm8x4, 12)],
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
        // a triangle that minimally covers the unit circle
        let mut positions = Vec::with_capacity(9);
        for i in 0..3 {
            let angle = i as f32 / 3.0 * std::f32::consts::TAU;
            positions.extend_from_slice(&[angle.cos() * 2.0, angle.sin() * 2.0, 0.0]);
        }
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(3);
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
        let u = model.uniforms("pointCloudUniforms")?;
        u.set_f32("radiusPixels", props.point_size)?;
        u.set_i32("sizeUnits", props.size_units.shader_value())?;
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

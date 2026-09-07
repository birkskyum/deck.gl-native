//! Port of `@deck.gl/layers/src/scatterplot-layer/scatterplot-layer.ts`.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_positions, resolve_vec2};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::{create_vertex_buffer_from, split_f64};
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/scatterplot_layer.wgsl");

/// Per-instance scalars interleaved in one buffer, to stay within the vertex buffer limit.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    radius: f32,
    line_width: f32,
    pixel_offset: [f32; 2],
    row_index: u32,
}

/// Properties of a [`ScatterplotLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterplotLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub radius_units: Unit,
    pub radius_scale: f32,
    pub radius_min_pixels: f32,
    pub radius_max_pixels: f32,
    pub line_width_units: Unit,
    pub line_width_scale: f32,
    pub line_width_min_pixels: f32,
    pub line_width_max_pixels: f32,
    pub stroked: bool,
    pub filled: bool,
    pub billboard: bool,
    pub antialiasing: bool,
    pub get_position: Accessor<Position>,
    pub get_radius: Accessor<f32>,
    pub get_fill_color: Accessor<Color>,
    pub get_line_color: Accessor<Color>,
    pub get_line_width: Accessor<f32>,
    pub get_pixel_offset: Accessor<[f32; 2]>,
}

impl Default for ScatterplotLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("ScatterplotLayer"),
            data: LayerData::default(),
            radius_units: Unit::Meters,
            radius_scale: 1.0,
            radius_min_pixels: 0.0,
            radius_max_pixels: f32::MAX,
            line_width_units: Unit::Meters,
            line_width_scale: 1.0,
            line_width_min_pixels: 0.0,
            line_width_max_pixels: f32::MAX,
            stroked: false,
            filled: true,
            billboard: false,
            antialiasing: true,
            get_position: Accessor::column("position"),
            get_radius: Accessor::Constant(1.0),
            get_fill_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_width: Accessor::Constant(1.0),
            get_pixel_offset: Accessor::Constant([0.0, 0.0]),
        }
    }
}

/// Renders circles at given coordinates.
pub struct ScatterplotLayer {
    props: ScatterplotLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl ScatterplotLayer {
    pub fn new(props: ScatterplotLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &ScatterplotLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: ScatterplotLayerProps) {
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

        let positions = resolve_positions(data, &props.get_position)?;
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let (hi, lo) = split_f64(&flat);
        let radius = resolve_f32(data, &props.get_radius)?;
        let line_widths = resolve_f32(data, &props.get_line_width)?;
        let fill_colors = resolve_colors(data, &props.get_fill_color)?;
        let line_colors = resolve_colors(data, &props.get_line_color)?;
        let pixel_offsets = resolve_vec2(data, &props.get_pixel_offset)?;

        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(device, "instancePositions", &hi),
        )?;
        model.set_vertex_buffer(
            "instancePositions64Low",
            create_vertex_buffer_from(device, "instancePositions64Low", &lo),
        )?;
        let instance_data: Vec<InstanceData> = (0..data.len())
            .map(|i| InstanceData {
                radius: radius[i],
                line_width: line_widths[i],
                pixel_offset: pixel_offsets[i],
                row_index: data.source_row(i),
            })
            .collect();
        model.set_vertex_buffer(
            "instanceFillColors",
            create_vertex_buffer_from(device, "instanceFillColors", &fill_colors),
        )?;
        model.set_vertex_buffer(
            "instanceLineColors",
            create_vertex_buffer_from(device, "instanceLineColors", &line_colors),
        )?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(device, "instanceData", &instance_data),
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl Layer for ScatterplotLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instancePositions", 1, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instancePositions64Low", 2, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceFillColors", 5, VertexFormat::Unorm8x4),
            VertexBufferLayout::instance("instanceLineColors", 6, VertexFormat::Unorm8x4),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<InstanceData>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (3, VertexFormat::Float32, 0),
                    (4, VertexFormat::Float32, 4),
                    (7, VertexFormat::Float32x2, 8),
                    (8, VertexFormat::Uint32, 16),
                ],
            ),
        ];
        let mut desc = ModelDescriptor::new(
            &self.props.base.id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        let mut model = Model::new(&ctx.device, &desc)?;
        // a square that minimally covers the unit circle
        let positions: [f32; 12] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(4);
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

        let u = model.uniforms("scatterplot")?;
        u.set_f32("radiusScale", props.radius_scale)?;
        u.set_f32("radiusMinPixels", props.radius_min_pixels)?;
        u.set_f32("radiusMaxPixels", props.radius_max_pixels)?;
        u.set_f32("lineWidthScale", props.line_width_scale)?;
        u.set_f32("lineWidthMinPixels", props.line_width_min_pixels)?;
        u.set_f32("lineWidthMaxPixels", props.line_width_max_pixels)?;
        u.set_f32("stroked", if props.stroked { 1.0 } else { 0.0 })?;
        u.set_i32("filled", props.filled as i32)?;
        u.set_i32("antialiasing", props.antialiasing as i32)?;
        u.set_i32("billboard", props.billboard as i32)?;
        u.set_i32("radiusUnits", props.radius_units.shader_value())?;
        u.set_i32("lineWidthUnits", props.line_width_units.shader_value())?;
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

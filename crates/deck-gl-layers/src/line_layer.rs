//! Port of `@deck.gl/layers/src/line-layer/line-layer.ts`.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_positions};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::{create_vertex_buffer_from, split_f64};
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/line_layer.wgsl");

/// Properties of a [`LineLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug)]
pub struct LineLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub width_units: Unit,
    pub width_scale: f32,
    pub width_min_pixels: f32,
    pub width_max_pixels: f32,
    pub get_source_position: Accessor<Position>,
    pub get_target_position: Accessor<Position>,
    pub get_color: Accessor<Color>,
    pub get_width: Accessor<f32>,
}

impl Default for LineLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("LineLayer"),
            data: LayerData::default(),
            width_units: Unit::Pixels,
            width_scale: 1.0,
            width_min_pixels: 0.0,
            width_max_pixels: f32::MAX,
            get_source_position: Accessor::column("source_position"),
            get_target_position: Accessor::column("target_position"),
            get_color: Accessor::Constant([0, 0, 0, 255]),
            get_width: Accessor::Constant(1.0),
        }
    }
}

/// Renders straight lines joining pairs of source and target coordinates.
pub struct LineLayer {
    props: LineLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl LineLayer {
    pub fn new(props: LineLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &LineLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: LineLayerProps) {
        self.props = props;
        self.data_dirty = true;
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;
        let model = self.model.as_mut().expect("initialized");

        let sources: Vec<f64> = resolve_positions(data, &props.get_source_position)?
            .iter()
            .flatten()
            .copied()
            .collect();
        let targets: Vec<f64> = resolve_positions(data, &props.get_target_position)?
            .iter()
            .flatten()
            .copied()
            .collect();
        let (source_hi, source_lo) = split_f64(&sources);
        let (target_hi, target_lo) = split_f64(&targets);
        let colors = resolve_colors(data, &props.get_color)?;
        let widths = resolve_f32(data, &props.get_width)?;

        model.set_vertex_buffer(
            "instanceSourcePositions",
            create_vertex_buffer_from(device, "instanceSourcePositions", &source_hi),
        )?;
        model.set_vertex_buffer(
            "instanceTargetPositions",
            create_vertex_buffer_from(device, "instanceTargetPositions", &target_hi),
        )?;
        model.set_vertex_buffer(
            "instanceSourcePositions64Low",
            create_vertex_buffer_from(device, "instanceSourcePositions64Low", &source_lo),
        )?;
        model.set_vertex_buffer(
            "instanceTargetPositions64Low",
            create_vertex_buffer_from(device, "instanceTargetPositions64Low", &target_lo),
        )?;
        model.set_vertex_buffer(
            "instanceColors",
            create_vertex_buffer_from(device, "instanceColors", &colors),
        )?;
        model.set_vertex_buffer(
            "instanceWidths",
            create_vertex_buffer_from(device, "instanceWidths", &widths),
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl Layer for LineLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceSourcePositions", 1, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceTargetPositions", 2, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceSourcePositions64Low", 3, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceTargetPositions64Low", 4, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceColors", 5, VertexFormat::Unorm8x4),
            VertexBufferLayout::instance("instanceWidths", 6, VertexFormat::Float32),
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
        //  (0, -1)-------------_(1, -1)
        //       |          _,-"  |
        //       o      _,-"      o
        //       |  _,-"          |
        //   (0, 1)"-------------(1, 1)
        let positions: [f32; 12] = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0];
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

        let u = model.uniforms("line")?;
        u.set_f32("widthScale", props.width_scale)?;
        u.set_f32("widthMinPixels", props.width_min_pixels)?;
        u.set_f32("widthMaxPixels", props.width_max_pixels)?;
        // TODO: wrapLongitude draws the layer twice with useShortestPath 1 and -1
        u.set_f32("useShortestPath", 0.0)?;
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
}

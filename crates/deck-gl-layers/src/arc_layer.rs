//! Port of `@deck.gl/layers/src/arc-layer/arc-layer.ts`.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_positions};
use deck_gl::layer::{depth_bias_for_layer, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/arc_layer.wgsl");

/// Source and target positions as high and low f32 parts, interleaved.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstancePositions {
    source: [f32; 3],
    source_low: [f32; 3],
    target: [f32; 3],
    target_low: [f32; 3],
}

/// Colors, width, height and tilt, interleaved.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    source_color: [u8; 4],
    target_color: [u8; 4],
    width: f32,
    height: f32,
    tilt: f32,
}

/// Properties of an [`ArcLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug)]
pub struct ArcLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Follow the great circle instead of a flat arc
    pub great_circle: bool,
    pub num_segments: u32,
    pub width_units: Unit,
    pub width_scale: f32,
    pub width_min_pixels: f32,
    pub width_max_pixels: f32,
    pub get_source_position: Accessor<Position>,
    pub get_target_position: Accessor<Position>,
    pub get_source_color: Accessor<Color>,
    pub get_target_color: Accessor<Color>,
    pub get_width: Accessor<f32>,
    /// Multiplier of the arc height (1 is a semicircle in flat mode)
    pub get_height: Accessor<f32>,
    /// Tilt of the arc plane in degrees
    pub get_tilt: Accessor<f32>,
}

impl Default for ArcLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("ArcLayer"),
            data: LayerData::default(),
            great_circle: false,
            num_segments: 50,
            width_units: Unit::Pixels,
            width_scale: 1.0,
            width_min_pixels: 0.0,
            width_max_pixels: f32::MAX,
            get_source_position: Accessor::column("source_position"),
            get_target_position: Accessor::column("target_position"),
            get_source_color: Accessor::Constant([0, 0, 0, 255]),
            get_target_color: Accessor::Constant([0, 0, 0, 255]),
            get_width: Accessor::Constant(1.0),
            get_height: Accessor::Constant(1.0),
            get_tilt: Accessor::Constant(0.0),
        }
    }
}

/// Renders raised arcs joining pairs of source and target coordinates.
pub struct ArcLayer {
    props: ArcLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl ArcLayer {
    pub fn new(props: ArcLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &ArcLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: ArcLayerProps) {
        self.props = props;
        self.data_dirty = true;
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;
        let model = self.model.as_mut().expect("initialized");

        let sources = resolve_positions(data, &props.get_source_position)?;
        let targets = resolve_positions(data, &props.get_target_position)?;
        let source_colors = resolve_colors(data, &props.get_source_color)?;
        let target_colors = resolve_colors(data, &props.get_target_color)?;
        let widths = resolve_f32(data, &props.get_width)?;
        let heights = resolve_f32(data, &props.get_height)?;
        let tilts = resolve_f32(data, &props.get_tilt)?;

        let split = |p: Position| -> ([f32; 3], [f32; 3]) {
            let hi = [p[0] as f32, p[1] as f32, p[2] as f32];
            let lo = [
                (p[0] - hi[0] as f64) as f32,
                (p[1] - hi[1] as f64) as f32,
                (p[2] - hi[2] as f64) as f32,
            ];
            (hi, lo)
        };
        let positions: Vec<InstancePositions> = sources
            .iter()
            .zip(&targets)
            .map(|(s, t)| {
                let (source, source_low) = split(*s);
                let (target, target_low) = split(*t);
                InstancePositions {
                    source,
                    source_low,
                    target,
                    target_low,
                }
            })
            .collect();
        let instance_data: Vec<InstanceData> = (0..data.len())
            .map(|i| InstanceData {
                source_color: source_colors[i],
                target_color: target_colors[i],
                width: widths[i],
                height: heights[i],
                tilt: tilts[i],
            })
            .collect();

        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(device, "instancePositions", &positions),
        )?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(device, "instanceData", &instance_data),
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl Layer for ArcLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let layouts = [
            VertexBufferLayout::interleaved(
                "instancePositions",
                std::mem::size_of::<InstancePositions>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (0, VertexFormat::Float32x3, 0),
                    (1, VertexFormat::Float32x3, 12),
                    (2, VertexFormat::Float32x3, 24),
                    (3, VertexFormat::Float32x3, 36),
                ],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<InstanceData>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (4, VertexFormat::Unorm8x4, 0),
                    (5, VertexFormat::Unorm8x4, 4),
                    (6, VertexFormat::Float32, 8),
                    (7, VertexFormat::Float32, 12),
                    (8, VertexFormat::Float32, 16),
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
        desc.depth_bias = depth_bias_for_layer(ctx.layer_index);
        desc.pickable = self.props.base.pickable;
        let mut model = Model::new(&ctx.device, &desc)?;
        model.set_vertex_count(self.props.num_segments.max(1) * 2);
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
        model.set_vertex_count(props.num_segments.max(1) * 2);
        update_standard_uniforms(model, ctx, viewport, &props.base)?;

        let u = model.uniforms("arc")?;
        u.set_f32("greatCircle", if props.great_circle { 1.0 } else { 0.0 })?;
        u.set_f32(
            "useShortestPath",
            if props.base.wrap_longitude { 1.0 } else { 0.0 },
        )?;
        u.set_f32("numSegments", props.num_segments.max(1) as f32)?;
        u.set_f32("widthScale", props.width_scale)?;
        u.set_f32("widthMinPixels", props.width_min_pixels)?;
        u.set_f32("widthMaxPixels", props.width_max_pixels)?;
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

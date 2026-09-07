//! Port of `@deck.gl/layers/src/line-layer/line-layer.ts`.

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/line_layer.wgsl");

/// Properties of a [`LineLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
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
/// `useShortestPath` values to draw with: 0 normally, 1 and -1 with `wrap_longitude`.
fn shortest_path_variants(wrap_longitude: bool) -> &'static [f32] {
    if wrap_longitude {
        &[1.0, -1.0]
    } else {
        &[0.0]
    }
}

/// The line's instance buffers: both end points with their low parts, colours and widths.
fn line_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::instance("instanceSourcePositions", "source", 1, VertexFormat::Float32x3),
        BufferSpec::instance("instanceTargetPositions", "target", 2, VertexFormat::Float32x3),
        BufferSpec::instance_low("instanceSourcePositions64Low", "source", 3),
        BufferSpec::instance_low("instanceTargetPositions64Low", "target", 4),
        BufferSpec::instance("instanceColors", "color", 5, VertexFormat::Unorm8x4),
        BufferSpec::instance("instanceWidths", "width", 6, VertexFormat::Float32),
    ])
}

pub struct LineLayer {
    props: LineLayerProps,
    model: Option<Model>,
    data_dirty: bool,
    attributes: AttributeManager,
}

impl LineLayer {
    pub fn new(props: LineLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
            attributes: line_attributes(),
        }
    }

    pub fn props(&self) -> &LineLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: LineLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        self.attributes.update(
            &ctx.device,
            model,
            &props.data,
            &[
                (
                    "source",
                    AttributeSource::Positions(props.get_source_position.clone()),
                ),
                (
                    "target",
                    AttributeSource::Positions(props.get_target_position.clone()),
                ),
                ("color", AttributeSource::Colors(props.get_color.clone())),
                ("width", AttributeSource::Floats(props.get_width.clone())),
            ],
        )?;
        model.set_instance_count(props.data.len() as u32);
        Ok(())
    }
}

impl Layer for LineLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let mut layouts = vec![VertexBufferLayout::vertex(
            "positions",
            0,
            VertexFormat::Float32x3,
        )];
        layouts.extend(self.attributes.layouts());
        let mut desc = ModelDescriptor::new(
            &self.props.base.id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        self.props.base.parameters.apply(&mut desc);
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
        self.attributes.invalidate_all();
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;

        // With wrapLongitude the layer draws twice, with useShortestPath 1 and -1, each in
        // its own uniform slot (two per world copy).
        for (variant, shortest_path) in shortest_path_variants(props.base.wrap_longitude)
            .iter()
            .enumerate()
        {
            model.set_uniform_slot(ctx.uniform_slot * 2 + variant);
            let u = model.uniforms("line")?;
            u.set_f32("widthScale", props.width_scale)?;
            u.set_f32("widthMinPixels", props.width_min_pixels)?;
            u.set_f32("widthMaxPixels", props.width_max_pixels)?;
            u.set_f32("useShortestPath", *shortest_path)?;
            u.set_i32("widthUnits", props.width_units.shader_value())?;
            model.upload_uniforms(&ctx.queue);
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        let variants = shortest_path_variants(self.props.base.wrap_longitude).len();
        if let Some(model) = &mut self.model {
            for variant in 0..variants {
                model.set_uniform_slot(ctx.uniform_slot * 2 + variant);
                model.draw(pass)?;
            }
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        let variants = shortest_path_variants(self.props.base.wrap_longitude).len();
        if let Some(model) = &mut self.model {
            for variant in 0..variants {
                model.set_uniform_slot(ctx.uniform_slot * 2 + variant);
                set_model_picking_active(model, &ctx.queue, active)?;
            }
        }
        Ok(())
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        let variants = shortest_path_variants(self.props.base.wrap_longitude).len();
        if let Some(model) = &mut self.model {
            for variant in 0..variants {
                model.set_uniform_slot(ctx.uniform_slot * 2 + variant);
                model.draw_picking(pass)?;
            }
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

//! Port of `@deck.gl/layers/src/scatterplot-layer/scatterplot-layer.ts`.

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/scatterplot_layer.wgsl");

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
/// The scatterplot's instance buffers: positions in two buffers, colours in their own, and
/// the scalars interleaved to stay within the vertex buffer limit.
fn scatterplot_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::instance("instancePositions", "position", 1, VertexFormat::Float32x3),
        BufferSpec::instance_low("instancePositions64Low", "position", 2),
        BufferSpec::instance("instanceFillColors", "fillColor", 5, VertexFormat::Unorm8x4),
        BufferSpec::instance("instanceLineColors", "lineColor", 6, VertexFormat::Unorm8x4),
        BufferSpec::interleaved(
            "instanceData",
            20,
            vec![
                Field::new("radius", 3, VertexFormat::Float32, 0),
                Field::new("lineWidth", 4, VertexFormat::Float32, 4),
                Field::new("pixelOffset", 7, VertexFormat::Float32x2, 8),
                Field::new("rowIndex", 8, VertexFormat::Uint32, 16),
            ],
        ),
    ])
}

pub struct ScatterplotLayer {
    props: ScatterplotLayerProps,
    model: Option<Model>,
    data_dirty: bool,
    attributes: AttributeManager,
}

impl ScatterplotLayer {
    pub fn new(props: ScatterplotLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
            attributes: scatterplot_attributes(),
        }
    }

    pub fn props(&self) -> &ScatterplotLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed, the
    /// model when the pipeline state or the extensions changed.
    pub fn set_props(&mut self, props: ScatterplotLayerProps) {
        if self.props != props {
            if self.props.base.needs_new_model(&props.base) {
                self.model = None;
            }
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        let mut sources = vec![
            ("position", AttributeSource::Positions(props.get_position.clone())),
            ("fillColor", AttributeSource::Colors(props.get_fill_color.clone())),
            ("lineColor", AttributeSource::Colors(props.get_line_color.clone())),
            ("radius", AttributeSource::Floats(props.get_radius.clone())),
            ("lineWidth", AttributeSource::Floats(props.get_line_width.clone())),
            (
                "pixelOffset",
                AttributeSource::Vec2(props.get_pixel_offset.clone()),
            ),
            ("rowIndex", AttributeSource::RowIndex),
        ];
        sources.extend(props.base.extensions.sources());
        self.attributes
            .update(&ctx.device, model, &props.data, &sources)?;
        model.set_instance_count(props.data.len() as u32);
        Ok(())
    }
}

impl Layer for ScatterplotLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let extensions = &self.props.base.extensions;
        let shader = extensions.assemble(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        self.attributes = scatterplot_attributes();
        self.attributes.extend(extensions.buffer_specs(&shader)?);
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
        if self.model.is_none() {
            self.initialize(ctx)?;
        }
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
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

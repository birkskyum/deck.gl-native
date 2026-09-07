//! Port of `@deck.gl/layers/src/arc-layer/arc-layer.ts`.

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::{assemble_shader, Model, ModelDescriptor};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/arc_layer.wgsl");

/// Properties of an [`ArcLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
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
/// The arc's instance buffers: both end points with their low parts interleaved, then colours,
/// width, height and tilt.
fn arc_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::interleaved(
            "instancePositions",
            48,
            vec![
                Field::new("source", 0, VertexFormat::Float32x3, 0),
                Field::low("source", 1, 12),
                Field::new("target", 2, VertexFormat::Float32x3, 24),
                Field::low("target", 3, 36),
            ],
        ),
        BufferSpec::interleaved(
            "instanceData",
            20,
            vec![
                Field::new("sourceColor", 4, VertexFormat::Unorm8x4, 0),
                Field::new("targetColor", 5, VertexFormat::Unorm8x4, 4),
                Field::new("width", 6, VertexFormat::Float32, 8),
                Field::new("height", 7, VertexFormat::Float32, 12),
                Field::new("tilt", 8, VertexFormat::Float32, 16),
            ],
        ),
    ])
}

pub struct ArcLayer {
    props: ArcLayerProps,
    model: Option<Model>,
    data_dirty: bool,
    attributes: AttributeManager,
}

impl ArcLayer {
    pub fn new(props: ArcLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
            attributes: arc_attributes(),
        }
    }

    pub fn props(&self) -> &ArcLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: ArcLayerProps) {
        if self.props == props {
            return;
        }
        if Self::attributes_changed(&self.props, &props) {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms (sizes, units, flags and the base props).
    fn attributes_changed(old: &ArcLayerProps, new: &ArcLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.width_units = old.width_units;
        probe.width_scale = old.width_scale;
        probe.width_min_pixels = old.width_min_pixels;
        probe.width_max_pixels = old.width_max_pixels;
        probe != *old
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        self.attributes.update(
            &ctx.device,
            model,
            data,
            &[
                (
                    "source",
                    AttributeSource::Positions(props.get_source_position.clone()),
                ),
                (
                    "target",
                    AttributeSource::Positions(props.get_target_position.clone()),
                ),
                (
                    "sourceColor",
                    AttributeSource::Colors(props.get_source_color.clone()),
                ),
                (
                    "targetColor",
                    AttributeSource::Colors(props.get_target_color.clone()),
                ),
                ("width", AttributeSource::Floats(props.get_width.clone())),
                ("height", AttributeSource::Floats(props.get_height.clone())),
                ("tilt", AttributeSource::Floats(props.get_tilt.clone())),
            ],
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl ArcLayer {
    /// deck.gl's GreatCircleLayer: an arc layer drawing flat great circle paths.
    pub fn great_circle(props: ArcLayerProps) -> ArcLayer {
        ArcLayer::new(ArcLayerProps {
            great_circle: true,
            num_segments: 100,
            get_height: Accessor::Constant(0.0),
            ..props
        })
    }
}

impl Layer for ArcLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let layouts = self.attributes.layouts();
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
        model.set_vertex_count(self.props.num_segments.max(1) * 2);
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

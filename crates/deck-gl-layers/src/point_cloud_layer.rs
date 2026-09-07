//! Port of `@deck.gl/layers/src/point-cloud-layer/point-cloud-layer.ts`: lit points with
//! normals.

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/point_cloud_layer.wgsl");

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
/// The point cloud's instance buffers: position with its low part, then normal and colour.
fn point_cloud_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::interleaved(
            "instancePositions",
            24,
            vec![
                Field::new("position", 1, VertexFormat::Float32x3, 0),
                Field::low("position", 2, 12),
            ],
        ),
        BufferSpec::interleaved(
            "instanceData",
            16,
            vec![
                Field::new("normal", 3, VertexFormat::Float32x3, 0),
                Field::new("color", 4, VertexFormat::Unorm8x4, 12),
            ],
        ),
    ])
}

pub struct PointCloudLayer {
    props: PointCloudLayerProps,
    model: Option<Model>,
    data_dirty: bool,
    attributes: AttributeManager,
}

impl PointCloudLayer {
    pub fn new(props: PointCloudLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
            attributes: point_cloud_attributes(),
        }
    }

    pub fn props(&self) -> &PointCloudLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: PointCloudLayerProps) {
        if self.props == props {
            return;
        }
        if self.props.base.needs_new_model(&props.base) {
            self.model = None;
        }
        if self.props.base.extensions != props.base.extensions
            || Self::attributes_changed(&self.props, &props)
        {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms (sizes, units, flags and the base props).
    fn attributes_changed(old: &PointCloudLayerProps, new: &PointCloudLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.size_units = old.size_units;
        probe.point_size = old.point_size;
        probe != *old
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        let mut sources = vec![
            ("position", AttributeSource::Positions(props.get_position.clone())),
            ("normal", AttributeSource::Vec3(props.get_normal.clone())),
            ("color", AttributeSource::Colors(props.get_color.clone())),
        ];
        sources.extend(props.base.extensions.sources());
        self.attributes.update(&ctx.device, model, data, &sources)?;
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
        let extensions = &self.props.base.extensions;
        let shader = extensions.assemble(&self.props.base.id, &modules, SHADER)?;
        self.attributes = point_cloud_attributes();
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
            wgpu::PrimitiveTopology::TriangleList,
            ctx.target,
        );
        ctx.configure(&mut desc, &self.props.base);
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

    fn bounds(&self) -> Option<[f64; 4]> {
        self.attributes.bounds()
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

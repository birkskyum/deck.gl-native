//! Port of `@deck.gl/layers/src/column-layer/column-layer.ts`: extruded cylinders
//! (tesselated regular polygons) at given coordinates.

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use glam::Vec2;
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from};
use luma_gl::{Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/column_layer.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GeometryVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Port of deck.gl's `tesselateColumn` for a regular disk: a triangle strip of the sides
/// (when extruded) followed by the top cap, plus line list indices for the wireframe.
struct ColumnGeometry {
    vertices: Vec<GeometryVertex>,
    wireframe_indices: Vec<u32>,
}

fn tesselate_column(nradial: u32, extruded: bool) -> ColumnGeometry {
    let nradial = nradial.max(3) as usize;
    let height = if extruded { 1.0f32 } else { 0.0 };
    let verts_around_edge = nradial + 1;
    let step = std::f32::consts::TAU / nradial as f32;
    let mut vertices = Vec::new();

    if extruded {
        // side: 0 - 2 - 4 top, 1 - 3 - 5 bottom
        for j in 0..verts_around_edge {
            let a = j as f32 * step;
            let (sin, cos) = a.sin_cos();
            for k in 0..2 {
                vertices.push(GeometryVertex {
                    position: [cos, sin, (0.5 - k as f32) * height],
                    normal: [cos, sin, 0.0],
                });
            }
        }
        // duplicate the last vertex to create a proper degenerate triangle
        if let Some(&last) = vertices.last() {
            vertices.push(last);
        }
    }
    // top: 0, -1, 1, -2, 2, -3, 3, ...
    for j in (if extruded { 0 } else { 1 })..verts_around_edge {
        let v = (j / 2) as i64 * if j.is_multiple_of(2) { 1 } else { -1 };
        let a = v as f32 * step;
        let (sin, cos) = a.sin_cos();
        vertices.push(GeometryVertex {
            position: [cos, sin, height / 2.0],
            normal: [0.0, 0.0, 1.0],
        });
    }

    let mut wireframe_indices = Vec::new();
    if extruded {
        for j in 0..nradial as u32 {
            wireframe_indices.extend_from_slice(&[
                j * 2,
                j * 2 + 2, // top loop
                j * 2,
                j * 2 + 1, // side vertical
                j * 2 + 1,
                j * 2 + 3, // bottom loop
            ]);
        }
    }
    ColumnGeometry {
        vertices,
        wireframe_indices,
    }
}

/// Properties of a [`ColumnLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Number of sides of the disk
    pub disk_resolution: u32,
    /// Disk radius in `radius_units`
    pub radius: f32,
    /// Disk rotation in degrees
    pub angle: f32,
    /// Disk offset from the position, relative to the radius
    pub offset: [f32; 2],
    /// Fraction of the radius that is drawn
    pub coverage: f32,
    pub elevation_scale: f32,
    pub radius_units: Unit,
    pub line_width_units: Unit,
    pub line_width_scale: f32,
    pub line_width_min_pixels: f32,
    pub line_width_max_pixels: f32,
    pub extruded: bool,
    pub wireframe: bool,
    pub filled: bool,
    pub stroked: bool,
    pub get_position: Accessor<Position>,
    pub get_fill_color: Accessor<Color>,
    pub get_line_color: Accessor<Color>,
    pub get_line_width: Accessor<f32>,
    pub get_elevation: Accessor<f32>,
}

impl Default for ColumnLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("ColumnLayer"),
            data: LayerData::default(),
            disk_resolution: 20,
            radius: 1000.0,
            angle: 0.0,
            offset: [0.0, 0.0],
            coverage: 1.0,
            elevation_scale: 1.0,
            radius_units: Unit::Meters,
            line_width_units: Unit::Meters,
            line_width_scale: 1.0,
            line_width_min_pixels: 0.0,
            line_width_max_pixels: f32::MAX,
            extruded: true,
            wireframe: false,
            filled: true,
            stroked: false,
            get_position: Accessor::column("position"),
            get_fill_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_width: Accessor::Constant(1.0),
            get_elevation: Accessor::Constant(1000.0),
        }
    }
}

/// Renders extruded columns or flat disks at given coordinates.
/// The column's instance buffers: position with its low part, then elevation, colours and
/// stroke width.
fn column_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::interleaved(
            "instancePositions",
            24,
            vec![
                Field::new("position", 2, VertexFormat::Float32x3, 0),
                Field::low("position", 3, 12),
            ],
        ),
        BufferSpec::interleaved(
            "instanceData",
            16,
            vec![
                Field::new("elevation", 4, VertexFormat::Float32, 0),
                Field::new("fillColor", 5, VertexFormat::Unorm8x4, 4),
                Field::new("lineColor", 6, VertexFormat::Unorm8x4, 8),
                Field::new("lineWidth", 7, VertexFormat::Float32, 12),
            ],
        ),
    ])
}

pub struct ColumnLayer {
    props: ColumnLayerProps,
    fill: Option<Model>,
    stroke: Option<Model>,
    wireframe: Option<Model>,
    data_dirty: bool,
    attributes: AttributeManager,
    models_dirty: bool,
}

impl ColumnLayer {
    pub fn new(props: ColumnLayerProps) -> Self {
        Self {
            props,
            fill: None,
            stroke: None,
            wireframe: None,
            data_dirty: true,
            attributes: column_attributes(),
            models_dirty: false,
        }
    }

    pub fn props(&self) -> &ColumnLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: ColumnLayerProps) {
        if self.props == props {
            return;
        }
        if Self::models_changed(&self.props, &props) {
            self.models_dirty = true;
        }
        if self.props.base.extensions != props.base.extensions
            || Self::attributes_changed(&self.props, &props)
        {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the models rebuilt: which of fill, stroke and wireframe
    /// exist, the tessellation, or anything the base props say about the pipeline.
    fn models_changed(old: &ColumnLayerProps, new: &ColumnLayerProps) -> bool {
        old.base.needs_new_model(&new.base)
            || old.filled != new.filled
            || old.stroked != new.stroked
            || old.extruded != new.extruded
            || old.wireframe != new.wireframe
            || old.disk_resolution != new.disk_resolution
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms (sizes, units, flags and the base props).
    fn attributes_changed(old: &ColumnLayerProps, new: &ColumnLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.radius = old.radius;
        probe.angle = old.angle;
        probe.offset = old.offset;
        probe.coverage = old.coverage;
        probe.elevation_scale = old.elevation_scale;
        probe.radius_units = old.radius_units;
        probe.line_width_units = old.line_width_units;
        probe.line_width_scale = old.line_width_scale;
        probe.line_width_min_pixels = old.line_width_min_pixels;
        probe.line_width_max_pixels = old.line_width_max_pixels;
        probe != *old
    }

    fn models(&mut self) -> impl Iterator<Item = &mut Model> {
        [&mut self.wireframe, &mut self.fill, &mut self.stroke]
            .into_iter()
            .flatten()
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let mut models: Vec<&mut Model> = [&mut self.fill, &mut self.stroke, &mut self.wireframe]
            .into_iter()
            .flatten()
            .collect();
        let mut sources = vec![
            ("position", AttributeSource::Positions(props.get_position.clone())),
            ("elevation", AttributeSource::Floats(props.get_elevation.clone())),
            ("fillColor", AttributeSource::Colors(props.get_fill_color.clone())),
            ("lineColor", AttributeSource::Colors(props.get_line_color.clone())),
            ("lineWidth", AttributeSource::Floats(props.get_line_width.clone())),
        ];
        sources.extend(props.base.extensions.sources(&props.data)?);
        self.attributes
            .update_many(&ctx.device, &ctx.queue, &mut models, data, &sources)?;
        for model in models {
            model.set_instance_count(data.len() as u32);
        }
        Ok(())
    }
}

impl Layer for ColumnLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        self.fill = None;
        self.stroke = None;
        self.wireframe = None;
        let props = &self.props;
        let modules: Vec<ShaderModuleSource> = STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect();
        let extensions = &props.base.extensions;
        let shader = extensions.assemble(&props.base.id, &modules, SHADER)?;
        self.attributes = column_attributes();
        self.attributes.extend(extensions.buffer_specs(&shader)?);
        let layouts = [VertexBufferLayout::interleaved(
            "geometry",
            std::mem::size_of::<GeometryVertex>() as u64,
            wgpu::VertexStepMode::Vertex,
            &[(0, VertexFormat::Float32x3, 0), (1, VertexFormat::Float32x3, 12)],
        )];
        let mut layouts = layouts.to_vec();
        layouts.extend(self.attributes.layouts());
        let geometry = tesselate_column(props.disk_resolution, props.extruded || props.stroked);
        let geometry_buffer = create_vertex_buffer_from(&ctx.device, "geometry", &geometry.vertices);
        let make = |label: String, topology: wgpu::PrimitiveTopology| -> Result<Model> {
            let mut desc = ModelDescriptor::new(&label, &shader, &layouts, topology, ctx.target);
            ctx.configure(&mut desc, &props.base);
            let mut model = Model::new(&ctx.device, &desc)?;
            model.set_vertex_buffer("geometry", geometry_buffer.clone())?;
            Ok(model)
        };

        if props.filled {
            let mut fill = make(
                format!("{}-fill", props.base.id),
                wgpu::PrimitiveTopology::TriangleStrip,
            )?;
            fill.set_vertex_count(geometry.vertices.len() as u32);
            self.fill = Some(fill);
        }
        if !props.extruded && props.stroked {
            let mut stroke = make(
                format!("{}-stroke", props.base.id),
                wgpu::PrimitiveTopology::TriangleStrip,
            )?;
            // remove the cap
            stroke
                .set_vertex_count((geometry.vertices.len() as u32).saturating_sub(props.disk_resolution + 1));
            self.stroke = Some(stroke);
        }
        if props.extruded && props.wireframe {
            let mut wireframe = make(
                format!("{}-wireframe", props.base.id),
                wgpu::PrimitiveTopology::LineList,
            )?;
            wireframe.set_index_buffer(
                create_index_buffer(&ctx.device, "wireframe", &geometry.wireframe_indices),
                wgpu::IndexFormat::Uint32,
                geometry.wireframe_indices.len() as u32,
            );
            self.wireframe = Some(wireframe);
        }
        self.data_dirty = true;
        self.models_dirty = false;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.models_dirty {
            self.initialize(ctx)?;
        }
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = self.props.clone();
        let edge_distance = (std::f32::consts::PI / props.disk_resolution.max(3) as f32).cos();
        let models = [
            (&mut self.wireframe, true),
            (&mut self.fill, false),
            (&mut self.stroke, true),
        ];
        for (model, is_stroke) in models {
            let Some(model) = model else { continue };
            update_standard_uniforms(model, ctx, viewport, &props.base)?;
            let u = model.uniforms("column")?;
            u.set_f32("radius", props.radius)?;
            u.set_f32("angle", props.angle.to_radians())?;
            u.set_vec2("offset", Vec2::from(props.offset))?;
            u.set_f32("extruded", if props.extruded { 1.0 } else { 0.0 })?;
            u.set_f32("stroked", if props.stroked { 1.0 } else { 0.0 })?;
            u.set_f32("isStroke", if is_stroke { 1.0 } else { 0.0 })?;
            u.set_f32("coverage", props.coverage)?;
            u.set_f32("elevationScale", props.elevation_scale)?;
            u.set_f32("edgeDistance", edge_distance)?;
            u.set_f32("widthScale", props.line_width_scale)?;
            u.set_f32("widthMinPixels", props.line_width_min_pixels)?;
            u.set_f32("widthMaxPixels", props.line_width_max_pixels)?;
            u.set_i32("radiusUnits", props.radius_units.shader_value())?;
            u.set_i32("widthUnits", props.line_width_units.shader_value())?;
            model.upload_uniforms(&ctx.queue);
        }
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        // When drawing 3d: draw wireframe first so it doesn't get occluded by depth test
        if let Some(wireframe) = &self.wireframe {
            wireframe.draw(pass)?;
        }
        if let Some(fill) = &self.fill {
            fill.draw(pass)?;
        }
        // When drawing 2d: draw fill before stroke so that the outline is always on top
        if let Some(stroke) = &self.stroke {
            stroke.draw(pass)?;
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for model in self.models() {
            set_model_picking_active(model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(fill) = &self.fill {
            fill.draw_picking(pass)?;
        }
        if let Some(stroke) = &self.stroke {
            stroke.draw_picking(pass)?;
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

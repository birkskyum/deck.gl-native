//! Port of `@deck.gl/layers/src/column-layer/column-layer.ts`: extruded cylinders
//! (tesselated regular polygons) at given coordinates.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_positions};
use deck_gl::layer::{depth_bias_for_layer, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use glam::Vec2;
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from};
use luma_gl::{assemble_shader, Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/column_layer.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GeometryVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstancePositions {
    position: [f32; 3],
    position_low: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    elevation: f32,
    fill_color: [u8; 4],
    line_color: [u8; 4],
    stroke_width: f32,
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
        let last = *vertices.last().unwrap();
        vertices.push(last);
    }
    // top: 0, -1, 1, -2, 2, -3, 3, ...
    for j in (if extruded { 0 } else { 1 })..verts_around_edge {
        let v = (j / 2) as i64 * if j % 2 == 0 { 1 } else { -1 };
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
#[derive(Clone, Debug)]
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
pub struct ColumnLayer {
    props: ColumnLayerProps,
    fill: Option<Model>,
    stroke: Option<Model>,
    wireframe: Option<Model>,
    data_dirty: bool,
}

impl ColumnLayer {
    pub fn new(props: ColumnLayerProps) -> Self {
        Self {
            props,
            fill: None,
            stroke: None,
            wireframe: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &ColumnLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: ColumnLayerProps) {
        self.props = props;
        self.data_dirty = true;
    }

    fn models(&mut self) -> impl Iterator<Item = &mut Model> {
        [&mut self.wireframe, &mut self.fill, &mut self.stroke]
            .into_iter()
            .flatten()
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;

        let positions = resolve_positions(data, &props.get_position)?;
        let elevations = resolve_f32(data, &props.get_elevation)?;
        let fill_colors = resolve_colors(data, &props.get_fill_color)?;
        let line_colors = resolve_colors(data, &props.get_line_color)?;
        let widths = resolve_f32(data, &props.get_line_width)?;

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
                elevation: elevations[i],
                fill_color: fill_colors[i],
                line_color: line_colors[i],
                stroke_width: widths[i],
            })
            .collect();
        let positions_buffer = create_vertex_buffer_from(device, "instancePositions", &instance_positions);
        let data_buffer = create_vertex_buffer_from(device, "instanceData", &instance_data);
        let count = data.len() as u32;
        for model in self.models() {
            model.set_vertex_buffer("instancePositions", positions_buffer.clone())?;
            model.set_vertex_buffer("instanceData", data_buffer.clone())?;
            model.set_instance_count(count);
        }
        Ok(())
    }
}

impl Layer for ColumnLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let modules: Vec<ShaderModuleSource> = STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect();
        let shader = assemble_shader(&props.base.id, &modules, SHADER)?;
        let layouts = [
            VertexBufferLayout::interleaved(
                "geometry",
                std::mem::size_of::<GeometryVertex>() as u64,
                wgpu::VertexStepMode::Vertex,
                &[(0, VertexFormat::Float32x3, 0), (1, VertexFormat::Float32x3, 12)],
            ),
            VertexBufferLayout::interleaved(
                "instancePositions",
                std::mem::size_of::<InstancePositions>() as u64,
                wgpu::VertexStepMode::Instance,
                &[(2, VertexFormat::Float32x3, 0), (3, VertexFormat::Float32x3, 12)],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<InstanceData>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (4, VertexFormat::Float32, 0),
                    (5, VertexFormat::Unorm8x4, 4),
                    (6, VertexFormat::Unorm8x4, 8),
                    (7, VertexFormat::Float32, 12),
                ],
            ),
        ];
        let geometry = tesselate_column(props.disk_resolution, props.extruded || props.stroked);
        let geometry_buffer = create_vertex_buffer_from(&ctx.device, "geometry", &geometry.vertices);
        let make = |label: String, topology: wgpu::PrimitiveTopology| -> Result<Model> {
            let mut desc = ModelDescriptor::new(&label, &shader, &layouts, topology, ctx.target);
            desc.depth_bias = depth_bias_for_layer(ctx.layer_index);
            desc.pickable = props.base.pickable;
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
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
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
}

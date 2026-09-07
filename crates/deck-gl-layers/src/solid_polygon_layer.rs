//! Port of `@deck.gl/layers/src/solid-polygon-layer/solid-polygon-layer.ts`.

use deck_gl::data::{resolve_colors, resolve_f32, resolve_polygons};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::math_gl::web_mercator::lng_lat_to_world;
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, CoordinateSystem, Layer, LayerContext, LayerData, LayerProps, Polygon, Result, Viewport,
};
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from, split_f64};
use luma_gl::{assemble_shader, Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::polygon::tesselate;

const COMMON: &str = include_str!("wgsl/solid_polygon_layer_common.wgsl");

/// Per-vertex data interleaved in one buffer, like deck.gl's `solid-polygon-instance-data`
/// buffer group. Keeps the side model within WebGPU's default limit of 8 vertex buffers.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexData {
    elevation: f32,
    fill_color: [u8; 4],
    line_color: [u8; 4],
    row_index: u32,
}

const VERTEX_DATA_STRIDE: u64 = std::mem::size_of::<VertexData>() as u64;

/// Layout of the interleaved buffer given the shader locations of its four attributes.
fn vertex_data_layout(step_mode: wgpu::VertexStepMode, locations: [u32; 4]) -> VertexBufferLayout {
    VertexBufferLayout::interleaved(
        "vertexData",
        VERTEX_DATA_STRIDE,
        step_mode,
        &[
            (locations[0], VertexFormat::Float32, 0),
            (locations[1], VertexFormat::Unorm8x4, 4),
            (locations[2], VertexFormat::Unorm8x4, 8),
            (locations[3], VertexFormat::Uint32, 12),
        ],
    )
}
const TOP: &str = include_str!("wgsl/solid_polygon_layer_top.wgsl");
const SIDE: &str = include_str!("wgsl/solid_polygon_layer_side.wgsl");

/// Properties of a [`SolidPolygonLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct SolidPolygonLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Whether to fill the polygons
    pub filled: bool,
    /// Whether to extrude the polygons by elevation
    pub extruded: bool,
    /// Whether to draw the outline of extruded polygons
    pub wireframe: bool,
    pub elevation_scale: f32,
    pub get_polygon: Accessor<Polygon>,
    /// Elevation in meters, used when `extruded`
    pub get_elevation: Accessor<f32>,
    pub get_fill_color: Accessor<Color>,
    pub get_line_color: Accessor<Color>,
}

impl Default for SolidPolygonLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("SolidPolygonLayer"),
            data: LayerData::default(),
            filled: true,
            extruded: false,
            wireframe: false,
            elevation_scale: 1.0,
            get_polygon: Accessor::column("polygon"),
            get_elevation: Accessor::Constant(1000.0),
            get_fill_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
        }
    }
}

/// Renders filled and optionally extruded polygons.
pub struct SolidPolygonLayer {
    props: SolidPolygonLayerProps,
    top: Option<Model>,
    side: Option<Model>,
    wireframe: Option<Model>,
    data_dirty: bool,
}

impl SolidPolygonLayer {
    pub fn new(props: SolidPolygonLayerProps) -> Self {
        Self {
            props,
            top: None,
            side: None,
            wireframe: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &SolidPolygonLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: SolidPolygonLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn modules(&self) -> Vec<ShaderModuleSource> {
        STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect()
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;

        let polygons = resolve_polygons(data, &props.get_polygon)?;
        let elevations = resolve_f32(data, &props.get_elevation)?;
        let fill_colors = resolve_colors(data, &props.get_fill_color)?;
        let line_colors = resolve_colors(data, &props.get_line_color)?;

        // When tesselating lnglat coordinates, project them to the common space for accuracy
        let geospatial = !matches!(props.base.coordinate_system, CoordinateSystem::Cartesian);
        let tesselated = tesselate(&polygons, |p| {
            if geospatial {
                lng_lat_to_world([p[0], p[1].clamp(-89.9, 89.9)])
            } else {
                [p[0], p[1]]
            }
        });

        let vertex_count = tesselated.vertex_count();
        let flat: Vec<f64> = tesselated.positions.iter().flatten().copied().collect();
        let (hi, lo) = split_f64(&flat);
        // nextVertexPositions is the position buffer shifted by one vertex
        let mut next: Vec<f64> = flat[3.min(flat.len())..].to_vec();
        next.extend_from_slice(&flat[flat.len().saturating_sub(3)..]);
        let (next_hi, next_lo) = split_f64(&next);

        let vertex_data: Vec<VertexData> = tesselated
            .row_index
            .iter()
            .map(|&r| VertexData {
                elevation: elevations[r as usize],
                fill_color: fill_colors[r as usize],
                line_color: line_colors[r as usize],
                row_index: data.source_row(r as usize),
            })
            .collect();

        let buffers = [
            (
                "vertexPositions",
                create_vertex_buffer_from(device, "vertexPositions", &hi),
            ),
            (
                "vertexPositions64Low",
                create_vertex_buffer_from(device, "vertexPositions64Low", &lo),
            ),
            (
                "nextVertexPositions",
                create_vertex_buffer_from(device, "nextVertexPositions", &next_hi),
            ),
            (
                "nextVertexPositions64Low",
                create_vertex_buffer_from(device, "nextVertexPositions64Low", &next_lo),
            ),
            (
                "vertexValid",
                create_vertex_buffer_from(device, "vertexValid", &tesselated.vertex_valid),
            ),
            (
                "vertexData",
                create_vertex_buffer_from(device, "vertexData", &vertex_data),
            ),
        ];

        if let Some(top) = &mut self.top {
            for (name, buffer) in &buffers {
                if matches!(
                    *name,
                    "nextVertexPositions" | "nextVertexPositions64Low" | "vertexValid"
                ) {
                    continue;
                }
                top.set_vertex_buffer(name, buffer.clone())?;
            }
            let index_buffer = create_index_buffer(device, "indices", &tesselated.indices);
            top.set_index_buffer(
                index_buffer,
                wgpu::IndexFormat::Uint32,
                tesselated.indices.len() as u32,
            );
        }
        for model in [&mut self.side, &mut self.wireframe].into_iter().flatten() {
            for (name, buffer) in &buffers {
                model.set_vertex_buffer(name, buffer.clone())?;
            }
            model.set_instance_count(vertex_count.saturating_sub(1) as u32);
        }
        Ok(())
    }
}

impl Layer for SolidPolygonLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let id = self.props.base.id.clone();
        let modules = self.modules();
        let depth_bias = ctx.depth_bias();
        let pickable = self.props.base.pickable;
        let parameters = self.props.base.parameters;
        let top_label = format!("{id}-top");
        let side_label = format!("{id}-side");
        let wireframe_label = format!("{id}-wireframe");

        if self.props.filled {
            let top_shader = assemble_shader(&format!("{id}-top"), &modules, &format!("{COMMON}\n{TOP}"))?;
            let layouts = [
                VertexBufferLayout::vertex("vertexPositions", 0, VertexFormat::Float32x3),
                VertexBufferLayout::vertex("vertexPositions64Low", 1, VertexFormat::Float32x3),
                vertex_data_layout(wgpu::VertexStepMode::Vertex, [2, 3, 4, 5]),
            ];
            let mut desc = ModelDescriptor::new(
                &top_label,
                &top_shader,
                &layouts,
                wgpu::PrimitiveTopology::TriangleList,
                ctx.target,
            );
            desc.depth_bias = depth_bias;
            desc.pickable = pickable;
            parameters.apply(&mut desc);
            self.top = Some(Model::new(&ctx.device, &desc)?);
        }

        if self.props.extruded {
            let side_shader = assemble_shader(&format!("{id}-side"), &modules, &format!("{COMMON}\n{SIDE}"))?;
            let layouts = [
                VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x2),
                VertexBufferLayout::instance("vertexPositions", 1, VertexFormat::Float32x3),
                VertexBufferLayout::instance("vertexPositions64Low", 2, VertexFormat::Float32x3),
                VertexBufferLayout::instance("nextVertexPositions", 3, VertexFormat::Float32x3),
                VertexBufferLayout::instance("nextVertexPositions64Low", 4, VertexFormat::Float32x3),
                VertexBufferLayout::instance("vertexValid", 5, VertexFormat::Float32),
                vertex_data_layout(wgpu::VertexStepMode::Instance, [6, 7, 8, 9]),
            ];
            let mut desc = ModelDescriptor::new(
                &side_label,
                &side_shader,
                &layouts,
                wgpu::PrimitiveTopology::TriangleStrip,
                ctx.target,
            );
            desc.depth_bias = depth_bias;
            desc.pickable = pickable;
            parameters.apply(&mut desc);
            let mut side = Model::new(&ctx.device, &desc)?;
            // top right - top left - bottom right - bottom left
            let side_positions: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            side.set_vertex_buffer(
                "positions",
                create_vertex_buffer_from(&ctx.device, "positions", &side_positions),
            )?;
            side.set_vertex_count(4);
            self.side = Some(side);

            if self.props.wireframe {
                let mut desc = ModelDescriptor::new(
                    &wireframe_label,
                    &side_shader,
                    &layouts,
                    wgpu::PrimitiveTopology::LineStrip,
                    ctx.target,
                );
                desc.depth_bias = depth_bias;
                desc.pickable = pickable;
                parameters.apply(&mut desc);
                let mut wireframe = Model::new(&ctx.device, &desc)?;
                // top right - top left - bottom left - bottom right
                let wire_positions: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
                wireframe.set_vertex_buffer(
                    "positions",
                    create_vertex_buffer_from(&ctx.device, "positions", &wire_positions),
                )?;
                wireframe.set_vertex_count(4);
                self.wireframe = Some(wireframe);
            }
        }
        self.data_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = &self.props;
        let models = [
            (&mut self.top, false),
            (&mut self.side, false),
            (&mut self.wireframe, true),
        ];
        for (model, is_wireframe) in models {
            let Some(model) = model else { continue };
            update_standard_uniforms(model, ctx, viewport, &props.base)?;
            let u = model.uniforms("solidPolygon")?;
            u.set_f32("extruded", if props.extruded { 1.0 } else { 0.0 })?;
            u.set_f32("isWireframe", if is_wireframe { 1.0 } else { 0.0 })?;
            u.set_f32("elevationScale", props.elevation_scale)?;
            model.upload_uniforms(&ctx.queue);
        }
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        // Same order as deck.gl: sides, wireframe, then top
        if self.props.extruded {
            if let Some(side) = &self.side {
                side.draw(pass)?;
            }
            if self.props.wireframe {
                if let Some(wireframe) = &self.wireframe {
                    wireframe.draw(pass)?;
                }
            }
        }
        if self.props.filled {
            if let Some(top) = &self.top {
                top.draw(pass)?;
            }
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for model in [&mut self.top, &mut self.side, &mut self.wireframe]
            .into_iter()
            .flatten()
        {
            set_model_picking_active(model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if self.props.extruded {
            if let Some(side) = &self.side {
                side.draw_picking(pass)?;
            }
        }
        if self.props.filled {
            if let Some(top) = &self.top {
                top.draw_picking(pass)?;
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

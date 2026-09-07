//! Port of `@deck.gl/aggregation-layers` ScreenGridLayer with CPU aggregation: points are
//! binned into a grid of screen pixels, re-aggregated whenever the view changes, and each
//! cell is drawn as a screen space square coloured by its aggregated weight.

use std::collections::HashMap;

use deck_gl::data::{resolve_f32, resolve_positions};
use deck_gl::glam::{DMat4, DVec3};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, Result, Viewport};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::aggregation::{
    aggregate, sample_color_range, AggregationOperation, ScaleType, DEFAULT_COLOR_RANGE,
};

const SHADER: &str = include_str!("wgsl/screen_grid_layer.wgsl");

/// Properties of a [`ScreenGridLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenGridLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Cell size in logical pixels
    pub cell_size_pixels: f32,
    /// Gap between cells in logical pixels
    pub cell_margin_pixels: f32,
    /// Value range mapped onto `color_range`; the data's min and max when unset
    pub color_domain: Option<[f32; 2]>,
    pub color_range: Vec<Color>,
    /// `Linear` or `Quantize`
    pub color_scale_type: ScaleType,
    pub aggregation: AggregationOperation,
    pub get_position: Accessor<Position>,
    pub get_weight: Accessor<f32>,
}

impl Default for ScreenGridLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("ScreenGridLayer"),
            data: LayerData::default(),
            cell_size_pixels: 100.0,
            cell_margin_pixels: 2.0,
            color_domain: None,
            color_range: DEFAULT_COLOR_RANGE.to_vec(),
            color_scale_type: ScaleType::Linear,
            aggregation: AggregationOperation::Sum,
            get_position: Accessor::column("position"),
            get_weight: Accessor::Constant(1.0),
        }
    }
}

/// One screen grid cell after aggregation; a pick's `index` refers into [`ScreenGridLayer::bins`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenGridBin {
    /// Column from the left of the viewport
    pub col: u32,
    /// Row from the top of the viewport
    pub row: u32,
    pub value: f32,
    pub count: usize,
    pub point_indices: Vec<usize>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CellInstance {
    cell: [f32; 2],
    color: [u8; 4],
    row_index: u32,
}

/// Aggregates points into cells of screen pixels.
pub struct ScreenGridLayer {
    props: ScreenGridLayerProps,
    model: Option<Model>,
    positions: Vec<Position>,
    weights: Vec<f32>,
    bins: Vec<ScreenGridBin>,
    color_domain: [f32; 2],
    data_dirty: bool,
    /// The view the bins were computed for
    last_view: Option<(DMat4, f64, f64)>,
}

impl ScreenGridLayer {
    pub fn new(props: ScreenGridLayerProps) -> Self {
        Self {
            props,
            model: None,
            positions: Vec::new(),
            weights: Vec::new(),
            bins: Vec::new(),
            color_domain: [0.0, 0.0],
            data_dirty: true,
            last_view: None,
        }
    }

    pub fn props(&self) -> &ScreenGridLayerProps {
        &self.props
    }

    /// Replace the props. Points are re-read on the next update when they changed.
    pub fn set_props(&mut self, props: ScreenGridLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    /// The cells after the last update.
    pub fn bins(&self) -> &[ScreenGridBin] {
        &self.bins
    }

    /// Min and max of the aggregated values after the last update.
    pub fn color_domain(&self) -> [f32; 2] {
        self.color_domain
    }

    /// Bin the points for a viewport. Returns the bins and the value domain.
    pub fn aggregate(
        positions: &[Position],
        weights: &[f32],
        viewport: &Viewport,
        cell_size: f32,
        operation: AggregationOperation,
    ) -> (Vec<ScreenGridBin>, [f32; 2]) {
        let mut index_of: HashMap<(u32, u32), usize> = HashMap::new();
        let mut ids = Vec::new();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, p) in positions.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                continue;
            }
            let screen = viewport.project(DVec3::new(p[0], p[1], p[2]), true);
            if screen.x < 0.0 || screen.x >= viewport.width || screen.y < 0.0 || screen.y >= viewport.height {
                continue;
            }
            let id = (
                (screen.x / cell_size as f64).floor() as u32,
                (screen.y / cell_size as f64).floor() as u32,
            );
            let slot = *index_of.entry(id).or_insert_with(|| {
                ids.push(id);
                members.push(Vec::new());
                ids.len() - 1
            });
            members[slot].push(i);
        }
        let (values, domain) = aggregate(&members, weights, operation);
        let bins = ids
            .into_iter()
            .zip(members)
            .zip(values)
            .map(|(((col, row), point_indices), value)| ScreenGridBin {
                col,
                row,
                value,
                count: point_indices.len(),
                point_indices,
            })
            .collect();
        (bins, domain)
    }

    fn update_cells(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        let props = &self.props;
        let (bins, domain) = Self::aggregate(
            &self.positions,
            &self.weights,
            viewport,
            props.cell_size_pixels,
            props.aggregation,
        );
        let color_domain = props.color_domain.unwrap_or(domain);
        let [d0, d1] = color_domain;
        let instances: Vec<CellInstance> = bins
            .iter()
            .enumerate()
            .map(|(i, bin)| {
                let ratio = if d1 == d0 {
                    1.0
                } else {
                    ((bin.value - d0) / (d1 - d0)).clamp(0.0, 1.0)
                };
                CellInstance {
                    cell: [bin.col as f32, bin.row as f32],
                    color: sample_color_range(ratio, &props.color_range, props.color_scale_type),
                    row_index: i as u32,
                }
            })
            .collect();
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(&ctx.device, "instanceData", &instances),
        )?;
        model.set_instance_count(instances.len() as u32);
        self.bins = bins;
        self.color_domain = color_domain;
        Ok(())
    }
}

impl Layer for ScreenGridLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x2),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<CellInstance>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (1, VertexFormat::Float32x2, 0),
                    (2, VertexFormat::Unorm8x4, 8),
                    (3, VertexFormat::Uint32, 12),
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
        // an overlay in screen space, drawn over whatever is below
        desc.depth_compare = wgpu::CompareFunction::Always;
        desc.depth_write_enabled = false;
        desc.pickable = self.props.base.pickable;
        self.props.base.parameters.apply(&mut desc);
        let mut model = Model::new(&ctx.device, &desc)?;
        let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(4);
        self.model = Some(model);
        self.data_dirty = true;
        self.last_view = None;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.positions = resolve_positions(&self.props.data, &self.props.get_position)?;
            self.weights = resolve_f32(&self.props.data, &self.props.get_weight)?;
            self.data_dirty = false;
            self.last_view = None;
        }
        let view = (viewport.view_projection_matrix, viewport.width, viewport.height);
        if self.last_view != Some(view) {
            self.update_cells(ctx, viewport)?;
            self.last_view = Some(view);
        }
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;
        let u = model.uniforms("screenGrid")?;
        u.set_f32("cellSizePixels", props.cell_size_pixels)?;
        u.set_f32("cellMarginPixels", props.cell_margin_pixels)?;
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

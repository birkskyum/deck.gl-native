//! Port of `@deck.gl/aggregation-layers/src/contour-layer/contour-layer.ts`: aggregates points
//! into a grid of `cell_size` meters and draws isolines and isobands through marching squares.

use std::collections::HashMap;
use std::sync::Arc;

use deck_gl::data::resolve_positions;
use deck_gl::glam::{DMat4, DVec3};
use deck_gl::{
    Accessor, Color, CoordinateSystem, Layer, LayerContext, LayerData, LayerProps, Position, Result,
    SubLayers, Unit, Viewport,
};

use crate::aggregation::{
    aggregate_points, common_frame, gridbin_origin, point_to_gridbin, AggregatedBin, Aggregation,
    AggregationOperation, AggregationProps, BinSpace,
};
use crate::marching_squares::{generate_contours, ContourGeometry, ContourThreshold};
use crate::{PathLayer, PathLayerProps, SolidPolygonLayer, SolidPolygonLayerProps};

pub const DEFAULT_CONTOUR_COLOR: Color = [255, 255, 255, 255];

/// One isoline or isoband to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contour {
    pub threshold: ContourThreshold,
    pub color: Color,
    /// Width of an isoline in pixels
    pub stroke_width: f32,
    /// Draw order; the contour's index when unset
    pub z_index: Option<i32>,
}

impl Contour {
    pub fn line(threshold: f32) -> Self {
        Self {
            threshold: ContourThreshold::Line(threshold),
            color: DEFAULT_CONTOUR_COLOR,
            stroke_width: 1.0,
            z_index: None,
        }
    }

    pub fn band(min: f32, max: f32) -> Self {
        Self {
            threshold: ContourThreshold::Band([min, max]),
            ..Self::line(0.0)
        }
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContourLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub get_position: Accessor<Position>,
    pub get_weight: Accessor<f32>,
    /// Size of each cell in meters
    pub cell_size: f64,
    /// Origin of the grid in common units
    pub grid_origin: [f64; 2],
    /// SUM, MEAN, MIN or MAX of the weights in a cell
    pub aggregation: AggregationOperation,
    pub contours: Vec<Contour>,
    /// A small z offset added per contour index so later contours draw above earlier ones
    pub z_offset: f64,
}

impl Default for ContourLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("contours"),
            data: LayerData::default(),
            get_position: Accessor::column("position"),
            get_weight: Accessor::Constant(1.0),
            cell_size: 1000.0,
            grid_origin: [0.0, 0.0],
            aggregation: AggregationOperation::Sum,
            contours: vec![Contour::line(1.0)],
            z_offset: 0.005,
        }
    }
}

/// The aggregated grid the contours are computed from.
#[derive(Clone, Debug, Default)]
pub struct ContourGrid {
    pub bins: Vec<AggregatedBin>,
    /// Cell size in common units
    pub cell_size_common: [f64; 2],
    /// Origin of cell (0, 0) in common units
    pub cell_origin_common: [f64; 2],
    /// Exclusive ranges of the column and row indices
    pub x_range: [i64; 2],
    pub y_range: [i64; 2],
}

impl ContourGrid {
    /// The aggregated value of a cell, NaN when empty.
    pub fn value_reader(&self) -> impl Fn(i64, i64) -> f32 + '_ {
        let values: HashMap<(i64, i64), f32> = self
            .bins
            .iter()
            .map(|b| ((b.col, b.row), b.color_value))
            .collect();
        move |x, y| values.get(&(x, y)).copied().unwrap_or(f32::NAN)
    }
}

pub struct ContourLayer {
    props: ContourLayerProps,
    sub_layers: SubLayers,
    dirty: bool,
    grid: ContourGrid,
}

impl ContourLayer {
    pub fn new(props: ContourLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            dirty: true,
            grid: ContourGrid::default(),
        }
    }

    pub fn props(&self) -> &ContourLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: ContourLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    /// The grid of the last update.
    pub fn grid(&self) -> &ContourGrid {
        &self.grid
    }

    /// Aggregate the data into the grid, without a GPU.
    pub fn aggregate(props: &ContourLayerProps) -> Result<ContourGrid> {
        let positions = resolve_positions(&props.data, &props.get_position)?;
        let Some((_, centroid_common, scales)) = common_frame(&positions) else {
            return Ok(ContourGrid::default());
        };
        let size = [
            scales.units_per_meter.x * props.cell_size,
            scales.units_per_meter.y * props.cell_size,
        ];
        let origin = [
            ((centroid_common[0] - props.grid_origin[0]) / size[0]).floor() * size[0] + props.grid_origin[0],
            ((centroid_common[1] - props.grid_origin[1]) / size[1]).floor() * size[1] + props.grid_origin[1],
        ];
        let space = BinSpace {
            bin_of: Box::new(move |p| point_to_gridbin(p, size)),
            anchor_of: Box::new(move |bin| gridbin_origin(bin, size)),
            origin,
        };
        let aggregation = AggregationProps {
            get_position: props.get_position.clone(),
            get_color_weight: props.get_weight.clone(),
            color_aggregation: props.aggregation,
            ..Default::default()
        };
        let Aggregation { bins, .. } = aggregate_points(&props.data, &aggregation, &space)?;
        let mut x_range = [i64::MAX, i64::MIN];
        let mut y_range = [i64::MAX, i64::MIN];
        for bin in &bins {
            x_range = [x_range[0].min(bin.col), x_range[1].max(bin.col + 1)];
            y_range = [y_range[0].min(bin.row), y_range[1].max(bin.row + 1)];
        }
        if bins.is_empty() {
            x_range = [0, 1];
            y_range = [0, 1];
        }
        Ok(ContourGrid {
            bins,
            cell_size_common: size,
            cell_origin_common: origin,
            x_range,
            y_range,
        })
    }

    fn render_layers(&mut self) -> Result<Vec<Box<dyn Layer>>> {
        let props = &self.props;
        self.grid = Self::aggregate(props)?;
        let grid = &self.grid;
        if grid.bins.is_empty() {
            return Ok(Vec::new());
        }
        let thresholds: Vec<(ContourThreshold, Option<i32>)> =
            props.contours.iter().map(|c| (c.threshold, c.z_index)).collect();
        let (lines, polygons) =
            generate_contours(&thresholds, &grid.value_reader(), grid.x_range, grid.y_range);
        let model_matrix = DMat4::from_translation(DVec3::new(
            grid.cell_origin_common[0],
            grid.cell_origin_common[1],
            0.0,
        )) * DMat4::from_scale(DVec3::new(
            grid.cell_size_common[0],
            grid.cell_size_common[1],
            props.z_offset,
        ));
        let sub_base = |suffix: &str| LayerProps {
            id: format!("{}-{suffix}", props.base.id),
            coordinate_system: CoordinateSystem::Cartesian,
            model_matrix: Some(model_matrix),
            ..props.base.clone()
        };
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        if !lines.is_empty() {
            let contours = props.contours.clone();
            let widths: Arc<Vec<f32>> =
                Arc::new(lines.iter().map(|l| contours[l.contour].stroke_width).collect());
            let colors: Arc<Vec<Color>> = Arc::new(lines.iter().map(|l| contours[l.contour].color).collect());
            let paths: Arc<Vec<ContourGeometry>> = Arc::new(lines);
            let count = paths.len();
            let p = paths.clone();
            layers.push(Box::new(PathLayer::new(PathLayerProps {
                base: sub_base("lines"),
                data: LayerData::with_length(count),
                get_path: Accessor::Func(Arc::new(move |i| p[i].vertices.clone())),
                get_color: Accessor::Func(Arc::new(move |i| colors[i])),
                get_width: Accessor::Func(Arc::new(move |i| widths[i])),
                width_units: Unit::Pixels,
                ..Default::default()
            })));
        }
        if !polygons.is_empty() {
            let contours = props.contours.clone();
            let colors: Arc<Vec<Color>> =
                Arc::new(polygons.iter().map(|p| contours[p.contour].color).collect());
            let rings: Arc<Vec<ContourGeometry>> = Arc::new(polygons);
            let count = rings.len();
            let r = rings.clone();
            layers.push(Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
                base: sub_base("bands"),
                data: LayerData::with_length(count),
                get_polygon: Accessor::Func(Arc::new(move |i| vec![r[i].vertices.clone()])),
                get_fill_color: Accessor::Func(Arc::new(move |i| colors[i])),
                filled: true,
                extruded: false,
                ..Default::default()
            })));
        }
        Ok(layers)
    }
}

impl Layer for ContourLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.dirty || self.sub_layers.is_empty() {
            let layers = self.render_layers()?;
            self.sub_layers.replace(layers);
            self.dirty = false;
        }
        self.sub_layers.update(ctx, viewport)
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.sub_layers.draw(ctx, pass)
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        self.sub_layers.set_picking_active(ctx, active)
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.sub_layers.draw_picking(ctx, pass)
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
        self.sub_layers.set_highlighted_object(index);
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

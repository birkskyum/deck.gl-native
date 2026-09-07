//! Port of `@deck.gl/aggregation-layers` GridLayer with CPU aggregation: points are binned
//! into square cells of `cell_size` meters and rendered as square columns coloured and
//! extruded by their aggregated weights.

use std::f32::consts::FRAC_1_SQRT_2;

use deck_gl::data::resolve_positions;
use deck_gl::{Accessor, Layer, LayerContext, LayerData, LayerProps, Result, SubLayers, Unit, Viewport};

use crate::aggregation::{
    aggregate_points, cell_accessors, common_frame, gridbin_origin, point_to_gridbin, AggregatedBin,
    Aggregation, AggregationProps, BinSpace,
};
use crate::column_layer::{ColumnLayer, ColumnLayerProps};

/// Properties of a [`GridLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct GridLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Cell size in meters
    pub cell_size: f64,
    pub aggregation: AggregationProps,
}

impl Default for GridLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("GridLayer"),
            data: LayerData::default(),
            cell_size: 1000.0,
            aggregation: AggregationProps::default(),
        }
    }
}

/// Aggregates points into square grid cells.
pub struct GridLayer {
    props: GridLayerProps,
    sub_layers: SubLayers,
    aggregation: Aggregation,
    dirty: bool,
}

impl GridLayer {
    pub fn new(props: GridLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            aggregation: Aggregation::default(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &GridLayerProps {
        &self.props
    }

    /// Replace the props. Bins are recomputed on the next update when they changed.
    pub fn set_props(&mut self, props: GridLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    /// The bins after the last update; a pick's `index` refers into this list.
    pub fn bins(&self) -> &[AggregatedBin] {
        &self.aggregation.bins
    }

    pub fn color_domain(&self) -> [f32; 2] {
        self.aggregation.color_domain
    }

    pub fn elevation_domain(&self) -> [f32; 2] {
        self.aggregation.elevation_domain
    }

    fn cell_size_common(props: &GridLayerProps, scales: &math_gl::web_mercator::DistanceScales) -> [f64; 2] {
        [
            scales.units_per_meter.x * props.cell_size,
            scales.units_per_meter.y * props.cell_size,
        ]
    }

    /// Aggregate the data without a GPU, for tests and hosts that want the bins.
    pub fn aggregate(props: &GridLayerProps) -> Result<Aggregation> {
        let positions = resolve_positions(&props.data, &props.aggregation.get_position)?;
        let Some((_, centroid_common, scales)) = common_frame(&positions) else {
            return Ok(Aggregation::default());
        };
        let size = Self::cell_size_common(props, &scales);
        let origin = [
            (centroid_common[0] / size[0]).floor() * size[0],
            (centroid_common[1] / size[1]).floor() * size[1],
        ];
        let space = BinSpace {
            bin_of: Box::new(move |p| point_to_gridbin(p, size)),
            anchor_of: Box::new(move |bin| gridbin_origin(bin, size)),
            origin,
        };
        aggregate_points(&props.data, &props.aggregation, &space)
    }

    fn render_layers(&mut self) -> Result<Vec<Box<dyn Layer>>> {
        let props = &self.props;
        self.aggregation = Self::aggregate(props)?;
        let positions = resolve_positions(&props.data, &props.aggregation.get_position)?;
        let Some((_, _, scales)) = common_frame(&positions) else {
            return Ok(Vec::new());
        };
        let size = Self::cell_size_common(props, &scales);
        let (data, get_position, get_fill_color, get_elevation) = cell_accessors(&self.aggregation);
        // A square inscribed in the circle of radius size / sqrt(2), rotated by 45 degrees and
        // shifted so that the cell's lower left corner is at the bin position.
        let cells = ColumnLayer::new(ColumnLayerProps {
            base: LayerProps {
                id: format!("{}-cells", props.base.id),
                ..props.base.clone()
            },
            data,
            disk_resolution: 4,
            radius: (size[0] * std::f64::consts::FRAC_1_SQRT_2) as f32,
            radius_units: Unit::Common,
            angle: 45.0,
            offset: [FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            coverage: props.aggregation.coverage,
            extruded: props.aggregation.extruded,
            elevation_scale: 1.0,
            filled: true,
            stroked: false,
            wireframe: false,
            get_position,
            get_fill_color,
            get_elevation,
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_width: Accessor::Constant(0.0),
            ..Default::default()
        });
        Ok(vec![Box::new(cells)])
    }
}

impl Layer for GridLayer {
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

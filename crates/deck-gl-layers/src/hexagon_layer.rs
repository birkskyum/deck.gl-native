//! Port of `@deck.gl/aggregation-layers` HexagonLayer with CPU aggregation: points are binned
//! into hexagons of `radius` meters, weights are aggregated per bin, and the bins render as
//! hexagonal columns coloured and extruded by their values.

use deck_gl::data::resolve_positions;
use deck_gl::{Accessor, Layer, LayerContext, LayerData, LayerProps, Result, SubLayers, Unit, Viewport};

use crate::aggregation::{
    aggregate_points, cell_accessors, common_frame, hexbin_centroid, point_to_hexbin, AggregatedBin,
    Aggregation, AggregationProps, BinSpace,
};
use crate::column_layer::{ColumnLayer, ColumnLayerProps};

/// Properties of a [`HexagonLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct HexagonLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Hexagon radius in meters
    pub radius: f64,
    pub aggregation: AggregationProps,
}

impl Default for HexagonLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("HexagonLayer"),
            data: LayerData::default(),
            radius: 1000.0,
            aggregation: AggregationProps::default(),
        }
    }
}

/// Aggregates points into hexagonal bins.
pub struct HexagonLayer {
    props: HexagonLayerProps,
    sub_layers: SubLayers,
    aggregation: Aggregation,
    dirty: bool,
}

impl HexagonLayer {
    pub fn new(props: HexagonLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            aggregation: Aggregation::default(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &HexagonLayerProps {
        &self.props
    }

    /// Replace the props. Bins are recomputed on the next update when they changed.
    pub fn set_props(&mut self, props: HexagonLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    /// The bins after the last update; a pick's `index` refers into this list.
    pub fn bins(&self) -> &[AggregatedBin] {
        &self.aggregation.bins
    }

    /// Data domain of the aggregated colour values (min and max over the bins).
    pub fn color_domain(&self) -> [f32; 2] {
        self.aggregation.color_domain
    }

    pub fn elevation_domain(&self) -> [f32; 2] {
        self.aggregation.elevation_domain
    }

    /// Aggregate the data without a GPU, for tests and hosts that want the bins.
    pub fn aggregate(props: &HexagonLayerProps) -> Result<Aggregation> {
        let positions = resolve_positions(&props.data, &props.aggregation.get_position)?;
        let Some((_, centroid_common, scales)) = common_frame(&positions) else {
            return Ok(Aggregation::default());
        };
        let radius_common = scales.units_per_meter.x * props.radius;
        let center = point_to_hexbin(centroid_common, radius_common);
        let origin = hexbin_centroid(center, radius_common);
        let space = BinSpace {
            bin_of: Box::new(move |p| point_to_hexbin(p, radius_common)),
            anchor_of: Box::new(move |bin| hexbin_centroid(bin, radius_common)),
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
        let radius_common = scales.units_per_meter.x * props.radius;
        let (data, get_position, get_fill_color, get_elevation) = cell_accessors(&self.aggregation);
        let cells = ColumnLayer::new(ColumnLayerProps {
            base: LayerProps {
                id: format!("{}-cells", props.base.id),
                ..props.base.clone()
            },
            data,
            disk_resolution: 6,
            radius: radius_common as f32,
            radius_units: Unit::Common,
            // pointy top hexagons, like d3-hexbin
            angle: 30.0,
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

impl Layer for HexagonLayer {
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

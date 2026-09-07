//! Port of `@deck.gl/layers` GridCellLayer: square cells of `cell_size` meters anchored at
//! their lower left corner, drawn by a [`ColumnLayer`] with a square footprint.

use std::f32::consts::FRAC_1_SQRT_2;

use deck_gl::{Accessor, Color, LayerData, LayerProps, Position, Unit};

use crate::column_layer::{ColumnLayer, ColumnLayerProps};

/// Properties of a grid cell layer. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct GridCellLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Cell size in meters
    pub cell_size: f32,
    /// Cell size as a fraction of `cell_size`, 0 to 1
    pub coverage: f32,
    pub elevation_scale: f32,
    pub extruded: bool,
    pub get_position: Accessor<Position>,
    pub get_fill_color: Accessor<Color>,
    pub get_elevation: Accessor<f32>,
}

impl Default for GridCellLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("GridCellLayer"),
            data: LayerData::default(),
            cell_size: 1000.0,
            coverage: 1.0,
            elevation_scale: 1.0,
            extruded: true,
            get_position: Accessor::column("position"),
            get_fill_color: Accessor::Constant([255, 0, 255, 255]),
            get_elevation: Accessor::Constant(1000.0),
        }
    }
}

impl From<GridCellLayerProps> for ColumnLayerProps {
    fn from(props: GridCellLayerProps) -> Self {
        ColumnLayerProps {
            base: props.base,
            data: props.data,
            disk_resolution: 4,
            // a square inscribed in the circle, rotated so its sides are axis aligned, and
            // shifted so that the position is its lower left corner
            radius: props.cell_size * FRAC_1_SQRT_2,
            radius_units: Unit::Meters,
            angle: 45.0,
            offset: [FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            coverage: props.coverage,
            elevation_scale: props.elevation_scale,
            extruded: props.extruded,
            filled: true,
            stroked: false,
            wireframe: false,
            get_position: props.get_position,
            get_fill_color: props.get_fill_color,
            get_elevation: props.get_elevation,
            ..Default::default()
        }
    }
}

/// A [`ColumnLayer`] configured as deck.gl's GridCellLayer.
pub type GridCellLayer = ColumnLayer;

impl ColumnLayer {
    pub fn grid_cells(props: GridCellLayerProps) -> ColumnLayer {
        ColumnLayer::new(props.into())
    }
}

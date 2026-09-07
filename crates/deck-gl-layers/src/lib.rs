//! Rust port of `@deck.gl/layers`: the core layer catalog on wgpu.
//!
//! Layer shaders are the WGSL sources deck.gl 9 ships, included verbatim from `src/wgsl`.

pub mod aggregation;
pub mod arc_layer;
pub mod bitmap_layer;
pub mod column_layer;
pub mod geojson_layer;
pub mod grid_cell_layer;
pub mod grid_layer;
pub mod hexagon_layer;
pub mod icon_layer;
pub mod line_layer;
pub mod path;
pub mod path_layer;
pub mod point_cloud_layer;
pub mod polygon;
pub mod polygon_layer;
pub mod scatterplot_layer;
pub mod screen_grid_layer;
pub mod solid_polygon_layer;
pub mod text;
pub mod text_layer;
pub mod trips_layer;

pub use aggregation::{
    AggregatedBin, AggregationOperation, AggregationProps, ScaleType, DEFAULT_COLOR_RANGE,
};
pub use arc_layer::{ArcLayer, ArcLayerProps};
pub use bitmap_layer::{BitmapImage, BitmapLayer, BitmapLayerProps};
pub use column_layer::{ColumnLayer, ColumnLayerProps};
pub use geojson_layer::{GeoJsonLayer, GeoJsonLayerProps};
pub use grid_cell_layer::{GridCellLayer, GridCellLayerProps};
pub use grid_layer::{GridLayer, GridLayerProps};
pub use hexagon_layer::{HexagonLayer, HexagonLayerProps};
pub use icon_layer::{IconAtlas, IconLayer, IconLayerProps, IconMapping};
pub use line_layer::{LineLayer, LineLayerProps};
pub use path_layer::{PathLayer, PathLayerProps};
pub use point_cloud_layer::{PointCloudLayer, PointCloudLayerProps};
pub use polygon_layer::{PolygonLayer, PolygonLayerProps};
pub use scatterplot_layer::{ScatterplotLayer, ScatterplotLayerProps};
pub use screen_grid_layer::{ScreenGridBin, ScreenGridLayer, ScreenGridLayerProps};
pub use solid_polygon_layer::{SolidPolygonLayer, SolidPolygonLayerProps};
pub use text::{CharacterSet, FontAtlas, FontSettings, FontSource, WordBreak};
pub use text_layer::{AlignmentBaseline, TextAnchor, TextLayer, TextLayerProps};
pub use trips_layer::{TripsLayer, TripsLayerProps};

//! Rust port of `@deck.gl/layers`: the core layer catalog on wgpu.
//!
//! Layer shaders are the WGSL sources deck.gl 9 ships, included verbatim from `src/wgsl`.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unreachable)
)]
pub mod aggregation;
pub mod arc_layer;
pub mod bitmap_layer;
pub mod column_layer;
pub mod contour_layer;
pub mod extensions;
pub mod fetch;
pub mod geo_cell_layer;
pub mod geojson_layer;
pub mod grid_cell_layer;
pub mod grid_layer;
pub mod heatmap_layer;
pub mod hexagon_layer;
pub mod icon_layer;
pub mod line_layer;
pub mod marching_squares;
pub mod marching_squares_codes;
pub mod mesh;
pub mod mvt;
pub mod mvt_layer;
pub mod path;
pub mod path_layer;
pub mod point_cloud_layer;
pub mod polygon;
pub mod polygon_layer;
pub mod scatterplot_layer;
pub mod scenegraph;
pub mod scenegraph_layer;
pub mod screen_grid_layer;
pub mod simple_mesh_layer;
pub mod solid_polygon_layer;
pub mod text;
pub mod text_layer;
pub mod tile_layer;
pub mod tileset;
pub mod trips_layer;
pub mod wms_layer;

pub use aggregation::{
    AggregatedBin, AggregationOperation, AggregationProps, ScaleType, DEFAULT_COLOR_RANGE,
};
pub use arc_layer::{ArcLayer, ArcLayerProps};
pub use bitmap_layer::{BitmapImage, BitmapLayer, BitmapLayerProps};
pub use column_layer::{ColumnLayer, ColumnLayerProps};
pub use contour_layer::{Contour, ContourGrid, ContourLayer, ContourLayerProps, DEFAULT_CONTOUR_COLOR};
pub use extensions::{
    BrushingExtension, BrushingTarget, ClipExtension, CollisionFilterExtension, DataFilterExtension,
    FillPattern, FillPatternAtlas, FillStyleExtension, FilterCategories, FilterValues, MaskExtension,
    PathStyleExtension, PathStyleTarget,
};
pub use fetch::{CancelToken, FetchHandle, FetchResult, FetchStats, FetchStatus, Fetcher};
pub use geo_cell_layer::{
    geohash_bounds, geohash_polygon, h3_polygon, quadkey_polygon, s2_polygon, CellKind, GeoCellLayer,
    GeoCellLayerProps,
};
pub use geojson_layer::{GeoJsonLayer, GeoJsonLayerProps};
pub use grid_cell_layer::{GridCellLayer, GridCellLayerProps};
pub use grid_layer::{GridLayer, GridLayerProps};
pub use heatmap_layer::{HeatmapAggregation, HeatmapLayer, HeatmapLayerProps};
pub use hexagon_layer::{HexagonLayer, HexagonLayerProps};
pub use icon_layer::{IconAtlas, IconLayer, IconLayerProps, IconMapping};
pub use line_layer::{LineLayer, LineLayerProps};
pub use marching_squares::{ContourGeometry, ContourThreshold};
pub use mesh::{Mesh, MeshVertex};
pub use mvt::{decode_tile, decode_tile_features, encode_tile, MvtLayerData};
pub use mvt_layer::{geojson_renderer, mvt_loader_with, MvtLayer, MvtLayerProps};
pub use path_layer::{PathLayer, PathLayerProps};
pub use point_cloud_layer::{PointCloudLayer, PointCloudLayerProps};
pub use polygon_layer::{PolygonLayer, PolygonLayerProps};
pub use scatterplot_layer::{ScatterplotLayer, ScatterplotLayerProps};
pub use scenegraph::{ScenePrimitive, Scenegraph};
pub use scenegraph_layer::{ScenegraphLayer, ScenegraphLayerProps, ScenegraphLighting};
pub use screen_grid_layer::{ScreenGridBin, ScreenGridLayer, ScreenGridLayerProps};
pub use simple_mesh_layer::{transform_matrix, SimpleMeshLayer, SimpleMeshLayerProps};
pub use solid_polygon_layer::{SolidPolygonLayer, SolidPolygonLayerProps};
pub use text::{CharacterSet, FontAtlas, FontSettings, FontSource, WordBreak};
pub use text_layer::{AlignmentBaseline, TextAnchor, TextLayer, TextLayerProps};
pub use tile_layer::{
    raster_renderer, LoadedTile, TileData, TileLayer, TileLayerProps, TileLoader, TileRenderer,
};
pub use tileset::{
    get_tile_indices, is_url_template, tile_bounds, url_from_template, RefinementStrategy, TileBounds,
    TileHeader, TileIndex, TileStatus, Tileset, TilesetOptions, TILE_SIZE,
};
pub use trips_layer::{TripsLayer, TripsLayerProps};
pub use wms_layer::{image_url as wms_image_url, WmsFetch, WmsLayer, WmsLayerProps, WmsServiceType, WmsSrs};

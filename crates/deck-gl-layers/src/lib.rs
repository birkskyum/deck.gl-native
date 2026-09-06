//! Rust port of `@deck.gl/layers`: the core layer catalog on wgpu.
//!
//! Layer shaders are the WGSL sources deck.gl 9 ships, included verbatim from `src/wgsl`.

pub mod arc_layer;
pub mod line_layer;
pub mod path;
pub mod path_layer;
pub mod polygon;
pub mod polygon_layer;
pub mod scatterplot_layer;
pub mod solid_polygon_layer;

pub use arc_layer::{ArcLayer, ArcLayerProps};
pub use line_layer::{LineLayer, LineLayerProps};
pub use path_layer::{PathLayer, PathLayerProps};
pub use polygon_layer::{PolygonLayer, PolygonLayerProps};
pub use scatterplot_layer::{ScatterplotLayer, ScatterplotLayerProps};
pub use solid_polygon_layer::{SolidPolygonLayer, SolidPolygonLayerProps};

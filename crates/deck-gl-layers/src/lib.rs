//! Rust port of `@deck.gl/layers`: the core layer catalog on wgpu.
//!
//! Layer shaders are the WGSL sources deck.gl 9 ships, included verbatim from `src/wgsl`.

pub mod line_layer;
pub mod polygon;
pub mod scatterplot_layer;
pub mod solid_polygon_layer;

pub use line_layer::{LineLayer, LineLayerProps};
pub use scatterplot_layer::{ScatterplotLayer, ScatterplotLayerProps};
pub use solid_polygon_layer::{SolidPolygonLayer, SolidPolygonLayerProps};

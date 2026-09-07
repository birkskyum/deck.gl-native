//! CPU aggregation shared by [`HexagonLayer`](crate::HexagonLayer) and
//! [`GridLayer`](crate::GridLayer): binning, weight aggregation and the colour and elevation
//! scales, ported from `@deck.gl/aggregation-layers`.

pub mod bin;
pub mod scale;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use deck_gl::data::{resolve_f32, resolve_positions};
use deck_gl::{Accessor, Color, LayerData, Position, Result};
use math_gl::web_mercator::{get_distance_scales, lng_lat_to_world, world_to_lng_lat};

pub use bin::{gridbin_origin, hexbin_centroid, hexbin_vertices, point_to_gridbin, point_to_hexbin};
pub use scale::{
    aggregate, interpolate_elevation, sample_color_range, AggregationOperation, ScaleType, ScaledValues,
    DEFAULT_COLOR_RANGE,
};

/// Props shared by the aggregation layers. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregationProps {
    /// Value range mapped onto `color_range`; the data's min and max when unset
    pub color_domain: Option<[f32; 2]>,
    pub color_range: Vec<Color>,
    pub color_scale_type: ScaleType,
    pub color_aggregation: AggregationOperation,
    /// Hide bins whose colour value is below this percentile (0 to 100)
    pub lower_percentile: f32,
    pub upper_percentile: f32,
    pub elevation_domain: Option<[f32; 2]>,
    /// Elevation in meters mapped from the elevation domain
    pub elevation_range: [f32; 2],
    pub elevation_scale: f32,
    pub elevation_scale_type: ScaleType,
    pub elevation_aggregation: AggregationOperation,
    pub elevation_lower_percentile: f32,
    pub elevation_upper_percentile: f32,
    pub extruded: bool,
    /// Cell size as a fraction of the bin, 0 to 1
    pub coverage: f32,
    pub get_position: Accessor<Position>,
    pub get_color_weight: Accessor<f32>,
    pub get_elevation_weight: Accessor<f32>,
}

impl Default for AggregationProps {
    fn default() -> Self {
        Self {
            color_domain: None,
            color_range: DEFAULT_COLOR_RANGE.to_vec(),
            color_scale_type: ScaleType::Quantize,
            color_aggregation: AggregationOperation::Sum,
            lower_percentile: 0.0,
            upper_percentile: 100.0,
            elevation_domain: None,
            elevation_range: [0.0, 1000.0],
            elevation_scale: 1.0,
            elevation_scale_type: ScaleType::Linear,
            elevation_aggregation: AggregationOperation::Sum,
            elevation_lower_percentile: 0.0,
            elevation_upper_percentile: 100.0,
            extruded: false,
            coverage: 1.0,
            get_position: Accessor::column("position"),
            get_color_weight: Accessor::Constant(1.0),
            get_elevation_weight: Accessor::Constant(1.0),
        }
    }
}

/// One aggregated bin, what a pick on an aggregation layer refers to.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregatedBin {
    pub col: i64,
    pub row: i64,
    /// Centre of a hexagon, or lower left corner of a grid cell, as longitude and latitude
    pub position: [f64; 2],
    pub color_value: f32,
    pub elevation_value: f32,
    pub count: usize,
    /// Rows of the layer's data that fell into this bin
    pub point_indices: Vec<usize>,
}

/// Result of aggregating a layer's data.
#[derive(Clone, Debug, Default)]
pub struct Aggregation {
    pub bins: Vec<AggregatedBin>,
    /// Data domain of the colour values before scaling
    pub color_domain: [f32; 2],
    pub elevation_domain: [f32; 2],
    /// Per bin fill colour, elevation in meters and bin index, for the bins that are shown
    pub cells: Vec<Cell>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub bin: usize,
    pub color: Color,
    pub elevation: f32,
}

/// Geometry of the bins in common (Web Mercator world) units.
pub struct BinSpace {
    /// Bin id of a point relative to `origin`
    pub bin_of: Box<dyn Fn([f64; 2]) -> [i64; 2] + Send + Sync>,
    /// Anchor position of a bin relative to `origin`
    pub anchor_of: Box<dyn Fn([i64; 2]) -> [f64; 2] + Send + Sync>,
    pub origin: [f64; 2],
}

/// Bounds of the positions and the units per meter at their centre, or None without data.
pub fn common_frame(
    positions: &[Position],
) -> Option<([f64; 2], [f64; 2], math_gl::web_mercator::DistanceScales)> {
    if positions.is_empty() {
        return None;
    }
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for p in positions {
        for k in 0..2 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    if !min[0].is_finite() {
        return None;
    }
    let centroid = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
    let scales = get_distance_scales(centroid[0], centroid[1], false);
    Some((centroid, lng_lat_to_world(centroid), scales))
}

/// The distinct bins of a slice, in the order they first appear.
fn distinct_bins(chunk: &[Option<[i64; 2]>]) -> Vec<[i64; 2]> {
    let mut seen: HashMap<[i64; 2], (), BuildHasherDefault<BinHasher>> = HashMap::default();
    let mut order = Vec::new();
    for id in chunk.iter().flatten() {
        if seen.insert(*id, ()).is_none() {
            order.push(*id);
        }
    }
    order
}

/// Hashes a bin's two integer coordinates by mixing rather than by a cryptographic round.
/// Bin ids are dense and small, and the default hasher costs more than the collisions it
/// avoids here.
#[derive(Default)]
pub struct BinHasher(u64);

impl Hasher for BinHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_u8(*byte);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.write_u64(value as u64);
    }

    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    fn write_u64(&mut self, value: u64) {
        // A round of the same mixing xxHash and rustc-hash use
        self.0 = (self.0 ^ value).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 ^= self.0 >> 29;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Bin the data and aggregate its weights.
pub fn aggregate_points(data: &LayerData, props: &AggregationProps, space: &BinSpace) -> Result<Aggregation> {
    let t0 = web_time::Instant::now();
    let positions = resolve_positions(data, &props.get_position)?;
    let color_weights = resolve_f32(data, &props.get_color_weight)?;
    let elevation_weights = resolve_f32(data, &props.get_elevation_weight)?;
    let t_resolve = t0.elapsed();
    let t1 = web_time::Instant::now();

    // Which bin every point falls in. A projection and a bin lookup each, with nothing
    // shared between them, so this runs on every core. It is most of the cost of binning,
    // and binning is what a hexagon layer redoes every time its radius moves.
    let bin_of = |p: &Position| -> Option<[i64; 2]> {
        if !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        let common = lng_lat_to_world([p[0], p[1]]);
        Some((space.bin_of)([
            common[0] - space.origin[0],
            common[1] - space.origin[1],
        ]))
    };
    #[cfg(feature = "parallel")]
    let point_bins: Vec<Option<[i64; 2]>> = {
        use rayon::prelude::*;
        positions.par_iter().map(bin_of).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let point_bins: Vec<Option<[i64; 2]>> = positions.iter().map(bin_of).collect();

    let t_map = t1.elapsed();

    // Grouping the points by bin. Done in one pass this is a hash lookup and a push per
    // point, all of it serial, which is what a hexagon layer's radius slider waits for. Done
    // in four it is mostly parallel: the distinct bins of each chunk, merged in chunk order
    // so the result is the same as the serial pass; then the slot of every point against a
    // map nobody is writing to; then exact sized member lists so no push ever reallocates.
    const CHUNK: usize = 1 << 16;
    const NONE: u32 = u32::MAX;

    #[cfg(feature = "parallel")]
    let distinct: Vec<Vec<[i64; 2]>> = {
        use rayon::prelude::*;
        point_bins.par_chunks(CHUNK).map(distinct_bins).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let distinct: Vec<Vec<[i64; 2]>> = point_bins.chunks(CHUNK).map(distinct_bins).collect();

    let mut index_of: HashMap<[i64; 2], u32, BuildHasherDefault<BinHasher>> = HashMap::default();
    let mut ids: Vec<[i64; 2]> = Vec::new();
    for chunk in distinct {
        for id in chunk {
            index_of.entry(id).or_insert_with(|| {
                ids.push(id);
                (ids.len() - 1) as u32
            });
        }
    }

    let slot_of = |id: &Option<[i64; 2]>| -> u32 { id.map_or(NONE, |id| index_of[&id]) };
    #[cfg(feature = "parallel")]
    let slots: Vec<u32> = {
        use rayon::prelude::*;
        point_bins.par_iter().map(slot_of).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let slots: Vec<u32> = point_bins.iter().map(slot_of).collect();

    let mut counts = vec![0usize; ids.len()];
    for slot in &slots {
        if *slot != NONE {
            counts[*slot as usize] += 1;
        }
    }
    let mut members: Vec<Vec<usize>> = counts.iter().map(|n| Vec::with_capacity(*n)).collect();
    for (i, slot) in slots.iter().enumerate() {
        if *slot != NONE {
            members[*slot as usize].push(i);
        }
    }

    let t_bin = t1.elapsed();
    let t2 = web_time::Instant::now();
    let (color_values, color_domain) = aggregate(&members, &color_weights, props.color_aggregation);
    let (elevation_values, elevation_domain) =
        aggregate(&members, &elevation_weights, props.elevation_aggregation);
    let colors = ScaledValues::new(
        color_values.clone(),
        color_domain,
        props.color_domain,
        props.color_scale_type,
        props.lower_percentile,
        props.upper_percentile,
    );
    let elevations = ScaledValues::new(
        elevation_values.clone(),
        elevation_domain,
        props.elevation_domain,
        props.elevation_scale_type,
        props.elevation_lower_percentile,
        props.elevation_upper_percentile,
    );

    let t_agg = t2.elapsed();
    if std::env::var_os("DECKGL_TIME_BINNING").is_some() {
        eprintln!(
            "    resolve {:.1} ms  bin {:.1} ms (map {:.1}, group {:.1})  aggregate {:.1} ms",
            t_resolve.as_secs_f64() * 1000.0,
            t_bin.as_secs_f64() * 1000.0,
            t_map.as_secs_f64() * 1000.0,
            (t_bin - t_map).as_secs_f64() * 1000.0,
            t_agg.as_secs_f64() * 1000.0
        );
    }
    let mut bins = Vec::with_capacity(ids.len());
    let mut cells = Vec::with_capacity(ids.len());
    for (b, id) in ids.iter().enumerate() {
        let anchor = (space.anchor_of)(*id);
        let position = world_to_lng_lat([anchor[0] + space.origin[0], anchor[1] + space.origin[1]]);
        bins.push(AggregatedBin {
            col: id[0],
            row: id[1],
            position,
            color_value: color_values[b],
            elevation_value: elevation_values[b],
            count: members[b].len(),
            point_indices: std::mem::take(&mut members[b]),
        });
        if colors.visible(b) && elevations.visible(b) {
            cells.push(Cell {
                bin: b,
                color: sample_color_range(colors.ratio(b), &props.color_range, props.color_scale_type),
                elevation: if props.extruded {
                    interpolate_elevation(elevations.ratio(b), props.elevation_range, props.elevation_scale)
                } else {
                    0.0
                },
            });
        }
    }
    Ok(Aggregation {
        bins,
        color_domain,
        elevation_domain,
        cells,
    })
}

/// Data and accessors for the cell sub layer: one row per shown cell, picking back to bins.
pub fn cell_accessors(
    aggregation: &Aggregation,
) -> (LayerData, Accessor<Position>, Accessor<Color>, Accessor<f32>) {
    let cells = Arc::new(aggregation.cells.clone());
    let positions: Arc<Vec<Position>> = Arc::new(
        aggregation
            .cells
            .iter()
            .map(|c| {
                let p = aggregation.bins[c.bin].position;
                [p[0], p[1], 0.0]
            })
            .collect(),
    );
    let rows: Arc<Vec<u32>> = Arc::new(aggregation.cells.iter().map(|c| c.bin as u32).collect());
    let data = LayerData::with_length(aggregation.cells.len()).with_source_rows(rows);
    let colors = cells.clone();
    let elevations = cells;
    (
        data,
        Accessor::func(move |i| positions[i]),
        Accessor::func(move |i| colors[i].color),
        Accessor::func(move |i| elevations[i].elevation),
    )
}

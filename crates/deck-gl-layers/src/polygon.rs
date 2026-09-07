//! Polygon normalization and tesselation. Port of the parts of
//! `@deck.gl/layers/src/solid-polygon-layer/polygon.ts` and `polygon-tesselator.ts` that the
//! native `SolidPolygonLayer` needs.

use deck_gl::{Polygon, Position};

/// A polygon flattened into one vertex list. `hole_indices` are vertex indices where each
/// hole ring starts. Rings are closed (last vertex equals first) and wound as deck.gl expects:
/// outer ring clockwise, holes counter clockwise (y up).
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedPolygon {
    pub positions: Vec<Position>,
    pub hole_indices: Vec<usize>,
}

/// math.gl `getPolygonSignedArea`: positive for clockwise rings in a y-up frame.
pub fn signed_area(ring: &[Position]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    let mut j = n - 1;
    for i in 0..n {
        area += (ring[i][0] - ring[j][0]) * (ring[i][1] + ring[j][1]);
        j = i;
    }
    area / 2.0
}

fn copy_ring(target: &mut Vec<Position>, ring: &[Position], clockwise: bool) {
    let start = target.len();
    target.extend_from_slice(ring);
    if let (Some(first), Some(last)) = (ring.first(), ring.last()) {
        if first != last {
            target.push(*first);
        }
    }
    let area = signed_area(&target[start..]);
    let is_clockwise = area > 0.0;
    if area != 0.0 && is_clockwise != clockwise {
        target[start..].reverse();
    }
}

/// Normalize any polygon into the flat format, fixing ring closure and winding.
pub fn normalize(polygon: &Polygon) -> NormalizedPolygon {
    let mut positions = Vec::new();
    let mut hole_indices = Vec::new();
    for (ring_index, ring) in polygon.iter().enumerate() {
        if ring.is_empty() {
            continue;
        }
        if ring_index > 0 {
            hole_indices.push(positions.len());
        }
        copy_ring(&mut positions, ring, ring_index == 0);
    }
    NormalizedPolygon {
        positions,
        hole_indices,
    }
}

/// Triangulate the polygon surface. `preproject` maps a position to the 2D plane used for
/// triangulation (for lng/lat data, the Web Mercator plane).
pub fn surface_indices(polygon: &NormalizedPolygon, preproject: impl Fn(&Position) -> [f64; 2]) -> Vec<u32> {
    let mut flat = Vec::with_capacity(polygon.positions.len() * 2);
    for p in &polygon.positions {
        let xy = preproject(p);
        flat.push(xy[0]);
        flat.push(xy[1]);
    }
    match earcutr::earcut(&flat, &polygon.hole_indices, 2) {
        Ok(indices) => indices.into_iter().map(|i| i as u32).collect(),
        Err(_) => Vec::new(),
    }
}

/// Per-vertex attributes for a batch of polygons, laid out the way deck.gl's
/// `PolygonTesselator` produces them.
#[derive(Clone, Debug, Default)]
pub struct TesselatedPolygons {
    /// All ring vertices of all polygons, in order
    pub positions: Vec<Position>,
    /// 1 for every vertex except the last vertex of each ring, which is 0 so the side wall
    /// from that vertex to the next ring's first vertex is not drawn.
    pub vertex_valid: Vec<f32>,
    /// Triangle list indices for the polygon surfaces
    pub indices: Vec<u32>,
    /// Index of the source polygon for every vertex
    pub row_index: Vec<u32>,
}

impl TesselatedPolygons {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

/// Polygons above which tessellation runs on all cores.
#[cfg(feature = "parallel")]
const PARALLEL_POLYGONS: usize = 64;

/// The tessellation of one polygon, before the parts are joined.
struct TesselatedPolygon {
    positions: Vec<Position>,
    vertex_valid: Vec<f32>,
    /// Indices relative to this polygon's first vertex
    indices: Vec<u32>,
}

fn tesselate_one(
    polygon: &Polygon,
    preproject: &(impl Fn(&Position) -> [f64; 2] + Sync),
) -> TesselatedPolygon {
    let normalized = normalize(polygon);
    let size = normalized.positions.len();
    if size == 0 {
        return TesselatedPolygon {
            positions: Vec::new(),
            vertex_valid: Vec::new(),
            indices: Vec::new(),
        };
    }
    let indices = surface_indices(&normalized, preproject);
    // The last vertex of every ring has no side wall to the next one
    let mut vertex_valid = vec![1.0f32; size];
    for &hole in &normalized.hole_indices {
        vertex_valid[hole - 1] = 0.0;
    }
    vertex_valid[size - 1] = 0.0;
    TesselatedPolygon {
        positions: normalized.positions,
        vertex_valid,
        indices,
    }
}

/// Tesselate polygons for the solid polygon layer. Large batches are tessellated on all
/// cores and joined afterwards, since earcut is the bulk of a polygon layer's upload.
pub fn tesselate(
    polygons: &[Polygon],
    preproject: impl Fn(&Position) -> [f64; 2] + Sync,
) -> TesselatedPolygons {
    #[cfg(feature = "parallel")]
    let parallel = polygons.len() >= PARALLEL_POLYGONS;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;
    let parts: Vec<TesselatedPolygon> = if parallel {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            polygons
                .par_iter()
                .map(|polygon| tesselate_one(polygon, &preproject))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        unreachable!()
    } else {
        polygons
            .iter()
            .map(|polygon| tesselate_one(polygon, &preproject))
            .collect()
    };
    let vertices: usize = parts.iter().map(|p| p.positions.len()).sum();
    let mut out = TesselatedPolygons {
        positions: Vec::with_capacity(vertices),
        vertex_valid: Vec::with_capacity(vertices),
        indices: Vec::with_capacity(parts.iter().map(|p| p.indices.len()).sum()),
        row_index: Vec::with_capacity(vertices),
    };
    for (row, part) in parts.into_iter().enumerate() {
        let base = out.positions.len() as u32;
        if part.positions.is_empty() {
            continue;
        }
        out.indices.extend(part.indices.into_iter().map(|i| i + base));
        out.row_index
            .extend(std::iter::repeat_n(row as u32, part.positions.len()));
        out.positions.extend(part.positions);
        out.vertex_valid.extend(part.vertex_valid);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Vec<Position> {
        vec![[x, y, 0.0], [x + s, y, 0.0], [x + s, y + s, 0.0], [x, y + s, 0.0]]
    }

    #[test]
    fn normalizes_winding_and_closes_rings() {
        // Counter clockwise input outer ring, clockwise input hole: both must be flipped.
        let polygon: Polygon = vec![
            square(0.0, 0.0, 10.0),
            square(2.0, 2.0, 2.0).into_iter().rev().collect(),
        ];
        let n = normalize(&polygon);
        assert_eq!(n.positions.len(), 10);
        assert_eq!(n.hole_indices, vec![5]);
        assert!(
            signed_area(&n.positions[0..5]) > 0.0,
            "outer ring must be clockwise"
        );
        assert!(
            signed_area(&n.positions[5..10]) < 0.0,
            "hole must be counter clockwise"
        );
        assert_eq!(n.positions[0], n.positions[4]);
    }

    #[test]
    fn tesselates_polygon_with_hole() {
        let polygon: Polygon = vec![square(0.0, 0.0, 10.0), square(2.0, 2.0, 2.0)];
        let t = tesselate(&[polygon], |p| [p[0], p[1]]);
        assert_eq!(t.vertex_count(), 10);
        // 8 triangles for a square with a square hole
        assert_eq!(t.indices.len(), 24);
        assert!(t.indices.iter().all(|&i| (i as usize) < 10));
        assert_eq!(
            t.vertex_valid,
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(t.row_index, vec![0; 10]);
    }

    #[test]
    fn offsets_indices_across_polygons() {
        let t = tesselate(&[vec![square(0.0, 0.0, 1.0)], vec![square(5.0, 5.0, 1.0)]], |p| {
            [p[0], p[1]]
        });
        assert_eq!(t.vertex_count(), 10);
        assert_eq!(t.indices.len(), 12);
        assert!(t.indices[6..].iter().all(|&i| i >= 5));
        assert_eq!(t.row_index[5], 1);
    }
}

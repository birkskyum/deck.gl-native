//! Path tesselation. Port of `@deck.gl/layers/src/path-layer/path-tesselator.ts`.
//!
//! Every path vertex becomes one instance of the path layer's segment geometry. Each instance
//! reads its own vertex and the three neighbors (previous, next and the one after) so the
//! shader can build joints and caps.

use deck_gl::{Path, Position};

const START_CAP: f32 = 1.0;
const END_CAP: f32 = 2.0;
const INVALID: f32 = 4.0;

/// Per-instance attributes for a batch of paths.
#[derive(Clone, Debug, Default)]
pub struct TesselatedPaths {
    /// One entry per instance: the path vertices, with closed paths repeating their second
    /// and third vertex at the end so the loop closes with proper joints.
    pub positions: Vec<Position>,
    /// Segment type flags per instance: 1 start cap, 2 end cap, 4 invalid (not drawn).
    pub segment_types: Vec<f32>,
    /// Index of the source path for every instance
    pub row_index: Vec<u32>,
}

impl TesselatedPaths {
    pub fn instance_count(&self) -> usize {
        self.positions.len()
    }

    /// The packed neighbor window deck.gl builds for WebGPU: for each instance, the previous,
    /// current, next and next-next positions as high f32 parts followed by their low parts,
    /// 24 floats per instance. Out of range neighbors are zero.
    pub fn packed_neighbor_positions(&self) -> Vec<f32> {
        let n = self.positions.len();
        let mut out = vec![0f32; n * 24];
        for i in 0..n {
            let base = i * 24;
            for (slot, offset) in [-1i64, 0, 1, 2].into_iter().enumerate() {
                let source = i as i64 + offset;
                if source < 0 || source >= n as i64 {
                    continue;
                }
                let position = self.positions[source as usize];
                for j in 0..3 {
                    let hi = position[j] as f32;
                    out[base + slot * 3 + j] = hi;
                    out[base + 12 + slot * 3 + j] = (position[j] - hi as f64) as f32;
                }
            }
        }
        out
    }
}

fn is_closed(path: &Path) -> bool {
    path.len() >= 2 && path[0] == path[path.len() - 1]
}

/// Number of instances a path produces.
fn geometry_size(path: &Path) -> usize {
    let n = path.len();
    if n < 2 {
        return 0;
    }
    if is_closed(path) {
        if n < 3 {
            0
        } else {
            n + 2
        }
    } else {
        n
    }
}

/// Tesselate paths for the path layer.
pub fn tesselate(paths: &[Path]) -> TesselatedPaths {
    let mut out = TesselatedPaths::default();
    for (row, path) in paths.iter().enumerate() {
        let size = geometry_size(path);
        if size == 0 {
            continue;
        }
        let n = path.len();
        let start = out.positions.len();
        for pt_index in 0..size {
            // positions   --  A0 A1 B0 B1 B2 B3 B0 B1 B2 --
            // segmentTypes     3  4  4  0  0  0  0  4  4
            let mut index = pt_index;
            if index >= n {
                // loop
                index += 1;
                index -= n;
            }
            out.positions.push(path[index]);
        }
        out.segment_types.extend(std::iter::repeat_n(0.0, size));
        if is_closed(path) {
            out.segment_types[start] = INVALID;
            out.segment_types[start + size - 2] = INVALID;
        } else {
            out.segment_types[start] += START_CAP;
            out.segment_types[start + size - 2] += END_CAP;
        }
        out.segment_types[start + size - 1] = INVALID;
        out.row_index.extend(std::iter::repeat_n(row as u32, size));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_path_has_caps_and_a_trailing_invalid_instance() {
        let t = tesselate(&[vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]]);
        assert_eq!(t.instance_count(), 3);
        assert_eq!(t.segment_types, vec![1.0, 2.0, 4.0]);
        assert_eq!(t.row_index, vec![0, 0, 0]);
    }

    #[test]
    fn two_point_path_is_start_and_end_cap() {
        let t = tesselate(&[vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]);
        assert_eq!(t.segment_types, vec![3.0, 4.0]);
    }

    #[test]
    fn closed_path_repeats_vertices() {
        let square = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let t = tesselate(&[square]);
        assert_eq!(t.instance_count(), 7);
        assert_eq!(t.positions[5], [1.0, 0.0, 0.0]);
        assert_eq!(t.positions[6], [1.0, 1.0, 0.0]);
        assert_eq!(t.segment_types, vec![4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0]);
    }

    #[test]
    fn packed_neighbors_follow_the_minus_one_to_two_window() {
        let t = tesselate(&[vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]);
        let packed = t.packed_neighbor_positions();
        assert_eq!(packed.len(), 3 * 24);
        // instance 1: left = P0, start = P1, end = P2, right = out of range (0)
        assert_eq!(
            &packed[24..24 + 12],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        // instance 0: left is out of range
        assert_eq!(&packed[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&packed[3..6], &[0.0, 0.0, 0.0]);
        assert_eq!(&packed[6..9], &[1.0, 0.0, 0.0]);
    }
}

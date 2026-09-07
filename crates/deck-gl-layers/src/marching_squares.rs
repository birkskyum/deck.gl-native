//! Port of `@deck.gl/aggregation-layers/src/contour-layer/marching-squares.ts` and
//! `contour-utils.ts`: isolines and isobands over a grid of aggregated values.

use crate::marching_squares_codes::{isobands, BandCase, LineCase, ISOLINES};

/// A single isoline threshold, or a `[min, max)` isoband.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContourThreshold {
    Line(f32),
    Band([f32; 2]),
}

/// A contour path (isoline) or ring (isoband) in cell units, with the contour it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourGeometry {
    pub vertices: Vec<[f64; 3]>,
    pub contour: usize,
}

fn vertex_code(weight: f32, threshold: ContourThreshold) -> u32 {
    if weight.is_nan() {
        return 0;
    }
    match threshold {
        ContourThreshold::Band([min, max]) => {
            if weight < min {
                0
            } else if weight < max {
                1
            } else {
                2
            }
        }
        ContourThreshold::Line(t) => (weight >= t) as u32,
    }
}

/// Marching squares code of the cell at `x`, `y` and the code of the mean of its corners.
/// The cell's corners are the values at (x, y), (x + 1, y), (x, y + 1) and (x + 1, y + 1);
/// cells outside the ranges count as below the threshold.
pub fn get_code(
    get_value: &dyn Fn(i64, i64) -> f32,
    threshold: ContourThreshold,
    x: i64,
    y: i64,
    x_range: [i64; 2],
    y_range: [i64; 2],
) -> (usize, u32) {
    let is_left = x < x_range[0];
    let is_right = x >= x_range[1] - 1;
    let is_bottom = y < y_range[0];
    let is_top = y >= y_range[1] - 1;
    let is_boundary = is_left || is_right || is_bottom || is_top;
    let mut weights = 0.0f32;
    let mut sample = |skip: bool, x: i64, y: i64| -> u32 {
        if skip {
            0
        } else {
            let w = get_value(x, y);
            weights += w;
            vertex_code(w, threshold)
        }
    };
    let top = sample(is_left || is_top, x, y + 1);
    let top_right = sample(is_right || is_top, x + 1, y + 1);
    let right = sample(is_right || is_bottom, x + 1, y);
    let current = sample(is_left || is_bottom, x, y);
    let code = match threshold {
        ContourThreshold::Line(_) => (top << 3) | (top_right << 2) | (right << 1) | current,
        ContourThreshold::Band(_) => (top << 6) | (top_right << 4) | (right << 2) | current,
    };
    // The mean code is only needed for saddle cases, which never occur on the boundary
    let mean_code = if is_boundary {
        0
    } else {
        vertex_code(weights / 4.0, threshold)
    };
    (code as usize, mean_code)
}

/// Isoline segments of a cell, each as two vertices at the reference corner plus the offsets.
pub fn get_lines(x: i64, y: i64, z: f64, code: usize, mean_code: u32) -> Vec<Vec<[f64; 3]>> {
    let segments = match &ISOLINES[code.min(15)] {
        LineCase::Fixed(segments) => *segments,
        LineCase::Saddle(cases) => cases[(mean_code as usize).min(1)],
    };
    let (rx, ry) = ((x + 1) as f64, (y + 1) as f64);
    segments
        .iter()
        .map(|segment| segment.iter().map(|o| [rx + o[0], ry + o[1], z]).collect())
        .collect()
}

/// Isoband polygons of a cell.
pub fn get_polygons(x: i64, y: i64, z: f64, code: usize, mean_code: u32) -> Vec<Vec<[f64; 3]>> {
    let polygons = match isobands(code) {
        BandCase::Fixed(polygons) => polygons,
        BandCase::Saddle(cases) => cases[(mean_code as usize).min(2)],
    };
    let (rx, ry) = ((x + 1) as f64, (y + 1) as f64);
    polygons
        .iter()
        .map(|polygon| polygon.iter().map(|o| [rx + o[0], ry + o[1], z]).collect())
        .collect()
}

/// Isolines and isobands for every contour over the value grid. `get_value` returns NaN for
/// empty cells; `x_range` and `y_range` are exclusive upper bounds of the cell indices.
pub fn generate_contours(
    contours: &[(ContourThreshold, Option<i32>)],
    get_value: &dyn Fn(i64, i64) -> f32,
    x_range: [i64; 2],
    y_range: [i64; 2],
) -> (Vec<ContourGeometry>, Vec<ContourGeometry>) {
    let mut lines = Vec::new();
    let mut polygons = Vec::new();
    for (index, &(threshold, z_index)) in contours.iter().enumerate() {
        let z = z_index.map_or(index as f64, f64::from);
        for x in (x_range[0] - 1)..x_range[1] {
            for y in (y_range[0] - 1)..y_range[1] {
                let (code, mean_code) = get_code(get_value, threshold, x, y, x_range, y_range);
                match threshold {
                    ContourThreshold::Band(_) => {
                        for vertices in get_polygons(x, y, z, code, mean_code) {
                            polygons.push(ContourGeometry {
                                vertices,
                                contour: index,
                            });
                        }
                    }
                    ContourThreshold::Line(_) => {
                        for vertices in get_lines(x, y, z, code, mean_code) {
                            lines.push(ContourGeometry {
                                vertices,
                                contour: index,
                            });
                        }
                    }
                }
            }
        }
    }
    (lines, polygons)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single cell at (0, 0) with value 2 in an otherwise empty grid
    fn single(x: i64, y: i64) -> f32 {
        if (x, y) == (0, 0) {
            2.0
        } else {
            f32::NAN
        }
    }

    #[test]
    fn a_lone_cell_is_ringed_by_four_segments() {
        let (lines, polygons) =
            generate_contours(&[(ContourThreshold::Line(1.0), None)], &single, [0, 1], [0, 1]);
        assert!(polygons.is_empty());
        assert_eq!(lines.len(), 4, "{lines:?}");
        // The segments form a diamond around the value, which sits at the cell centre (0.5, 0.5)
        let mut points: Vec<[i64; 2]> = lines
            .iter()
            .flat_map(|l| {
                l.vertices
                    .iter()
                    .map(|v| [(v[0] * 2.0) as i64, (v[1] * 2.0) as i64])
            })
            .collect();
        points.sort();
        points.dedup();
        assert_eq!(points, [[0, 1], [1, 0], [1, 2], [2, 1]]);
        assert!(lines.iter().all(|l| l.vertices.iter().all(|v| v[2] == 0.0)));
    }

    #[test]
    fn a_lone_cell_inside_a_band_gives_four_triangles() {
        let (lines, polygons) = generate_contours(
            &[(ContourThreshold::Band([1.0, 3.0]), Some(5))],
            &single,
            [0, 1],
            [0, 1],
        );
        assert!(lines.is_empty());
        assert_eq!(polygons.len(), 4, "{polygons:?}");
        assert!(polygons
            .iter()
            .all(|p| p.vertices.len() == 3 && p.vertices[0][2] == 5.0));
        // A value above the band leaves a ring of trapezoids where the band is crossed
        let (_, above) = generate_contours(
            &[(ContourThreshold::Band([0.0, 1.0]), None)],
            &single,
            [0, 1],
            [0, 1],
        );
        assert_eq!(above.len(), 4);
        assert!(above.iter().all(|p| p.vertices.len() == 4));
    }

    #[test]
    fn saddle_cases_use_the_mean() {
        // Two opposite corners above the threshold
        let checker = |x: i64, y: i64| if (x + y) % 2 == 0 { 3.0 } else { 0.0 };
        let (code, mean) = get_code(&checker, ContourThreshold::Line(1.0), 0, 0, [-1, 3], [-1, 3]);
        assert!(code == 5 || code == 10, "{code}");
        assert_eq!(mean, 1, "mean 1.5 is above the threshold");
        assert_eq!(get_lines(0, 0, 0.0, code, mean).len(), 2);
    }
}

//! Binning of points into hexagons (a port of d3-hexbin as used by deck.gl's `hexbin.ts`) and
//! square grid cells, in common (Web Mercator world) units.

use std::f64::consts::PI;

/// Horizontal distance between hexagon centres, in radii.
pub const HEX_DIST_X: f64 = 2.0 * 0.866_025_403_784_438_6; // 2 sin(pi / 3)
/// Vertical distance between hexagon rows, in radii.
pub const HEX_DIST_Y: f64 = 1.5;

/// The six vertices of a pointy top hexagon inscribed in the unit circle, starting at the
/// bottom, counter clockwise like `HexbinVertices` in deck.gl.
pub fn hexbin_vertices() -> [[f64; 2]; 6] {
    let mut v = [[0.0; 2]; 6];
    for (i, vertex) in v.iter_mut().enumerate() {
        let angle = i as f64 * PI / 3.0;
        *vertex = [angle.sin(), -angle.cos()];
    }
    v
}

/// The hexagon containing `p`, as column and row indices.
pub fn point_to_hexbin(p: [f64; 2], radius: f64) -> [i64; 2] {
    let py = p[1] / radius / HEX_DIST_Y;
    let mut pj = py.round() as i64;
    let px = p[0] / radius / HEX_DIST_X - (pj & 1) as f64 / 2.0;
    let mut pi = px.round() as i64;
    let py1 = py - pj as f64;
    if py1.abs() * 3.0 > 1.0 {
        let px1 = px - pi as f64;
        let pi2 = pi as f64 + if px < pi as f64 { -0.5 } else { 0.5 };
        let pj2 = pj + if py < pj as f64 { -1 } else { 1 };
        let px2 = px - pi2;
        let py2 = py - pj2 as f64;
        if px1 * px1 + py1 * py1 > px2 * px2 + py2 * py2 {
            pi = (pi2 + if pj & 1 != 0 { 0.5 } else { -0.5 }) as i64;
            pj = pj2;
        }
    }
    [pi, pj]
}

/// Centre of a hexagon bin.
pub fn hexbin_centroid(bin: [i64; 2], radius: f64) -> [f64; 2] {
    let [i, j] = bin;
    [
        (i as f64 + (j & 1) as f64 / 2.0) * radius * HEX_DIST_X,
        j as f64 * radius * HEX_DIST_Y,
    ]
}

/// The grid cell containing `p`, for cells of `size` anchored at the origin.
pub fn point_to_gridbin(p: [f64; 2], size: [f64; 2]) -> [i64; 2] {
    [(p[0] / size[0]).floor() as i64, (p[1] / size[1]).floor() as i64]
}

/// Lower left corner of a grid cell.
pub fn gridbin_origin(bin: [i64; 2], size: [f64; 2]) -> [f64; 2] {
    [bin[0] as f64 * size[0], bin[1] as f64 * size[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hexbin_centroids_round_trip() {
        let radius = 10.0;
        for i in -3..4 {
            for j in -3..4 {
                let c = hexbin_centroid([i, j], radius);
                assert_eq!(point_to_hexbin(c, radius), [i, j], "centroid of ({i}, {j})");
                // a little off the centre still lands in the same bin
                assert_eq!(point_to_hexbin([c[0] + 3.0, c[1] - 2.0], radius), [i, j]);
            }
        }
    }

    #[test]
    fn hexbins_are_pointy_top_and_offset_every_other_row() {
        let radius = 1.0;
        let a = hexbin_centroid([0, 0], radius);
        let b = hexbin_centroid([0, 1], radius);
        assert!(
            (b[0] - a[0] - HEX_DIST_X / 2.0).abs() < 1e-12,
            "odd rows shift by half a cell"
        );
        assert!((b[1] - a[1] - 1.5).abs() < 1e-12);
        let v = hexbin_vertices();
        assert!(
            (v[0][0]).abs() < 1e-12 && (v[0][1] + 1.0).abs() < 1e-12,
            "first vertex at the bottom"
        );
    }

    #[test]
    fn grid_bins_floor_to_the_cell() {
        assert_eq!(point_to_gridbin([2.5, -0.5], [1.0, 1.0]), [2, -1]);
        assert_eq!(point_to_gridbin([0.0, 0.0], [10.0, 5.0]), [0, 0]);
        assert_eq!(gridbin_origin([2, -1], [10.0, 5.0]), [20.0, -5.0]);
    }
}

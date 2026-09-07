//! Terrain meshes from elevation images, the part of `@loaders.gl/terrain` the
//! [`TerrainLayer`](crate::TerrainLayer) needs: decode heights out of an RGB image and
//! triangulate them with Martini (a right triangulated irregular network), so a tile becomes
//! a mesh whose error stays under a given number of meters.

use crate::mesh::Mesh;
use crate::BitmapImage;

/// How an RGB pixel becomes a height in meters, deck.gl's `elevationDecoder`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevationDecoder {
    pub r_scaler: f32,
    pub g_scaler: f32,
    pub b_scaler: f32,
    pub offset: f32,
}

impl Default for ElevationDecoder {
    /// deck.gl's default: the red channel as meters.
    fn default() -> Self {
        Self {
            r_scaler: 1.0,
            g_scaler: 0.0,
            b_scaler: 0.0,
            offset: 0.0,
        }
    }
}

impl ElevationDecoder {
    /// Mapbox Terrain-RGB: `-10000 + (r * 256 * 256 + g * 256 + b) * 0.1`.
    pub fn mapbox() -> Self {
        Self {
            r_scaler: 6553.6,
            g_scaler: 25.6,
            b_scaler: 0.1,
            offset: -10000.0,
        }
    }

    /// Mapzen and AWS Terrarium: `(r * 256 + g + b / 256) - 32768`.
    pub fn terrarium() -> Self {
        Self {
            r_scaler: 256.0,
            g_scaler: 1.0,
            b_scaler: 1.0 / 256.0,
            offset: -32768.0,
        }
    }

    /// The height of one RGB pixel, in meters.
    pub fn decode(&self, r: u8, g: u8, b: u8) -> f32 {
        r as f32 * self.r_scaler + g as f32 * self.g_scaler + b as f32 * self.b_scaler + self.offset
    }
}

/// A square grid of heights in meters, `size` x `size` samples, as Martini needs it: the side
/// is a power of two plus one.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightGrid {
    pub size: usize,
    pub heights: Vec<f32>,
}

impl HeightGrid {
    /// Decode an image into a grid, padding the last row and column so the side is `2^k + 1`
    /// as Martini requires (the loaders.gl terrain loader does the same).
    pub fn from_image(image: &BitmapImage, decoder: &ElevationDecoder) -> Result<Self, String> {
        let (width, height) = (image.width as usize, image.height as usize);
        if width == 0 || height == 0 {
            return Err("elevation image is empty".into());
        }
        if image.rgba.len() < width * height * 4 {
            return Err(format!(
                "elevation image has {} bytes, expected {}",
                image.rgba.len(),
                width * height * 4
            ));
        }
        let size = width.max(height) + 1;
        let mut heights = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                // The extra row and column repeat the edge samples
                let sx = x.min(width - 1);
                let sy = y.min(height - 1);
                let i = (sy * width + sx) * 4;
                heights[y * size + x] = decoder.decode(image.rgba[i], image.rgba[i + 1], image.rgba[i + 2]);
            }
        }
        Ok(Self { size, heights })
    }

    pub fn height(&self, x: usize, y: usize) -> f32 {
        self.heights[y.min(self.size - 1) * self.size + x.min(self.size - 1)]
    }

    /// The lowest and highest sample.
    pub fn range(&self) -> (f32, f32) {
        self.heights
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), h| (lo.min(*h), hi.max(*h)))
    }
}

/// Martini's error map: how far the terrain strays from the mesh when a triangle is left
/// undivided. Built once per grid, then read while triangulating at any error.
pub struct Martini {
    grid_size: usize,
    num_triangles: usize,
    num_parent_triangles: usize,
    indices: Vec<u32>,
    coords: Vec<u32>,
    errors: Vec<f32>,
}

impl Martini {
    /// Build the error map of a grid whose side is `2^k + 1`.
    pub fn new(grid: &HeightGrid) -> Result<Self, String> {
        let size = grid.size;
        if size < 5 || (size - 1) & (size - 2) != 0 {
            return Err(format!("terrain grid side must be 2^k + 1, got {size}"));
        }
        let tile_size = size - 1;
        let num_triangles = tile_size * tile_size * 2 - 2;
        let num_parent_triangles = num_triangles - tile_size * tile_size;
        // Martini's coordinate table: the two corners of every triangle's hypotenuse
        let mut coords = vec![0u32; num_triangles * 4];
        for i in (0..num_triangles).rev() {
            let mut id = i + 2;
            let (mut ax, mut ay, mut bx, mut by, mut cx, mut cy) =
                (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            if id & 1 != 0 {
                // bottom right triangle
                bx = tile_size;
                by = tile_size;
                cx = tile_size;
            } else {
                // top left triangle
                ax = tile_size;
                ay = tile_size;
                cy = tile_size;
            }
            id >>= 1;
            while id > 1 {
                let mx = (ax + bx) >> 1;
                let my = (ay + by) >> 1;
                if id & 1 != 0 {
                    // left half
                    bx = ax;
                    by = ay;
                    ax = cx;
                    ay = cy;
                } else {
                    // right half
                    ax = bx;
                    ay = by;
                    bx = cx;
                    by = cy;
                }
                cx = mx;
                cy = my;
                id >>= 1;
            }
            let k = i * 4;
            coords[k] = ax as u32;
            coords[k + 1] = ay as u32;
            coords[k + 2] = bx as u32;
            coords[k + 3] = by as u32;
        }
        let mut martini = Self {
            grid_size: size,
            num_triangles,
            num_parent_triangles,
            indices: vec![0; size * size],
            coords,
            errors: vec![0.0; size * size],
        };
        martini.build_errors(grid);
        Ok(martini)
    }

    /// Martini's `update`: the error of every possible split point, from the smallest
    /// triangles up so a parent inherits its children's error.
    fn build_errors(&mut self, grid: &HeightGrid) {
        let size = self.grid_size;
        for i in (0..self.num_triangles).rev() {
            let k = i * 4;
            let (ax, ay) = (self.coords[k] as usize, self.coords[k + 1] as usize);
            let (bx, by) = (self.coords[k + 2] as usize, self.coords[k + 3] as usize);
            let mx = (ax + bx) >> 1;
            let my = (ay + by) >> 1;
            let cx = mx + my - ay;
            let cy = my + ax - mx;
            // The height the mesh would give the middle of the hypotenuse
            let interpolated = (grid.height(ax, ay) + grid.height(bx, by)) / 2.0;
            let middle = my * size + mx;
            let error = (interpolated - grid.heights[middle]).abs();
            self.errors[middle] = self.errors[middle].max(error);
            if i < self.num_parent_triangles {
                // A parent's error is at least the error of both children
                let left = ((ay + cy) >> 1) * size + ((ax + cx) >> 1);
                let right = ((by + cy) >> 1) * size + ((bx + cx) >> 1);
                self.errors[middle] = self.errors[middle].max(self.errors[left]).max(self.errors[right]);
            }
        }
    }

    /// The error of the split point of a triangle, in meters.
    pub fn error_at(&self, x: usize, y: usize) -> f32 {
        self.errors[y * self.grid_size + x]
    }

    /// Martini's `getMesh`: the vertices and triangles that keep the error under
    /// `max_error` meters.
    pub fn mesh(&mut self, max_error: f32) -> (Vec<[u32; 2]>, Vec<u32>) {
        let size = self.grid_size;
        let max = size - 1;
        self.indices.iter_mut().for_each(|i| *i = 0);
        let mut vertices: Vec<[u32; 2]> = Vec::new();
        let mut triangles: Vec<u32> = Vec::new();
        // Count the vertices of the mesh, numbering them as they are met
        let mut stack: Vec<(usize, usize, usize, usize, usize, usize)> =
            vec![(0, 0, max, max, max, 0), (max, max, 0, 0, 0, max)];
        while let Some((ax, ay, bx, by, cx, cy)) = stack.pop() {
            let mx = (ax + bx) >> 1;
            let my = (ay + by) >> 1;
            if (ax as i64 - cx as i64).abs() + (ay as i64 - cy as i64).abs() > 1
                && self.errors[my * size + mx] > max_error
            {
                stack.push((cx, cy, ax, ay, mx, my));
                stack.push((bx, by, cx, cy, mx, my));
            } else {
                for (x, y) in [(ax, ay), (bx, by), (cx, cy)] {
                    let index = y * size + x;
                    if self.indices[index] == 0 {
                        vertices.push([x as u32, y as u32]);
                        self.indices[index] = vertices.len() as u32;
                    }
                    triangles.push(self.indices[index] - 1);
                }
            }
        }
        (vertices, triangles)
    }
}

/// Build a mesh out of an elevation image.
///
/// `bounds` is `[min_x, min_y, max_x, max_y]` in the coordinates the mesh is drawn in, and
/// the mesh's positions are `[x, y, elevation]` with the elevation in meters. Texture
/// coordinates run from `[0, 0]` at the bottom left to `[1, 1]` at the top right, so a
/// texture of the same area drapes over it. `max_error` is the largest deviation from the
/// height map, in meters.
pub fn terrain_mesh(
    image: &BitmapImage,
    decoder: &ElevationDecoder,
    bounds: [f64; 4],
    max_error: f32,
) -> Result<Mesh, String> {
    let grid = HeightGrid::from_image(image, decoder)?;
    let mut martini = Martini::new(&grid)?;
    let (vertices, indices) = martini.mesh(max_error.max(1.0));
    let size = grid.size as f64 - 1.0;
    let (min_x, min_y, max_x, max_y) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let (span_x, span_y) = (max_x - min_x, max_y - min_y);
    let positions: Vec<[f32; 3]> = vertices
        .iter()
        .map(|[x, y]| {
            let u = *x as f64 / size;
            // Image rows run from the north, the mesh's y from the south
            let v = *y as f64 / size;
            [
                (min_x + u * span_x) as f32,
                (max_y - v * span_y) as f32,
                grid.height(*x as usize, *y as usize),
            ]
        })
        .collect();
    let tex_coords: Vec<[f32; 2]> = vertices
        .iter()
        .map(|[x, y]| [*x as f32 / size as f32, *y as f32 / size as f32])
        .collect();
    // No normals: x and y are common space units while the elevation is in meters, so a
    // normal computed here would be meaningless. The mesh layer shades the surface from the
    // screen space derivatives of the common position instead, as deck.gl does for terrain.
    Ok(Mesh {
        positions,
        normals: None,
        colors: None,
        tex_coords: Some(tex_coords),
        indices: Some(indices),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// An image whose heights are `f(x, y)` through the terrarium decoder.
    fn image(size: u32, f: impl Fn(u32, u32) -> f32) -> BitmapImage {
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let h = f(x, y) + 32768.0;
                let r = (h / 256.0).floor().clamp(0.0, 255.0) as u8;
                let g = (h - r as f32 * 256.0).floor().clamp(0.0, 255.0) as u8;
                let b = ((h - r as f32 * 256.0 - g as f32) * 256.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        BitmapImage {
            width: size,
            height: size,
            rgba: Arc::new(rgba),
        }
    }

    #[test]
    fn decoders_match_their_encodings() {
        assert_eq!(ElevationDecoder::default().decode(120, 200, 30), 120.0);
        // Mapbox: -10000 + (r * 65536 + g * 256 + b) * 0.1
        let mapbox = ElevationDecoder::mapbox().decode(1, 134, 160);
        assert!(
            (mapbox - (-10000.0 + (65536.0 + 134.0 * 256.0 + 160.0) * 0.1)).abs() < 0.01,
            "{mapbox}"
        );
        // Terrarium: (r * 256 + g + b / 256) - 32768
        let terrarium = ElevationDecoder::terrarium().decode(128, 100, 128);
        assert!(
            (terrarium - (128.0 * 256.0 + 100.0 + 0.5 - 32768.0)).abs() < 0.01,
            "{terrarium}"
        );
    }

    #[test]
    fn grids_pad_to_a_power_of_two_plus_one() {
        let grid = HeightGrid::from_image(&image(4, |x, _| x as f32 * 10.0), &ElevationDecoder::terrarium())
            .unwrap();
        assert_eq!(grid.size, 5, "4 samples pad to 5");
        assert!((grid.height(3, 0) - 30.0).abs() < 0.01);
        assert!((grid.height(4, 0) - 30.0).abs() < 0.01, "the last column repeats");
        assert!((grid.height(4, 4) - 30.0).abs() < 0.01, "and the last row");
        let (lo, hi) = grid.range();
        assert!((lo - 0.0).abs() < 0.01 && (hi - 30.0).abs() < 0.01);
    }

    #[test]
    fn martini_simplifies_flat_ground_and_keeps_detail_where_it_matters() {
        // A flat tile needs two triangles however small the error
        let flat = HeightGrid::from_image(&image(8, |_, _| 100.0), &ElevationDecoder::terrarium()).unwrap();
        assert_eq!(flat.size, 9);
        let mut martini = Martini::new(&flat).unwrap();
        let (vertices, indices) = martini.mesh(1.0);
        assert_eq!(indices.len(), 6, "two triangles");
        assert_eq!(vertices.len(), 4, "the corners");
        // A peak in the middle is kept when the error allows less than its height
        let peak = HeightGrid::from_image(
            &image(8, |x, y| if x == 4 && y == 4 { 500.0 } else { 0.0 }),
            &ElevationDecoder::terrarium(),
        )
        .unwrap();
        let mut martini = Martini::new(&peak).unwrap();
        assert!(
            martini.error_at(4, 4) > 100.0,
            "the peak's split point carries its error"
        );
        let (detailed, detailed_indices) = martini.mesh(10.0);
        assert!(detailed.contains(&[4, 4]), "the peak is a vertex");
        assert!(detailed_indices.len() > 6);
        // A large error tolerance drops it again
        let (coarse, coarse_indices) = martini.mesh(1000.0);
        assert_eq!(coarse.len(), 4);
        assert_eq!(coarse_indices.len(), 6);
        // Grids that are not 2^k + 1 are rejected
        assert!(Martini::new(&HeightGrid {
            size: 6,
            heights: vec![0.0; 36]
        })
        .is_err());
    }

    #[test]
    fn terrain_mesh_spans_its_bounds_with_texture_coordinates() {
        let image = image(8, |x, y| (x + y) as f32 * 10.0);
        let mesh = terrain_mesh(
            &image,
            &ElevationDecoder::terrarium(),
            [10.0, 20.0, 11.0, 21.0],
            5.0,
        )
        .unwrap();
        mesh.validate().unwrap();
        // Terrain is flat shaded from the common space derivatives, so it carries no normals
        assert!(!mesh.has_normals());
        let bounds = mesh.bounds().unwrap();
        assert_eq!(
            [bounds[0][0], bounds[0][1]],
            [10.0, 20.0],
            "the mesh starts at the bounds"
        );
        assert_eq!([bounds[1][0], bounds[1][1]], [11.0, 21.0]);
        assert!(bounds[1][2] > bounds[0][2], "the elevation varies");
        let tex = mesh.tex_coords.as_ref().unwrap();
        assert!(tex
            .iter()
            .all(|[u, v]| (0.0..=1.0).contains(u) && (0.0..=1.0).contains(v)));
        // The image's first row is the north edge, so v 0 is the top of the bounds
        let north = mesh
            .positions
            .iter()
            .zip(tex)
            .find(|(_, [u, v])| *u == 0.0 && *v == 0.0)
            .expect("the north west corner");
        assert_eq!([north.0[0], north.0[1]], [10.0, 21.0]);
    }
}

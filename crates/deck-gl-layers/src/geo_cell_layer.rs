//! Ports of deck.gl's geo cell layers: `H3HexagonLayer`, `S2Layer`, `GeohashLayer` and
//! `QuadkeyLayer`. Each row names a cell of a spatial index; the layer turns the cells into
//! polygons and draws them with a `PolygonLayer`, so all polygon props apply.

use std::str::FromStr;
use std::sync::Arc;

use deck_gl::data::resolve_strings;
use deck_gl::{Accessor, Layer, LayerContext, LayerProps, Polygon, Position, Result, SubLayers, Viewport};
use math_gl::web_mercator::world_to_lng_lat;

use crate::{PolygonLayer, PolygonLayerProps};

/// Which spatial index the cells belong to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellKind {
    /// H3 cell indexes such as `"8928308280fffff"`; `coverage` scales each hexagon around its
    /// centre (1 is the full cell)
    H3 { coverage: f64 },
    /// S2 cell tokens such as `"80858004"`
    S2,
    /// A5 pentagon indexes, hexadecimal as in `"7b1400000000000"` or decimal
    A5,
    /// Geohashes such as `"9q8yy"`
    Geohash,
    /// Bing quadkeys such as `"0231"`; `coverage` scales each tile from its top left corner
    Quadkey { coverage: f64 },
}

impl CellKind {
    /// The ring of a cell as longitude and latitude, or `None` for an invalid index.
    pub fn polygon(&self, cell: &str) -> Option<Vec<Position>> {
        match *self {
            CellKind::H3 { coverage } => h3_polygon(cell, coverage),
            CellKind::S2 => s2_polygon(cell),
            CellKind::A5 => a5_polygon(cell),
            CellKind::Geohash => geohash_polygon(cell),
            CellKind::Quadkey { coverage } => quadkey_polygon(cell, coverage),
        }
    }
}

/// The boundary of an H3 cell, scaled by `coverage` around its centre.
pub fn h3_polygon(index: &str, coverage: f64) -> Option<Vec<Position>> {
    let cell = h3o::CellIndex::from_str(index.trim()).ok()?;
    let boundary = cell.boundary();
    let vertices: Vec<[f64; 2]> = boundary.iter().map(|ll| [ll.lng(), ll.lat()]).collect();
    if vertices.is_empty() {
        return None;
    }
    let centre = h3o::LatLng::from(cell);
    let (cx, cy) = (centre.lng(), centre.lat());
    let mut ring: Vec<Position> = vertices
        .iter()
        .map(|v| {
            // Unwrap longitudes across the antimeridian relative to the centre
            let mut lng = v[0];
            if lng - cx > 180.0 {
                lng -= 360.0;
            } else if cx - lng > 180.0 {
                lng += 360.0;
            }
            [cx + (lng - cx) * coverage, cy + (v[1] - cy) * coverage, 0.0]
        })
        .collect();
    ring.push(ring[0]);
    Some(ring)
}

/// The four corners of an S2 cell given by its token. Needs the `s2-cells` feature, which is
/// on by default; without it S2 cells resolve to nothing (the `s2` crate has no wasm32 build).
#[cfg(feature = "s2-cells")]
pub fn s2_polygon(token: &str) -> Option<Vec<Position>> {
    let id = s2::cellid::CellID::from_token(token.trim());
    if !id.is_valid() {
        return None;
    }
    let cell = s2::cell::Cell::from(&id);
    let mut ring: Vec<Position> = (0..4)
        .map(|k| {
            let ll = s2::latlng::LatLng::from(cell.vertex(k));
            [ll.lng.deg(), ll.lat.deg(), 0.0]
        })
        .collect();
    ring.push(ring[0]);
    Some(ring)
}

/// Without the `s2-cells` feature there is no S2 support, so every token is unknown.
#[cfg(not(feature = "s2-cells"))]
pub fn s2_polygon(_token: &str) -> Option<Vec<Position>> {
    None
}

/// The boundary of an A5 pentagon given by its index, hexadecimal or decimal.
pub fn a5_polygon(index: &str) -> Option<Vec<Position>> {
    let index = index.trim();
    let cell = match a5::hex_to_u64(index) {
        Ok(cell) => cell,
        Err(_) => index.parse::<u64>().ok()?,
    };
    let boundary = a5::cell_to_boundary(cell, None).ok()?;
    if boundary.is_empty() {
        return None;
    }
    // The ring comes back closed; unwrap it across the antimeridian relative to the first
    // vertex so a pentagon there stays in one piece
    let mut reference = f64::NAN;
    Some(
        boundary
            .into_iter()
            .map(|ll| {
                let mut lng = ll.longitude();
                if reference.is_nan() {
                    reference = lng;
                }
                while lng - reference > 180.0 {
                    lng -= 360.0;
                }
                while reference - lng > 180.0 {
                    lng += 360.0;
                }
                reference = lng;
                [lng, ll.latitude(), 0.0]
            })
            .collect(),
    )
}

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// `[south, west, north, east]` of a geohash, port of deck.gl's `getGeohashBounds`.
pub fn geohash_bounds(geohash: &str) -> Option<[f64; 4]> {
    let (mut min_lat, mut max_lat, mut min_lng, mut max_lng) = (-90.0, 90.0, -180.0, 180.0);
    let mut is_lng = true;
    if geohash.is_empty() {
        return None;
    }
    for ch in geohash.trim().bytes() {
        let value = GEOHASH_BASE32
            .iter()
            .position(|&c| c == ch.to_ascii_lowercase())? as u32;
        for bits in (0..5).rev() {
            let bit = (value >> bits) & 1;
            if is_lng {
                let mid = (max_lng + min_lng) / 2.0;
                if bit == 1 {
                    min_lng = mid;
                } else {
                    max_lng = mid;
                }
            } else {
                let mid = (max_lat + min_lat) / 2.0;
                if bit == 1 {
                    min_lat = mid;
                } else {
                    max_lat = mid;
                }
            }
            is_lng = !is_lng;
        }
    }
    Some([min_lat, min_lng, max_lat, max_lng])
}

pub fn geohash_polygon(geohash: &str) -> Option<Vec<Position>> {
    let [s, w, n, e] = geohash_bounds(geohash)?;
    Some(vec![
        [e, n, 0.0],
        [e, s, 0.0],
        [w, s, 0.0],
        [w, n, 0.0],
        [e, n, 0.0],
    ])
}

/// Top left and bottom right of a quadkey in Web Mercator world units, port of deck.gl's
/// `quadkeyToWorldBounds`.
pub fn quadkey_world_bounds(quadkey: &str, coverage: f64) -> Option<([f64; 2], [f64; 2])> {
    const TILE_SIZE: f64 = 512.0;
    let quadkey = quadkey.trim();
    if quadkey.is_empty() || quadkey.len() > 30 {
        return None;
    }
    let (mut x, mut y) = (0u64, 0u64);
    let mut mask = 1u64 << quadkey.len();
    let scale = mask as f64 / TILE_SIZE;
    for ch in quadkey.bytes() {
        mask >>= 1;
        let q = (ch as char).to_digit(10)?;
        if q > 3 {
            return None;
        }
        if q % 2 == 1 {
            x |= mask;
        }
        if q > 1 {
            y |= mask;
        }
    }
    Some((
        [x as f64 / scale, TILE_SIZE - y as f64 / scale],
        [
            (x as f64 + coverage) / scale,
            TILE_SIZE - (y as f64 + coverage) / scale,
        ],
    ))
}

pub fn quadkey_polygon(quadkey: &str, coverage: f64) -> Option<Vec<Position>> {
    let (top_left, bottom_right) = quadkey_world_bounds(quadkey, coverage)?;
    let [w, n] = world_to_lng_lat(top_left);
    let [e, s] = world_to_lng_lat(bottom_right);
    Some(vec![
        [e, n, 0.0],
        [e, s, 0.0],
        [w, s, 0.0],
        [w, n, 0.0],
        [e, n, 0.0],
    ])
}

/// Props of a geo cell layer: the polygon props plus the cell accessor and the index kind.
/// `polygon.get_polygon` is ignored; the cells define the geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoCellLayerProps {
    pub polygon: PolygonLayerProps,
    /// The cell index or token of each row
    pub get_cell: Accessor<String>,
    pub kind: CellKind,
}

impl GeoCellLayerProps {
    /// deck.gl's `H3HexagonLayer` defaults: extruded, cells from the `hexagon` field.
    pub fn h3() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("h3-hexagons"),
                extruded: true,
                ..Default::default()
            },
            get_cell: Accessor::column("hexagon"),
            kind: CellKind::H3 { coverage: 1.0 },
        }
    }

    /// deck.gl's `A5Layer` defaults: pentagons from the `pentagon` field.
    pub fn a5() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("a5-cells"),
                ..Default::default()
            },
            get_cell: Accessor::column("pentagon"),
            kind: CellKind::A5,
        }
    }

    pub fn s2() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("s2-cells"),
                ..Default::default()
            },
            get_cell: Accessor::column("token"),
            kind: CellKind::S2,
        }
    }

    pub fn geohash() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("geohashes"),
                ..Default::default()
            },
            get_cell: Accessor::column("geohash"),
            kind: CellKind::Geohash,
        }
    }

    pub fn quadkey() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("quadkeys"),
                ..Default::default()
            },
            get_cell: Accessor::column("quadkey"),
            kind: CellKind::Quadkey { coverage: 1.0 },
        }
    }
}

impl Default for GeoCellLayerProps {
    fn default() -> Self {
        Self::geohash()
    }
}

/// A composite layer drawing spatial index cells as polygons.
pub struct GeoCellLayer {
    props: GeoCellLayerProps,
    sub_layers: SubLayers,
    dirty: bool,
}

impl GeoCellLayer {
    pub fn new(props: GeoCellLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &GeoCellLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: GeoCellLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    /// The polygons of the rows' cells; invalid cells become empty polygons.
    pub fn polygons(props: &GeoCellLayerProps) -> Result<Vec<Polygon>> {
        let cells = resolve_strings(&props.polygon.data, &props.get_cell)?;
        Ok(cells
            .iter()
            .map(|cell| {
                props
                    .kind
                    .polygon(cell)
                    .map(|ring| vec![ring])
                    .unwrap_or_default()
            })
            .collect())
    }

    fn render_layers(&self) -> Result<Vec<Box<dyn Layer>>> {
        let polygons = Arc::new(Self::polygons(&self.props)?);
        let props = &self.props.polygon;
        let layer = PolygonLayer::new(PolygonLayerProps {
            base: LayerProps {
                id: format!("{}-cells", props.base.id),
                ..props.base.clone()
            },
            get_polygon: Accessor::Func(Arc::new(move |i| polygons[i].clone())),
            ..props.clone()
        });
        Ok(vec![Box::new(layer)])
    }
}

impl Layer for GeoCellLayer {
    fn props(&self) -> &LayerProps {
        &self.props.polygon.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.dirty || self.sub_layers.is_empty() {
            let layers = self.render_layers()?;
            self.sub_layers.replace(layers);
            self.dirty = false;
        }
        self.sub_layers.update(ctx, viewport)
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.sub_layers.draw(ctx, pass)
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        self.sub_layers.set_picking_active(ctx, active)
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.sub_layers.draw_picking(ctx, pass)
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.polygon.base.highlighted_object_index = index;
        self.sub_layers.set_highlighted_object(index);
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn update_from(&mut self, incoming: &mut dyn Layer) -> bool {
        match incoming.as_any_mut().downcast_mut::<Self>() {
            Some(other) => {
                self.set_props(std::mem::take(&mut other.props));
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geohash_and_quadkey_bounds_match_deck_gl() {
        // deck.gl's test fixture: geohash "9q8yy" covers part of San Francisco
        let [s, w, n, e] = geohash_bounds("9q8yy").unwrap();
        assert!(
            w < -122.4 && e > -122.4 && s < 37.78 && n > 37.78,
            "{s} {w} {n} {e}"
        );
        assert!((e - w - 0.0439453125).abs() < 1e-9 && (n - s - 0.0439453125).abs() < 1e-9);
        assert!(geohash_bounds("9a!").is_none());
        // Quadkey "0" is the north west quarter of the world
        let ring = quadkey_polygon("0", 1.0).unwrap();
        assert!((ring[0][0] - 0.0).abs() < 1e-9 && ring[0][1] > 85.0, "{ring:?}");
        assert!(
            (ring[2][0] + 180.0).abs() < 1e-9 && ring[2][1].abs() < 1e-9,
            "{ring:?}"
        );
        assert!(quadkey_polygon("05", 1.0).is_none());
        // Half coverage keeps the top left corner
        let half = quadkey_polygon("0", 0.5).unwrap();
        assert!(
            (half[3][0] + 180.0).abs() < 1e-9 && (half[1][0] + 90.0).abs() < 1e-9,
            "{half:?}"
        );
    }

    #[test]
    fn a5_pentagons_come_back_as_closed_rings() {
        // The A5 cell of the view centre at a middling resolution
        let cell = a5::lonlat_to_cell(a5::LonLat::new(-122.4, 37.8), 8).unwrap();
        let index = a5::u64_to_hex(cell);
        let ring = a5_polygon(&index).expect("a boundary");
        assert!(ring.len() >= 6, "a pentagon with split edges: {}", ring.len());
        assert_eq!(ring[0], ring[ring.len() - 1], "the ring is closed");
        // Every vertex is within a few degrees of the cell it came from
        for p in &ring {
            assert!((p[0] - -122.4).abs() < 5.0 && (p[1] - 37.8).abs() < 5.0, "{p:?}");
        }
        // Decimal indexes work too, and nonsense is skipped
        assert_eq!(a5_polygon(&cell.to_string()), Some(ring));
        assert!(a5_polygon("not a cell").is_none());
    }

    #[test]
    fn h3_and_s2_cells_become_rings() {
        let ring = h3_polygon("8928308280fffff", 1.0).unwrap();
        assert_eq!(ring.len(), 7);
        assert!(
            ring.iter()
                .all(|p| (p[0] + 122.4).abs() < 0.1 && (p[1] - 37.8).abs() < 0.1),
            "{ring:?}"
        );
        let small = h3_polygon("8928308280fffff", 0.5).unwrap();
        let span = |r: &[Position]| {
            r.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max)
                - r.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min)
        };
        assert!((span(&small) - span(&ring) / 2.0).abs() < 1e-9);
        assert!(h3_polygon("not a cell", 1.0).is_none());
        let s2 = s2_polygon("80858004").unwrap();
        assert_eq!(s2.len(), 5);
        assert!(
            s2.iter()
                .all(|p| (p[0] + 122.0).abs() < 2.0 && (p[1] - 37.0).abs() < 2.0),
            "{s2:?}"
        );
        assert!(s2_polygon("zz").is_none());
    }
}

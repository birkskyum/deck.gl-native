//! The 3D Tiles pieces the [`Tile3DLayer`](crate::Tile3DLayer) needs: a tileset read from
//! its JSON, the traversal that picks which tiles to draw for a view, and the `b3dm` and
//! `pnts` container formats.
//!
//! Positions in 3D Tiles are earth centred and earth fixed (ECEF) metres. The layer draws
//! them as metre offsets around a origin, so this module also carries the WGS84 ellipsoid
//! maths that converts between the two.

use glam::{DMat4, DVec3, DVec4};
use serde_json::Value;

/// Semi major axis of the WGS84 ellipsoid, in metres.
pub const WGS84_A: f64 = 6_378_137.0;
/// Semi minor axis of the WGS84 ellipsoid, in metres.
pub const WGS84_B: f64 = 6_356_752.314_245_179;

/// Longitude and latitude in degrees with a height in metres, as an ECEF position.
pub fn cartographic_to_cartesian(lng: f64, lat: f64, height: f64) -> DVec3 {
    let (lng, lat) = (lng.to_radians(), lat.to_radians());
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lng, cos_lng) = lng.sin_cos();
    let e2 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A);
    // Radius of curvature in the prime vertical
    let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    DVec3::new(
        (n + height) * cos_lat * cos_lng,
        (n + height) * cos_lat * sin_lng,
        (n * (1.0 - e2) + height) * sin_lat,
    )
}

/// An ECEF position as longitude and latitude in degrees with a height in metres.
pub fn cartesian_to_cartographic(position: DVec3) -> DVec3 {
    let e2 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A);
    let p = (position.x * position.x + position.y * position.y).sqrt();
    let lng = position.y.atan2(position.x);
    if p < 1e-9 {
        // On the axis: the latitude is a pole and the height follows the minor axis
        let sign = if position.z < 0.0 { -1.0 } else { 1.0 };
        return DVec3::new(lng.to_degrees(), sign * 90.0, position.z.abs() - WGS84_B);
    }
    // Bowring's method, two rounds of it are plenty for the heights tiles use
    let mut lat = (position.z / (p * (1.0 - e2))).atan();
    for _ in 0..5 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        lat = (position.z + e2 * n * sin_lat).atan2(p);
    }
    let sin_lat = lat.sin();
    let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let height = p / lat.cos() - n;
    DVec3::new(lng.to_degrees(), lat.to_degrees(), height)
}

/// The frame with x east, y north and z up at an ECEF position: the matrix that takes local
/// metre offsets to ECEF.
pub fn east_north_up(origin: DVec3) -> DMat4 {
    let carto = cartesian_to_cartographic(origin);
    let (lng, lat) = (carto.x.to_radians(), carto.y.to_radians());
    let (sin_lng, cos_lng) = lng.sin_cos();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let east = DVec3::new(-sin_lng, cos_lng, 0.0);
    let north = DVec3::new(-sin_lat * cos_lng, -sin_lat * sin_lng, cos_lat);
    let up = DVec3::new(cos_lat * cos_lng, cos_lat * sin_lng, sin_lat);
    DMat4::from_cols(
        east.extend(0.0),
        north.extend(0.0),
        up.extend(0.0),
        origin.extend(1.0),
    )
}

/// How a tile's children relate to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Refine {
    /// The children are drawn as well as the tile
    Add,
    /// The children replace the tile
    #[default]
    Replace,
}

/// The volume a tile's content lies in, in the tile's own frame before its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundingVolume {
    /// `[west, south, east, north, min height, max height]`, angles in radians
    Region([f64; 6]),
    /// A centre and three half axes
    Box {
        center: DVec3,
        half_axes: [DVec3; 3],
    },
    Sphere {
        center: DVec3,
        radius: f64,
    },
}

impl BoundingVolume {
    /// The centre of the volume in ECEF metres, after `transform`.
    pub fn center(&self, transform: &DMat4) -> DVec3 {
        match self {
            // A region is already geographic, so the tile transform does not apply
            Self::Region([west, south, east, north, min_h, max_h]) => cartographic_to_cartesian(
                (west + east).to_degrees() / 2.0,
                (south + north).to_degrees() / 2.0,
                (min_h + max_h) / 2.0,
            ),
            Self::Box { center, .. } => transform.transform_point3(*center),
            Self::Sphere { center, .. } => transform.transform_point3(*center),
        }
    }

    /// A radius that covers the volume, in metres.
    pub fn radius(&self, transform: &DMat4) -> f64 {
        match self {
            Self::Region([west, south, east, north, min_h, max_h]) => {
                let corner = cartographic_to_cartesian(west.to_degrees(), south.to_degrees(), *min_h);
                let far = cartographic_to_cartesian(east.to_degrees(), north.to_degrees(), *max_h);
                (far - corner).length() / 2.0
            }
            Self::Box { half_axes, .. } => half_axes
                .iter()
                .map(|axis| transform.transform_vector3(*axis).length())
                .sum::<f64>(),
            Self::Sphere { center: _, radius } => {
                let scale = transform.transform_vector3(DVec3::X).length();
                radius * scale
            }
        }
    }

    /// The volume as longitude, latitude and height of its centre.
    pub fn cartographic_center(&self, transform: &DMat4) -> DVec3 {
        cartesian_to_cartographic(self.center(transform))
    }
}

/// One tile of a tileset, with its children.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile3D {
    pub bounding_volume: BoundingVolume,
    /// The error, in metres, of not refining into the children
    pub geometric_error: f64,
    pub refine: Refine,
    /// The tile's content, relative to the tileset
    pub content_uri: Option<String>,
    /// The tile's transform composed with its parents'
    pub transform: DMat4,
    pub children: Vec<Tile3D>,
}

/// A tileset read from its JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct Tileset3D {
    pub root: Tile3D,
    pub geometric_error: f64,
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64()
}

fn numbers(value: &Value, expected: usize) -> Option<Vec<f64>> {
    let items = value.as_array()?;
    if items.len() != expected {
        return None;
    }
    items.iter().map(number).collect()
}

fn parse_bounding_volume(value: &Value) -> Result<BoundingVolume, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "boundingVolume must be an object".to_string())?;
    if let Some(region) = object.get("region") {
        let v = numbers(region, 6).ok_or_else(|| "region needs six numbers".to_string())?;
        return Ok(BoundingVolume::Region([v[0], v[1], v[2], v[3], v[4], v[5]]));
    }
    if let Some(bbox) = object.get("box") {
        let v = numbers(bbox, 12).ok_or_else(|| "box needs twelve numbers".to_string())?;
        return Ok(BoundingVolume::Box {
            center: DVec3::new(v[0], v[1], v[2]),
            half_axes: [
                DVec3::new(v[3], v[4], v[5]),
                DVec3::new(v[6], v[7], v[8]),
                DVec3::new(v[9], v[10], v[11]),
            ],
        });
    }
    if let Some(sphere) = object.get("sphere") {
        let v = numbers(sphere, 4).ok_or_else(|| "sphere needs four numbers".to_string())?;
        return Ok(BoundingVolume::Sphere {
            center: DVec3::new(v[0], v[1], v[2]),
            radius: v[3],
        });
    }
    Err("boundingVolume must be a region, box or sphere".into())
}

fn parse_tile(value: &Value, parent_transform: DMat4, parent_refine: Refine) -> Result<Tile3D, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "a tile must be an object".to_string())?;
    let bounding_volume = parse_bounding_volume(
        object
            .get("boundingVolume")
            .ok_or_else(|| "a tile needs a boundingVolume".to_string())?,
    )?;
    let transform = match object.get("transform").and_then(|t| numbers(t, 16)) {
        // 3D Tiles matrices are column major, as glam wants them
        Some(v) => parent_transform * DMat4::from_cols_array(&std::array::from_fn(|i| v[i])),
        None => parent_transform,
    };
    let refine = match object.get("refine").and_then(Value::as_str) {
        Some(r) if r.eq_ignore_ascii_case("add") => Refine::Add,
        Some(r) if r.eq_ignore_ascii_case("replace") => Refine::Replace,
        // A tile without one inherits its parent's
        _ => parent_refine,
    };
    let content_uri = object
        .get("content")
        .and_then(Value::as_object)
        .and_then(|c| c.get("uri").or_else(|| c.get("url")))
        .and_then(Value::as_str)
        .map(str::to_string);
    let children = match object.get("children") {
        Some(Value::Array(items)) => items
            .iter()
            .map(|child| parse_tile(child, transform, refine))
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    Ok(Tile3D {
        bounding_volume,
        geometric_error: object.get("geometricError").and_then(number).unwrap_or(0.0),
        refine,
        content_uri,
        transform,
        children,
    })
}

/// Read a `tileset.json`.
pub fn parse_tileset(json: &Value) -> Result<Tileset3D, String> {
    let object = json
        .as_object()
        .ok_or_else(|| "a tileset must be an object".to_string())?;
    let root = object
        .get("root")
        .ok_or_else(|| "a tileset needs a root tile".to_string())?;
    Ok(Tileset3D {
        root: parse_tile(root, DMat4::IDENTITY, Refine::Replace)?,
        geometric_error: object.get("geometricError").and_then(number).unwrap_or(0.0),
    })
}

/// A tile the traversal picked, with what it takes to draw it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedTile {
    /// Where the tile sits in the tileset's tree, as the indexes taken at each level
    pub path: Vec<usize>,
    pub content_uri: String,
    pub transform: DMat4,
    /// Longitude, latitude and height of the tile's centre
    pub center: DVec3,
}

/// How the traversal sees the view.
#[derive(Clone, Copy, Debug)]
pub struct TraversalView {
    /// The camera in ECEF metres
    pub camera: DVec3,
    /// Height of the viewport in pixels
    pub height: f64,
    /// Vertical field of view in radians
    pub fovy: f64,
    /// deck.gl's `maximumScreenSpaceError`
    pub max_error: f64,
}

impl TraversalView {
    /// The screen space error of a geometric error at a distance, in pixels.
    fn screen_error(&self, geometric_error: f64, distance: f64) -> f64 {
        if geometric_error <= 0.0 {
            return 0.0;
        }
        if distance <= f64::EPSILON {
            return f64::INFINITY;
        }
        // loaders.gl's perspective screen space error
        geometric_error * self.height / (distance * 2.0 * (self.fovy / 2.0).tan())
    }
}

/// Pick the tiles to draw, deck.gl's screen space error traversal: a tile is refined into its
/// children when its own error is too large on screen, and drawn otherwise.
pub fn select_tiles(tileset: &Tileset3D, view: &TraversalView) -> Vec<SelectedTile> {
    let mut selected = Vec::new();
    visit(&tileset.root, view, &mut Vec::new(), &mut selected);
    selected
}

fn visit(tile: &Tile3D, view: &TraversalView, path: &mut Vec<usize>, out: &mut Vec<SelectedTile>) {
    let center = tile.bounding_volume.center(&tile.transform);
    let radius = tile.bounding_volume.radius(&tile.transform);
    // Distance to the volume, not to its centre
    let distance = ((view.camera - center).length() - radius).max(0.0);
    let error = view.screen_error(tile.geometric_error, distance);
    let refine = error > view.max_error && !tile.children.is_empty();
    let draw = !refine || tile.refine == Refine::Add;
    if draw {
        if let Some(uri) = &tile.content_uri {
            out.push(SelectedTile {
                path: path.clone(),
                content_uri: uri.clone(),
                transform: tile.transform,
                center: cartesian_to_cartographic(center),
            });
        }
    }
    if refine {
        for (index, child) in tile.children.iter().enumerate() {
            path.push(index);
            visit(child, view, path, out);
            path.pop();
        }
    }
}

/// The payload of a `b3dm` or `pnts` file: the glTF or point data with the centre its
/// positions are relative to.
#[derive(Clone, Debug, PartialEq)]
pub struct TileContent {
    /// The embedded glTF of a `b3dm`, empty for other kinds
    pub gltf: Vec<u8>,
    /// The feature table's `RTC_CENTER`, in the tile's frame
    pub rtc_center: Option<DVec3>,
}

/// Read a 3D Tiles container. `b3dm` gives up its glTF; a bare `glTF` or `glb` passes
/// through. Other kinds are an error for now.
pub fn parse_tile_content(bytes: &[u8]) -> Result<TileContent, String> {
    if bytes.len() < 4 {
        return Err("tile content is empty".into());
    }
    let magic = &bytes[0..4];
    if magic == b"glTF" {
        return Ok(TileContent {
            gltf: bytes.to_vec(),
            rtc_center: None,
        });
    }
    if magic != b"b3dm" {
        return Err(format!(
            "tile content `{}` is not supported yet",
            String::from_utf8_lossy(magic)
        ));
    }
    if bytes.len() < 28 {
        return Err("b3dm header is truncated".into());
    }
    let word = |offset: usize| -> usize {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize
    };
    let feature_json = word(12);
    let feature_binary = word(16);
    let batch_json = word(20);
    let batch_binary = word(24);
    let start = 28 + feature_json;
    let rtc_center = bytes
        .get(28..start)
        .and_then(|json| serde_json::from_slice::<Value>(json).ok())
        .and_then(|table| {
            let rtc = numbers(table.get("RTC_CENTER")?, 3)?;
            Some(DVec3::new(rtc[0], rtc[1], rtc[2]))
        });
    let gltf_start = start + feature_binary + batch_json + batch_binary;
    let gltf = bytes
        .get(gltf_start..)
        .ok_or_else(|| "b3dm has no glTF payload".to_string())?
        .to_vec();
    if gltf.len() < 4 || &gltf[0..4] != b"glTF" {
        return Err("b3dm payload is not glTF".into());
    }
    Ok(TileContent { gltf, rtc_center })
}

/// The matrix that takes a tile's ECEF positions to metre offsets around `origin`.
///
/// glTF in 3D Tiles is y up while the tiles themselves are z up, so the y up to z up
/// rotation is part of it, as is the tile's own transform and any `RTC_CENTER`.
pub fn tile_to_meter_offsets(transform: &DMat4, rtc_center: Option<DVec3>, origin: DVec3) -> DMat4 {
    let origin_ecef = cartographic_to_cartesian(origin.x, origin.y, origin.z);
    let to_local = east_north_up(origin_ecef).inverse();
    let rtc = rtc_center.map_or(DMat4::IDENTITY, DMat4::from_translation);
    // glTF's y up to the z up of 3D Tiles
    let y_up_to_z_up = DMat4::from_cols(
        DVec4::new(1.0, 0.0, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 1.0, 0.0),
        DVec4::new(0.0, -1.0, 0.0, 0.0),
        DVec4::W,
    );
    to_local * *transform * rtc * y_up_to_z_up
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cartographic_and_cartesian_round_trip() {
        for (lng, lat, height) in [
            (0.0, 0.0, 0.0),
            (-122.4, 37.8, 120.0),
            (13.4, 52.5, -30.0),
            (179.9, -66.5, 4000.0),
        ] {
            let ecef = cartographic_to_cartesian(lng, lat, height);
            let back = cartesian_to_cartographic(ecef);
            assert!((back.x - lng).abs() < 1e-9, "{lng} became {}", back.x);
            assert!((back.y - lat).abs() < 1e-9, "{lat} became {}", back.y);
            assert!((back.z - height).abs() < 1e-6, "{height} became {}", back.z);
        }
        // The equator sits on the semi major axis, the pole on the semi minor one
        assert!((cartographic_to_cartesian(0.0, 0.0, 0.0).x - WGS84_A).abs() < 1e-6);
        assert!((cartographic_to_cartesian(0.0, 90.0, 0.0).z - WGS84_B).abs() < 1e-6);
    }

    #[test]
    fn east_north_up_points_the_way_it_says() {
        let origin = cartographic_to_cartesian(-122.4, 37.8, 0.0);
        let frame = east_north_up(origin);
        let east = frame.transform_vector3(DVec3::X);
        let north = frame.transform_vector3(DVec3::Y);
        let up = frame.transform_vector3(DVec3::Z);
        // The axes are unit length and at right angles
        for axis in [east, north, up] {
            assert!((axis.length() - 1.0).abs() < 1e-9);
        }
        assert!(east.dot(north).abs() < 1e-9);
        assert!(east.dot(up).abs() < 1e-9);
        // Up points away from the centre of the earth, east has no vertical component
        assert!(up.dot(origin.normalize()) > 0.999);
        assert!(east.z.abs() < 1e-9);
        // A hundred metres north lands a hundred metres from the origin
        let moved = frame.transform_point3(DVec3::new(0.0, 100.0, 0.0));
        assert!(((moved - origin).length() - 100.0).abs() < 1e-6);
    }

    fn tileset_json() -> Value {
        json!({
            "asset": {"version": "1.0"},
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {"sphere": [0.0, 0.0, 0.0, 100.0]},
                "geometricError": 100.0,
                "refine": "REPLACE",
                "transform": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    4000000.0, -4000000.0, 4000000.0, 1.0
                ],
                "content": {"uri": "root.b3dm"},
                "children": [
                    {
                        "boundingVolume": {"sphere": [0.0, 0.0, 0.0, 50.0]},
                        "geometricError": 0.0,
                        "content": {"uri": "child.b3dm"}
                    }
                ]
            }
        })
    }

    #[test]
    fn tilesets_parse_with_inherited_transforms_and_refinement() {
        let tileset = parse_tileset(&tileset_json()).unwrap();
        assert_eq!(tileset.geometric_error, 500.0);
        assert_eq!(tileset.root.content_uri.as_deref(), Some("root.b3dm"));
        assert_eq!(tileset.root.refine, Refine::Replace);
        let child = &tileset.root.children[0];
        // The child has no transform or refine of its own, so it takes the root's
        assert_eq!(child.transform, tileset.root.transform);
        assert_eq!(child.refine, Refine::Replace);
        assert_eq!(child.content_uri.as_deref(), Some("child.b3dm"));
        // The root's translation moves its bounding sphere there
        let center = tileset.root.bounding_volume.center(&tileset.root.transform);
        assert_eq!(center, DVec3::new(4000000.0, -4000000.0, 4000000.0));
        assert!(parse_tileset(&json!({"asset": {}})).is_err());
    }

    #[test]
    fn the_traversal_refines_close_up_and_stops_far_away() {
        let tileset = parse_tileset(&tileset_json()).unwrap();
        let center = tileset.root.bounding_volume.center(&tileset.root.transform);
        let view = |distance: f64| TraversalView {
            camera: center + DVec3::new(distance, 0.0, 0.0),
            height: 1000.0,
            fovy: 0.8,
            max_error: 16.0,
        };
        // Far away the root's own error is small enough, so only it is drawn
        let far = select_tiles(&tileset, &view(1_000_000.0));
        assert_eq!(far.len(), 1);
        assert_eq!(far[0].content_uri, "root.b3dm");
        assert!(far[0].path.is_empty());
        // Close up it refines into the child, and replacement drops the root
        let near = select_tiles(&tileset, &view(1_000.0));
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].content_uri, "child.b3dm");
        assert_eq!(near[0].path, vec![0]);
        // With additive refinement both are drawn
        let mut additive = tileset.clone();
        additive.root.refine = Refine::Add;
        let both = select_tiles(&additive, &view(1_000.0));
        assert_eq!(both.len(), 2);
        // The selected tile knows where it is on the globe
        let carto = near[0].center;
        assert!(carto.y.abs() <= 90.0 && carto.x.abs() <= 180.0, "{carto:?}");
    }

    #[test]
    fn b3dm_gives_up_its_gltf_and_centre() {
        let gltf = b"glTF\x02\x00\x00\x00rest of the model";
        let feature_table = br#"{"RTC_CENTER":[1.0,2.0,3.0]}"#;
        let mut bytes = Vec::new();
        bytes.extend(b"b3dm");
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes()); // byte length, unused here
        bytes.extend((feature_table.len() as u32).to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(feature_table);
        bytes.extend(gltf);
        let content = parse_tile_content(&bytes).unwrap();
        assert_eq!(content.gltf, gltf);
        assert_eq!(content.rtc_center, Some(DVec3::new(1.0, 2.0, 3.0)));
        // A bare glTF passes through
        let plain = parse_tile_content(gltf).unwrap();
        assert_eq!(plain.gltf, gltf);
        assert_eq!(plain.rtc_center, None);
        // Other containers say so rather than producing nonsense
        let pnts = parse_tile_content(b"pnts\x01\x00\x00\x00").unwrap_err();
        assert!(pnts.contains("pnts"), "{pnts}");
    }

    #[test]
    fn tile_positions_become_metre_offsets_around_the_origin() {
        // A tile whose frame sits at the origin: a vertex there is at no offset
        let origin = DVec3::new(-122.4, 37.8, 0.0);
        let origin_ecef = cartographic_to_cartesian(origin.x, origin.y, origin.z);
        let transform = east_north_up(origin_ecef);
        let matrix = tile_to_meter_offsets(&transform, None, origin);
        let at_origin = matrix.transform_point3(DVec3::ZERO);
        assert!(at_origin.length() < 1e-6, "{at_origin:?}");
        // glTF y up: ten metres along its y is ten metres up here
        let up = matrix.transform_point3(DVec3::new(0.0, 10.0, 0.0));
        assert!((up - DVec3::new(0.0, 0.0, 10.0)).length() < 1e-6, "{up:?}");
        // And its x stays east
        let east = matrix.transform_point3(DVec3::new(10.0, 0.0, 0.0));
        assert!((east - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-6, "{east:?}");
        // A relative centre shifts everything by it
        let shifted = tile_to_meter_offsets(&transform, Some(DVec3::new(5.0, 0.0, 0.0)), origin);
        let moved = shifted.transform_point3(DVec3::ZERO);
        assert!((moved - DVec3::new(5.0, 0.0, 0.0)).length() < 1e-6, "{moved:?}");
    }
}

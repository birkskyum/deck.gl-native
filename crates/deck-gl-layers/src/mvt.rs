//! A decoder for Mapbox Vector Tiles (the `.mvt` / `.pbf` protobuf format) into GeoJSON
//! features in longitude and latitude, for the [`crate::MvtLayer`].

use std::collections::HashMap;

use deck_gl::geojson::{Feature, FeatureCollection, Geometry};
use deck_gl::Position;
use serde_json::{Map, Value};

use crate::tileset::TileBounds;

/// A protobuf wire type reader over a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self.bytes.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too long".into());
            }
        }
    }

    /// Field number and wire type of the next key.
    fn key(&mut self) -> Result<(u32, u8), String> {
        let key = self.varint()?;
        Ok(((key >> 3) as u32, (key & 7) as u8))
    }

    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|e| *e <= self.bytes.len())
            .ok_or("truncated bytes")?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, wire_type: u8) -> Result<(), String> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.pos += 4,
            other => return Err(format!("unsupported wire type {other}")),
        }
        if self.pos > self.bytes.len() {
            return Err("truncated field".into());
        }
        Ok(())
    }

    fn fixed64(&mut self) -> Result<u64, String> {
        let end = self.pos + 8;
        let bytes = self.bytes.get(self.pos..end).ok_or("truncated fixed64")?;
        self.pos = end;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn fixed32(&mut self) -> Result<u32, String> {
        let end = self.pos + 4;
        let bytes = self.bytes.get(self.pos..end).ok_or("truncated fixed32")?;
        self.pos = end;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
}

fn zigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// One layer of a tile with its features as GeoJSON.
#[derive(Clone, Debug, Default)]
pub struct MvtLayerData {
    pub name: String,
    pub extent: u32,
    pub features: Vec<Feature>,
}

fn decode_value(bytes: &[u8]) -> Result<Value, String> {
    let mut r = Reader::new(bytes);
    let mut value = Value::Null;
    while !r.done() {
        let (field, wire) = r.key()?;
        value = match (field, wire) {
            (1, 2) => Value::String(String::from_utf8_lossy(r.bytes()?).into_owned()),
            (2, 5) => Value::from(f32::from_bits(r.fixed32()?) as f64),
            (3, 1) => Value::from(f64::from_bits(r.fixed64()?)),
            (4, 0) => Value::from(r.varint()? as i64),
            (5, 0) => Value::from(r.varint()?),
            (6, 0) => Value::from(zigzag(r.varint()?)),
            (7, 0) => Value::Bool(r.varint()? != 0),
            (_, wire) => {
                r.skip(wire)?;
                value
            }
        };
    }
    Ok(value)
}

struct RawFeature {
    id: Option<u64>,
    tags: Vec<u32>,
    kind: u32,
    geometry: Vec<u32>,
}

fn decode_feature(bytes: &[u8]) -> Result<RawFeature, String> {
    let mut r = Reader::new(bytes);
    let mut feature = RawFeature {
        id: None,
        tags: Vec::new(),
        kind: 0,
        geometry: Vec::new(),
    };
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 0) => feature.id = Some(r.varint()?),
            (2, 2) => {
                let mut packed = Reader::new(r.bytes()?);
                while !packed.done() {
                    feature.tags.push(packed.varint()? as u32);
                }
            }
            (2, 0) => feature.tags.push(r.varint()? as u32),
            (3, 0) => feature.kind = r.varint()? as u32,
            (4, 2) => {
                let mut packed = Reader::new(r.bytes()?);
                while !packed.done() {
                    feature.geometry.push(packed.varint()? as u32);
                }
            }
            (4, 0) => feature.geometry.push(r.varint()? as u32),
            (_, wire) => r.skip(wire)?,
        }
    }
    Ok(feature)
}

/// Walk the geometry commands into rings of tile coordinates (MoveTo starts a new part).
fn decode_rings(commands: &[u32]) -> Result<Vec<Vec<[i64; 2]>>, String> {
    let mut rings: Vec<Vec<[i64; 2]>> = Vec::new();
    let (mut x, mut y) = (0i64, 0i64);
    let mut i = 0;
    while i < commands.len() {
        let command = commands[i] & 7;
        let count = (commands[i] >> 3) as usize;
        i += 1;
        match command {
            1 | 2 => {
                for _ in 0..count {
                    let (dx, dy) = (
                        *commands.get(i).ok_or("truncated geometry")?,
                        *commands.get(i + 1).ok_or("truncated geometry")?,
                    );
                    i += 2;
                    x += zigzag(dx as u64);
                    y += zigzag(dy as u64);
                    if command == 1 {
                        rings.push(vec![[x, y]]);
                    } else if let Some(ring) = rings.last_mut() {
                        ring.push([x, y]);
                    } else {
                        return Err("LineTo before MoveTo".into());
                    }
                }
            }
            7 => {
                if let Some(ring) = rings.last_mut() {
                    if let Some(first) = ring.first().copied() {
                        ring.push(first);
                    }
                }
            }
            other => return Err(format!("unknown geometry command {other}")),
        }
    }
    Ok(rings)
}

/// Twice the signed area of a ring in tile coordinates (y down): positive is clockwise.
fn signed_area(ring: &[[i64; 2]]) -> i64 {
    let mut area = 0i64;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area
}

/// Converts tile coordinates to longitude and latitude within the tile's bounds.
struct Projector {
    west: f64,
    north_y: f64,
    lng_span: f64,
    y_span: f64,
    extent: f64,
}

impl Projector {
    fn new(bounds: TileBounds, extent: u32) -> Self {
        let [west, south, east, north] = bounds.as_array();
        // Tile rows are linear in Web Mercator y, not in latitude
        let mercator_y = |lat: f64| (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln();
        let (north_y, south_y) = (mercator_y(north), mercator_y(south));
        Self {
            west,
            north_y,
            lng_span: east - west,
            y_span: south_y - north_y,
            extent: extent as f64,
        }
    }

    fn position(&self, p: [i64; 2]) -> Position {
        let lng = self.west + p[0] as f64 / self.extent * self.lng_span;
        let y = self.north_y + p[1] as f64 / self.extent * self.y_span;
        let lat = (2.0 * y.exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees();
        [lng, lat, 0.0]
    }
}

fn feature_geometry(kind: u32, rings: Vec<Vec<[i64; 2]>>, projector: &Projector) -> Option<Geometry> {
    let convert =
        |ring: &[[i64; 2]]| -> Vec<Position> { ring.iter().map(|p| projector.position(*p)).collect() };
    match kind {
        1 => {
            let points: Vec<Position> = rings.iter().flatten().map(|p| projector.position(*p)).collect();
            match points.len() {
                0 => None,
                1 => Some(Geometry::Point(points[0])),
                _ => Some(Geometry::MultiPoint(points)),
            }
        }
        2 => {
            let lines: Vec<Vec<Position>> = rings
                .iter()
                .filter(|r| r.len() >= 2)
                .map(|r| convert(r))
                .collect();
            match lines.len() {
                0 => None,
                1 => Some(Geometry::LineString(lines.into_iter().next().unwrap())),
                _ => Some(Geometry::MultiLineString(lines)),
            }
        }
        3 => {
            // Exterior rings are clockwise in tile space, holes counter clockwise
            let mut polygons: Vec<Vec<Vec<Position>>> = Vec::new();
            for ring in rings.iter().filter(|r| r.len() >= 4) {
                if signed_area(ring) > 0 || polygons.is_empty() {
                    polygons.push(vec![convert(ring)]);
                } else if let Some(last) = polygons.last_mut() {
                    last.push(convert(ring));
                }
            }
            match polygons.len() {
                0 => None,
                1 => Some(Geometry::Polygon(polygons.into_iter().next().unwrap())),
                _ => Some(Geometry::MultiPolygon(polygons)),
            }
        }
        _ => None,
    }
}

fn decode_layer(bytes: &[u8], bounds: TileBounds) -> Result<MvtLayerData, String> {
    let mut r = Reader::new(bytes);
    let mut name = String::new();
    let mut extent = 4096u32;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut raw_features = Vec::new();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => name = String::from_utf8_lossy(r.bytes()?).into_owned(),
            (2, 2) => raw_features.push(decode_feature(r.bytes()?)?),
            (3, 2) => keys.push(String::from_utf8_lossy(r.bytes()?).into_owned()),
            (4, 2) => values.push(decode_value(r.bytes()?)?),
            (5, 0) => extent = r.varint()? as u32,
            (15, 0) => {
                r.varint()?;
            }
            (_, wire) => r.skip(wire)?,
        }
    }
    let projector = Projector::new(bounds, extent.max(1));
    let mut features = Vec::with_capacity(raw_features.len());
    for raw in raw_features {
        let mut properties = Map::new();
        for pair in raw.tags.chunks(2) {
            if let [k, v] = pair {
                if let (Some(key), Some(value)) = (keys.get(*k as usize), values.get(*v as usize)) {
                    properties.insert(key.clone(), value.clone());
                }
            }
        }
        properties.insert("layerName".to_string(), Value::String(name.clone()));
        let rings = decode_rings(&raw.geometry)?;
        features.push(Feature {
            id: raw.id.map(Value::from),
            geometry: feature_geometry(raw.kind, rings, &projector),
            properties,
        });
    }
    Ok(MvtLayerData {
        name,
        extent,
        features,
    })
}

/// Inflate gzip encoded tile bytes (servers often store tiles compressed); other bytes pass
/// through.
pub fn maybe_gunzip(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if bytes.len() > 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| format!("gunzip: {e}"))?;
        return Ok(out);
    }
    Ok(bytes)
}

/// Decode a tile into its layers, with geometry in longitude and latitude within `bounds`.
pub fn decode_tile(bytes: &[u8], bounds: TileBounds) -> Result<Vec<MvtLayerData>, String> {
    let mut r = Reader::new(bytes);
    let mut layers = Vec::new();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (3, 2) => layers.push(decode_layer(r.bytes()?, bounds)?),
            (_, wire) => r.skip(wire)?,
        }
    }
    Ok(layers)
}

/// Decode a tile into one feature collection; every feature carries a `layerName` property.
pub fn decode_tile_features(bytes: &[u8], bounds: TileBounds) -> Result<FeatureCollection, String> {
    let layers = decode_tile(bytes, bounds)?;
    Ok(FeatureCollection {
        features: layers.into_iter().flat_map(|l| l.features).collect(),
    })
}

/// A feature to encode: geometry type (1 point, 2 line, 3 polygon), rings of tile
/// coordinates and properties.
pub type EncodeFeature = (u32, Vec<Vec<[i64; 2]>>, HashMap<String, Value>);
/// A layer to encode: name, extent and features.
pub type EncodeLayer<'a> = (&'a str, u32, Vec<EncodeFeature>);

/// Encode a tile for tests and tools: layers of features given as tile coordinate rings.
pub fn encode_tile(layers: &[EncodeLayer<'_>]) -> Vec<u8> {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
        varint(out, u64::from(field << 3 | 2));
        varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }
    fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        varint(out, u64::from(field << 3));
        varint(out, v);
    }
    fn zig(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    let mut tile = Vec::new();
    for (name, extent, features) in layers {
        let mut layer = Vec::new();
        field_varint(&mut layer, 15, 2);
        field_bytes(&mut layer, 1, name.as_bytes());
        let mut keys: Vec<String> = Vec::new();
        let mut values: Vec<Value> = Vec::new();
        for (kind, rings, props) in features {
            let mut feature = Vec::new();
            let mut tags = Vec::new();
            for (k, v) in props {
                let ki = keys.iter().position(|x| x == k).unwrap_or_else(|| {
                    keys.push(k.clone());
                    keys.len() - 1
                });
                let vi = values.iter().position(|x| x == v).unwrap_or_else(|| {
                    values.push(v.clone());
                    values.len() - 1
                });
                varint(&mut tags, ki as u64);
                varint(&mut tags, vi as u64);
            }
            field_bytes(&mut feature, 2, &tags);
            field_varint(&mut feature, 3, u64::from(*kind));
            let mut geometry = Vec::new();
            let (mut cx, mut cy) = (0i64, 0i64);
            for ring in rings {
                let closed = *kind == 3;
                let points = if closed && ring.len() > 1 && ring.first() == ring.last() {
                    &ring[..ring.len() - 1]
                } else {
                    &ring[..]
                };
                for (i, p) in points.iter().enumerate() {
                    if i == 0 {
                        varint(&mut geometry, 1 | (1 << 3));
                    } else if i == 1 {
                        varint(&mut geometry, 2 | (((points.len() - 1) as u64) << 3));
                    }
                    varint(&mut geometry, zig(p[0] - cx));
                    varint(&mut geometry, zig(p[1] - cy));
                    cx = p[0];
                    cy = p[1];
                }
                if closed {
                    varint(&mut geometry, 7 | (1 << 3));
                }
            }
            field_bytes(&mut feature, 4, &geometry);
            field_bytes(&mut layer, 2, &feature);
        }
        for key in &keys {
            field_bytes(&mut layer, 3, key.as_bytes());
        }
        for value in &values {
            let mut encoded = Vec::new();
            match value {
                Value::String(s) => field_bytes(&mut encoded, 1, s.as_bytes()),
                Value::Bool(b) => field_varint(&mut encoded, 7, u64::from(*b)),
                Value::Number(n) if n.is_i64() => {
                    varint(&mut encoded, 6 << 3);
                    varint(&mut encoded, zig(n.as_i64().unwrap()));
                }
                Value::Number(n) => {
                    varint(&mut encoded, 3 << 3 | 1);
                    encoded.extend_from_slice(&n.as_f64().unwrap_or(0.0).to_bits().to_le_bytes());
                }
                _ => field_bytes(&mut encoded, 1, b""),
            }
            field_bytes(&mut layer, 4, &encoded);
        }
        field_varint(&mut layer, 5, u64::from(*extent));
        field_bytes(&mut tile, 3, &layer);
    }
    tile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_points_lines_and_polygons() {
        let bounds = TileBounds::Geo {
            west: -180.0,
            north: 85.0511287798,
            east: 0.0,
            south: 0.0,
        };
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::String("A".into()));
        props.insert("height".to_string(), Value::from(12));
        props.insert("ratio".to_string(), Value::from(0.5));
        props.insert("ok".to_string(), Value::Bool(true));
        let tile = encode_tile(&[(
            "test",
            4096,
            vec![
                (1, vec![vec![[2048, 2048]]], props.clone()),
                (2, vec![vec![[0, 0], [4096, 0]]], HashMap::new()),
                (
                    3,
                    vec![vec![[0, 0], [4096, 0], [4096, 4096], [0, 4096], [0, 0]]],
                    HashMap::new(),
                ),
            ],
        )]);
        let layers = decode_tile(&tile, bounds).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(
            (
                layers[0].name.as_str(),
                layers[0].extent,
                layers[0].features.len()
            ),
            ("test", 4096, 3)
        );
        let f = &layers[0].features;
        match &f[0].geometry {
            Some(Geometry::Point(p)) => assert!(
                (p[0] + 90.0).abs() < 1e-9 && (p[1] - 66.5132604431).abs() < 1e-6,
                "{p:?}"
            ),
            other => panic!("{other:?}"),
        }
        assert_eq!(f[0].properties["name"], "A");
        assert_eq!(f[0].properties["height"], 12);
        assert_eq!(f[0].properties["ratio"], 0.5);
        assert_eq!(f[0].properties["ok"], true);
        assert_eq!(f[0].properties["layerName"], "test");
        match &f[1].geometry {
            Some(Geometry::LineString(line)) => {
                assert!((line[0][0] + 180.0).abs() < 1e-9 && (line[1][0] - 0.0).abs() < 1e-9);
                assert!((line[0][1] - 85.0511287798).abs() < 1e-6);
            }
            other => panic!("{other:?}"),
        }
        match &f[2].geometry {
            Some(Geometry::Polygon(rings)) => {
                assert_eq!(rings.len(), 1);
                assert_eq!(rings[0].len(), 5);
                assert!((rings[0][2][1] - 0.0).abs() < 1e-9, "{:?}", rings[0]);
            }
            other => panic!("{other:?}"),
        }
        assert!(decode_tile(&[0x1a, 0x05, 0x01], bounds).is_err());
    }
}

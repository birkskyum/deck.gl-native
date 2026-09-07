//! A Well Known Binary (WKB) decoder for the geometry columns of GeoParquet, PostGIS and
//! GeoArrow data: ISO and PostGIS EWKB flavours, with or without Z, M and an SRID.

use crate::geojson::Geometry;
use crate::{Path, Polygon, Position};

/// Decode a Well Known Binary geometry (ISO and PostGIS EWKB flavours, with or without Z, M
/// and an SRID).
pub fn wkb_geometry(bytes: &[u8]) -> Result<Geometry, String> {
    let mut reader = WkbReader { bytes, pos: 0 };
    let geometry = reader.geometry()?;
    Ok(geometry)
}

struct WkbReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl WkbReader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.pos + n;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| format!("WKB ends after {} of {} bytes", self.bytes.len(), end))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, little_endian: bool) -> Result<u32, String> {
        let b = self.take(4)?;
        let array = [b[0], b[1], b[2], b[3]];
        Ok(if little_endian {
            u32::from_le_bytes(array)
        } else {
            u32::from_be_bytes(array)
        })
    }

    fn f64(&mut self, little_endian: bool) -> Result<f64, String> {
        let b = self.take(8)?;
        let array = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Ok(if little_endian {
            f64::from_le_bytes(array)
        } else {
            f64::from_be_bytes(array)
        })
    }

    fn geometry(&mut self) -> Result<Geometry, String> {
        let little_endian = match self.u8()? {
            0 => false,
            1 => true,
            other => return Err(format!("WKB byte order {other} is not 0 or 1")),
        };
        let mut kind = self.u32(little_endian)?;
        let mut has_z = false;
        let mut has_m = false;
        // PostGIS EWKB flags
        if kind & 0x8000_0000 != 0 {
            has_z = true;
            kind &= !0x8000_0000;
        }
        if kind & 0x4000_0000 != 0 {
            has_m = true;
            kind &= !0x4000_0000;
        }
        if kind & 0x2000_0000 != 0 {
            kind &= !0x2000_0000;
            self.u32(little_endian)?; // SRID
        }
        // ISO WKB dimension offsets
        match kind / 1000 {
            1 => has_z = true,
            2 => has_m = true,
            3 => {
                has_z = true;
                has_m = true;
            }
            _ => {}
        }
        kind %= 1000;
        let position = |reader: &mut Self| -> Result<Position, String> {
            let x = reader.f64(little_endian)?;
            let y = reader.f64(little_endian)?;
            let z = if has_z { reader.f64(little_endian)? } else { 0.0 };
            if has_m {
                reader.f64(little_endian)?;
            }
            Ok([x, y, z])
        };
        let path = |reader: &mut Self| -> Result<Path, String> {
            let count = reader.u32(little_endian)? as usize;
            (0..count).map(|_| position(reader)).collect()
        };
        let polygon = |reader: &mut Self| -> Result<Polygon, String> {
            let rings = reader.u32(little_endian)? as usize;
            (0..rings).map(|_| path(reader)).collect()
        };
        let members = |reader: &mut Self| -> Result<Vec<Geometry>, String> {
            let count = reader.u32(little_endian)? as usize;
            (0..count).map(|_| reader.geometry()).collect()
        };
        Ok(match kind {
            1 => Geometry::Point(position(self)?),
            2 => Geometry::LineString(path(self)?),
            3 => Geometry::Polygon(polygon(self)?),
            4 => Geometry::MultiPoint(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::Point(p) => Ok(p),
                        other => Err(format!("WKB MultiPoint holds a {}", geometry_name(&other))),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            5 => Geometry::MultiLineString(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::LineString(l) => Ok(l),
                        other => Err(format!("WKB MultiLineString holds a {}", geometry_name(&other))),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            6 => Geometry::MultiPolygon(
                members(self)?
                    .into_iter()
                    .map(|g| match g {
                        Geometry::Polygon(p) => Ok(p),
                        other => Err(format!("WKB MultiPolygon holds a {}", geometry_name(&other))),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            7 => Geometry::GeometryCollection(members(self)?),
            other => return Err(format!("WKB geometry type {other} is not supported")),
        })
    }
}

fn geometry_name(geometry: &Geometry) -> &'static str {
    match geometry {
        Geometry::Point(_) => "Point",
        Geometry::MultiPoint(_) => "MultiPoint",
        Geometry::LineString(_) => "LineString",
        Geometry::MultiLineString(_) => "MultiLineString",
        Geometry::Polygon(_) => "Polygon",
        Geometry::MultiPolygon(_) => "MultiPolygon",
        Geometry::GeometryCollection(_) => "GeometryCollection",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_point(x: f64, y: f64) -> Vec<u8> {
        let mut bytes = vec![1u8, 1, 0, 0, 0];
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes
    }

    #[test]
    fn decodes_wkb_points_polygons_and_ewkb_flags() {
        assert_eq!(
            wkb_geometry(&le_point(1.5, -2.0)).unwrap(),
            Geometry::Point([1.5, -2.0, 0.0])
        );
        // big endian polygon with one ring of four points
        let mut bytes = vec![0u8, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4];
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)] {
            bytes.extend_from_slice(&f64::to_be_bytes(x));
            bytes.extend_from_slice(&f64::to_be_bytes(y));
        }
        match wkb_geometry(&bytes).unwrap() {
            Geometry::Polygon(rings) => assert_eq!(rings[0][2], [1.0, 1.0, 0.0]),
            other => panic!("{other:?}"),
        }
        // EWKB point with Z and an SRID
        let mut ewkb = vec![1u8];
        ewkb.extend_from_slice(&(1u32 | 0x8000_0000 | 0x2000_0000).to_le_bytes());
        ewkb.extend_from_slice(&4326u32.to_le_bytes());
        for v in [3.0f64, 4.0, 5.0] {
            ewkb.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(wkb_geometry(&ewkb).unwrap(), Geometry::Point([3.0, 4.0, 5.0]));
        // ISO WKB multipoint with Z (type 1004)
        let mut iso = vec![1u8];
        iso.extend_from_slice(&1004u32.to_le_bytes());
        iso.extend_from_slice(&1u32.to_le_bytes());
        iso.push(1);
        iso.extend_from_slice(&1001u32.to_le_bytes());
        for v in [1.0f64, 2.0, 3.0] {
            iso.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            wkb_geometry(&iso).unwrap(),
            Geometry::MultiPoint(vec![[1.0, 2.0, 3.0]])
        );
        assert!(wkb_geometry(&[1, 9, 0, 0, 0]).unwrap_err().contains("type 9"));
        assert!(wkb_geometry(&le_point(1.0, 2.0)[..10])
            .unwrap_err()
            .contains("ends"));
    }
}

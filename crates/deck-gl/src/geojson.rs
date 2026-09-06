//! A small GeoJSON reader for layer input. Positions are `[lng, lat, elevation]`.

use std::path::Path as FsPath;

use serde_json::{Map, Value};

use crate::data::{Path, Polygon, Position};
use crate::{DeckError, Result};

/// A GeoJSON geometry.
#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    Point(Position),
    MultiPoint(Vec<Position>),
    LineString(Path),
    MultiLineString(Vec<Path>),
    Polygon(Polygon),
    MultiPolygon(Vec<Polygon>),
    GeometryCollection(Vec<Geometry>),
}

/// A GeoJSON feature: geometry plus properties.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub id: Option<Value>,
    pub geometry: Option<Geometry>,
    pub properties: Map<String, Value>,
}

impl Feature {
    /// A numeric property, if present and numeric.
    pub fn number(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(Value::as_f64)
    }

    /// A string property, if present and a string.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(Value::as_str)
    }
}

/// A GeoJSON feature collection. Single features and bare geometries parse into a
/// collection of one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureCollection {
    pub features: Vec<Feature>,
}

impl FeatureCollection {
    pub fn parse(text: &str) -> Result<Self> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| DeckError::Data(format!("invalid JSON: {e}")))?;
        Self::from_value(&value)
    }

    pub fn from_file(path: impl AsRef<FsPath>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| DeckError::Data(format!("{}: {e}", path.as_ref().display())))?;
        Self::parse(&text)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let kind = value.get("type").and_then(Value::as_str).unwrap_or_default();
        match kind {
            "FeatureCollection" => {
                let features = value
                    .get("features")
                    .and_then(Value::as_array)
                    .ok_or_else(|| DeckError::Data("FeatureCollection without features".into()))?;
                Ok(Self {
                    features: features.iter().map(parse_feature).collect::<Result<_>>()?,
                })
            }
            "Feature" => Ok(Self {
                features: vec![parse_feature(value)?],
            }),
            _ => Ok(Self {
                features: vec![Feature {
                    id: None,
                    geometry: Some(parse_geometry(value)?),
                    properties: Map::new(),
                }],
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

fn parse_feature(value: &Value) -> Result<Feature> {
    let geometry = match value.get("geometry") {
        Some(Value::Null) | None => None,
        Some(geometry) => Some(parse_geometry(geometry)?),
    };
    let properties = match value.get("properties") {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    };
    Ok(Feature {
        id: value.get("id").cloned(),
        geometry,
        properties,
    })
}

fn parse_geometry(value: &Value) -> Result<Geometry> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| DeckError::Data("geometry without type".into()))?;
    let coordinates = || {
        value
            .get("coordinates")
            .ok_or_else(|| DeckError::Data(format!("{kind} without coordinates")))
    };
    Ok(match kind {
        "Point" => Geometry::Point(parse_position(coordinates()?)?),
        "MultiPoint" => Geometry::MultiPoint(parse_positions(coordinates()?)?),
        "LineString" => Geometry::LineString(parse_positions(coordinates()?)?),
        "MultiLineString" => Geometry::MultiLineString(parse_list(coordinates()?, parse_positions)?),
        "Polygon" => Geometry::Polygon(parse_list(coordinates()?, parse_positions)?),
        "MultiPolygon" => Geometry::MultiPolygon(parse_list(coordinates()?, |rings| {
            parse_list(rings, parse_positions)
        })?),
        "GeometryCollection" => {
            let geometries = value
                .get("geometries")
                .and_then(Value::as_array)
                .ok_or_else(|| DeckError::Data("GeometryCollection without geometries".into()))?;
            Geometry::GeometryCollection(geometries.iter().map(parse_geometry).collect::<Result<_>>()?)
        }
        other => return Err(DeckError::Data(format!("unsupported geometry type {other}"))),
    })
}

fn parse_list<T>(value: &Value, item: impl Fn(&Value) -> Result<T>) -> Result<Vec<T>> {
    value
        .as_array()
        .ok_or_else(|| DeckError::Data("expected an array".into()))?
        .iter()
        .map(item)
        .collect()
}

fn parse_positions(value: &Value) -> Result<Vec<Position>> {
    parse_list(value, parse_position)
}

fn parse_position(value: &Value) -> Result<Position> {
    let items = value
        .as_array()
        .ok_or_else(|| DeckError::Data("position must be an array".into()))?;
    if items.len() < 2 {
        return Err(DeckError::Data("position needs at least two numbers".into()));
    }
    let n = |i: usize| items.get(i).and_then(Value::as_f64);
    Ok([
        n(0).ok_or_else(|| DeckError::Data("position longitude is not a number".into()))?,
        n(1).ok_or_else(|| DeckError::Data("position latitude is not a number".into()))?,
        n(2).unwrap_or(0.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_mixed_collection() {
        let text = r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"name":"a","value":1.5},"geometry":{"type":"Point","coordinates":[1,2]}},
            {"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":[[0,0],[1,1,5]]}},
            {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,0]],[[1,1],[1.5,1],[1,1.5],[1,1]]]}},
            {"type":"Feature","properties":{},"geometry":{"type":"MultiPolygon","coordinates":[[[[0,0],[1,0],[1,1],[0,0]]]]}},
            {"type":"Feature","properties":{},"geometry":null}
        ]}"#;
        let fc = FeatureCollection::parse(text).unwrap();
        assert_eq!(fc.len(), 5);
        assert_eq!(fc.features[0].geometry, Some(Geometry::Point([1.0, 2.0, 0.0])));
        assert_eq!(fc.features[0].string("name"), Some("a"));
        assert_eq!(fc.features[0].number("value"), Some(1.5));
        assert_eq!(
            fc.features[1].geometry,
            Some(Geometry::LineString(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]]))
        );
        match &fc.features[2].geometry {
            Some(Geometry::Polygon(rings)) => assert_eq!(rings.len(), 2),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(fc.features[3].geometry, Some(Geometry::MultiPolygon(_))));
        assert_eq!(fc.features[4].geometry, None);
    }

    #[test]
    fn bare_geometry_becomes_one_feature() {
        let fc = FeatureCollection::parse(r#"{"type":"Point","coordinates":[3,4]}"#).unwrap();
        assert_eq!(fc.len(), 1);
    }
}

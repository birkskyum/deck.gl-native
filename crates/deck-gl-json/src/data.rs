//! Loading `data`, `image` and `iconAtlas` props from inline JSON, local files or URLs.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use deck_gl::FeatureCollection;
use deck_gl_layers::BitmapImage;
use serde_json::Value;

use crate::props::Rows;
use crate::{ConvertOptions, JsonError, Result};

fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

/// Resolve a path against the base directory of the spec.
pub fn resolve_path(source: &str, options: &ConvertOptions) -> PathBuf {
    let path = Path::new(source.strip_prefix("file://").unwrap_or(source));
    match (&options.base_dir, path.is_relative()) {
        (Some(base), true) => base.join(path),
        _ => path.to_path_buf(),
    }
}

/// Read a URL or file path to bytes.
pub fn load_bytes(source: &str, options: &ConvertOptions) -> Result<Vec<u8>> {
    if is_url(source) {
        return fetch(source);
    }
    let path = resolve_path(source, options);
    std::fs::read(&path).map_err(|e| JsonError::Load {
        url: path.display().to_string(),
        message: e.to_string(),
    })
}

/// Read a URL or file path to text.
pub fn load_text(source: &str, options: &ConvertOptions) -> Result<String> {
    let bytes = load_bytes(source, options)?;
    String::from_utf8(bytes).map_err(|e| JsonError::Load {
        url: source.to_string(),
        message: format!("not valid UTF-8: {e}"),
    })
}

#[cfg(feature = "fetch")]
fn fetch(url: &str) -> Result<Vec<u8>> {
    let load_error = |message: String| JsonError::Load {
        url: url.to_string(),
        message,
    };
    let mut response = ureq::get(url).call().map_err(|e| load_error(e.to_string()))?;
    response
        .body_mut()
        .with_config()
        .limit(u64::MAX)
        .read_to_vec()
        .map_err(|e| load_error(e.to_string()))
}

#[cfg(not(feature = "fetch"))]
fn fetch(url: &str) -> Result<Vec<u8>> {
    Err(JsonError::Load {
        url: url.to_string(),
        message: "deck-gl-json was built without the `fetch` feature, so URLs cannot be loaded".into(),
    })
}

/// A JSON prop that is either inline or a string pointing at a JSON document. CSV, TSV and
/// newline delimited JSON files (by extension) load as arrays of row objects.
pub fn load_json<'a>(value: &'a Value, options: &ConvertOptions) -> Result<Cow<'a, Value>> {
    match value {
        Value::String(source) => {
            let text = load_text(source, options)?;
            if let Some(format) = crate::tabular::TabularFormat::from_source(source) {
                let rows = crate::tabular::parse(&text, format, source)?;
                return Ok(Cow::Owned(Value::Array(rows)));
            }
            let parsed = serde_json::from_str(&text).map_err(|e| JsonError::Load {
                url: source.clone(),
                message: format!("invalid JSON: {e}"),
            })?;
            Ok(Cow::Owned(parsed))
        }
        other => Ok(Cow::Borrowed(other)),
    }
}

/// The rows of a `data` prop: an array of objects, or the features of GeoJSON.
pub fn rows_from_value(value: Cow<'_, Value>) -> Result<Rows> {
    let is_geojson_object = |map: &serde_json::Map<String, Value>| {
        matches!(
            map.get("type").and_then(Value::as_str),
            Some("FeatureCollection" | "Feature")
        )
    };
    match value {
        Cow::Owned(Value::Array(items)) => Ok(Arc::new(items)),
        Cow::Borrowed(Value::Array(items)) => Ok(Arc::new(items.clone())),
        Cow::Owned(Value::Object(mut map)) if is_geojson_object(&map) => match map.remove("features") {
            Some(Value::Array(features)) => Ok(Arc::new(features)),
            _ => Ok(Arc::new(vec![Value::Object(map)])),
        },
        Cow::Borrowed(Value::Object(map)) if is_geojson_object(map) => match map.get("features") {
            Some(Value::Array(features)) => Ok(Arc::new(features.clone())),
            _ => Ok(Arc::new(vec![Value::Object(map.clone())])),
        },
        other => Err(JsonError::Parse(format!(
            "`data` must be an array of rows or GeoJSON, got {}",
            crate::props::describe(&other)
        ))),
    }
}

/// GeoJSON `data`: the parsed collection plus the raw features as accessor rows.
pub fn geojson_from_value(value: Cow<'_, Value>) -> Result<(Arc<FeatureCollection>, Rows)> {
    let value = match value {
        Cow::Owned(Value::Array(items)) => Value::Object(
            [
                ("type".to_string(), Value::String("FeatureCollection".into())),
                ("features".to_string(), Value::Array(items)),
            ]
            .into_iter()
            .collect(),
        ),
        Cow::Borrowed(Value::Array(items)) => Value::Object(
            [
                ("type".to_string(), Value::String("FeatureCollection".into())),
                ("features".to_string(), Value::Array(items.clone())),
            ]
            .into_iter()
            .collect(),
        ),
        other => other.into_owned(),
    };
    let collection = FeatureCollection::from_value(&value)?;
    let rows = rows_from_value(Cow::Owned(value))?;
    Ok((Arc::new(collection), rows))
}

/// Decode a PNG or JPEG from a URL or file path.
pub fn load_image(source: &str, options: &ConvertOptions) -> Result<BitmapImage> {
    let bytes = load_bytes(source, options)?;
    let image = image::load_from_memory(&bytes)
        .map_err(|e| JsonError::Load {
            url: source.to_string(),
            message: format!("could not decode image: {e}"),
        })?
        .to_rgba8();
    let (width, height) = image.dimensions();
    Ok(BitmapImage::new(width, height, image.into_raw()))
}

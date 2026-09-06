//! Conversion of one JSON layer object into a layer.

use std::sync::Arc;

use deck_gl::{FeatureCollection, Layer, LayerData};
use deck_gl_layers::{
    ArcLayer, ArcLayerProps, BitmapLayer, BitmapLayerProps, ColumnLayer, ColumnLayerProps, GeoJsonLayer,
    GeoJsonLayerProps, IconAtlas, IconLayer, IconLayerProps, LineLayer, LineLayerProps, PathLayer,
    PathLayerProps, PointCloudLayer, PointCloudLayerProps, PolygonLayer, PolygonLayerProps, ScatterplotLayer,
    ScatterplotLayerProps, SolidPolygonLayer, SolidPolygonLayerProps,
};
use serde_json::Value;

use crate::data;
use crate::props::{convert, Props, TYPE_KEY};
use crate::{ConvertOptions, JsonConverter, JsonError, Result};

/// Convert a layer object. Unknown layer types are skipped with a warning, like deck.gl does.
pub fn convert_layer(
    converter: &JsonConverter,
    value: &Value,
    warnings: &mut Vec<String>,
) -> Result<Option<Box<dyn Layer>>> {
    let object = value
        .as_object()
        .ok_or_else(|| JsonError::Parse(format!("each layer must be an object with `{TYPE_KEY}`")))?;
    let layer_type = object
        .get(TYPE_KEY)
        .and_then(Value::as_str)
        .ok_or_else(|| JsonError::Parse(format!("layer object is missing `{TYPE_KEY}`")))?;
    let mut props = Props::new(layer_type, object);
    let options = &converter.options;
    let layer: Box<dyn Layer> = match layer_type {
        "ScatterplotLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(ScatterplotLayer::new(scatterplot(&props, data)?))
        }
        "LineLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(LineLayer::new(line(&props, data)?))
        }
        "ArcLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(ArcLayer::new(arc(&props, data)?))
        }
        "PathLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(PathLayer::new(path(&props, data)?))
        }
        "SolidPolygonLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(SolidPolygonLayer::new(solid_polygon(&props, data)?))
        }
        "PolygonLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(PolygonLayer::new(polygon(&props, data)?))
        }
        "ColumnLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(ColumnLayer::new(column(&props, data)?))
        }
        "PointCloudLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(PointCloudLayer::new(point_cloud(&props, data)?))
        }
        "IconLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(IconLayer::new(icon(&props, data, options)?))
        }
        "GeoJsonLayer" => {
            let collection = load_geojson(&mut props, options)?;
            Box::new(GeoJsonLayer::new(geojson(&props, collection)?))
        }
        "BitmapLayer" => Box::new(BitmapLayer::new(bitmap(&props, options)?)),
        _ => {
            warnings.push(format!(
                "layer `{}`: layer type `{layer_type}` is not available yet and was skipped",
                props.id
            ));
            return Ok(None);
        }
    };
    props.finish(warnings);
    Ok(Some(layer))
}

/// Attach the layer id to load and parse errors.
fn in_layer(props: &Props, error: JsonError) -> JsonError {
    match error {
        JsonError::Load { .. } | JsonError::Parse(_) => JsonError::Layer {
            layer: props.id.clone(),
            message: error.to_string(),
        },
        other => other,
    }
}

fn load_rows(props: &mut Props, options: &ConvertOptions) -> Result<LayerData> {
    let rows = match props.get("data") {
        None | Some(Value::Null) => Arc::new(Vec::new()),
        Some(value) => data::load_json(value, options)
            .and_then(data::rows_from_value)
            .map_err(|e| in_layer(props, e))?,
    };
    let length = rows.len();
    props.set_rows(rows);
    Ok(LayerData::with_length(length))
}

fn load_geojson(props: &mut Props, options: &ConvertOptions) -> Result<Arc<FeatureCollection>> {
    let (collection, rows) = match props.get("data") {
        None | Some(Value::Null) => (Arc::new(FeatureCollection::default()), Arc::new(Vec::new())),
        Some(value) => data::load_json(value, options)
            .and_then(data::geojson_from_value)
            .map_err(|e| in_layer(props, e))?,
    };
    props.set_rows(rows);
    Ok(collection)
}

fn scatterplot(p: &Props, data: LayerData) -> Result<ScatterplotLayerProps> {
    let d = ScatterplotLayerProps::default();
    Ok(ScatterplotLayerProps {
        base: p.base()?,
        data,
        radius_units: p.unit("radiusUnits", d.radius_units)?,
        radius_scale: p.f32("radiusScale", d.radius_scale)?,
        radius_min_pixels: p.f32("radiusMinPixels", d.radius_min_pixels)?,
        radius_max_pixels: p.f32("radiusMaxPixels", d.radius_max_pixels)?,
        line_width_units: p.unit("lineWidthUnits", d.line_width_units)?,
        line_width_scale: p.f32("lineWidthScale", d.line_width_scale)?,
        line_width_min_pixels: p.f32("lineWidthMinPixels", d.line_width_min_pixels)?,
        line_width_max_pixels: p.f32("lineWidthMaxPixels", d.line_width_max_pixels)?,
        stroked: p.bool("stroked", d.stroked)?,
        filled: p.bool("filled", d.filled)?,
        billboard: p.bool("billboard", d.billboard)?,
        antialiasing: p.bool("antialiasing", d.antialiasing)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_radius: p.accessor("getRadius", &d.get_radius, convert::f32)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
        get_line_width: p.accessor("getLineWidth", &d.get_line_width, convert::f32)?,
        get_pixel_offset: p.accessor("getPixelOffset", &d.get_pixel_offset, convert::vec2)?,
    })
}

fn line(p: &Props, data: LayerData) -> Result<LineLayerProps> {
    let d = LineLayerProps::default();
    Ok(LineLayerProps {
        base: p.base()?,
        data,
        width_units: p.unit("widthUnits", d.width_units)?,
        width_scale: p.f32("widthScale", d.width_scale)?,
        width_min_pixels: p.f32("widthMinPixels", d.width_min_pixels)?,
        width_max_pixels: p.f32("widthMaxPixels", d.width_max_pixels)?,
        get_source_position: p.accessor("getSourcePosition", "sourcePosition", convert::position)?,
        get_target_position: p.accessor("getTargetPosition", "targetPosition", convert::position)?,
        get_color: p.accessor("getColor", &d.get_color, convert::color)?,
        get_width: p.accessor("getWidth", &d.get_width, convert::f32)?,
    })
}

fn arc(p: &Props, data: LayerData) -> Result<ArcLayerProps> {
    let d = ArcLayerProps::default();
    Ok(ArcLayerProps {
        base: p.base()?,
        data,
        great_circle: p.bool("greatCircle", d.great_circle)?,
        num_segments: p.u32("numSegments", d.num_segments)?,
        width_units: p.unit("widthUnits", d.width_units)?,
        width_scale: p.f32("widthScale", d.width_scale)?,
        width_min_pixels: p.f32("widthMinPixels", d.width_min_pixels)?,
        width_max_pixels: p.f32("widthMaxPixels", d.width_max_pixels)?,
        get_source_position: p.accessor("getSourcePosition", "sourcePosition", convert::position)?,
        get_target_position: p.accessor("getTargetPosition", "targetPosition", convert::position)?,
        get_source_color: p.accessor("getSourceColor", &d.get_source_color, convert::color)?,
        get_target_color: p.accessor("getTargetColor", &d.get_target_color, convert::color)?,
        get_width: p.accessor("getWidth", &d.get_width, convert::f32)?,
        get_height: p.accessor("getHeight", &d.get_height, convert::f32)?,
        get_tilt: p.accessor("getTilt", &d.get_tilt, convert::f32)?,
    })
}

fn path(p: &Props, data: LayerData) -> Result<PathLayerProps> {
    let d = PathLayerProps::default();
    Ok(PathLayerProps {
        base: p.base()?,
        data,
        width_units: p.unit("widthUnits", d.width_units)?,
        width_scale: p.f32("widthScale", d.width_scale)?,
        width_min_pixels: p.f32("widthMinPixels", d.width_min_pixels)?,
        width_max_pixels: p.f32("widthMaxPixels", d.width_max_pixels)?,
        joint_rounded: p.bool("jointRounded", d.joint_rounded)?,
        cap_rounded: p.bool("capRounded", d.cap_rounded)?,
        miter_limit: p.f32("miterLimit", d.miter_limit)?,
        billboard: p.bool("billboard", d.billboard)?,
        get_path: p.accessor("getPath", "path", convert::path)?,
        get_color: p.accessor("getColor", &d.get_color, convert::color)?,
        get_width: p.accessor("getWidth", &d.get_width, convert::f32)?,
    })
}

fn solid_polygon(p: &Props, data: LayerData) -> Result<SolidPolygonLayerProps> {
    let d = SolidPolygonLayerProps::default();
    Ok(SolidPolygonLayerProps {
        base: p.base()?,
        data,
        filled: p.bool("filled", d.filled)?,
        extruded: p.bool("extruded", d.extruded)?,
        wireframe: p.bool("wireframe", d.wireframe)?,
        elevation_scale: p.f32("elevationScale", d.elevation_scale)?,
        get_polygon: p.accessor("getPolygon", "polygon", convert::polygon)?,
        get_elevation: p.accessor("getElevation", &d.get_elevation, convert::f32)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
    })
}

fn polygon(p: &Props, data: LayerData) -> Result<PolygonLayerProps> {
    let d = PolygonLayerProps::default();
    Ok(PolygonLayerProps {
        base: p.base()?,
        data,
        stroked: p.bool("stroked", d.stroked)?,
        filled: p.bool("filled", d.filled)?,
        extruded: p.bool("extruded", d.extruded)?,
        wireframe: p.bool("wireframe", d.wireframe)?,
        elevation_scale: p.f32("elevationScale", d.elevation_scale)?,
        line_width_units: p.unit("lineWidthUnits", d.line_width_units)?,
        line_width_scale: p.f32("lineWidthScale", d.line_width_scale)?,
        line_width_min_pixels: p.f32("lineWidthMinPixels", d.line_width_min_pixels)?,
        line_width_max_pixels: p.f32("lineWidthMaxPixels", d.line_width_max_pixels)?,
        line_joint_rounded: p.bool("lineJointRounded", d.line_joint_rounded)?,
        line_miter_limit: p.f32("lineMiterLimit", d.line_miter_limit)?,
        get_polygon: p.accessor("getPolygon", "polygon", convert::polygon)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
        get_line_width: p.accessor("getLineWidth", &d.get_line_width, convert::f32)?,
        get_elevation: p.accessor("getElevation", &d.get_elevation, convert::f32)?,
    })
}

fn column(p: &Props, data: LayerData) -> Result<ColumnLayerProps> {
    let d = ColumnLayerProps::default();
    Ok(ColumnLayerProps {
        base: p.base()?,
        data,
        disk_resolution: p.u32("diskResolution", d.disk_resolution)?,
        radius: p.f32("radius", d.radius)?,
        angle: p.f32("angle", d.angle)?,
        offset: p.vec2("offset", d.offset)?,
        coverage: p.f32("coverage", d.coverage)?,
        elevation_scale: p.f32("elevationScale", d.elevation_scale)?,
        radius_units: p.unit("radiusUnits", d.radius_units)?,
        line_width_units: p.unit("lineWidthUnits", d.line_width_units)?,
        line_width_scale: p.f32("lineWidthScale", d.line_width_scale)?,
        line_width_min_pixels: p.f32("lineWidthMinPixels", d.line_width_min_pixels)?,
        line_width_max_pixels: p.f32("lineWidthMaxPixels", d.line_width_max_pixels)?,
        extruded: p.bool("extruded", d.extruded)?,
        wireframe: p.bool("wireframe", d.wireframe)?,
        filled: p.bool("filled", d.filled)?,
        stroked: p.bool("stroked", d.stroked)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
        get_line_width: p.accessor("getLineWidth", &d.get_line_width, convert::f32)?,
        get_elevation: p.accessor("getElevation", &d.get_elevation, convert::f32)?,
    })
}

fn point_cloud(p: &Props, data: LayerData) -> Result<PointCloudLayerProps> {
    let d = PointCloudLayerProps::default();
    Ok(PointCloudLayerProps {
        base: p.base()?,
        data,
        size_units: p.unit("sizeUnits", d.size_units)?,
        point_size: p.f32("pointSize", d.point_size)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_normal: p.accessor("getNormal", &d.get_normal, convert::vec3)?,
        get_color: p.accessor("getColor", &d.get_color, convert::color)?,
    })
}

fn icon(p: &Props, data: LayerData, options: &ConvertOptions) -> Result<IconLayerProps> {
    let d = IconLayerProps::default();
    let atlas = match (p.string("iconAtlas")?, p.get("iconMapping")) {
        (Some(source), Some(mapping)) => {
            let image = data::load_image(&source, options).map_err(|e| in_layer(p, e))?;
            let mapping = data::load_json(mapping, options).map_err(|e| in_layer(p, e))?;
            let mapping = IconAtlas::mapping_from_json(&mapping.to_string())
                .map_err(|e| p.error("iconMapping", e.to_string()))?;
            Some(Arc::new(IconAtlas { image, mapping }))
        }
        (None, None) => None,
        _ => return Err(p.error("iconAtlas", "iconAtlas and iconMapping must be given together")),
    };
    Ok(IconLayerProps {
        base: p.base()?,
        data,
        atlas,
        size_units: p.unit("sizeUnits", d.size_units)?,
        size_scale: p.f32("sizeScale", d.size_scale)?,
        size_min_pixels: p.f32("sizeMinPixels", d.size_min_pixels)?,
        size_max_pixels: p.f32("sizeMaxPixels", d.size_max_pixels)?,
        size_by_height: d.size_by_height,
        billboard: p.bool("billboard", d.billboard)?,
        alpha_cutoff: p.f32("alphaCutoff", d.alpha_cutoff)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_icon: p.accessor("getIcon", "icon", convert::string)?,
        get_color: p.accessor("getColor", &d.get_color, convert::color)?,
        get_size: p.accessor("getSize", &d.get_size, convert::f32)?,
        get_angle: p.accessor("getAngle", &d.get_angle, convert::f32)?,
        get_pixel_offset: p.accessor("getPixelOffset", &d.get_pixel_offset, convert::vec2)?,
    })
}

fn geojson(p: &Props, collection: Arc<FeatureCollection>) -> Result<GeoJsonLayerProps> {
    let d = GeoJsonLayerProps::default();
    Ok(GeoJsonLayerProps {
        base: p.base()?,
        data: collection,
        filled: p.bool("filled", d.filled)?,
        stroked: p.bool("stroked", d.stroked)?,
        extruded: p.bool("extruded", d.extruded)?,
        wireframe: p.bool("wireframe", d.wireframe)?,
        elevation_scale: p.f32("elevationScale", d.elevation_scale)?,
        line_width_units: p.unit("lineWidthUnits", d.line_width_units)?,
        line_width_scale: p.f32("lineWidthScale", d.line_width_scale)?,
        line_width_min_pixels: p.f32("lineWidthMinPixels", d.line_width_min_pixels)?,
        line_width_max_pixels: p.f32("lineWidthMaxPixels", d.line_width_max_pixels)?,
        line_joint_rounded: p.bool("lineJointRounded", d.line_joint_rounded)?,
        line_cap_rounded: p.bool("lineCapRounded", d.line_cap_rounded)?,
        line_miter_limit: p.f32("lineMiterLimit", d.line_miter_limit)?,
        point_radius_units: p.unit("pointRadiusUnits", d.point_radius_units)?,
        point_radius_scale: p.f32("pointRadiusScale", d.point_radius_scale)?,
        point_radius_min_pixels: p.f32("pointRadiusMinPixels", d.point_radius_min_pixels)?,
        point_radius_max_pixels: p.f32("pointRadiusMaxPixels", d.point_radius_max_pixels)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
        get_line_width: p.accessor("getLineWidth", &d.get_line_width, convert::f32)?,
        get_point_radius: p.accessor("getPointRadius", &d.get_point_radius, convert::f32)?,
        get_elevation: p.accessor("getElevation", &d.get_elevation, convert::f32)?,
    })
}

fn bitmap(p: &Props, options: &ConvertOptions) -> Result<BitmapLayerProps> {
    let d = BitmapLayerProps::default();
    let image = match p.string("image")? {
        Some(source) => Some(data::load_image(&source, options).map_err(|e| in_layer(p, e))?),
        None => None,
    };
    let bounds = match p.get("bounds") {
        None | Some(Value::Null) => d.bounds,
        Some(value) => bounds(value).map_err(|m| p.error("bounds", m))?,
    };
    let tint_color = match p.get("tintColor") {
        None | Some(Value::Null) => d.tint_color,
        Some(value) => convert::rgb(value).map_err(|m| p.error("tintColor", m))?,
    };
    Ok(BitmapLayerProps {
        base: p.base()?,
        image,
        bounds,
        desaturate: p.f32("desaturate", d.desaturate)?,
        transparent_color: p.color("transparentColor", d.transparent_color)?,
        tint_color,
    })
}

/// `[left, bottom, right, top]`, or four corners which must be axis aligned.
fn bounds(value: &Value) -> std::result::Result<[f64; 4], String> {
    let items = value
        .as_array()
        .ok_or_else(|| "expected [left, bottom, right, top] or four corners".to_string())?;
    if items.iter().all(Value::is_number) {
        let v = convert::numbers(value, 4, 4)?;
        return Ok([v[0], v[1], v[2], v[3]]);
    }
    if items.len() != 4 {
        return Err(format!("expected four corners, got {}", items.len()));
    }
    let mut bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for corner in items {
        let [x, y, _] = convert::position(corner)?;
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x);
        bounds[3] = bounds[3].max(y);
    }
    Ok(bounds)
}

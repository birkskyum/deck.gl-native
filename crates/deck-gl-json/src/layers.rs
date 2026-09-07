//! Conversion of one JSON layer object into a layer.

use std::sync::Arc;

use arrow_array::RecordBatch;
use deck_gl::{FeatureCollection, Layer, LayerData, Polygon};
use deck_gl_layers::{
    AggregationOperation, AggregationProps, AlignmentBaseline, ArcLayer, ArcLayerProps, BitmapLayer,
    BitmapLayerProps, CellKind, CharacterSet, ColumnLayer, ColumnLayerProps, Contour, ContourLayer,
    ContourLayerProps, ContourThreshold, FontSettings, FontSource, GeoCellLayer, GeoCellLayerProps,
    GeoJsonLayer, GeoJsonLayerProps, GridCellLayerProps, GridLayer, GridLayerProps, HeatmapAggregation,
    HeatmapLayer, HeatmapLayerProps, HexagonLayer, HexagonLayerProps, IconAtlas, IconLayer, IconLayerProps,
    LineLayer, LineLayerProps, PathLayer, PathLayerProps, PointCloudLayer, PointCloudLayerProps,
    PolygonLayer, PolygonLayerProps, RefinementStrategy, ScaleType, ScatterplotLayer, ScatterplotLayerProps,
    ScreenGridLayer, ScreenGridLayerProps, SolidPolygonLayer, SolidPolygonLayerProps, TextAnchor, TextLayer,
    TextLayerProps, TileLayer, TileLayerProps, TripsLayer, TripsLayerProps, WordBreak,
};
use serde_json::Value;

use crate::data;
use crate::props::{convert, Props, TABLE_IDENTIFIER, TYPE_KEY};
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
        "TripsLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = TripsLayerProps::default();
            Box::new(TripsLayer::new(TripsLayerProps {
                path: path(&props, data)?,
                fade_trail: props.bool("fadeTrail", d.fade_trail)?,
                trail_length: props.f32("trailLength", d.trail_length)?,
                current_time: props.f32("currentTime", d.current_time)?,
                get_timestamps: props.accessor("getTimestamps", "timestamps", convert::f32_list)?,
            }))
        }
        "GreatCircleLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(ArcLayer::great_circle(arc(&props, data)?))
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
        "HexagonLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = HexagonLayerProps::default();
            Box::new(HexagonLayer::new(HexagonLayerProps {
                base: props.base()?,
                data,
                radius: props.f64("radius", d.radius)?,
                aggregation: aggregation(&props)?,
            }))
        }
        "GridLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = GridLayerProps::default();
            Box::new(GridLayer::new(GridLayerProps {
                base: props.base()?,
                data,
                cell_size: props.f64("cellSize", d.cell_size)?,
                aggregation: aggregation(&props)?,
            }))
        }
        "H3HexagonLayer" | "S2Layer" | "GeohashLayer" | "QuadkeyLayer" => {
            let data = load_rows(&mut props, options)?;
            props.get("highPrecision");
            props.get("centerHexagon");
            let (mut defaults, accessor, field) = match props.layer_type.as_str() {
                "H3HexagonLayer" => (GeoCellLayerProps::h3(), "getHexagon", "hexagon"),
                "S2Layer" => (GeoCellLayerProps::s2(), "getS2Token", "token"),
                "GeohashLayer" => (GeoCellLayerProps::geohash(), "getGeohash", "geohash"),
                _ => (GeoCellLayerProps::quadkey(), "getQuadkey", "quadkey"),
            };
            let coverage = props.f64("coverage", 1.0)?;
            defaults.kind = match defaults.kind {
                CellKind::H3 { .. } => CellKind::H3 { coverage },
                CellKind::Quadkey { .. } => CellKind::Quadkey { coverage },
                other => other,
            };
            let mut polygon = polygon_with(&props, data, Accessor::Constant(Vec::new()))?;
            polygon.extruded = props.bool("extruded", defaults.polygon.extruded)?;
            Box::new(GeoCellLayer::new(GeoCellLayerProps {
                polygon,
                get_cell: props.accessor(accessor, field, convert::string)?,
                kind: defaults.kind,
            }))
        }
        "ContourLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = ContourLayerProps::default();
            props.get("gpuAggregation");
            let aggregation = match props.get("aggregation") {
                None | Some(Value::Null) => d.aggregation,
                Some(Value::String(name)) => AggregationOperation::parse(name)
                    .filter(|op| !matches!(op, AggregationOperation::Count))
                    .ok_or_else(|| {
                        props.error(
                            "aggregation",
                            format!("expected SUM, MEAN, MIN or MAX, got `{name}`"),
                        )
                    })?,
                Some(other) => {
                    return Err(props.error(
                        "aggregation",
                        format!("expected a string, got {}", crate::props::describe(other)),
                    ))
                }
            };
            let contours = match props.get("contours") {
                None | Some(Value::Null) => d.contours,
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|item| contour_from_value(item).map_err(|m| props.error("contours", m)))
                    .collect::<Result<Vec<_>>>()?,
                Some(other) => {
                    return Err(props.error(
                        "contours",
                        format!("expected an array, got {}", crate::props::describe(other)),
                    ))
                }
            };
            let grid_origin = match props.get("gridOrigin") {
                None | Some(Value::Null) => d.grid_origin,
                Some(v) => {
                    let n = convert::numbers(v, 2, 2).map_err(|m| props.error("gridOrigin", m))?;
                    [n[0], n[1]]
                }
            };
            Box::new(ContourLayer::new(ContourLayerProps {
                base: props.base()?,
                data,
                cell_size: props.f32("cellSize", d.cell_size as f32)? as f64,
                grid_origin,
                aggregation,
                contours,
                z_offset: props.f32("zOffset", d.z_offset as f32)? as f64,
                get_position: props.accessor("getPosition", "position", convert::position)?,
                get_weight: props.accessor("getWeight", &d.get_weight, convert::f32)?,
            }))
        }
        "TileLayer" => {
            let d = TileLayerProps::default();
            let templates: Vec<String> = match props.get("data") {
                Some(Value::String(url)) => vec![url.clone()],
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or_else(|| props.error("data", "expected URL templates"))
                    })
                    .collect::<Result<Vec<_>>>()?,
                Some(other) => {
                    return Err(props.error(
                        "data",
                        format!(
                            "expected a tile URL template, got {}",
                            crate::props::describe(other)
                        ),
                    ))
                }
                None => Vec::new(),
            };
            if !templates.iter().all(|t| deck_gl_layers::is_url_template(t)) {
                return Err(props.error("data", "expected URL templates with {z}, {x} and {y} (or {-y})"));
            }
            props.get("renderSubLayers");
            props.get("onViewportLoad");
            let refinement_strategy = match props.get("refinementStrategy") {
                None | Some(Value::Null) => d.refinement_strategy,
                Some(Value::String(name)) => RefinementStrategy::parse(name)
                    .ok_or_else(|| props.error("refinementStrategy", format!("unknown strategy `{name}`")))?,
                Some(other) => {
                    return Err(props.error(
                        "refinementStrategy",
                        format!("expected a string, got {}", crate::props::describe(other)),
                    ))
                }
            };
            let optional_zoom = |key: &str| -> Result<Option<u32>> {
                match props.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(v) => convert::f32(v)
                        .map(|z| Some(z as u32))
                        .map_err(|m| props.error(key, m)),
                }
            };
            let extent = match props.get("extent") {
                None | Some(Value::Null) => None,
                Some(v) => {
                    let n = convert::numbers(v, 4, 4).map_err(|m| props.error("extent", m))?;
                    Some([n[0], n[1], n[2], n[3]])
                }
            };
            let get_tile_data = if templates.is_empty() {
                None
            } else {
                Some(tile_loader(templates))
            };
            Box::new(TileLayer::new(TileLayerProps {
                base: props.base()?,
                get_tile_data,
                render_sub_layers: deck_gl_layers::raster_renderer(),
                tile_size: props.f32("tileSize", d.tile_size as f32)? as f64,
                min_zoom: optional_zoom("minZoom")?.or(d.min_zoom),
                max_zoom: optional_zoom("maxZoom")?,
                zoom_offset: props.f32("zoomOffset", d.zoom_offset as f32)? as f64,
                extent,
                refinement_strategy,
                max_cache_size: props
                    .f32("maxCacheSize", 0.0)
                    .map(|n| (n > 0.0).then_some(n as usize))?,
                max_requests: props.f32("maxRequests", d.max_requests as f32)? as usize,
            }))
        }
        "HeatmapLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = HeatmapLayerProps::default();
            props.get("debounceTimeout");
            let color_range = match props.get("colorRange") {
                None | Some(Value::Null) => d.color_range,
                Some(Value::Array(items)) => items
                    .iter()
                    .map(convert::color)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|m| props.error("colorRange", m))?,
                Some(other) => {
                    return Err(props.error(
                        "colorRange",
                        format!(
                            "expected an array of colors, got {}",
                            crate::props::describe(other)
                        ),
                    ))
                }
            };
            let aggregation = match props.get("aggregation") {
                None | Some(Value::Null) => d.aggregation,
                Some(Value::String(name)) => HeatmapAggregation::parse(name).ok_or_else(|| {
                    props.error("aggregation", format!("expected SUM or MEAN, got `{name}`"))
                })?,
                Some(other) => {
                    return Err(props.error(
                        "aggregation",
                        format!("expected a string, got {}", crate::props::describe(other)),
                    ))
                }
            };
            Box::new(HeatmapLayer::new(HeatmapLayerProps {
                base: props.base()?,
                data,
                radius_pixels: props.f32("radiusPixels", d.radius_pixels)?,
                intensity: props.f32("intensity", d.intensity)?,
                threshold: props.f32("threshold", d.threshold)?,
                color_domain: domain(&props, "colorDomain")?,
                color_range,
                aggregation,
                weights_texture_size: props.f32("weightsTextureSize", d.weights_texture_size as f32)? as u32,
                get_position: props.accessor("getPosition", "position", convert::position)?,
                get_weight: props.accessor("getWeight", &d.get_weight, convert::f32)?,
            }))
        }
        "ScreenGridLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = ScreenGridLayerProps::default();
            props.get("gpuAggregation");
            let color_range = match props.get("colorRange") {
                None | Some(Value::Null) => d.color_range,
                Some(Value::Array(items)) => items
                    .iter()
                    .map(convert::color)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|m| props.error("colorRange", m))?,
                Some(other) => {
                    return Err(props.error(
                        "colorRange",
                        format!(
                            "expected an array of colors, got {}",
                            crate::props::describe(other)
                        ),
                    ))
                }
            };
            Box::new(ScreenGridLayer::new(ScreenGridLayerProps {
                base: props.base()?,
                data,
                cell_size_pixels: props.f32("cellSizePixels", d.cell_size_pixels)?,
                cell_margin_pixels: props.f32("cellMarginPixels", d.cell_margin_pixels)?,
                color_domain: domain(&props, "colorDomain")?,
                color_range,
                color_scale_type: scale_type(&props, "colorScaleType", d.color_scale_type)?,
                aggregation: operation(&props, "aggregation", d.aggregation)?,
                get_position: props.accessor("getPosition", "position", convert::position)?,
                get_weight: props.accessor("getWeight", &d.get_weight, convert::f32)?,
            }))
        }
        "GridCellLayer" => {
            let data = load_rows(&mut props, options)?;
            let d = GridCellLayerProps::default();
            Box::new(ColumnLayer::grid_cells(GridCellLayerProps {
                base: props.base()?,
                data,
                cell_size: props.f32("cellSize", d.cell_size)?,
                coverage: props.f32("coverage", d.coverage)?,
                elevation_scale: props.f32("elevationScale", d.elevation_scale)?,
                extruded: props.bool("extruded", d.extruded)?,
                get_position: props.accessor("getPosition", "position", convert::position)?,
                get_fill_color: props.accessor("getFillColor", &d.get_fill_color, convert::color)?,
                get_elevation: props.accessor("getElevation", &d.get_elevation, convert::f32)?,
            }))
        }
        "TextLayer" => {
            let data = load_rows(&mut props, options)?;
            Box::new(TextLayer::new(text(&props, data, options)?))
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

/// The Arrow table a `data` string refers to, if it does.
fn table_for<'a>(
    props: &Props,
    value: &Value,
    options: &'a ConvertOptions,
) -> Result<Option<&'a RecordBatch>> {
    let Some(name) = value.as_str().and_then(|s| s.strip_prefix(TABLE_IDENTIFIER)) else {
        return Ok(None);
    };
    options
        .tables
        .get(name)
        .map(Some)
        .ok_or_else(|| props.error("data", format!("no Arrow table named `{name}` was registered")))
}

fn load_rows(props: &mut Props, options: &ConvertOptions) -> Result<LayerData> {
    let value = match props.get("data") {
        None | Some(Value::Null) => {
            props.set_rows(Arc::new(Vec::new()));
            return Ok(LayerData::with_length(0));
        }
        Some(value) => value,
    };
    if let Some(batch) = table_for(props, value, options)? {
        props.set_table();
        return Ok(LayerData::from_batch(batch.clone()));
    }
    let rows = data::load_json(value, options)
        .and_then(data::rows_from_value)
        .map_err(|e| in_layer(props, e))?;
    let length = rows.len();
    props.set_rows(rows);
    Ok(LayerData::with_length(length))
}

fn load_geojson(props: &mut Props, options: &ConvertOptions) -> Result<Arc<FeatureCollection>> {
    let (collection, rows) = match props.get("data") {
        None | Some(Value::Null) => (Arc::new(FeatureCollection::default()), Arc::new(Vec::new())),
        Some(value) if table_for(props, value, options)?.is_some() => {
            return Err(props.error("data", "GeoJsonLayer takes GeoJSON, not an Arrow table"));
        }
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
        get_polygon,
        get_elevation: p.accessor("getElevation", &d.get_elevation, convert::f32)?,
        get_fill_color: p.accessor("getFillColor", &d.get_fill_color, convert::color)?,
        get_line_color: p.accessor("getLineColor", &d.get_line_color, convert::color)?,
    })
}

fn polygon(p: &Props, data: LayerData) -> Result<PolygonLayerProps> {
    let get_polygon = p.accessor("getPolygon", "polygon", convert::polygon)?;
    polygon_with(p, data, get_polygon)
}

/// PolygonLayer props with a geometry accessor supplied by the caller (cell layers).
fn polygon_with(p: &Props, data: LayerData, get_polygon: Accessor<Polygon>) -> Result<PolygonLayerProps> {
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

fn scale_type(p: &Props, key: &str, default: ScaleType) -> Result<ScaleType> {
    match p.string(key)? {
        None => Ok(default),
        Some(name) => ScaleType::parse(&name).ok_or_else(|| {
            p.error(
                key,
                format!("expected quantize, linear, quantile or ordinal, got `{name}`"),
            )
        }),
    }
}

fn operation(p: &Props, key: &str, default: AggregationOperation) -> Result<AggregationOperation> {
    match p.string(key)? {
        None => Ok(default),
        Some(name) => AggregationOperation::parse(&name).ok_or_else(|| {
            p.error(
                key,
                format!("expected SUM, MEAN, MIN, MAX or COUNT, got `{name}`"),
            )
        }),
    }
}

fn domain(p: &Props, key: &str) -> Result<Option<[f32; 2]>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let d = convert::numbers(v, 2, 2).map_err(|m| p.error(key, m))?;
            Ok(Some([d[0] as f32, d[1] as f32]))
        }
    }
}

/// Props shared by HexagonLayer and GridLayer.
fn aggregation(p: &Props) -> Result<AggregationProps> {
    let d = AggregationProps::default();
    // accepted but not implemented: aggregation always runs on the CPU here
    p.get("gpuAggregation");
    let color_range = match p.get("colorRange") {
        None | Some(Value::Null) => d.color_range,
        Some(Value::Array(items)) => items
            .iter()
            .map(convert::color)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|m| p.error("colorRange", m))?,
        Some(other) => {
            return Err(p.error(
                "colorRange",
                format!(
                    "expected an array of colors, got {}",
                    crate::props::describe(other)
                ),
            ))
        }
    };
    let elevation_range = match p.get("elevationRange") {
        None | Some(Value::Null) => d.elevation_range,
        Some(v) => {
            let r = convert::numbers(v, 2, 2).map_err(|m| p.error("elevationRange", m))?;
            [r[0] as f32, r[1] as f32]
        }
    };
    Ok(AggregationProps {
        color_domain: domain(p, "colorDomain")?,
        color_range,
        color_scale_type: scale_type(p, "colorScaleType", d.color_scale_type)?,
        color_aggregation: operation(p, "colorAggregation", d.color_aggregation)?,
        lower_percentile: p.f32("lowerPercentile", d.lower_percentile)?,
        upper_percentile: p.f32("upperPercentile", d.upper_percentile)?,
        elevation_domain: domain(p, "elevationDomain")?,
        elevation_range,
        elevation_scale: p.f32("elevationScale", d.elevation_scale)?,
        elevation_scale_type: scale_type(p, "elevationScaleType", d.elevation_scale_type)?,
        elevation_aggregation: operation(p, "elevationAggregation", d.elevation_aggregation)?,
        elevation_lower_percentile: p.f32("elevationLowerPercentile", d.elevation_lower_percentile)?,
        elevation_upper_percentile: p.f32("elevationUpperPercentile", d.elevation_upper_percentile)?,
        extruded: p.bool("extruded", d.extruded)?,
        coverage: p.f32("coverage", d.coverage)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_color_weight: p.accessor("getColorWeight", &d.get_color_weight, convert::f32)?,
        get_elevation_weight: p.accessor("getElevationWeight", &d.get_elevation_weight, convert::f32)?,
    })
}

fn text_anchor(value: &Value) -> std::result::Result<TextAnchor, String> {
    let name = convert::string(value)?;
    TextAnchor::parse(&name).ok_or_else(|| format!("expected start, middle or end, got `{name}`"))
}

fn alignment_baseline(value: &Value) -> std::result::Result<AlignmentBaseline, String> {
    let name = convert::string(value)?;
    AlignmentBaseline::parse(&name).ok_or_else(|| format!("expected top, center or bottom, got `{name}`"))
}

/// `[x, y]` or `[left, top, right, bottom]`.
fn box_sides(value: &Value) -> std::result::Result<[f32; 4], String> {
    let v = convert::numbers(value, 2, 4)?;
    Ok(match v.len() {
        2 => [v[0] as f32, v[1] as f32, v[0] as f32, v[1] as f32],
        4 => [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32],
        n => return Err(format!("expected 2 or 4 numbers, got {n}")),
    })
}

fn text(p: &Props, data: LayerData, options: &ConvertOptions) -> Result<TextLayerProps> {
    let d = TextLayerProps::default();
    let mut font = FontSettings::default();
    if let Some(family) = p.string("fontFamily")? {
        // deck.gl takes a CSS family; here a font file (path or URL) can be named instead,
        // anything else uses the bundled font.
        let lower = family.to_ascii_lowercase();
        if lower.ends_with(".ttf") || lower.ends_with(".otf") {
            let bytes = data::load_bytes(&family, options).map_err(|e| in_layer(p, e))?;
            font.font = FontSource::Bytes(Arc::new(bytes));
        }
    }
    p.get("fontWeight");
    font.character_set = match p.get("characterSet") {
        None | Some(Value::Null) => font.character_set,
        Some(Value::String(s)) if s == "auto" => CharacterSet::Auto,
        Some(Value::String(s)) => CharacterSet::Chars(s.clone()),
        Some(Value::Array(items)) => CharacterSet::Chars(
            items
                .iter()
                .map(convert::string)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|m| p.error("characterSet", m))?
                .concat(),
        ),
        Some(other) => {
            return Err(p.error(
                "characterSet",
                format!(
                    "expected \"auto\", a string or an array, got {}",
                    crate::props::describe(other)
                ),
            ))
        }
    };
    if let Some(settings) = p.get("fontSettings") {
        let object = settings
            .as_object()
            .ok_or_else(|| p.error("fontSettings", "expected an object"))?;
        let number = |key: &str, default: f32| -> Result<f32> {
            match object.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(v) => convert::f32(v).map_err(|m| p.error("fontSettings", format!("{key}: {m}"))),
            }
        };
        font.font_size = number("fontSize", font.font_size)?;
        font.buffer = number("buffer", font.buffer as f32)?.max(0.0) as u32;
        font.cutoff = number("cutoff", font.cutoff)?;
        font.radius = number("radius", font.radius)?;
        font.smoothing = number("smoothing", font.smoothing)?;
        font.sdf = match object.get("sdf") {
            None | Some(Value::Null) => font.sdf,
            Some(Value::Bool(b)) => *b,
            Some(v) => convert::number(v).map_err(|m| p.error("fontSettings", format!("sdf: {m}")))? != 0.0,
        };
    }
    let word_break = match p.string("wordBreak")?.as_deref() {
        None => d.word_break,
        Some("break-word") => WordBreak::BreakWord,
        Some("break-all") => WordBreak::BreakAll,
        Some(other) => {
            return Err(p.error(
                "wordBreak",
                format!("expected break-word or break-all, got `{other}`"),
            ))
        }
    };
    let background_padding = match p.get("backgroundPadding") {
        None | Some(Value::Null) => d.background_padding,
        Some(v) => box_sides(v).map_err(|m| p.error("backgroundPadding", m))?,
    };
    let background_border_radius = match p.get("backgroundBorderRadius") {
        None | Some(Value::Null) => d.background_border_radius,
        Some(Value::Number(n)) => [n.as_f64().unwrap_or(0.0) as f32; 4],
        Some(v) => {
            let r = convert::numbers(v, 4, 4).map_err(|m| p.error("backgroundBorderRadius", m))?;
            [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
        }
    };
    Ok(TextLayerProps {
        base: p.base()?,
        data,
        billboard: p.bool("billboard", d.billboard)?,
        size_scale: p.f32("sizeScale", d.size_scale)?,
        size_units: p.unit("sizeUnits", d.size_units)?,
        size_min_pixels: p.f32("sizeMinPixels", d.size_min_pixels)?,
        size_max_pixels: p.f32("sizeMaxPixels", d.size_max_pixels)?,
        background: p.bool("background", d.background)?,
        get_background_color: p.accessor("getBackgroundColor", &d.get_background_color, convert::color)?,
        get_border_color: p.accessor("getBorderColor", &d.get_border_color, convert::color)?,
        get_border_width: p.accessor("getBorderWidth", &d.get_border_width, convert::f32)?,
        background_border_radius,
        background_padding,
        font,
        line_height: p.f32("lineHeight", d.line_height)?,
        outline_width: p.f32("outlineWidth", d.outline_width)?,
        outline_color: p.color("outlineColor", d.outline_color)?,
        word_break,
        max_width: p.f32("maxWidth", d.max_width)?,
        get_text: p.accessor("getText", "text", convert::string)?,
        get_position: p.accessor("getPosition", "position", convert::position)?,
        get_color: p.accessor("getColor", &d.get_color, convert::color)?,
        get_size: p.accessor("getSize", &d.get_size, convert::f32)?,
        get_angle: p.accessor("getAngle", &d.get_angle, convert::f32)?,
        get_text_anchor: p.accessor("getTextAnchor", &d.get_text_anchor, text_anchor)?,
        get_alignment_baseline: p.accessor(
            "getAlignmentBaseline",
            &d.get_alignment_baseline,
            alignment_baseline,
        )?,
        get_pixel_offset: p.accessor("getPixelOffset", &d.get_pixel_offset, convert::vec2)?,
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

/// One entry of a ContourLayer's `contours`: `threshold` (a number for an isoline or
/// `[min, max]` for an isoband), `color`, `strokeWidth` and `zIndex`.
fn contour_from_value(value: &Value) -> std::result::Result<Contour, String> {
    let map = value
        .as_object()
        .ok_or_else(|| format!("expected an object, got {}", crate::props::describe(value)))?;
    let threshold = match map.get("threshold") {
        Some(Value::Array(_)) => {
            let n = convert::numbers(map.get("threshold").unwrap(), 2, 2)?;
            ContourThreshold::Band([n[0] as f32, n[1] as f32])
        }
        Some(v) if v.is_number() => ContourThreshold::Line(convert::f32(v)?),
        _ => return Err("threshold must be a number or [min, max]".to_string()),
    };
    let mut contour = Contour {
        threshold,
        ..Contour::line(0.0)
    };
    if let Some(color) = map.get("color").filter(|v| !v.is_null()) {
        contour.color = convert::color(color)?;
    }
    if let Some(width) = map.get("strokeWidth").filter(|v| !v.is_null()) {
        contour.stroke_width = convert::f32(width)?;
    }
    if let Some(z) = map.get("zIndex").filter(|v| !v.is_null()) {
        contour.z_index = Some(convert::f32(z)? as i32);
    }
    Ok(contour)
}

#[cfg(feature = "fetch")]
fn tile_loader(templates: Vec<String>) -> deck_gl_layers::TileLoader {
    deck_gl_layers::tile_layer::raster_loader(templates)
}

#[cfg(not(feature = "fetch"))]
fn tile_loader(_templates: Vec<String>) -> deck_gl_layers::TileLoader {
    deck_gl_layers::TileLoader::new(|_, _| Err("built without the fetch feature".to_string()))
}

//! Port of `@deck.gl/layers/src/geojson-layer/geojson-layer.ts`: features are split by
//! geometry type into a polygon layer, a path layer and a scatterplot layer.

use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32};
use deck_gl::geojson::{FeatureCollection, Geometry};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Path, Polygon, Position, Result, SubLayers,
    Unit, Viewport,
};

use crate::{
    PathLayer, PathLayerProps, PolygonLayer, PolygonLayerProps, ScatterplotLayer, ScatterplotLayerProps,
};

/// Properties of a [`GeoJsonLayer`]. Accessors are called with the feature index. Defaults
/// match deck.gl with `pointType: 'circle'`.
#[derive(Clone, Debug)]
pub struct GeoJsonLayerProps {
    pub base: LayerProps,
    pub data: Arc<FeatureCollection>,
    pub filled: bool,
    pub stroked: bool,
    pub extruded: bool,
    pub wireframe: bool,
    pub elevation_scale: f32,
    pub line_width_units: Unit,
    pub line_width_scale: f32,
    pub line_width_min_pixels: f32,
    pub line_width_max_pixels: f32,
    pub line_joint_rounded: bool,
    pub line_cap_rounded: bool,
    pub line_miter_limit: f32,
    pub point_radius_units: Unit,
    pub point_radius_scale: f32,
    pub point_radius_min_pixels: f32,
    pub point_radius_max_pixels: f32,
    pub get_fill_color: Accessor<Color>,
    pub get_line_color: Accessor<Color>,
    pub get_line_width: Accessor<f32>,
    pub get_point_radius: Accessor<f32>,
    pub get_elevation: Accessor<f32>,
}

impl Default for GeoJsonLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("GeoJsonLayer"),
            data: Arc::new(FeatureCollection::default()),
            filled: true,
            stroked: true,
            extruded: false,
            wireframe: false,
            elevation_scale: 1.0,
            line_width_units: Unit::Meters,
            line_width_scale: 1.0,
            line_width_min_pixels: 0.0,
            line_width_max_pixels: f32::MAX,
            line_joint_rounded: false,
            line_cap_rounded: false,
            line_miter_limit: 4.0,
            point_radius_units: Unit::Meters,
            point_radius_scale: 1.0,
            point_radius_min_pixels: 0.0,
            point_radius_max_pixels: f32::MAX,
            get_fill_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_width: Accessor::Constant(1.0),
            get_point_radius: Accessor::Constant(1.0),
            get_elevation: Accessor::Constant(1000.0),
        }
    }
}

/// Renders GeoJSON features as polygons, paths and points.
pub struct GeoJsonLayer {
    props: GeoJsonLayerProps,
    sub_layers: SubLayers,
    dirty: bool,
}

/// Geometry parts of a feature collection, each tagged with its feature index.
#[derive(Default)]
struct Parts {
    polygons: Vec<(u32, Polygon)>,
    paths: Vec<(u32, Path)>,
    points: Vec<(u32, Position)>,
}

fn collect(parts: &mut Parts, row: u32, geometry: &Geometry) {
    match geometry {
        Geometry::Point(p) => parts.points.push((row, *p)),
        Geometry::MultiPoint(ps) => parts.points.extend(ps.iter().map(|p| (row, *p))),
        Geometry::LineString(path) => parts.paths.push((row, path.clone())),
        Geometry::MultiLineString(paths) => parts.paths.extend(paths.iter().map(|p| (row, p.clone()))),
        Geometry::Polygon(polygon) => parts.polygons.push((row, polygon.clone())),
        Geometry::MultiPolygon(polygons) => parts.polygons.extend(polygons.iter().map(|p| (row, p.clone()))),
        Geometry::GeometryCollection(geometries) => {
            for geometry in geometries {
                collect(parts, row, geometry);
            }
        }
    }
}

/// Resolve a per-feature accessor, then remap it to a per-part accessor.
fn remap<T: Clone + Send + Sync + 'static>(
    data: &LayerData,
    accessor: &Accessor<T>,
    rows: &Arc<Vec<u32>>,
    resolve: impl Fn(&LayerData, &Accessor<T>) -> Result<Vec<T>>,
) -> Result<Accessor<T>> {
    Ok(match accessor {
        Accessor::Constant(v) => Accessor::Constant(v.clone()),
        other => {
            let values = Arc::new(resolve(data, other)?);
            let rows = rows.clone();
            Accessor::func(move |i| values[rows[i] as usize].clone())
        }
    })
}

impl GeoJsonLayer {
    pub fn new(props: GeoJsonLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &GeoJsonLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: GeoJsonLayerProps) {
        self.props = props;
        self.dirty = true;
    }

    fn sub_props(&self, suffix: &str) -> LayerProps {
        LayerProps {
            id: format!("{}-{suffix}", self.props.base.id),
            ..self.props.base.clone()
        }
    }

    fn render_layers(&self) -> Result<Vec<Box<dyn Layer>>> {
        let props = &self.props;
        let mut parts = Parts::default();
        for (row, feature) in props.data.features.iter().enumerate() {
            if let Some(geometry) = &feature.geometry {
                collect(&mut parts, row as u32, geometry);
            }
        }
        let feature_data = LayerData::with_length(props.data.len());
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        if !parts.polygons.is_empty() && (props.filled || props.stroked || props.extruded) {
            let rows = Arc::new(parts.polygons.iter().map(|(row, _)| *row).collect::<Vec<_>>());
            let polygons = Arc::new(parts.polygons.into_iter().map(|(_, p)| p).collect::<Vec<_>>());
            layers.push(Box::new(PolygonLayer::new(PolygonLayerProps {
                base: self.sub_props("polygons"),
                data: LayerData::with_length(polygons.len()).with_source_rows(rows.clone()),
                stroked: props.stroked,
                filled: props.filled,
                extruded: props.extruded,
                wireframe: props.wireframe,
                elevation_scale: props.elevation_scale,
                line_width_units: props.line_width_units,
                line_width_scale: props.line_width_scale,
                line_width_min_pixels: props.line_width_min_pixels,
                line_width_max_pixels: props.line_width_max_pixels,
                line_joint_rounded: props.line_joint_rounded,
                line_miter_limit: props.line_miter_limit,
                get_polygon: Accessor::func(move |i| polygons[i].clone()),
                get_fill_color: remap(&feature_data, &props.get_fill_color, &rows, resolve_colors)?,
                get_line_color: remap(&feature_data, &props.get_line_color, &rows, resolve_colors)?,
                get_line_width: remap(&feature_data, &props.get_line_width, &rows, resolve_f32)?,
                get_elevation: remap(&feature_data, &props.get_elevation, &rows, resolve_f32)?,
            })));
        }

        if !parts.paths.is_empty() && props.stroked {
            let rows = Arc::new(parts.paths.iter().map(|(row, _)| *row).collect::<Vec<_>>());
            let paths = Arc::new(parts.paths.into_iter().map(|(_, p)| p).collect::<Vec<_>>());
            layers.push(Box::new(PathLayer::new(PathLayerProps {
                base: self.sub_props("lines"),
                data: LayerData::with_length(paths.len()).with_source_rows(rows.clone()),
                width_units: props.line_width_units,
                width_scale: props.line_width_scale,
                width_min_pixels: props.line_width_min_pixels,
                width_max_pixels: props.line_width_max_pixels,
                joint_rounded: props.line_joint_rounded,
                cap_rounded: props.line_cap_rounded,
                miter_limit: props.line_miter_limit,
                get_path: Accessor::func(move |i| paths[i].clone()),
                get_color: remap(&feature_data, &props.get_line_color, &rows, resolve_colors)?,
                get_width: remap(&feature_data, &props.get_line_width, &rows, resolve_f32)?,
                ..Default::default()
            })));
        }

        if !parts.points.is_empty() {
            let rows = Arc::new(parts.points.iter().map(|(row, _)| *row).collect::<Vec<_>>());
            let points = Arc::new(parts.points.into_iter().map(|(_, p)| p).collect::<Vec<_>>());
            layers.push(Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
                base: self.sub_props("points"),
                data: LayerData::with_length(points.len()).with_source_rows(rows.clone()),
                radius_units: props.point_radius_units,
                radius_scale: props.point_radius_scale,
                radius_min_pixels: props.point_radius_min_pixels,
                radius_max_pixels: props.point_radius_max_pixels,
                line_width_units: props.line_width_units,
                line_width_scale: props.line_width_scale,
                line_width_min_pixels: props.line_width_min_pixels,
                line_width_max_pixels: props.line_width_max_pixels,
                stroked: props.stroked,
                filled: props.filled,
                get_position: Accessor::func(move |i| points[i]),
                get_radius: remap(&feature_data, &props.get_point_radius, &rows, resolve_f32)?,
                get_fill_color: remap(&feature_data, &props.get_fill_color, &rows, resolve_colors)?,
                get_line_color: remap(&feature_data, &props.get_line_color, &rows, resolve_colors)?,
                get_line_width: remap(&feature_data, &props.get_line_width, &rows, resolve_f32)?,
                ..Default::default()
            })));
        }

        Ok(layers)
    }
}

impl Layer for GeoJsonLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
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
        self.props.base.highlighted_object_index = index;
        self.sub_layers.set_highlighted_object(index);
    }
}

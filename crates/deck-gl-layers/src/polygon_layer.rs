//! Port of `@deck.gl/layers/src/polygon-layer/polygon-layer.ts`: a composite of
//! `SolidPolygonLayer` for the fill and `PathLayer` for the outline.

use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32, resolve_polygons};
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Path, Polygon, Result, SubLayers, Unit,
    Viewport,
};

use crate::{PathLayer, PathLayerProps, SolidPolygonLayer, SolidPolygonLayerProps};

/// Properties of a [`PolygonLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug)]
pub struct PolygonLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// Draw the outline. Ignored when extruded (use `wireframe`).
    pub stroked: bool,
    pub filled: bool,
    pub extruded: bool,
    pub wireframe: bool,
    pub elevation_scale: f32,
    pub line_width_units: Unit,
    pub line_width_scale: f32,
    pub line_width_min_pixels: f32,
    pub line_width_max_pixels: f32,
    pub line_joint_rounded: bool,
    pub line_miter_limit: f32,
    pub get_polygon: Accessor<Polygon>,
    pub get_fill_color: Accessor<Color>,
    pub get_line_color: Accessor<Color>,
    pub get_line_width: Accessor<f32>,
    pub get_elevation: Accessor<f32>,
}

impl Default for PolygonLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("PolygonLayer"),
            data: LayerData::default(),
            stroked: true,
            filled: true,
            extruded: false,
            wireframe: false,
            elevation_scale: 1.0,
            line_width_units: Unit::Meters,
            line_width_scale: 1.0,
            line_width_min_pixels: 0.0,
            line_width_max_pixels: f32::MAX,
            line_joint_rounded: false,
            line_miter_limit: 4.0,
            get_polygon: Accessor::column("polygon"),
            get_fill_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_color: Accessor::Constant([0, 0, 0, 255]),
            get_line_width: Accessor::Constant(1.0),
            get_elevation: Accessor::Constant(1000.0),
        }
    }
}

/// Renders filled, stroked or extruded polygons.
pub struct PolygonLayer {
    props: PolygonLayerProps,
    sub_layers: SubLayers,
    dirty: bool,
}

impl PolygonLayer {
    pub fn new(props: PolygonLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &PolygonLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: PolygonLayerProps) {
        self.props = props;
        self.dirty = true;
    }

    fn sub_props(&self, suffix: &str) -> LayerProps {
        LayerProps {
            id: format!("{}-{suffix}", self.props.base.id),
            ..self.props.base.clone()
        }
    }

    /// Build the fill and stroke sub layers from the current props.
    fn render_layers(&self) -> Result<Vec<Box<dyn Layer>>> {
        let props = &self.props;
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        if props.filled || props.extruded {
            layers.push(Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
                base: self.sub_props("fill"),
                data: props.data.clone(),
                filled: props.filled,
                extruded: props.extruded,
                wireframe: props.wireframe,
                elevation_scale: props.elevation_scale,
                get_polygon: props.get_polygon.clone(),
                get_elevation: props.get_elevation.clone(),
                get_fill_color: props.get_fill_color.clone(),
                get_line_color: if props.extruded && props.wireframe {
                    props.get_line_color.clone()
                } else {
                    Accessor::Constant([0, 0, 0, 255])
                },
            })));
        }

        if props.stroked && !props.extruded {
            // Every ring of every polygon becomes one closed path. Ring to source row mapping
            // is kept so colors and widths can be looked up per ring.
            let polygons = resolve_polygons(&props.data, &props.get_polygon)?;
            let line_colors = resolve_colors(&props.data, &props.get_line_color)?;
            let line_widths = resolve_f32(&props.data, &props.get_line_width)?;
            let mut rings: Vec<Path> = Vec::new();
            let mut rows: Vec<usize> = Vec::new();
            for (row, polygon) in polygons.iter().enumerate() {
                for ring in polygon {
                    if ring.len() < 2 {
                        continue;
                    }
                    let mut path = ring.clone();
                    if path.first() != path.last() {
                        path.push(path[0]);
                    }
                    rings.push(path);
                    rows.push(row);
                }
            }
            let ring_count = rings.len();
            let rings = Arc::new(rings);
            let rows = Arc::new(rows);
            let colors = Arc::new(line_colors);
            let widths = Arc::new(line_widths);
            layers.push(Box::new(PathLayer::new(PathLayerProps {
                base: self.sub_props("stroke"),
                data: LayerData::with_length(ring_count),
                width_units: props.line_width_units,
                width_scale: props.line_width_scale,
                width_min_pixels: props.line_width_min_pixels,
                width_max_pixels: props.line_width_max_pixels,
                joint_rounded: props.line_joint_rounded,
                miter_limit: props.line_miter_limit,
                get_path: Accessor::func({
                    let rings = rings.clone();
                    move |i| rings[i].clone()
                }),
                get_color: Accessor::func({
                    let rows = rows.clone();
                    let colors = colors.clone();
                    move |i| colors[rows[i]]
                }),
                get_width: Accessor::func(move |i| widths[rows[i]]),
                ..Default::default()
            })));
        }

        Ok(layers)
    }
}

impl Layer for PolygonLayer {
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
}

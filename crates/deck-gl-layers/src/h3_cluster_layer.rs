//! Port of `@deck.gl/geo-layers/src/h3-layers/h3-cluster-layer.ts`: sets of H3 cells merged
//! into the outline of the area they cover, drawn as polygons.

use std::str::FromStr;
use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32, resolve_string_lists};
use deck_gl::{
    Accessor, Layer, LayerContext, LayerData, LayerProps, Polygon, Position, Result, SubLayers, Viewport,
};

use crate::polygon_layer::{PolygonLayer, PolygonLayerProps};

/// Properties of an [`H3ClusterLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct H3ClusterLayerProps {
    pub polygon: PolygonLayerProps,
    /// The H3 cells of each row, as indexes in the usual hexadecimal form
    pub get_hexagons: Accessor<Vec<String>>,
}

impl Default for H3ClusterLayerProps {
    fn default() -> Self {
        Self {
            polygon: PolygonLayerProps {
                base: LayerProps::new("h3-clusters"),
                ..Default::default()
            },
            get_hexagons: Accessor::column("hexagons"),
        }
    }
}

/// The outline of a set of H3 cells: the boundary of their union, with one ring per part and
/// holes after it, as h3-js's `cellsToMultiPolygon` returns.
///
/// Cells that are not valid indexes are skipped. Rings are unwrapped across the antimeridian
/// relative to their first vertex, deck.gl's `normalizeLongitudes`.
pub fn cluster_polygons(hexagons: &[String]) -> Vec<Polygon> {
    // Duplicates would corrupt the vertex graph, so the cells are deduplicated first
    let mut cells: Vec<h3o::CellIndex> = hexagons
        .iter()
        .filter_map(|index| h3o::CellIndex::from_str(index.trim()).ok())
        .collect();
    cells.sort_unstable();
    cells.dedup();
    if cells.is_empty() {
        return Vec::new();
    }
    let solvent = h3o::geom::SolventBuilder::new().build();
    let Ok(multi) = solvent.dissolve(cells) else {
        return Vec::new();
    };
    multi
        .into_iter()
        .map(|polygon| {
            let (exterior, interiors) = polygon.into_inner();
            std::iter::once(exterior)
                .chain(interiors)
                .map(|ring| normalize_ring(ring.into_inner().into_iter().map(|c| (c.x, c.y))))
                .collect()
        })
        .collect()
}

/// deck.gl's `normalizeLongitudes`: keep every vertex within half a world of the first, so a
/// ring that crosses the antimeridian stays in one piece.
fn normalize_ring(coords: impl Iterator<Item = (f64, f64)>) -> Vec<Position> {
    let mut reference = f64::NAN;
    coords
        .map(|(x, y)| {
            let mut lng = x;
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
            [lng, y, 0.0]
        })
        .collect()
}

/// Draws the outline of sets of H3 cells.
pub struct H3ClusterLayer {
    props: H3ClusterLayerProps,
    sub_layers: SubLayers,
    dirty: bool,
}

impl H3ClusterLayer {
    pub fn new(props: H3ClusterLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &H3ClusterLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: H3ClusterLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    /// One polygon per part of every row's outline, with the row each part came from.
    fn parts(props: &H3ClusterLayerProps) -> Result<(Vec<Polygon>, Vec<u32>)> {
        let data = &props.polygon.data;
        let hexagons = resolve_string_lists(data, &props.get_hexagons)?;
        let mut polygons = Vec::new();
        let mut rows = Vec::new();
        for (row, cells) in hexagons.iter().enumerate() {
            for polygon in cluster_polygons(cells) {
                polygons.push(polygon);
                rows.push(data.source_row(row));
            }
        }
        Ok((polygons, rows))
    }

    fn render_layers(&self) -> Result<Vec<Box<dyn Layer>>> {
        let props = &self.props;
        let (polygons, rows) = Self::parts(props)?;
        let rows = Arc::new(rows);
        let polygons = Arc::new(polygons);
        let data = &props.polygon.data;
        // Every other accessor is read for the source row of each part
        let remap = |accessor: &Accessor<deck_gl::Color>| -> Result<Accessor<deck_gl::Color>> {
            Ok(match accessor {
                Accessor::Constant(v) => Accessor::Constant(*v),
                other => {
                    let values = Arc::new(resolve_colors(data, other)?);
                    let rows = rows.clone();
                    Accessor::func(move |i| values[rows[i] as usize])
                }
            })
        };
        let remap_f32 = |accessor: &Accessor<f32>| -> Result<Accessor<f32>> {
            Ok(match accessor {
                Accessor::Constant(v) => Accessor::Constant(*v),
                other => {
                    let values = Arc::new(resolve_f32(data, other)?);
                    let rows = rows.clone();
                    Accessor::func(move |i| values[rows[i] as usize])
                }
            })
        };
        let base = &props.polygon;
        let layer = PolygonLayer::new(PolygonLayerProps {
            base: LayerProps {
                id: format!("{}-clusters", base.base.id),
                ..base.base.clone()
            },
            data: LayerData::with_length(polygons.len()).with_source_rows(rows.clone()),
            get_polygon: {
                let polygons = polygons.clone();
                Accessor::func(move |i| polygons[i].clone())
            },
            get_fill_color: remap(&base.get_fill_color)?,
            get_line_color: remap(&base.get_line_color)?,
            get_line_width: remap_f32(&base.get_line_width)?,
            get_elevation: remap_f32(&base.get_elevation)?,
            ..base.clone()
        });
        Ok(vec![Box::new(layer)])
    }
}

impl Layer for H3ClusterLayer {
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

    fn bounds(&self) -> Option<[f64; 4]> {
        self.sub_layers.bounds()
    }

    fn in_transition(&self) -> bool {
        self.sub_layers.in_transition()
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

impl Default for H3ClusterLayer {
    fn default() -> Self {
        Self::new(H3ClusterLayerProps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cell and its six neighbours: one filled hexagon of seven cells.
    fn cluster() -> Vec<String> {
        let centre = h3o::CellIndex::from_str("8928308280fffff").unwrap();
        std::iter::once(centre)
            .chain(centre.grid_disk::<Vec<_>>(1))
            .map(|c| c.to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    #[test]
    fn cells_merge_into_one_outline() {
        let polygons = cluster_polygons(&cluster());
        assert_eq!(polygons.len(), 1, "seven touching cells make one polygon");
        let rings = &polygons[0];
        assert_eq!(rings.len(), 1, "and it has no holes");
        // Six cells around one: the outline has more vertices than a single hexagon
        assert!(rings[0].len() > 7, "{} vertices", rings[0].len());
        // A ring around a missing centre leaves a hole
        let centre = h3o::CellIndex::from_str("8928308280fffff").unwrap();
        let ring_only: Vec<String> = centre
            .grid_ring_fast(1)
            .flatten()
            .map(|c| c.to_string())
            .collect();
        let polygons = cluster_polygons(&ring_only);
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 2, "an outer ring and a hole");
        // Two cells far apart stay two polygons
        let at = |lng: f64, lat: f64| {
            h3o::LatLng::new(lat, lng)
                .unwrap()
                .to_cell(h3o::Resolution::Seven)
                .to_string()
        };
        assert_eq!(cluster_polygons(&[at(-122.4, 37.8), at(-74.0, 40.7)]).len(), 2);
        // The same cell twice is one outline, not a corrupted graph
        let centre_index = "8928308280fffff".to_string();
        let twice = cluster_polygons(&[centre_index.clone(), centre_index.clone()]);
        let once = cluster_polygons(&[centre_index]);
        assert_eq!(twice.len(), 1);
        assert_eq!(twice[0].len(), 1);
        assert_eq!(twice[0][0].len(), once[0][0].len());
        // Nonsense is skipped
        assert!(cluster_polygons(&["not a cell".to_string()]).is_empty());
        assert!(cluster_polygons(&[]).is_empty());
    }

    #[test]
    fn rings_stay_in_one_piece_across_the_antimeridian() {
        // A cell on the antimeridian: every vertex must land within half a world of the first
        let cell = h3o::LatLng::new(0.0, 179.999)
            .unwrap()
            .to_cell(h3o::Resolution::Five);
        let polygons = cluster_polygons(&[cell.to_string()]);
        assert_eq!(polygons.len(), 1);
        let ring = &polygons[0][0];
        let first = ring[0][0];
        assert!(ring.iter().all(|p| (p[0] - first).abs() < 180.0), "{ring:?}");
    }
}

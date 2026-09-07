//! Port of `@deck.gl/geo-layers/src/mvt-layer`: vector tiles decoded into GeoJSON features and
//! drawn per tile with a `GeoJsonLayer`, on top of the [`TileLayer`].

use std::sync::Arc;

use deck_gl::geojson::FeatureCollection;
use deck_gl::{Layer, LayerContext, LayerProps, Result, Viewport};

use crate::tile_layer::{LoadedTile, TileData, TileLayer, TileLayerProps, TileLoader, TileRenderer};
use crate::{GeoJsonLayer, GeoJsonLayerProps};

/// Props of a vector tile layer: the tile props plus the GeoJSON props every tile is drawn
/// with (`geojson.data` and `geojson.base.id` are ignored).
#[derive(Clone, Debug, PartialEq)]
pub struct MvtLayerProps {
    pub tiles: TileLayerProps,
    pub geojson: GeoJsonLayerProps,
    /// Only draw features of these source layers when set
    pub layers: Option<Vec<String>>,
}

impl Default for MvtLayerProps {
    fn default() -> Self {
        Self {
            tiles: TileLayerProps {
                base: LayerProps::new("vector-tiles"),
                ..Default::default()
            },
            geojson: GeoJsonLayerProps::default(),
            layers: None,
        }
    }
}

/// A loader decoding `.mvt` / `.pbf` tiles fetched from URL templates. Needs the `fetch`
/// feature.
#[cfg(feature = "fetch")]
pub fn mvt_loader(templates: Vec<String>) -> TileLoader {
    TileLoader::new(move |index, bounds| {
        let Some(url) = crate::tileset::url_from_template(&templates, index) else {
            return Ok(None);
        };
        let bytes = crate::tileset::fetch_bytes(&url)?;
        let bytes = crate::mvt::maybe_gunzip(bytes)?;
        let collection =
            crate::mvt::decode_tile_features(&bytes, bounds).map_err(|e| format!("{url}: {e}"))?;
        Ok(Some(Arc::new(collection) as TileData))
    })
}

/// A loader decoding tiles from bytes produced by `fetch` (any source).
pub fn mvt_loader_with(
    fetch: impl Fn(crate::tileset::TileIndex) -> std::result::Result<Option<Vec<u8>>, String>
        + Send
        + Sync
        + 'static,
) -> TileLoader {
    TileLoader::new(move |index, bounds| {
        let Some(bytes) = fetch(index)? else {
            return Ok(None);
        };
        let bytes = crate::mvt::maybe_gunzip(bytes)?;
        let collection = crate::mvt::decode_tile_features(&bytes, bounds)?;
        Ok(Some(Arc::new(collection) as TileData))
    })
}

/// Draw a tile's `FeatureCollection` with a `GeoJsonLayer` using `geojson` props.
pub fn geojson_renderer(geojson: GeoJsonLayerProps, layers: Option<Vec<String>>) -> TileRenderer {
    let geojson = Arc::new(geojson);
    let layers = Arc::new(layers);
    TileRenderer::new(move |tile: &LoadedTile, base: &LayerProps| {
        let Some(collection) = tile.data::<FeatureCollection>() else {
            return Vec::new();
        };
        let data = match layers.as_ref() {
            Some(wanted) => Arc::new(FeatureCollection {
                features: collection
                    .features
                    .iter()
                    .filter(|f| {
                        f.string("layerName")
                            .is_some_and(|n| wanted.iter().any(|w| w == n))
                    })
                    .cloned()
                    .collect(),
            }),
            None => Arc::new(collection.clone()),
        };
        if data.is_empty() {
            return Vec::new();
        }
        vec![Box::new(GeoJsonLayer::new(GeoJsonLayerProps {
            base: LayerProps {
                id: tile.sub_layer_id(&base.id, "geojson"),
                ..base.clone()
            },
            data,
            ..(*geojson).clone()
        })) as Box<dyn Layer>]
    })
}

/// Vector tiles drawn as GeoJSON per tile.
pub struct MvtLayer {
    props: MvtLayerProps,
    inner: TileLayer,
}

impl MvtLayer {
    pub fn new(props: MvtLayerProps) -> Self {
        let inner = TileLayer::new(Self::tile_props(&props));
        Self { props, inner }
    }

    /// Tiles from URL templates such as `https://host/tiles/{z}/{x}/{y}.pbf`. Needs `fetch`.
    #[cfg(feature = "fetch")]
    pub fn from_templates(id: impl Into<String>, templates: Vec<String>, geojson: GeoJsonLayerProps) -> Self {
        let mut props = MvtLayerProps {
            geojson,
            ..Default::default()
        };
        props.tiles.base = LayerProps::new(id);
        props.tiles.get_tile_data = Some(mvt_loader(templates));
        Self::new(props)
    }

    fn tile_props(props: &MvtLayerProps) -> TileLayerProps {
        TileLayerProps {
            render_sub_layers: geojson_renderer(props.geojson.clone(), props.layers.clone()),
            ..props.tiles.clone()
        }
    }

    pub fn props(&self) -> &MvtLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: MvtLayerProps) {
        if self.props != props {
            self.props = props;
            self.inner.set_props(Self::tile_props(&self.props));
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    pub fn tile_layer(&self) -> &TileLayer {
        &self.inner
    }
}

impl Layer for MvtLayer {
    fn props(&self) -> &LayerProps {
        &self.props.tiles.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        self.inner.initialize(ctx)
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        self.inner.update(ctx, viewport)
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.inner.draw(ctx, pass)
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        self.inner.set_picking_active(ctx, active)
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.inner.draw_picking(ctx, pass)
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.tiles.base.highlighted_object_index = index;
        self.inner.set_highlighted_object(index);
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

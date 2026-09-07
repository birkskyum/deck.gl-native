//! Port of `@deck.gl/geo-layers/src/terrain-layer/terrain-layer.ts`: elevation tiles become
//! terrain meshes with an optional texture draped over them.
//!
//! With URL templates the layer is a [`TileLayer`](crate::TileLayer) whose loader fetches an
//! elevation image per tile, decodes it with the [`ElevationDecoder`] and triangulates it
//! ([`terrain_mesh`]); each tile is drawn by a [`SimpleMeshLayer`](crate::SimpleMeshLayer).
//! With a plain URL and `bounds` a single mesh covers those bounds.

use std::sync::Arc;

use deck_gl::{Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Result, Viewport};
use math_gl::web_mercator::lng_lat_to_world;

use crate::terrain::{terrain_mesh, ElevationDecoder};
use crate::tile_layer::{LoadedTile, TileData, TileLayer, TileLayerProps, TileLoader, TileRenderer};
use crate::tileset::{is_url_template, url_from_template, TileBounds, TileIndex};
use crate::{BitmapImage, Mesh, SimpleMeshLayer, SimpleMeshLayerProps};

/// A loaded terrain tile: its mesh and the texture to drape over it.
#[derive(Clone, Debug)]
pub struct TerrainTile {
    pub mesh: Arc<Mesh>,
    pub texture: Option<BitmapImage>,
}

/// Properties of a [`TerrainLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainLayerProps {
    pub base: LayerProps,
    /// URL templates (`{z}/{x}/{y}`) of the elevation tiles, or one image URL with `bounds`
    pub elevation_data: Vec<String>,
    /// URL templates or a URL of the image draped over the terrain
    pub texture: Vec<String>,
    pub elevation_decoder: ElevationDecoder,
    /// The largest deviation of the mesh from the height map, in meters
    pub mesh_max_error: f32,
    /// `[west, south, east, north]` of a single elevation image
    pub bounds: Option<[f64; 4]>,
    /// Used where there is no texture
    pub color: Color,
    pub wireframe: bool,
    pub min_zoom: Option<u32>,
    pub max_zoom: Option<u32>,
    /// Pixel size of a tile at its zoom
    pub tile_size: f64,
    /// Concurrent tile loads
    pub max_requests: usize,
}

impl Default for TerrainLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("TerrainLayer"),
            elevation_data: Vec::new(),
            texture: Vec::new(),
            elevation_decoder: ElevationDecoder::default(),
            mesh_max_error: 4.0,
            bounds: None,
            color: [255, 255, 255, 255],
            wireframe: false,
            min_zoom: None,
            max_zoom: None,
            tile_size: 256.0,
            max_requests: 6,
        }
    }
}

/// The mesh of one tile, in the common space its bounds project to.
fn tile_mesh(
    image: &BitmapImage,
    decoder: &ElevationDecoder,
    bounds: TileBounds,
    max_error: f32,
) -> std::result::Result<Mesh, String> {
    let [min_x, min_y, max_x, max_y] = match bounds {
        TileBounds::Geo {
            west,
            south,
            east,
            north,
        } => {
            let [x0, y0] = lng_lat_to_world([west, south]);
            let [x1, y1] = lng_lat_to_world([east, north]);
            [x0, y0, x1, y1]
        }
        cartesian => cartesian.as_array(),
    };
    terrain_mesh(image, decoder, [min_x, min_y, max_x, max_y], max_error)
}

/// The loader of a tiled terrain: an elevation image, optionally a texture.
fn terrain_loader(props: &TerrainLayerProps) -> TileLoader {
    let elevation = props.elevation_data.clone();
    let texture = props.texture.clone();
    let decoder = props.elevation_decoder;
    let max_error = props.mesh_max_error;
    TileLoader::cancellable(move |index: TileIndex, bounds, cancel| {
        let Some(url) = url_from_template(&elevation, index) else {
            return Ok(None);
        };
        let bytes = crate::tileset::fetch_bytes_with(&url, cancel)?;
        let image = image::load_from_memory(&bytes)
            .map_err(|e| format!("{url}: {e}"))?
            .to_rgba8();
        let elevation_image = BitmapImage {
            width: image.width(),
            height: image.height(),
            rgba: Arc::new(image.into_raw()),
        };
        let mesh = tile_mesh(&elevation_image, &decoder, bounds, max_error)?;
        // A texture that fails to load leaves the tile with its flat colour
        let texture = url_from_template(&texture, index).and_then(|url| {
            let bytes = crate::tileset::fetch_bytes_with(&url, cancel).ok()?;
            let image = image::load_from_memory(&bytes).ok()?.to_rgba8();
            Some(BitmapImage {
                width: image.width(),
                height: image.height(),
                rgba: Arc::new(image.into_raw()),
            })
        });
        Ok(Some(Arc::new(TerrainTile {
            mesh: Arc::new(mesh),
            texture,
        }) as TileData))
    })
}

/// One `SimpleMeshLayer` per loaded tile, drawing its mesh in common space.
fn terrain_renderer(props: &TerrainLayerProps) -> TileRenderer {
    let color = props.color;
    let wireframe = props.wireframe;
    TileRenderer::new(move |tile: &LoadedTile, base: &LayerProps| {
        let Some(terrain) = tile.data::<TerrainTile>() else {
            return Vec::new();
        };
        vec![Box::new(SimpleMeshLayer::new(SimpleMeshLayerProps {
            base: LayerProps {
                id: tile.sub_layer_id(&base.id, "mesh"),
                // The mesh positions are already projected, deck.gl's `CARTESIAN`
                coordinate_system: deck_gl::CoordinateSystem::Cartesian,
                ..base.clone()
            },
            data: LayerData::with_length(1),
            mesh: Some(terrain.mesh.clone()),
            texture: terrain.texture.clone(),
            wireframe,
            // The mesh carries world positions, not offsets around an anchor
            instanced: false,
            get_position: Accessor::Constant([0.0, 0.0, 0.0]),
            get_color: Accessor::Constant(color),
            ..Default::default()
        })) as Box<dyn Layer>]
    })
}

/// Renders terrain meshes from elevation tiles.
pub struct TerrainLayer {
    props: TerrainLayerProps,
    /// Tiled terrain, when the elevation data is a URL template
    tiles: Option<TileLayer>,
    /// A single mesh, when it is one image over `bounds`
    single: Option<SimpleMeshLayer>,
    /// Whether the single mesh has been loaded
    loaded: bool,
    dirty: bool,
}

impl TerrainLayer {
    pub fn new(props: TerrainLayerProps) -> Self {
        let mut layer = Self {
            props,
            tiles: None,
            single: None,
            loaded: false,
            dirty: true,
        };
        layer.rebuild();
        layer
    }

    pub fn props(&self) -> &TerrainLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: TerrainLayerProps) {
        if self.props == props {
            return;
        }
        self.props = props;
        self.rebuild();
    }

    /// Whether the elevation data names tiles rather than one image.
    pub fn is_tiled(&self) -> bool {
        self.props.elevation_data.iter().any(|url| is_url_template(url))
    }

    /// Whether every tile in view (or the single mesh) has finished loading.
    pub fn is_loaded(&self) -> bool {
        match &self.tiles {
            Some(tiles) => tiles.is_loaded(),
            None => self.loaded,
        }
    }

    /// The tileset of a tiled terrain, for tests and tools.
    pub fn tile_layer(&self) -> Option<&TileLayer> {
        self.tiles.as_ref()
    }

    fn rebuild(&mut self) {
        self.dirty = true;
        self.tiles = None;
        self.single = None;
        self.loaded = false;
        if self.is_tiled() {
            let props = &self.props;
            self.tiles = Some(TileLayer::new(TileLayerProps {
                base: props.base.clone(),
                get_tile_data: Some(terrain_loader(props)),
                render_sub_layers: terrain_renderer(props),
                tile_size: props.tile_size,
                min_zoom: props.min_zoom,
                max_zoom: props.max_zoom,
                max_requests: props.max_requests,
                ..Default::default()
            }));
        }
    }

    /// Load the single elevation image and build its mesh, once.
    fn load_single(&mut self) -> Result<()> {
        self.loaded = true;
        let props = &self.props;
        let (Some(url), Some(bounds)) = (props.elevation_data.first(), props.bounds) else {
            return Ok(());
        };
        let load = |url: &str| -> std::result::Result<BitmapImage, String> {
            let bytes = crate::tileset::fetch_bytes(url)?;
            let image = image::load_from_memory(&bytes)
                .map_err(|e| format!("{url}: {e}"))?
                .to_rgba8();
            Ok(BitmapImage {
                width: image.width(),
                height: image.height(),
                rgba: Arc::new(image.into_raw()),
            })
        };
        let elevation = match load(url) {
            Ok(image) => image,
            Err(message) => {
                tracing::warn!("terrain elevation `{url}` failed to load: {message}");
                return Ok(());
            }
        };
        let [x0, y0] = lng_lat_to_world([bounds[0], bounds[1]]);
        let [x1, y1] = lng_lat_to_world([bounds[2], bounds[3]]);
        let mesh = match terrain_mesh(
            &elevation,
            &props.elevation_decoder,
            [x0, y0, x1, y1],
            props.mesh_max_error,
        ) {
            Ok(mesh) => mesh,
            Err(message) => {
                tracing::warn!("terrain `{url}`: {message}");
                return Ok(());
            }
        };
        let texture = props.texture.first().and_then(|url| load(url).ok());
        self.single = Some(SimpleMeshLayer::new(SimpleMeshLayerProps {
            base: LayerProps {
                id: format!("{}-mesh", props.base.id),
                coordinate_system: deck_gl::CoordinateSystem::Cartesian,
                ..props.base.clone()
            },
            data: LayerData::with_length(1),
            mesh: Some(Arc::new(mesh)),
            texture,
            wireframe: props.wireframe,
            instanced: false,
            get_position: Accessor::Constant([0.0, 0.0, 0.0]),
            get_color: Accessor::Constant(props.color),
            ..Default::default()
        }));
        Ok(())
    }
}

impl Layer for TerrainLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if let Some(tiles) = self.tiles.as_mut() {
            return tiles.update(ctx, viewport);
        }
        if !self.loaded {
            self.load_single()?;
        }
        if let Some(single) = self.single.as_mut() {
            if self.dirty {
                single.initialize(ctx)?;
                self.dirty = false;
            }
            single.update(ctx, viewport)?;
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(tiles) = self.tiles.as_mut() {
            return tiles.draw(ctx, pass);
        }
        if let Some(single) = self.single.as_mut() {
            single.draw(ctx, pass)?;
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        if let Some(tiles) = self.tiles.as_mut() {
            return tiles.set_picking_active(ctx, active);
        }
        if let Some(single) = self.single.as_mut() {
            single.set_picking_active(ctx, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(tiles) = self.tiles.as_mut() {
            return tiles.draw_picking(ctx, pass);
        }
        if let Some(single) = self.single.as_mut() {
            single.draw_picking(ctx, pass)?;
        }
        Ok(())
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
        if let Some(tiles) = self.tiles.as_mut() {
            tiles.set_highlighted_object(index);
        }
        if let Some(single) = self.single.as_mut() {
            single.set_highlighted_object(index);
        }
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

impl Default for TerrainLayer {
    fn default() -> Self {
        Self::new(TerrainLayerProps::default())
    }
}

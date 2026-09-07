//! Port of `@deck.gl/geo-layers/src/tile-3d-layer/tile-3d-layer.ts`: a 3D Tiles tileset,
//! traversed for the view and drawn tile by tile.
//!
//! The tileset and its content load in the background through the shared
//! [`Fetcher`](crate::Fetcher). Positions in 3D Tiles are earth centred metres; each tile's
//! geometry is placed as metre offsets around the tileset's origin, which keeps the numbers
//! small enough for the shaders. `b3dm` and plain glTF content are drawn; the point cloud and
//! composite containers are not read yet.

use std::collections::HashMap;
use std::sync::Arc;

use deck_gl::glam::{DMat4, DVec3, Mat4};
use deck_gl::{
    Accessor, CoordinateSystem, Layer, LayerContext, LayerData, LayerProps, Result, SubLayers, Viewport,
};

use crate::fetch::{FetchHandle, Fetcher};
use crate::scenegraph::Scenegraph;
use crate::scenegraph_layer::{ScenegraphLayer, ScenegraphLayerProps, ScenegraphLighting};
use crate::tiles3d::{
    cartographic_to_cartesian, parse_tile_content, parse_tileset, select_tiles, tile_to_meter_offsets,
    SelectedTile, Tileset3D, TraversalView,
};

/// Properties of a [`Tile3DLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile3DLayerProps {
    pub base: LayerProps,
    /// URL of the tileset's `tileset.json`
    pub data: String,
    /// deck.gl's `maximumScreenSpaceError`: how many pixels of error a tile may show before
    /// it is refined into its children
    pub maximum_screen_space_error: f64,
    /// Multiplied with the material colours of the tiles
    pub color: deck_gl::Color,
    pub lighting: ScenegraphLighting,
}

impl Default for Tile3DLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("Tile3DLayer"),
            data: String::new(),
            maximum_screen_space_error: 16.0,
            color: [255, 255, 255, 255],
            lighting: ScenegraphLighting::Flat,
        }
    }
}

/// Renders a 3D Tiles tileset.
pub struct Tile3DLayer {
    props: Tile3DLayerProps,
    /// The tileset once its JSON arrived
    tileset: Option<Tileset3D>,
    /// Longitude, latitude and height every tile is placed around
    origin: DVec3,
    /// The request for the tileset JSON
    tileset_request: Option<FetchHandle>,
    /// Requests for tile content, by URL
    requests: HashMap<String, FetchHandle>,
    /// Content that arrived and parsed, by URL
    contents: HashMap<String, Arc<Scenegraph>>,
    /// URLs that failed or hold something we cannot read
    failed: HashMap<String, String>,
    /// The tiles the last traversal picked
    selected: Vec<SelectedTile>,
    sub_layers: SubLayers,
    dirty: bool,
}

impl Tile3DLayer {
    pub fn new(props: Tile3DLayerProps) -> Self {
        Self {
            props,
            tileset: None,
            origin: DVec3::ZERO,
            tileset_request: None,
            requests: HashMap::new(),
            contents: HashMap::new(),
            failed: HashMap::new(),
            selected: Vec::new(),
            sub_layers: SubLayers::new(),
            dirty: true,
        }
    }

    pub fn props(&self) -> &Tile3DLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: Tile3DLayerProps) {
        if self.props == props {
            return;
        }
        if self.props.data != props.data {
            // A different tileset: everything about the old one goes
            self.tileset = None;
            self.tileset_request = None;
            self.requests.clear();
            self.contents.clear();
            self.failed.clear();
            self.selected.clear();
        }
        self.props = props;
        self.dirty = true;
    }

    /// The tileset, once it has loaded.
    pub fn tileset(&self) -> Option<&Tileset3D> {
        self.tileset.as_ref()
    }

    /// The tiles picked for the last view, whether or not their content has arrived.
    pub fn selected(&self) -> &[SelectedTile] {
        &self.selected
    }

    /// Whether every tile picked for the view has its content.
    pub fn is_loaded(&self) -> bool {
        self.tileset.is_some()
            && self.selected.iter().all(|tile| {
                let url = self.content_url(&tile.content_uri);
                self.contents.contains_key(&url) || self.failed.contains_key(&url)
            })
    }

    /// A tile's content URL, resolved against the tileset's.
    fn content_url(&self, uri: &str) -> String {
        if uri.contains("://") || uri.starts_with('/') {
            return uri.to_string();
        }
        match self.props.data.rfind('/') {
            Some(slash) => format!("{}{uri}", &self.props.data[..=slash]),
            None => uri.to_string(),
        }
    }

    /// Take in the tileset JSON once it arrives.
    fn poll_tileset(&mut self) {
        if self.tileset.is_some() || self.props.data.is_empty() {
            return;
        }
        let request = self
            .tileset_request
            .get_or_insert_with(|| Fetcher::global().fetch(&self.props.data));
        let Some(result) = request.result() else {
            return;
        };
        self.tileset_request = None;
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(message) => {
                tracing::warn!("tileset `{}` failed to load: {message}", self.props.data);
                self.failed.insert(self.props.data.clone(), message);
                return;
            }
        };
        let parsed = serde_json::from_slice(&bytes)
            .map_err(|e| e.to_string())
            .and_then(|json| parse_tileset(&json));
        match parsed {
            Ok(tileset) => {
                // Every tile is placed around the centre of the whole tileset
                self.origin = tileset
                    .root
                    .bounding_volume
                    .cartographic_center(&tileset.root.transform);
                self.tileset = Some(tileset);
                self.dirty = true;
            }
            Err(message) => {
                tracing::warn!("tileset `{}`: {message}", self.props.data);
                self.failed.insert(self.props.data.clone(), message);
            }
        }
    }

    /// Ask for the content of the tiles in view, and take in what arrived.
    fn poll_content(&mut self) {
        let urls: Vec<String> = self
            .selected
            .iter()
            .map(|tile| self.content_url(&tile.content_uri))
            .collect();
        for (tile, url) in self.selected.clone().iter().zip(urls) {
            if self.contents.contains_key(&url) || self.failed.contains_key(&url) {
                continue;
            }
            let request = self
                .requests
                .entry(url.clone())
                .or_insert_with(|| Fetcher::global().fetch(&url));
            let Some(result) = request.result() else {
                continue;
            };
            self.requests.remove(&url);
            match result.map_err(|e| e.to_string()).and_then(|bytes| {
                let content = parse_tile_content(&bytes)?;
                let mut scene = Scenegraph::from_gltf(&content.gltf)?;
                // Place the tile's geometry as metre offsets around the tileset's origin
                let matrix = tile_to_meter_offsets(&tile.transform, content.rtc_center, self.origin);
                for primitive in &mut scene.primitives {
                    let placed = matrix
                        * DMat4::from_cols_array(&primitive.model_matrix.to_cols_array().map(f64::from));
                    primitive.model_matrix = Mat4::from_cols_array(&placed.to_cols_array().map(|v| v as f32));
                }
                Ok(scene)
            }) {
                Ok(scene) => {
                    self.contents.insert(url, Arc::new(scene));
                    self.dirty = true;
                }
                Err(message) => {
                    tracing::warn!("3D tile `{url}`: {message}");
                    self.failed.insert(url, message);
                    self.dirty = true;
                }
            }
        }
    }

    /// How the traversal sees the current view.
    fn traversal_view(&self, viewport: &Viewport) -> TraversalView {
        let camera = viewport.unproject_position(viewport.camera_position);
        TraversalView {
            camera: cartographic_to_cartesian(camera.x, camera.y, camera.z),
            height: viewport.height.max(1.0),
            fovy: viewport.fovy.to_radians().max(1e-3),
            max_error: self.props.maximum_screen_space_error.max(1.0),
        }
    }

    fn render_layers(&self) -> Vec<Box<dyn Layer>> {
        let props = &self.props;
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        for (index, tile) in self.selected.iter().enumerate() {
            let url = self.content_url(&tile.content_uri);
            let Some(scene) = self.contents.get(&url) else {
                continue;
            };
            layers.push(Box::new(ScenegraphLayer::new(ScenegraphLayerProps {
                base: LayerProps {
                    id: format!("{}-tile-{index}", props.base.id),
                    // The geometry is metres around the tileset's origin
                    coordinate_system: CoordinateSystem::MeterOffsets,
                    coordinate_origin: [self.origin.x, self.origin.y, self.origin.z],
                    ..props.base.clone()
                },
                data: LayerData::with_length(1),
                scenegraph: Some(scene.clone()),
                lighting: props.lighting,
                get_position: Accessor::Constant([0.0, 0.0, 0.0]),
                get_color: Accessor::Constant(props.color),
                ..Default::default()
            })));
        }
        layers
    }
}

impl Layer for Tile3DLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if ctx.uniform_slot == 0 {
            self.poll_tileset();
            if let Some(tileset) = &self.tileset {
                let selected = select_tiles(tileset, &self.traversal_view(viewport));
                if selected != self.selected {
                    self.selected = selected;
                    self.dirty = true;
                }
                self.poll_content();
            }
            if self.dirty {
                let layers = self.render_layers();
                self.sub_layers.replace(layers);
                self.dirty = false;
            }
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

impl Default for Tile3DLayer {
    fn default() -> Self {
        Self::new(Tile3DLayerProps::default())
    }
}

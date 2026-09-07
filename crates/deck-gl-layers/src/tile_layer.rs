//! Port of `@deck.gl/geo-layers/src/tile-layer/tile-layer.ts`: loads tiles for the visible area
//! in background threads and draws each one through sub layers.
//!
//! Give the layer a [`TileLoader`] that returns the content of a tile (any `Send + Sync` value)
//! and a [`TileRenderer`] that turns a loaded tile into layers; [`TileLayer::raster`] does both
//! for image tiles from a URL template. Loads run on a small thread pool and finished tiles are
//! picked up on the next update, so frames never block on the network.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use deck_gl::{Layer, LayerContext, LayerProps, Result, SubLayers, Viewport};

use crate::tileset::{
    RefinementStrategy, TileBounds, TileIndex, TileStatus, Tileset, TilesetOptions, TILE_SIZE,
};
use crate::{BitmapImage, BitmapLayer, BitmapLayerProps};

/// Content of a loaded tile, whatever the loader produced.
pub type TileData = Arc<dyn Any + Send + Sync>;

/// A loaded tile handed to the renderer.
#[derive(Clone, Debug)]
pub struct LoadedTile {
    pub index: TileIndex,
    pub bounds: TileBounds,
    pub data: TileData,
}

impl LoadedTile {
    /// The content as the type the loader produced.
    pub fn data<T: 'static>(&self) -> Option<&T> {
        self.data.downcast_ref::<T>()
    }

    /// `"<layer id>-<x>-<y>-<z>"` style ids for sub layers.
    pub fn sub_layer_id(&self, layer_id: &str, suffix: &str) -> String {
        format!("{layer_id}-{}-{suffix}", self.index.id())
    }
}

/// Loads a tile's content, deck.gl's `getTileData`. Runs on a worker thread. `Ok(None)` is an
/// empty tile (nothing to draw), `Err` a failed load.
#[derive(Clone)]
pub struct TileLoader(pub Arc<TileLoaderFn>);
pub type TileLoaderFn =
    dyn Fn(TileIndex, TileBounds) -> std::result::Result<Option<TileData>, String> + Send + Sync;

impl TileLoader {
    pub fn new(
        f: impl Fn(TileIndex, TileBounds) -> std::result::Result<Option<TileData>, String> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(f))
    }
}

impl PartialEq for TileLoader {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for TileLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TileLoader")
    }
}

/// Turns a loaded tile into layers, deck.gl's `renderSubLayers`.
#[derive(Clone)]
pub struct TileRenderer(pub Arc<TileRendererFn>);
pub type TileRendererFn = dyn Fn(&LoadedTile, &LayerProps) -> Vec<Box<dyn Layer>> + Send + Sync;

impl TileRenderer {
    pub fn new(f: impl Fn(&LoadedTile, &LayerProps) -> Vec<Box<dyn Layer>> + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl PartialEq for TileRenderer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for TileRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TileRenderer")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TileLayerProps {
    pub base: LayerProps,
    pub get_tile_data: Option<TileLoader>,
    pub render_sub_layers: TileRenderer,
    /// Pixel size of a tile at its zoom; 256 loads one zoom level higher than 512
    pub tile_size: f64,
    pub min_zoom: Option<u32>,
    pub max_zoom: Option<u32>,
    pub zoom_offset: f64,
    /// `[west, south, east, north]` (or cartesian `[min_x, min_y, max_x, max_y]`) limits
    pub extent: Option<[f64; 4]>,
    pub refinement_strategy: RefinementStrategy,
    pub max_cache_size: Option<usize>,
    /// Concurrent loads
    pub max_requests: usize,
}

impl Default for TileLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("tiles"),
            get_tile_data: None,
            render_sub_layers: raster_renderer(),
            tile_size: TILE_SIZE,
            min_zoom: Some(0),
            max_zoom: None,
            zoom_offset: 0.0,
            extent: None,
            refinement_strategy: RefinementStrategy::BestAvailable,
            max_cache_size: None,
            max_requests: 6,
        }
    }
}

/// Draw tiles whose content is a [`BitmapImage`] with a `BitmapLayer` each.
pub fn raster_renderer() -> TileRenderer {
    TileRenderer::new(|tile, base| {
        let Some(image) = tile.data::<BitmapImage>() else {
            return Vec::new();
        };
        vec![Box::new(BitmapLayer::new(BitmapLayerProps {
            base: LayerProps {
                id: tile.sub_layer_id(&base.id, "bitmap"),
                ..base.clone()
            },
            image: Some(image.clone()),
            bounds: tile.bounds.as_array(),
            ..Default::default()
        })) as Box<dyn Layer>]
    })
}

/// Tiles waiting for a worker, with the condition variable that wakes the workers.
type LoadQueue = Arc<(Mutex<VecDeque<(TileIndex, TileBounds)>>, Condvar)>;

/// A small thread pool: tiles queue up and `max_requests` workers load them.
struct LoadPool {
    queue: LoadQueue,
    results: Receiver<(TileIndex, std::result::Result<Option<TileData>, String>)>,
    /// Generation of the loader; results from older loaders are dropped
    generation: u64,
}

impl LoadPool {
    fn new(loader: TileLoader, workers: usize, generation: u64) -> Self {
        let queue: LoadQueue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (tx, results) = channel();
        for _ in 0..workers.max(1) {
            let queue = queue.clone();
            let tx: Sender<(TileIndex, std::result::Result<Option<TileData>, String>)> = tx.clone();
            let loader = loader.clone();
            std::thread::Builder::new()
                .name("deck-gl tile loader".into())
                .spawn(move || loop {
                    let job = {
                        let (lock, cvar) = &*queue;
                        let mut q = lock.lock().unwrap();
                        loop {
                            if let Some(job) = q.pop_front() {
                                break Some(job);
                            }
                            // Stop once the layer dropped the queue (only workers hold it)
                            if Arc::strong_count(&queue) <= workers.max(1) {
                                break None;
                            }
                            q = cvar.wait(q).unwrap();
                        }
                    };
                    let Some((index, bounds)) = job else { return };
                    let result = (loader.0)(index, bounds);
                    if tx.send((index, result)).is_err() {
                        return;
                    }
                })
                .expect("spawn tile loader");
        }
        Self {
            queue,
            results,
            generation,
        }
    }

    fn submit(&self, index: TileIndex, bounds: TileBounds) {
        let (lock, cvar) = &*self.queue;
        lock.lock().unwrap().push_back((index, bounds));
        cvar.notify_one();
    }
}

impl Drop for LoadPool {
    fn drop(&mut self) {
        // Wake the workers so they notice the queue is gone
        let (_, cvar) = &*self.queue;
        cvar.notify_all();
    }
}

pub struct TileLayer {
    props: TileLayerProps,
    tileset: Tileset,
    pool: Option<LoadPool>,
    generation: u64,
    contents: HashMap<TileIndex, TileData>,
    sub_layers: HashMap<TileIndex, SubLayers>,
    visible: Vec<TileIndex>,
    dirty: bool,
}

impl TileLayer {
    pub fn new(props: TileLayerProps) -> Self {
        let tileset = Tileset::new(Self::tileset_options(&props));
        Self {
            props,
            tileset,
            pool: None,
            generation: 0,
            contents: HashMap::new(),
            sub_layers: HashMap::new(),
            visible: Vec::new(),
            dirty: true,
        }
    }

    /// Image tiles from a URL template such as `https://tile.openstreetmap.org/{z}/{x}/{y}.png`
    /// (several templates spread the load), drawn with bitmap layers. Needs the `fetch` feature.
    #[cfg(feature = "fetch")]
    pub fn raster(id: impl Into<String>, templates: Vec<String>) -> Self {
        Self::new(TileLayerProps {
            base: LayerProps::new(id),
            get_tile_data: Some(raster_loader(templates)),
            ..Default::default()
        })
    }

    fn tileset_options(props: &TileLayerProps) -> TilesetOptions {
        TilesetOptions {
            tile_size: props.tile_size,
            min_zoom: props.min_zoom,
            max_zoom: props.max_zoom,
            zoom_offset: props.zoom_offset,
            extent: props.extent,
            z_range: None,
            refinement_strategy: props.refinement_strategy,
            max_cache_size: props.max_cache_size,
        }
    }

    pub fn props(&self) -> &TileLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: TileLayerProps) {
        if self.props == props {
            return;
        }
        let reload = self.props.get_tile_data != props.get_tile_data
            || self.props.tile_size != props.tile_size
            || self.props.extent != props.extent;
        self.props = props;
        self.tileset.opts = Self::tileset_options(&self.props);
        if reload {
            self.generation += 1;
            self.pool = None;
            self.contents.clear();
            self.tileset = Tileset::new(self.tileset.opts.clone());
        }
        self.sub_layers.clear();
        self.dirty = true;
    }

    /// Whether every selected tile has finished loading.
    pub fn is_loaded(&self) -> bool {
        self.tileset.is_loaded()
    }

    pub fn tileset(&self) -> &Tileset {
        &self.tileset
    }

    /// Start loads for selected tiles and take in finished ones.
    fn poll_loads(&mut self) -> bool {
        let Some(loader) = self.props.get_tile_data.clone() else {
            return false;
        };
        if self.pool.is_none() {
            self.pool = Some(LoadPool::new(loader, self.props.max_requests, self.generation));
        }
        let pool = self.pool.as_mut().expect("pool");
        for index in self.tileset.pending() {
            if let Some(tile) = self.tileset.tile(index) {
                pool.submit(index, tile.bounds);
                self.tileset.set_status(index, TileStatus::Loading, false);
            }
        }
        let mut changed = false;
        while let Ok((index, result)) = pool.results.try_recv() {
            if pool.generation != self.generation {
                continue;
            }
            match result {
                Ok(Some(data)) => {
                    self.contents.insert(index, data);
                    self.tileset.set_status(index, TileStatus::Loaded, true);
                }
                Ok(None) => self.tileset.set_status(index, TileStatus::Loaded, false),
                Err(message) => {
                    eprintln!("deck.gl-native: tile {} failed to load: {message}", index.id());
                    self.tileset.set_status(index, TileStatus::Failed, false);
                }
            }
            changed = true;
        }
        changed
    }

    fn rebuild_sub_layers(&mut self) {
        self.visible = self.tileset.visible();
        let ids: std::collections::HashSet<TileIndex> = self.visible.iter().copied().collect();
        self.sub_layers.retain(|index, _| ids.contains(index));
        for &index in &self.visible {
            if self.sub_layers.contains_key(&index) {
                continue;
            }
            let (Some(data), Some(header)) = (self.contents.get(&index), self.tileset.tile(index)) else {
                continue;
            };
            let tile = LoadedTile {
                index,
                bounds: header.bounds,
                data: data.clone(),
            };
            let layers = (self.props.render_sub_layers.0)(&tile, &self.props.base);
            let mut sub = SubLayers::new();
            sub.replace(layers);
            self.sub_layers.insert(index, sub);
        }
    }
}

impl Layer for TileLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if ctx.uniform_slot == 0 {
            let selection_changed = self.tileset.update(viewport);
            let loads_changed = self.poll_loads();
            if loads_changed {
                self.tileset.update_visibility();
            }
            if selection_changed || loads_changed || self.dirty {
                self.rebuild_sub_layers();
                self.dirty = false;
            }
        }
        for index in &self.visible {
            if let Some(sub) = self.sub_layers.get_mut(index) {
                sub.update(ctx, viewport)?;
            }
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for index in &self.visible {
            if let Some(sub) = self.sub_layers.get_mut(index) {
                sub.draw(ctx, pass)?;
            }
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for sub in self.sub_layers.values_mut() {
            sub.set_picking_active(ctx, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for index in &self.visible {
            if let Some(sub) = self.sub_layers.get_mut(index) {
                sub.draw_picking(ctx, pass)?;
            }
        }
        Ok(())
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
        for sub in self.sub_layers.values_mut() {
            sub.set_highlighted_object(index);
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

impl Default for TileLayer {
    fn default() -> Self {
        Self::new(TileLayerProps::default())
    }
}

/// A loader that fetches PNG or JPEG tiles from URL templates.
#[cfg(feature = "fetch")]
pub fn raster_loader(templates: Vec<String>) -> TileLoader {
    TileLoader::new(move |index, _bounds| {
        let Some(url) = crate::tileset::url_from_template(&templates, index) else {
            return Ok(None);
        };
        let bytes = crate::tileset::fetch_bytes(&url)?;
        let image = image::load_from_memory(&bytes)
            .map_err(|e| format!("{url}: {e}"))?
            .to_rgba8();
        Ok(Some(Arc::new(BitmapImage {
            width: image.width(),
            height: image.height(),
            rgba: Arc::new(image.into_raw()),
        }) as TileData))
    })
}

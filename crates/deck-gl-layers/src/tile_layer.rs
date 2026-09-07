//! Port of `@deck.gl/geo-layers/src/tile-layer/tile-layer.ts`: loads tiles for the visible area
//! in background threads and draws each one through sub layers.
//!
//! Give the layer a [`TileLoader`] that returns the content of a tile (any `Send + Sync` value)
//! and a [`TileRenderer`] that turns a loaded tile into layers; [`TileLayer::raster`] does both
//! for image tiles from a URL template. Loads run on a small thread pool and finished tiles are
//! picked up on the next update, so frames never block on the network.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use deck_gl::{Layer, LayerContext, LayerProps, Result, SubLayers, Viewport};

use crate::fetch::CancelToken;
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
/// empty tile (nothing to draw), `Err` a failed load. The [`CancelToken`] is set when the
/// tile stopped being needed; loaders that fetch should pass it on and give up early.
#[derive(Clone)]
pub struct TileLoader(pub Arc<TileLoaderFn>);
pub type TileLoaderFn = dyn Fn(TileIndex, TileBounds, &CancelToken) -> std::result::Result<Option<TileData>, String>
    + Send
    + Sync;

impl TileLoader {
    pub fn new(
        f: impl Fn(TileIndex, TileBounds) -> std::result::Result<Option<TileData>, String> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(move |index, bounds, _| f(index, bounds)))
    }

    /// A loader that is told when its tile is no longer wanted.
    pub fn cancellable(
        f: impl Fn(TileIndex, TileBounds, &CancelToken) -> std::result::Result<Option<TileData>, String>
            + Send
            + Sync
            + 'static,
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
type LoadQueue = Arc<(
    Mutex<VecDeque<(TileIndex, TileBounds, CancelToken)>>,
    Condvar,
    AtomicBool,
)>;
/// What a worker reports for a tile: `None` when the load was cancelled.
type LoadOutcome = Option<std::result::Result<Option<TileData>, String>>;

/// A small thread pool: tiles queue up and `max_requests` workers load them.
struct LoadPool {
    queue: LoadQueue,
    results: Receiver<(TileIndex, LoadOutcome)>,
    /// wasm32 has no worker threads, so the pool keeps the loader and runs it itself
    #[cfg(target_arch = "wasm32")]
    loader: TileLoader,
    #[cfg(target_arch = "wasm32")]
    sender: Sender<(TileIndex, LoadOutcome)>,
    /// Generation of the loader; results from older loaders are dropped
    generation: u64,
    /// Tokens of the tiles queued or loading, deck.gl's `AbortController` per tile
    tokens: HashMap<TileIndex, CancelToken>,
}

impl LoadPool {
    fn new(loader: TileLoader, workers: usize, generation: u64) -> Self {
        let queue: LoadQueue = Arc::new((
            Mutex::new(VecDeque::new()),
            Condvar::new(),
            AtomicBool::new(false),
        ));
        let (tx, results) = channel();
        #[cfg(target_arch = "wasm32")]
        let (pool_loader, pool_sender) = (loader.clone(), tx.clone());
        #[cfg(target_arch = "wasm32")]
        let _ = workers;
        #[cfg(not(target_arch = "wasm32"))]
        for _ in 0..workers.max(1) {
            let queue = queue.clone();
            let tx: Sender<(TileIndex, LoadOutcome)> = tx.clone();
            let loader = loader.clone();
            // A failed spawn leaves fewer workers; the queue still drains through the others
            std::thread::Builder::new()
                .name("deck-gl tile loader".into())
                .spawn(move || loop {
                    let job = {
                        let (lock, cvar, closed) = &*queue;
                        let mut q = lock.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = q.pop_front() {
                                break Some(job);
                            }
                            // Stop once the layer dropped the pool
                            if closed.load(Ordering::SeqCst) {
                                break None;
                            }
                            q = cvar.wait(q).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let Some((index, bounds, cancel)) = job else {
                        return;
                    };
                    let outcome = if cancel.is_cancelled() {
                        None
                    } else {
                        let result = (loader.0)(index, bounds, &cancel);
                        (!cancel.is_cancelled()).then_some(result)
                    };
                    if tx.send((index, outcome)).is_err() {
                        return;
                    }
                })
                .ok();
        }
        Self {
            queue,
            results,
            #[cfg(target_arch = "wasm32")]
            loader: pool_loader,
            #[cfg(target_arch = "wasm32")]
            sender: pool_sender,
            generation,
            tokens: HashMap::new(),
        }
    }

    /// wasm32: run the queued loads on the main thread. A load whose bytes have not arrived
    /// yet says so rather than failing, and stays queued for a later frame; the browser is
    /// meanwhile fetching them. Off the web the worker threads do this.
    #[cfg(target_arch = "wasm32")]
    fn pump(&mut self) {
        let (lock, _, _) = &*self.queue;
        let jobs: Vec<(TileIndex, TileBounds, CancelToken)> = {
            let mut queue = lock.lock().unwrap_or_else(|e| e.into_inner());
            queue.drain(..).collect()
        };
        let mut waiting = VecDeque::new();
        for (index, bounds, cancel) in jobs {
            if cancel.is_cancelled() {
                self.sender.send((index, None)).ok();
                continue;
            }
            match (self.loader.0)(index, bounds, &cancel) {
                Err(message) if crate::fetch::is_still_loading(&message) => {
                    waiting.push_back((index, bounds, cancel));
                }
                outcome => {
                    self.sender.send((index, Some(outcome))).ok();
                }
            }
        }
        let mut queue = lock.lock().unwrap_or_else(|e| e.into_inner());
        queue.extend(waiting);
    }

    fn submit(&mut self, index: TileIndex, bounds: TileBounds) {
        let cancel = CancelToken::new();
        self.tokens.insert(index, cancel.clone());
        let (lock, cvar, _) = &*self.queue;
        lock.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back((index, bounds, cancel));
        cvar.notify_one();
    }

    /// Stop loading a tile; the worker reports it as cancelled.
    fn cancel(&mut self, index: TileIndex) {
        if let Some(token) = self.tokens.remove(&index) {
            token.cancel();
        }
    }
}

impl Drop for LoadPool {
    fn drop(&mut self) {
        // Tell the workers to stop and wake them; loads in flight are dropped by generation
        let (lock, cvar, closed) = &*self.queue;
        let _queue = lock.lock().unwrap_or_else(|e| e.into_inner());
        closed.store(true, Ordering::SeqCst);
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
        let Some(pool) = self.pool.as_mut() else {
            return false;
        };
        for index in self.tileset.pending() {
            if let Some(tile) = self.tileset.tile(index) {
                pool.submit(index, tile.bounds);
                self.tileset.set_status(index, TileStatus::Loading, false);
            }
        }
        // deck.gl's `_pruneRequests`: with more loads on the way than `maxRequests`, give up
        // on the ones for tiles that are neither selected nor visible
        let max_requests = self.props.max_requests.max(1);
        if pool.tokens.len() > max_requests {
            let mut unneeded: Vec<TileIndex> = pool
                .tokens
                .keys()
                .copied()
                .filter(|index| {
                    self.tileset
                        .tile(*index)
                        .is_none_or(|tile| !tile.is_selected && !tile.is_visible)
                })
                .collect();
            unneeded.sort();
            for index in unneeded {
                if pool.tokens.len() <= max_requests {
                    break;
                }
                pool.cancel(index);
                self.tileset.set_status(index, TileStatus::Pending, false);
            }
        }
        #[cfg(target_arch = "wasm32")]
        pool.pump();
        let mut changed = false;
        while let Ok((index, outcome)) = pool.results.try_recv() {
            if pool.generation != self.generation {
                continue;
            }
            pool.tokens.remove(&index);
            match outcome {
                Some(Ok(Some(data))) => {
                    self.contents.insert(index, data);
                    self.tileset.set_status(index, TileStatus::Loaded, true);
                }
                Some(Ok(None)) => self.tileset.set_status(index, TileStatus::Loaded, false),
                Some(Err(message)) => {
                    tracing::warn!("tile {} failed to load: {message}", index.id());
                    self.tileset.set_status(index, TileStatus::Failed, false);
                }
                // Cancelled: back to pending, so it loads again if it gets selected
                None => {
                    if self
                        .tileset
                        .tile(index)
                        .is_some_and(|t| t.status == TileStatus::Loading)
                    {
                        self.tileset.set_status(index, TileStatus::Pending, false);
                    }
                    continue;
                }
            }
            changed = true;
        }
        changed
    }

    /// Tiles queued or loading right now.
    pub fn loading_count(&self) -> usize {
        self.pool.as_ref().map_or(0, |pool| pool.tokens.len())
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

/// A loader that fetches PNG or JPEG tiles from URL templates through the shared
/// [`Fetcher`](crate::fetch::Fetcher), giving up on tiles that stopped being needed.
pub fn raster_loader(templates: Vec<String>) -> TileLoader {
    TileLoader::cancellable(move |index, _bounds, cancel| {
        let Some(url) = crate::tileset::url_from_template(&templates, index) else {
            return Ok(None);
        };
        let bytes = crate::tileset::fetch_bytes_with(&url, cancel)?;
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

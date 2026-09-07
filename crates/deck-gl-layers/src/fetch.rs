//! Shared HTTP loading for data and tiles: a small thread pool, an in memory cache with a
//! byte budget, one request per URL however many layers ask for it, cancellation and
//! progress counters. deck.gl leans on the browser's `fetch` for all of this; native hosts
//! get a [`Fetcher`] instead, one per process through [`Fetcher::global`] or their own.
//!
//! A load is asked for with [`Fetcher::fetch`], which returns at once with a [`FetchHandle`].
//! The handle reports the status, hands out the bytes when they arrived and cancels the
//! request when nothing else wants it. The fetcher's [`generation`](Fetcher::generation)
//! moves every time a request finishes, so a frame loop can poll it cheaply and rebuild
//! what depended on the data.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

/// Cancels a load. Clones share the flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// The bytes of a URL, shared with the cache, or why they could not be loaded.
pub type FetchResult = std::result::Result<Arc<Vec<u8>>, String>;

/// Where a request is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchStatus {
    /// Waiting for a worker
    Queued,
    Loading,
    Done,
    Failed,
    Cancelled,
}

impl FetchStatus {
    /// Whether the request will not change any more.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

struct Request {
    url: String,
    state: Mutex<(FetchStatus, Option<FetchResult>)>,
    finished: Condvar,
    cancel: CancelToken,
    /// Handles that still want the result; the request is cancelled when it drops to zero
    interest: AtomicUsize,
}

impl Request {
    fn new(url: &str, status: FetchStatus, result: Option<FetchResult>) -> Arc<Self> {
        Arc::new(Self {
            url: url.to_string(),
            state: Mutex::new((status, result)),
            finished: Condvar::new(),
            cancel: CancelToken::new(),
            interest: AtomicUsize::new(1),
        })
    }

    fn lock(&self) -> MutexGuard<'_, (FetchStatus, Option<FetchResult>)> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn finish(&self, status: FetchStatus, result: FetchResult) {
        let mut state = self.lock();
        if !state.0.is_finished() {
            *state = (status, Some(result));
        }
        drop(state);
        self.finished.notify_all();
    }
}

/// A request handed out by [`Fetcher::fetch`].
#[derive(Clone)]
pub struct FetchHandle(Arc<Request>);

impl std::fmt::Debug for FetchHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FetchHandle")
            .field("url", &self.0.url)
            .field("status", &self.status())
            .finish()
    }
}

impl FetchHandle {
    pub fn url(&self) -> &str {
        &self.0.url
    }

    pub fn status(&self) -> FetchStatus {
        self.0.lock().0
    }

    /// The outcome, once the request finished.
    pub fn result(&self) -> Option<FetchResult> {
        self.0.lock().1.clone()
    }

    /// Block until the request finished.
    pub fn wait(&self) -> FetchResult {
        let mut state = self.0.lock();
        while !state.0.is_finished() {
            state = self.0.finished.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state
            .1
            .clone()
            .unwrap_or_else(|| Err(format!("{}: no result", self.0.url)))
    }

    /// Give up on the result. The request is cancelled when no other handle wants it: a
    /// queued request never starts, a loading one stops at the next chunk.
    pub fn cancel(&self) {
        if self.0.interest.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.cancel.cancel();
            let cancelled = {
                let mut state = self.0.lock();
                if state.0 == FetchStatus::Queued {
                    *state = (
                        FetchStatus::Cancelled,
                        Some(Err(format!("{}: cancelled", self.0.url))),
                    );
                    true
                } else {
                    false
                }
            };
            if cancelled {
                self.0.finished.notify_all();
            }
        }
    }
}

/// Progress counters of a [`Fetcher`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FetchStats {
    /// Requests waiting for a worker
    pub queued: usize,
    /// Requests being read
    pub loading: usize,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// Bytes received over the network
    pub bytes: u64,
    /// Requests answered from the cache
    pub cache_hits: u64,
    /// Bytes held by the cache
    pub cached_bytes: usize,
}

impl FetchStats {
    /// Requests that have not finished.
    pub fn pending(&self) -> usize {
        self.queued + self.loading
    }
}

#[derive(Default)]
struct Counters {
    completed: u64,
    failed: u64,
    cancelled: u64,
    bytes: u64,
    cache_hits: u64,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<String, (Arc<Vec<u8>>, u64)>,
    bytes: usize,
    tick: u64,
}

impl Cache {
    fn get(&mut self, url: &str) -> Option<Arc<Vec<u8>>> {
        self.tick += 1;
        let tick = self.tick;
        let (bytes, used) = self.entries.get_mut(url)?;
        *used = tick;
        Some(bytes.clone())
    }

    fn insert(&mut self, url: &str, bytes: Arc<Vec<u8>>, limit: usize) {
        if bytes.len() > limit {
            return;
        }
        if let Some((old, _)) = self.entries.remove(url) {
            self.bytes -= old.len();
        }
        while self.bytes + bytes.len() > limit {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(url, _)| url.clone())
            else {
                break;
            };
            if let Some((old, _)) = self.entries.remove(&oldest) {
                self.bytes -= old.len();
            }
        }
        self.tick += 1;
        self.bytes += bytes.len();
        self.entries.insert(url.to_string(), (bytes, self.tick));
    }
}

struct Inner {
    queue: Mutex<VecDeque<Arc<Request>>>,
    wake: Condvar,
    /// Queued or loading requests by URL, so a URL is fetched once
    inflight: Mutex<HashMap<String, Arc<Request>>>,
    cache: Mutex<Cache>,
    counters: Mutex<Counters>,
    loading: AtomicUsize,
    generation: AtomicU64,
    workers: usize,
    spawned: AtomicBool,
    cache_limit: usize,
    max_bytes: usize,
}

impl Inner {
    fn queue(&self) -> MutexGuard<'_, VecDeque<Arc<Request>>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn inflight(&self) -> MutexGuard<'_, HashMap<String, Arc<Request>>> {
        self.inflight.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn cache(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn counters(&self) -> MutexGuard<'_, Counters> {
        self.counters.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take the next request that was not cancelled while queued; `None` once the fetcher
    /// is gone.
    fn next(&self, inner: &Arc<Inner>) -> Option<Arc<Request>> {
        let mut queue = self.queue();
        loop {
            while let Some(request) = queue.pop_front() {
                if request.cancel.is_cancelled() {
                    drop(queue);
                    self.settle(
                        &request,
                        FetchStatus::Cancelled,
                        Err(format!("{}: cancelled", request.url)),
                    );
                    queue = self.queue();
                    continue;
                }
                return Some(request);
            }
            // Only workers hold the fetcher: stop
            if Arc::strong_count(inner) <= self.workers {
                return None;
            }
            queue = self.wake.wait(queue).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn settle(&self, request: &Request, status: FetchStatus, result: FetchResult) {
        self.inflight().remove(&request.url);
        {
            let mut counters = self.counters();
            match status {
                FetchStatus::Done => counters.completed += 1,
                FetchStatus::Failed => counters.failed += 1,
                _ => counters.cancelled += 1,
            }
            if let Ok(bytes) = &result {
                counters.bytes += bytes.len() as u64;
            }
        }
        request.finish(status, result);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    fn work(inner: Arc<Inner>) {
        while let Some(request) = inner.next(&inner) {
            {
                let mut state = request.lock();
                if state.0 == FetchStatus::Queued {
                    state.0 = FetchStatus::Loading;
                }
            }
            inner.loading.fetch_add(1, Ordering::SeqCst);
            let result = http_get(&request.url, &request.cancel, inner.max_bytes);
            inner.loading.fetch_sub(1, Ordering::SeqCst);
            match result {
                Ok(bytes) => {
                    let bytes = Arc::new(bytes);
                    inner
                        .cache()
                        .insert(&request.url, bytes.clone(), inner.cache_limit);
                    inner.settle(&request, FetchStatus::Done, Ok(bytes));
                }
                Err(message) if request.cancel.is_cancelled() => {
                    inner.settle(&request, FetchStatus::Cancelled, Err(message))
                }
                Err(message) => inner.settle(&request, FetchStatus::Failed, Err(message)),
            }
        }
    }
}

/// Loads URLs on worker threads and keeps what it loaded. Clones share the pool and cache.
#[derive(Clone)]
pub struct Fetcher {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Fetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fetcher").field("stats", &self.stats()).finish()
    }
}

impl Default for Fetcher {
    fn default() -> Self {
        Self::new(DEFAULT_WORKERS)
    }
}

/// Workers of [`Fetcher::global`] and [`Fetcher::default`], the connection limit browsers
/// apply per host.
pub const DEFAULT_WORKERS: usize = 6;
/// Cache budget of [`Fetcher::global`] and [`Fetcher::default`].
pub const DEFAULT_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Responses above this size fail instead of exhausting memory.
pub const MAX_RESPONSE_BYTES: usize = 512 * 1024 * 1024;

impl Fetcher {
    /// A fetcher with `workers` threads, started when the first request comes in.
    pub fn new(workers: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                queue: Mutex::new(VecDeque::new()),
                wake: Condvar::new(),
                inflight: Mutex::new(HashMap::new()),
                cache: Mutex::new(Cache::default()),
                counters: Mutex::new(Counters::default()),
                loading: AtomicUsize::new(0),
                generation: AtomicU64::new(0),
                workers: workers.max(1),
                spawned: AtomicBool::new(false),
                cache_limit: DEFAULT_CACHE_BYTES,
                max_bytes: MAX_RESPONSE_BYTES,
            }),
        }
    }

    /// Keep at most `bytes` of responses. Only valid before the first request.
    pub fn with_cache_limit(mut self, bytes: usize) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.cache_limit = bytes;
        }
        self
    }

    /// The fetcher shared by the tile loaders and the JSON converter.
    pub fn global() -> &'static Fetcher {
        static GLOBAL: OnceLock<Fetcher> = OnceLock::new();
        GLOBAL.get_or_init(Fetcher::default)
    }

    /// Ask for a URL. Cached bytes come back finished at once; a URL already on its way is
    /// shared with the earlier request.
    pub fn fetch(&self, url: &str) -> FetchHandle {
        if let Some(bytes) = self.inner.cache().get(url) {
            self.inner.counters().cache_hits += 1;
            return FetchHandle(Request::new(url, FetchStatus::Done, Some(Ok(bytes))));
        }
        let mut inflight = self.inner.inflight();
        if let Some(request) = inflight.get(url) {
            request.interest.fetch_add(1, Ordering::SeqCst);
            return FetchHandle(request.clone());
        }
        let request = Request::new(url, FetchStatus::Queued, None);
        inflight.insert(url.to_string(), request.clone());
        drop(inflight);
        self.spawn_workers();
        self.inner.queue().push_back(request.clone());
        self.inner.wake.notify_one();
        FetchHandle(request)
    }

    /// Ask for a URL and block until it arrived. Loads through the pool, so other requests
    /// go on meanwhile and the bytes end up in the cache.
    pub fn fetch_blocking(&self, url: &str) -> FetchResult {
        self.fetch(url).wait()
    }

    /// Like [`fetch_blocking`](Self::fetch_blocking), giving up when `cancel` is set.
    pub fn fetch_blocking_with(&self, url: &str, cancel: &CancelToken) -> FetchResult {
        let handle = self.fetch(url);
        let mut state = handle.0.lock();
        while !state.0.is_finished() {
            if cancel.is_cancelled() {
                drop(state);
                handle.cancel();
                return Err(format!("{url}: cancelled"));
            }
            let (next, _) = handle
                .0
                .finished
                .wait_timeout(state, std::time::Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }
        state
            .1
            .clone()
            .unwrap_or_else(|| Err(format!("{url}: no result")))
    }

    /// The cached bytes of a URL, without requesting it.
    pub fn cached(&self, url: &str) -> Option<Arc<Vec<u8>>> {
        self.inner.cache().get(url)
    }

    /// Put bytes in the cache, for example a response read some other way.
    pub fn insert(&self, url: &str, bytes: Vec<u8>) {
        let limit = self.inner.cache_limit;
        self.inner.cache().insert(url, Arc::new(bytes), limit);
    }

    /// Drop everything cached.
    pub fn clear_cache(&self) {
        *self.inner.cache() = Cache::default();
    }

    pub fn stats(&self) -> FetchStats {
        let counters = self.inner.counters();
        FetchStats {
            queued: self.inner.queue().len(),
            loading: self.inner.loading.load(Ordering::SeqCst),
            completed: counters.completed,
            failed: counters.failed,
            cancelled: counters.cancelled,
            bytes: counters.bytes,
            cache_hits: counters.cache_hits,
            cached_bytes: self.inner.cache().bytes,
        }
    }

    /// Whether no request is queued or loading.
    pub fn is_idle(&self) -> bool {
        self.inner.inflight().is_empty()
    }

    /// A counter that moves every time a request finishes. Poll it once per frame and
    /// rebuild what waited for data when it changed.
    pub fn generation(&self) -> u64 {
        self.inner.generation.load(Ordering::SeqCst)
    }

    /// Block until every request finished, for tools and tests.
    pub fn wait_idle(&self) {
        let requests: Vec<Arc<Request>> = self.inner.inflight().values().cloned().collect();
        for request in requests {
            FetchHandle(request).wait().ok();
        }
    }

    fn spawn_workers(&self) {
        if self.inner.spawned.swap(true, Ordering::SeqCst) {
            return;
        }
        for _ in 0..self.inner.workers {
            let inner = self.inner.clone();
            // A failed spawn leaves fewer workers; the queue still drains through the others
            std::thread::Builder::new()
                .name("deck-gl fetch".into())
                .spawn(move || Inner::work(inner))
                .ok();
        }
    }
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        // Wake the workers so they notice when only they hold the pool
        self.inner.wake.notify_all();
    }
}

/// GET a URL, reading in chunks so a cancelled request stops early.
#[cfg(feature = "fetch")]
fn http_get(url: &str, cancel: &CancelToken, max_bytes: usize) -> std::result::Result<Vec<u8>, String> {
    use std::io::Read;
    if cancel.is_cancelled() {
        return Err(format!("{url}: cancelled"));
    }
    let response = ureq::get(url)
        .header(
            "User-Agent",
            concat!("deck.gl-native/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|e| format!("{url}: {e}"))?;
    if response.status().as_u16() >= 400 {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let expected = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if expected > max_bytes {
        return Err(format!(
            "{url}: {expected} bytes is more than the {max_bytes} byte limit"
        ));
    }
    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::with_capacity(expected.min(max_bytes));
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        if cancel.is_cancelled() {
            return Err(format!("{url}: cancelled"));
        }
        let read = reader.read(&mut chunk).map_err(|e| format!("{url}: {e}"))?;
        if read == 0 {
            break;
        }
        if bytes.len() + read > max_bytes {
            return Err(format!("{url}: more than the {max_bytes} byte limit"));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    Ok(bytes)
}

#[cfg(not(feature = "fetch"))]
fn http_get(url: &str, _cancel: &CancelToken, _max_bytes: usize) -> std::result::Result<Vec<u8>, String> {
    Err(format!(
        "{url}: deck-gl-layers was built without the `fetch` feature, so URLs cannot be loaded"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_evicts_least_recently_used() {
        let mut cache = Cache::default();
        cache.insert("a", Arc::new(vec![0; 100]), 250);
        cache.insert("b", Arc::new(vec![0; 100]), 250);
        assert!(cache.get("a").is_some(), "a is now the most recently used");
        cache.insert("c", Arc::new(vec![0; 100]), 250);
        assert!(cache.get("b").is_none(), "b was the least recently used");
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
        assert_eq!(cache.bytes, 200);
        cache.insert("d", Arc::new(vec![0; 300]), 250);
        assert!(cache.get("d").is_none(), "larger than the budget");
        cache.insert("a", Arc::new(vec![0; 50]), 250);
        assert_eq!(cache.bytes, 150, "replacing an entry accounts for the old size");
    }

    #[test]
    fn cancelling_a_queued_request_finishes_it() {
        let fetcher = Fetcher::new(1);
        // No worker ever runs this: the URL never resolves, but the request is only queued
        let request = Request::new("http://example.invalid/x", FetchStatus::Queued, None);
        fetcher
            .inner
            .inflight()
            .insert(request.url.clone(), request.clone());
        let a = FetchHandle(request.clone());
        let b = FetchHandle(request.clone());
        request.interest.fetch_add(1, Ordering::SeqCst);
        a.cancel();
        assert_eq!(b.status(), FetchStatus::Queued, "b still wants it");
        b.cancel();
        assert_eq!(b.status(), FetchStatus::Cancelled);
        assert!(b.wait().is_err());
    }
}

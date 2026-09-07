//! Port of `@deck.gl/geo-layers/src/wms-layer`: one image covering the visible area, requested
//! from a WMS `GetMap` endpoint (or any URL template) whenever the view settles, and drawn
//! with a `BitmapLayer`. Requests run on a thread; the previous image stays until the new one
//! arrives.

use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use web_time::Instant;

use deck_gl::{Layer, LayerContext, LayerProps, Result, SubLayers, Viewport};

use crate::{BitmapImage, BitmapLayer, BitmapLayerProps};

/// How the image URL is built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WmsServiceType {
    /// `data` is a WMS endpoint; a `GetMap` request is appended
    #[default]
    Wms,
    /// `data` is a URL template with `{west}`, `{south}`, `{east}`, `{north}`, `{bbox}`,
    /// `{width}`, `{height}`, `{layers}` and `{crs}`
    Template,
}

impl WmsServiceType {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "wms" | "auto" => Some(Self::Wms),
            "template" => Some(Self::Template),
            _ => None,
        }
    }
}

/// Coordinate reference system of the requested image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WmsSrs {
    /// Web Mercator, the natural fit for the map view: the image maps linearly onto it
    #[default]
    Epsg3857,
    /// Plate carree (linear in longitude and latitude)
    Epsg4326,
}

impl WmsSrs {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "EPSG:3857" | "auto" => Some(Self::Epsg3857),
            "EPSG:4326" => Some(Self::Epsg4326),
            _ => None,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::Epsg3857 => "EPSG:3857",
            Self::Epsg4326 => "EPSG:4326",
        }
    }
}

/// The function behind a [`WmsFetch`]: URL in, image bytes out.
pub type WmsFetchFn = dyn Fn(&str) -> std::result::Result<Vec<u8>, String> + Send + Sync;

/// Fetches the bytes of an image URL; the default uses HTTP (`fetch` feature).
#[derive(Clone)]
pub struct WmsFetch(pub Arc<WmsFetchFn>);

impl WmsFetch {
    pub fn new(f: impl Fn(&str) -> std::result::Result<Vec<u8>, String> + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl PartialEq for WmsFetch {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for WmsFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WmsFetch")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WmsLayerProps {
    pub base: LayerProps,
    /// The WMS endpoint or URL template
    pub data: String,
    pub service_type: WmsServiceType,
    /// WMS layer names, joined with commas
    pub layers: Vec<String>,
    pub srs: WmsSrs,
    /// Image format requested from a WMS, `image/png` by default
    pub format: String,
    pub transparent: bool,
    /// Milliseconds the view must stay still before a new image is requested
    pub debounce_ms: f64,
    /// Custom fetcher (tests, offline sources); HTTP by default
    pub fetch: Option<WmsFetch>,
}

impl Default for WmsLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("wms"),
            data: String::new(),
            service_type: WmsServiceType::Wms,
            layers: Vec::new(),
            srs: WmsSrs::Epsg3857,
            format: "image/png".to_string(),
            transparent: true,
            debounce_ms: 250.0,
            fetch: None,
        }
    }
}

/// Longitude and latitude to Web Mercator meters.
pub fn lng_lat_to_mercator_meters(lng: f64, lat: f64) -> [f64; 2] {
    let r = 6378137.0;
    let lat = lat.clamp(-85.051129, 85.051129);
    [
        lng.to_radians() * r,
        ((std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan()).ln() * r,
    ]
}

/// The URL of the image for `bounds` (`[west, south, east, north]`) at `width` x `height`.
pub fn image_url(props: &WmsLayerProps, bounds: [f64; 4], width: u32, height: u32) -> String {
    let [west, south, east, north] = bounds;
    let bbox = match props.srs {
        // WMS 1.3.0 lists EPSG:4326 as latitude, longitude
        WmsSrs::Epsg4326 => format!("{south},{west},{north},{east}"),
        WmsSrs::Epsg3857 => {
            // Millimetre precision keeps the URL short and drops noise such as -0.0000000007
            let mm = |v: f64| (v * 1000.0).round() / 1000.0 + 0.0;
            let min = lng_lat_to_mercator_meters(west, south);
            let max = lng_lat_to_mercator_meters(east, north);
            format!("{},{},{},{}", mm(min[0]), mm(min[1]), mm(max[0]), mm(max[1]))
        }
    };
    let layers = props.layers.join(",");
    match props.service_type {
        WmsServiceType::Template => props
            .data
            .replace("{west}", &west.to_string())
            .replace("{south}", &south.to_string())
            .replace("{east}", &east.to_string())
            .replace("{north}", &north.to_string())
            .replace("{bbox}", &bbox)
            .replace("{width}", &width.to_string())
            .replace("{height}", &height.to_string())
            .replace("{layers}", &layers)
            .replace("{crs}", props.srs.code()),
        WmsServiceType::Wms => {
            let separator = if props.data.contains('?') { '&' } else { '?' };
            format!(
                "{}{separator}SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={}&STYLES=&CRS={}&BBOX={bbox}&WIDTH={width}&HEIGHT={height}&FORMAT={}&TRANSPARENT={}",
                props.data,
                layers,
                props.srs.code(),
                props.format,
                if props.transparent { "TRUE" } else { "FALSE" }
            )
        }
    }
}

struct Request {
    id: u64,
    bounds: [f64; 4],
    receiver: Receiver<std::result::Result<BitmapImage, String>>,
}

pub struct WmsLayer {
    props: WmsLayerProps,
    sub_layers: SubLayers,
    /// Bounds and size of the last request started
    requested: Option<([f64; 4], u32, u32)>,
    /// When the view last moved to bounds that were not requested yet
    changed_at: Option<Instant>,
    pending: Option<Request>,
    next_request: u64,
    shown_request: u64,
}

impl WmsLayer {
    pub fn new(props: WmsLayerProps) -> Self {
        Self {
            props,
            sub_layers: SubLayers::new(),
            requested: None,
            changed_at: None,
            pending: None,
            next_request: 1,
            shown_request: 0,
        }
    }

    pub fn props(&self) -> &WmsLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: WmsLayerProps) {
        if self.props != props {
            self.props = props;
            self.requested = None;
            self.changed_at = None;
            self.pending = None;
        }
    }

    /// Whether an image request is in flight.
    pub fn is_loading(&self) -> bool {
        self.pending.is_some()
    }

    fn fetcher(&self) -> WmsFetch {
        match &self.props.fetch {
            Some(fetch) => fetch.clone(),
            None => WmsFetch::new(default_fetch),
        }
    }

    fn start_request(&mut self, bounds: [f64; 4], width: u32, height: u32) {
        let url = image_url(&self.props, bounds, width, height);
        let fetch = self.fetcher();
        let (sender, receiver) = channel();
        let id = self.next_request;
        self.next_request += 1;
        let spawned = std::thread::Builder::new()
            .name("deck-gl wms".into())
            .spawn(move || {
                let result = (fetch.0)(&url).and_then(|bytes| {
                    let image = image::load_from_memory(&bytes)
                        .map_err(|e| format!("{url}: {e}"))?
                        .to_rgba8();
                    Ok(BitmapImage {
                        width: image.width(),
                        height: image.height(),
                        rgba: Arc::new(image.into_raw()),
                    })
                });
                let _ = sender.send(result);
            })
            .is_ok();
        if spawned {
            self.pending = Some(Request { id, bounds, receiver });
            self.requested = Some((bounds, width, height));
        }
    }

    fn poll(&mut self) {
        let Some(request) = &self.pending else { return };
        match request.receiver.try_recv() {
            Ok(Ok(image)) => {
                if request.id > self.shown_request {
                    self.shown_request = request.id;
                    let layer = BitmapLayer::new(BitmapLayerProps {
                        base: LayerProps {
                            id: format!("{}-image", self.props.base.id),
                            ..self.props.base.clone()
                        },
                        image: Some(image),
                        bounds: request.bounds,
                        ..Default::default()
                    });
                    self.sub_layers.replace(vec![Box::new(layer)]);
                }
                self.pending = None;
            }
            Ok(Err(message)) => {
                tracing::warn!("WMS image failed: {message}");
                self.pending = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
        }
    }
}

fn default_fetch(url: &str) -> std::result::Result<Vec<u8>, String> {
    crate::tileset::fetch_bytes(url).map(|bytes| (*bytes).clone())
}

fn bounds_differ(a: [f64; 4], b: [f64; 4]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9)
}

impl Layer for WmsLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if ctx.uniform_slot == 0 {
            let b = viewport.get_bounds(0.0);
            let bounds = [
                b[0].max(-180.0),
                b[1].max(-85.051129),
                b[2].min(180.0),
                b[3].min(85.051129),
            ];
            let (width, height) = (
                viewport.width.round().max(1.0) as u32,
                viewport.height.round().max(1.0) as u32,
            );
            let wanted = self.props.service_type == WmsServiceType::Template || !self.props.layers.is_empty();
            let is_new = self
                .requested
                .is_none_or(|(rb, rw, rh)| bounds_differ(rb, bounds) || rw != width || rh != height);
            if wanted && !self.props.data.is_empty() && is_new {
                let now = Instant::now();
                let changed_at = *self.changed_at.get_or_insert(now);
                if self.pending.is_none()
                    && now.duration_since(changed_at).as_secs_f64() * 1000.0 >= self.props.debounce_ms
                {
                    self.changed_at = None;
                    self.start_request(bounds, width, height);
                }
            }
            self.poll();
        }
        self.sub_layers.update(ctx, viewport)
    }

    fn draw(&mut self, ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        self.sub_layers.draw(ctx, pass)
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

impl Default for WmsLayer {
    fn default() -> Self {
        Self::new(WmsLayerProps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_map_and_template_urls() {
        let props = WmsLayerProps {
            data: "https://wms.example/ows?map=x".to_string(),
            layers: vec!["a".to_string(), "b".to_string()],
            srs: WmsSrs::Epsg4326,
            ..Default::default()
        };
        let url = image_url(&props, [-10.0, 40.0, 10.0, 50.0], 800, 600);
        assert!(url.starts_with("https://wms.example/ows?map=x&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=a,b&STYLES=&CRS=EPSG:4326&BBOX=40,-10,50,10&WIDTH=800&HEIGHT=600&FORMAT=image/png&TRANSPARENT=TRUE"), "{url}");
        let mercator = WmsLayerProps {
            data: "https://wms.example/ows".to_string(),
            layers: vec!["a".to_string()],
            ..Default::default()
        };
        let url = image_url(&mercator, [0.0, 0.0, 1.0, 1.0], 10, 10);
        assert!(
            url.contains("?SERVICE=WMS") && url.contains("CRS=EPSG:3857&BBOX=0,0,111319.491,111325.143"),
            "{url}"
        );
        let template = WmsLayerProps {
            data: "https://img/{layers}/{west},{south},{east},{north}/{width}x{height}?crs={crs}".to_string(),
            service_type: WmsServiceType::Template,
            layers: vec!["roads".to_string()],
            srs: WmsSrs::Epsg4326,
            ..Default::default()
        };
        assert_eq!(
            image_url(&template, [1.0, 2.0, 3.0, 4.0], 5, 6),
            "https://img/roads/1,2,3,4/5x6?crs=EPSG:4326"
        );
    }
}

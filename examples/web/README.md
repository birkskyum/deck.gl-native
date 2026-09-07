# deck.gl-native in a browser

The crates compiled to `wasm32-unknown-unknown`, drawing with wgpu through the browser's
WebGPU, or through WebGL2 when built with the `webgl` feature. Same layers, same WGSL
shaders, same projection math as the native build; naga lowers the shaders to GLSL ES 3.00
on the WebGL2 path.

```sh
./build.sh                  # WebGPU, into www/pkg
./build.sh webgl            # WebGL2, for the maplibre-gl-js custom layer
python3 -m http.server 8787 --directory www
```

Then open <http://localhost:8787/>. Drag to pan, hold shift to tilt and turn, scroll to zoom.

`build.sh` needs the `wasm32-unknown-unknown` target (`rustup target add
wasm32-unknown-unknown`) and a `wasm-bindgen` whose version matches the `wasm-bindgen` crate
in `Cargo.lock` (`cargo install wasm-bindgen-cli --version <that version>`). Set
`WASM_BINDGEN` to use a particular one.

## What is different from the native build

- **No threads.** The `parallel` feature is off, so the tessellation and upload paths that
  fan out over rayon run in order instead, and the tile loader runs on the frame loop.
- **Loading goes through the browser.** `Fetcher` has no worker threads on wasm: a request
  becomes a `fetch` that settles it when it resolves. Nothing may block the main thread, so
  `fetch_blocking` reports `still loading` and the caller asks again on a later frame.
- **`Instant` is `web-time`'s**, which is `performance.now()` here.
- **Constant attributes are off on WebGL2.** A vertex buffer with a zero stride means "every
  instance reads element zero" in WebGPU, Metal, Vulkan and D3D12, but "tightly packed" in
  OpenGL, so `DeckProps::constant_attributes` is false on that path.

## Inside a maplibre-gl-js map

`www/maplibre.html` adds the layers to a maplibre-gl-js map as a custom layer. maplibre hands
the layer its live WebGL2 context, wgpu adopts that context rather than making one of its own
(`wgpu_hal::gles::Adapter::new_external`) and draws into the framebuffer maplibre already has
bound (`TextureInner::DefaultRenderbuffer`). Colour and depth are therefore shared: an arc is
hidden where it passes behind a tower and visible where it clears one, in one frame, without
compositing two canvases.

Two things that path needs:

- `DeckProps::clip_origin` is `BottomLeft`. A canvas' default framebuffer starts at the bottom
  left, while wgpu draws for a top left origin, so without it the layers come out upside down.
- The map is created with `antialias: false`. A multisampled default framebuffer cannot be
  drawn into this way.

The layer also puts the pixel store parameters back to their defaults after drawing. wgpu sets
`UNPACK_ROW_LENGTH` and friends while it uploads a texture and leaves them set; maplibre never
touches them, so it never sets them back, and every atlas it uploads afterwards fails with
`invalid unpack params combination` and draws as a solid block. The symbols already uploaded
when the layer first drew keep working, which makes it look as though panning breaks them.

The browser logs `INVALID_OPERATION: drawBuffers: BACK or NONE` once per frame: wgpu's GL
backend names a colour attachment where WebGL2 wants `BACK` for a default framebuffer. WebGL
ignores the call and the draw buffer stays where it should be, so it is noise rather than a
fault, but it is wgpu's to fix.

A map with terrain or the globe projection renders into a framebuffer of its own, which the
layers cannot attach a depth buffer to without taking the map's away; the overlay reports that
rather than doing it.

### The depth planes do not agree yet

Sharing a depth buffer means agreeing on what a depth value stands for, and deck and MapLibre
put their near and far planes in different places. The layers therefore win and lose depth
comparisons they should not at some camera angles: an arc well above a tower can vanish behind
it.

The camera comes from the custom layer's render parameters, `nearZ` and `farZ`, which is the
public way to reach it. `@deck.gl/maplibre` divides them by the viewport height, the unit
deck's own planes are in, and hands them to the view. Doing the same here makes the layers
disappear entirely, so something else differs as well and the planes are not used yet: the
overlay keeps deck's own. Finding the rest of that difference is the next thing to do.

The example runs on MapLibre GL JS 6, which is ESM only, WebGL2 only, and no longer has the
private `map.transform` that older integrations read. It loads MapLibre's own `.mjs` build
rather than a rebundled one, because MapLibre's worker only starts from that file.

## Racing deck.gl JS on the same file

`www/compare.html` runs deck.gl JS over the same Arrow file the native window reads, doing the
same thing: bin five million points into hexagons, then change the radius and bin them again.
Changing the radius is what dragging a kepler.gl radius slider costs, and it is the number
worth comparing, because it is per row CPU work that no amount of GPU speed removes.

```sh
cargo run --release --bin gen_bench_data -- points 5000000 /tmp/points5m.arrow
cp /tmp/points5m.arrow examples/web/www/data5m.arrow
python3 -m http.server 8787 --directory examples/web/www
```

Then open <http://localhost:8787/compare.html> beside the native window:

```sh
DECKGL_JSON=/tmp/points5m.arrow DECKGL_HEXBIN=200 cargo run --release --bin window
cargo run --release --bin load_race -- /tmp/points5m.arrow --hexbin 200
```

`?radii=200,100,400,200` chooses the sweep, `?gpu=0` asks deck.gl for its CPU aggregator
instead of the GPU one it picks by default, and `?file=` reads a different file.

deck.gl is given its fastest path rather than its most convenient one. The coordinates go in
as a binary attribute straight out of Arrow, so no accessor is ever called and no row object
is ever made, and the `data` object is built once and passed by reference, because deck.gl
diffs it by identity: a fresh object literal per radius makes it re-upload all forty megabytes
of positions as well, which is not what moving a slider does. That one difference is worth
more than everything else on the page.

The page will refuse to report anything if the browser is not painting the tab. deck.gl
updates its layers on an animation frame, and a tab that is not on screen gets one animation
frame a second, so the timings would be that throttle rather than the work.

## The JavaScript API

```js
import init, { create } from './pkg/deck_gl_web.js'

await init()
const deck = await create(canvas)
deck.set_spec(await (await fetch('scenes/osm-tiles.json')).text())
deck.render()
```

`create` takes the canvas and returns a deck. `set_spec` takes a deck.gl JSON description,
the same one `json_render` reads. The camera is deck.gl's own `MapController`, driven by
`pan_start`/`pan`/`pan_end`, `rotate_start`/`rotate`/`rotate_end`, `zoom_by` and `tick`, so
the browser gets the same drag, inertia and zoom behaviour as the native window rather than a
second implementation in JavaScript.

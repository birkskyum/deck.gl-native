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

The browser logs `INVALID_OPERATION: drawBuffers: BACK or NONE` once per frame: wgpu's GL
backend names a colour attachment where WebGL2 wants `BACK` for a default framebuffer. WebGL
ignores the call and the draw buffer stays where it should be, so it is noise rather than a
fault, but it is wgpu's to fix.

A map with terrain or the globe projection renders into a framebuffer of its own, which the
layers cannot attach a depth buffer to without taking the map's away; the overlay reports that
rather than doing it.

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

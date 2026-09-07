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

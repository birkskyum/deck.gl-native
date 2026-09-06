# deck.gl-native

A native implementation of [deck.gl](https://deck.gl) for native hardware: Rust on
[wgpu](https://wgpu.rs), rendering with deck.gl's own WGSL shaders and taking Apache Arrow
and GeoArrow data directly.

This repository started in 2020 as a C++ prototype on dawn (see
[the original C++ prototype](#original-c-prototype-2020) below). That prototype stalled for
lack of funding and is kept in `cpp/` as a design reference. The Rust port in `crates/` is a
fresh port of deck.gl 9 and is the active code base.

## Status

Working today, headless and verified pixel by pixel in tests:

- `ScatterplotLayer`, `LineLayer` and `SolidPolygonLayer` (filled, extruded, wireframe, holes)
- Web Mercator viewport math ported from `@math.gl/web-mercator` and tested against it
- deck.gl's `project` and `project32` shader modules, picking uniforms and lighting
- Arrow record batches as layer data, with column, constant and function accessors
- Rendering into any caller-owned `wgpu` render pass, or into textures you provide

- Rendering into a maplibre-native map, either inside maplibre-native's own Metal backend
  through a C API, or from an all-Rust host through maplibre-native-ffi. See
  [docs/maplibre-native.md](docs/maplibre-native.md).

Not yet: picking passes, transitions, and the wider layer catalog. See
[docs/rust-port.md](docs/rust-port.md) for the design and the open decisions.

![maplibre-native with the deck.gl-native overlay](docs/images/maplibre-overlay.png)

*maplibre-native's GLFW app on Metal with deck.gl-native layers drawn into the same frame.*

## Crates

| Crate | JavaScript counterpart | Contents |
| --- | --- | --- |
| `math-gl` | `@math.gl/web-mercator` | Web Mercator projection and camera math, f64 |
| `luma-gl` | `@luma.gl/core`, `@luma.gl/shadertools` | Shader assembly, uniform blocks, `Model`, headless device helpers |
| `deck-gl` | `@deck.gl/core` | `Deck`, `Layer`, `Viewport`, the `project` shader module, Arrow data accessors |
| `deck-gl-layers` | `@deck.gl/layers` | `ScatterplotLayer`, `LineLayer`, `SolidPolygonLayer` |
| `deck-gl-ffi` | `@deck.gl/mapbox` | C API (`libdeckgl.a`) for host renderers; Metal device and texture interop |
| `deck-gl-examples` | | Example binaries |

Shader sources under `src/wgsl` in each crate are copied from deck.gl 9.4 and luma.gl (MIT).

## Building and running

Requires a stable Rust toolchain (1.87 or newer) and a GPU with Metal, Vulkan or DirectX 12.

```sh
cargo test --workspace                     # unit tests plus headless GPU render tests
cargo run --release --bin texture_render   # renders the example scene to target/texture-render.png
cargo run --release --bin window           # the same scene in a window with an orbiting camera
cargo run --release --manifest-path examples/maplibre-ffi/Cargo.toml   # on a maplibre-native basemap
```

## Using the library

```rust
use deck_gl::{Accessor, Deck, DeckProps, LayerData, LayerProps, ViewState};
use deck_gl_layers::{ScatterplotLayer, ScatterplotLayerProps};

let layer = ScatterplotLayer::new(ScatterplotLayerProps {
    base: LayerProps::new("points"),
    data: LayerData::from_batch(record_batch),      // an arrow RecordBatch
    get_position: Accessor::column("geometry"),     // FixedSizeList<f64, 2 | 3>
    get_fill_color: Accessor::column("color"),      // FixedSizeList<u8, 3 | 4>
    get_radius: Accessor::Constant(50.0),           // meters
    ..Default::default()
});

let mut deck = Deck::new(&device, &queue, render_target, DeckProps {
    width, height,
    view_state: ViewState { longitude: -122.4, latitude: 37.8, zoom: 12.0, pitch: 45.0, bearing: 0.0 },
    layers: vec![Box::new(layer)],
    ..Default::default()
})?;

// Either let the deck begin a pass on your textures...
deck.render(&mut encoder, &color_view, Some(&depth_view), Some(wgpu::Color::TRANSPARENT))?;
// ...or draw into a pass you already own, for instance a basemap's:
deck.update()?;
deck.draw(&mut render_pass)?;
```

`Deck::set_viewport` accepts a caller-built `Viewport`, which is how a host map renderer will
drive the deck camera from its own.

---

## Original C++ prototype (2020)

> The text below describes the archived C++ code in `cpp/`. It does not build on current
> toolchains and is kept for reference only.

This is an open-source C++ implementation of deck.gl.

> This project is no longer active. It was an experiment to understand what it would take to build a native version of deck.gl. 
> There are discussions about restarting a deck.gl-native effort (this time probably in Rust instead of C++) to help us integrate more non-JavaScript code, better support mobile platforms, and collaborate better with basemaps like maplibre-gl. 
> If you have more than a passing interest in this topic you are welcome to join us at https://www.openvisualization.org/#get-involved.

<br />
<table style="border: 0;" align="center">
  <thead>
    <tr>
      <th colspan="3" style="text-align: center;">
        Layer examples running on an iOS device
      </td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img style="max-height:200px" src="https://raw.githubusercontent.com/UnfoldedInc/deck.gl-data/master/images/line-scatterplot-demo.gif" />
        <p><i>LineLayer + ScatterplotLayer</i></p>
      </td>
      <td>
        <img style="max-height:200px" src="https://raw.githubusercontent.com/UnfoldedInc/deck.gl-data/master/images/polygon-demo.gif" />
        <p><i>SolidPolygonLayer</i></p>
      </td>
      <td>
        <img style="max-height:200px" src="https://raw.githubusercontent.com/UnfoldedInc/deck.gl-data/master/images/scatterplot-demo.gif" />
        <p><i>ScatterplotLayer</i></p>
      </td>
    </tr>
  </tbody>
</table>

This project targeted a minimal, proof-of-concept prototype. This initial `deck.gl-native` release is unlikely to meet the requirements of most applications. 

## Scope

- Render to texture (not screen)
- Support deck.gl JSON API
- Layers: Only `ScatterplotLayer`, `LineLayer`, `SolidPolygonLayer`
- Tables: Only Apache Arrow tables will be supported, apps will need to use Arrow libraries if data is not already in Arrow format.
- loader support will be limited to CSV and line-delimited JSON.
- iOS only: will only be tested on iOS (no serious attempts will be done to make deck.gl-native work on other platforms).


## Not Planned

Many features normally considered fundamental by deck.gl applications are not even being addressed at this stage

- No Render to screen
- No Transitions
- No Base map support (e.g. Mapbox integration)
- No Interactivity (event handling, picking etc)
- No Extensive layer catalog
- No Performance Optimizations
- etc

## Supported Platforms

### iOS

- iOS12+ is required when building for a device
- iOS13+ is required when building for a simulator, as `Metal` is available on the simulator starting with iOS13
- Currently has to be built with bitcode disabled (under investigation)

### macOS

- macOS10.14+

#### Linux

- Tested and runs on Ubuntu Bionic

## Software Architecture

To get oriented on the deck.gl software architecture:
- [deck.gl Technical Deep-Dive](https://docs.google.com/presentation/d/1qtXUQzMuIa8NYIKUa1RKfSwvgpeccY-wrPrYqsb_8rE) is a good initial read. Written for the JavaScript version, however most concepts apply directly to the C++ port as well.
- [deck.gl cross-platform architecture](https://docs.google.com/presentation/d/1fzSHJxstSfNQe8VCtLLZvQ8D3OuJAfS7_LqPg0QPPvc) describes approaches to keeping multi-language implementations of deck.gl compatible (still a work-in-progress but gives a sense of the higher level direction).

### API Design Principles and Cross-Platform Concerns

A primary concern for the deck.gl-native port, especially once it matures, is how to keep it develop in sync with the various implementations and variants of the deck.gl API, such as the core JavaScript codebase, the pydeck Python API etc.

Currently the two main guidelines are:

- **deck.gl JSON parity** The deck.gl JSON API (See e.g. [deck.gl playground](https://deck.gl/playground/)) is the "common API" that will be supported by both the JavaScript and C++ implementations of deck.gl. This help ensure that both the C++ and JavaScript renderers evolve in sync, always providing compatible features (when their layer catalogs overlap etc). 
- **module and class parity** Within reason, the same modules, classes (and often even filenames) are used in both implementations. This 1-to-1 mapping really helps developers move quickly between the two code bases. 

Beyond the above rules, it is certainly acceptable C++ and JavaScript APIs to diverge, as long as it makes sense, to ensure they are both natural to use for programmers with backgrounds in the respective programming language. (E.g. functional style UI programming is common in the JS community while C++ programmers on the balance are likely favor a more imperative API style).

Top-level API also includes exception-less counterparts for core functionality across the modules.

### C++ Module Structure

The deck.gl-native code bases exposes a set of C++ "module header files" following a `<framework>/<module>.h` structure. 

| Module            | Description |
| --- | --- |
| `probe.gl/core.h` | Platform/Compiler detection, assertions, logging and timers |
| `math.gl/core.h`  | Vectors and Matrices |
| `math.gl/web-mercator.h` | Geospatial math for Web Mercator projection |
| `loaders.gl/csv.h`. | A CSV table loader |
| `loaders.gl/json.h`. | A JSON table loader |
| `luma.gl/core.h` | `Model`, `AnimationLoop` etc implemented on dawn API |
| `luma.gl/webgpu.h` | Internal utilities for working with the WebGPU API (adaptions from the dawn repo) |
| `luma.gl/garrow.h` | GPU-based implementation of Arrow-like API |
| `deck.gl/core.h` | The core deck.gl classes:\ `Deck`, `Layer`, `View`, etc. |
| `deck.gl/layers.h` | The initial layer catalog |
| `deck.gl/json.h` | Classes for parsing (deck.gl) JSON into C++ objects |

Remarks:
- The `framework` part of the C++ module header file name corresponds on the JavaScript side to a framework from the [vis.gl](https://vis.gl/) framework suite, and the `module` part corresponds to a submodule in that framework monorepo.
- Each of the modules is built as a separate library, in a format suitable for consumption on the OS it's being built for

### Apache Arrow

All in-memory table processing is based on the Apache Arrow C++ API. This will be almost invisible to applications if they use the provided loaders, however for more advanced table processing applications may need to work directly with the Arrow API.

To get started with Arrow, useful resources might be:
- [Arrow C++ API Docs](https://arrow.apache.org/docs/cpp/api.html)
- [Arrow Test Suite](https://github.com/apache/arrow/blob/master/cpp/src/arrow/table_test.cc) - Examples of working code.

CSV and JSON table loaders are provided as part of the deck.gl library. To support additional table formats, the envisioned approach is to implement additional "loaders" that load various formats and "convert" the loaded tables to Arrow representation.

### Graphics Backend: dawn/WebGPU

deck.gl-native is being built on top of the C++ WebGPU API using the [dawn](https://dawn.googlesource.com/dawn) framework.

The dawn framework is a compelling choice for deck.gl:

- The JavaScript version of deck.gl will inevitably move from WebGL to WebGPU, so having both C++ and JavaScript work against a common 3D API will signficantly increase the ease of aligning the JavaScript and C++ code bases.
- The Dawn project has the ambition to provide backends on basically all platforms/rendering APIs of interest, including Vulkan, Metal, D3D12, and OpenGL. Ideally meaning that deck.gl-native itself will only have to implement a single backend, namely dawn.

Note that Dawn is still a work in progress (with different levels of support for different platforms - the prototype is only being tested on iOS) and there is some risk with this technology choice. However, given the momentum behind WebGPU in browsers, we feel that the prospects are currently looking good.


## Supporting this Effort

This porting project is led by [Unfolded, Inc](https://www.unfolded.ai), and currently relies on initial funding provided by a customer as well as external contributions. At this stage, this is not an fully or independenly resourced project. It only targets a proof-of-concept prototype, it does not have a maintenance plan and is not set up to address feature requests etc. 

Our hope is to see this project quickly grow into a living part of the core deck.gl project. If this project reaches a sufficient level of completeness / critical mass, the ambition is to make this project part of the main deck.gl project and transfer it to an open governance setup. 


## Setting up a Development Environment

Development can be done on macOS which is the primary environment, or Linux, which is supported mainly for CI testing. (Windows is not a supported dev env.)

**NOTE:** Building using `gcc` currently doesn't work, this is to be updated

## Quickstart

### Xcode (macOS only)

If you haven't already configured Xcode: 
- Install Xcode
- Launch it
- accept the license agreement
- Run `sudo xcode-select -s /Applications/Xcode.app` (Subsitute with path to your Xcode app if different).

### clang

For macOS:

```sh
brew install clang cmake clang-format lcov
```

### gcc

For macOS:

```sh
brew install gcc cmake clang-format lcov
```

- `scripts/bootstrap.sh` to fetch the dependencies and perform setup. `bootstrap.sh` can/should be called after pulling a branch to ensure environment is up to date
- `scripts/build-clang` or `scripts/build-gcc` to build

## Dependencies

All the dependencies for macOS and Linux are bundled in a dependency repository that's being used as a submodule. Running `scripts/bootstrap.sh`, among other things, fetches all the submodules. Alternatively, you can fetch them by running `git submodule update --init --recursive`.

| Dependency                                                	| Description                                                                         	|
|-----------------------------------------------------------	|-------------------------------------------------------------------------------------	|
| [Google Test](https://github.com/google/googletest)       	| Testing framework                                                                   	|
| [jsoncpp](https://github.com/open-source-parsers/jsoncpp) 	| JSON parser                                                                         	|
| [arrow](https://github.com/apache/arrow)                  	| Columnar in-memory storage                                                          	|
| [dawn](https://dawn.googlesource.com/dawn)                	| C++ WebGPU implementation with a maturing list of backends for most platforms       	|
| [shaderc](https://github.com/google/shaderc)              	| GLSL shader compilation                                                             	|
| [glfw](https://github.com/glfw/glfw)                      	| Portable library for creating OS windows to render graphics in, and handling events 	|

## Building

```
mkdir build
cd build
cmake ..
make -j 16
```

To use different compilers, set the build options `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` on the `cmake` command line. There is a number of build scripts for different compilers available in `scripts` directory.

## Testing

For Google Test formatted output, run `./deckgl-bundle-tests`.
For CTest formatted output, run `ctest`.

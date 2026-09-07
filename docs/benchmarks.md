# Benchmarks

`crates/deck-gl-layers/benches/layers.rs` measures, for each of three large layers, the
**upload** (resolving accessors, tessellating, creating the GPU buffers and drawing the first
frame) and the **frame** (drawing again with everything resident). Both include reading the
1024 x 1024 frame back to the CPU through `Deck::snapshot`, so real frame times are lower.

```sh
cargo bench -p deck-gl-layers --bench layers
```

Results on an Apple M series laptop (Metal, release build, 2026-09-07 evening). Run to run
variation is around 10%, so treat these as orders of magnitude rather than exact figures:

| Layer | Objects | Upload | Frame |
| --- | ---: | ---: | ---: |
| `ScatterplotLayer` (function accessors) | 1,000,000 points | 20.8 ms | 2.9 ms |
| `ScatterplotLayer` (Arrow `Float32` positions, `UInt8` colours) | 1,000,000 points | 14.3 ms | 2.9 ms |
| `ScatterplotLayer` (Arrow positions, constant styling) | 1,000,000 points | 14.5 ms | 2.9 ms |
| `SolidPolygonLayer` (extruded) | 100,000 polygons | 33.7 ms | 2.0 ms |
| `PathLayer` (20 vertices each) | 10,000 paths | 12.8 ms | 1.8 ms |

Constant accessors upload a single element with a zero vertex stride instead of one value per
object, so the colour buffer of that third row is four bytes rather than four megabytes. The
upload time barely moves between the last two rows, since writing a buffer is cheap next to
resolving the positions, but the memory and the buffer creation go away: a layer whose styling
is all constants keeps only the buffers that vary per object.

The same applies to the low half of the positions. A `Float32` position column has no low
half, so that buffer is a single zero element rather than twelve megabytes, which is most of
the difference between the first Arrow row and the function accessors above it.

Prop updates on the million points, drawn again through a snapshot: a `radius_scale` change
(uniform only) costs 5.7 ms and a fill colour accessor change 8.5 ms, against the full upload
above. Attribute updates write into the existing GPU buffers when the size is unchanged
(`AttributeManager`), so they allocate nothing; the colour update is bound by evaluating the
accessor for a million rows.

Many small layers (`many_layers/init`, 200 scatterplot layers of 10 points, created and drawn
once): 165 ms, against 384 ms when every layer compiled its own shader module and pipeline.
Layers of one type share a pipeline through the deck's `PipelineCache`, which is why deck.gl's
per layer depth bias is a uniform here rather than pipeline state.

The upload numbers are dominated by CPU work: accessor evaluation through boxed closures,
`f64` splitting for the high precision positions, earcut for polygons and the path
tessellation, plus the shader compilation of a fresh deck (a long lived deck compiles once).
Polygons are tessellated on all cores above 64 of them, which took the hundred thousand
extruded polygons from 54.7 ms to 33.7 ms.
Run to run noise on the laptop is around 10 percent.

## Getting a large file on screen

`cargo run --release --bin load_race -- <file.parquet>` reports the four costs between a file
on disk and a drawn frame. The point of the numbers is not the frame rate, which a browser on
WebGPU matches, but how quickly a file becomes a picture and how large that file may be.

Apple M series laptop, release build, files written with DuckDB, 1024 x 1024 headless frames
read back to the CPU (so the frame times are an upper bound):

| File | Rows | Read | Build | Upload | Frame | On screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 MB GeoParquet, separated coordinates | 5,000,000 points | 43.5 ms | 0.1 ms | 199.2 ms | 6.8 ms | **243 ms** |
| 400 MB GeoParquet, separated coordinates | 25,000,000 points | 192.0 ms | 0.1 ms | 814.4 ms | 36.4 ms | **1.0 s** |
| 85 MB GeoParquet, WKB polygons of 17 vertices | 500,000 polygons | 90.3 ms | 0.1 ms | 772.4 ms | 2.4 ms | **863 ms** |

- **read** is the Parquet decode into an Arrow record batch. Nothing becomes rows of objects
  on the way.
- **build** is that record batch becoming a layer, and it is a tenth of a millisecond for
  twenty five million rows because the batch is moved in as it is. This is the difference
  columnar data makes: there is no conversion step to measure.
- **upload** resolves the accessors, tessellates where the layer needs it, creates the GPU
  buffers and draws once.
- The twenty five million point run puts 400 MB of Arrow in memory and 1.1 GB in GPU buffers.
  A browser tab has neither.

Reaching the second row needed a fix rather than a faster machine: the device was asking for
wgpu's default limits, whose 256 MB maximum buffer size a layer of a few million rows runs
into on hardware that has no such limit. `create_headless_context` now asks for what the
adapter offers.

## Comparing with deck.gl JS

The same scenes can be built in deck.gl JS on WebGPU (`new Deck({deviceProps: {type:
'webgpu'}})`) with the layers above and the same accessors, timing `deck.redraw('force')`
around `device.queue.onSubmittedWorkDone()` for the frame and the first `redraw` after
`setProps({layers})` for the upload. Run both on the same machine and browser; the numbers
here are the native side of that table.

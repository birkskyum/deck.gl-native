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

## Re-binning, the cost of moving a slider

A hexagon layer bins every point again whenever its radius changes, which is what a radius
slider costs to drag. The work is a mercator projection and a bin lookup per point, then
grouping the points by bin, then a value per bin. All of it is per element and none of it
shares state, which is the shape of work a single threaded renderer cannot make faster.

`cargo run --release --bin load_race -- points.arrow --hexbin 200` measures it, and
`DECKGL_TIME_BINNING=1` splits it into its three parts.

Five million points, 200 m hexagons, Apple M series laptop with 18 cores:

| | Before | After |
| --- | ---: | ---: |
| project and find the bin of every point | 210 ms | 11 ms |
| group the points by bin | (part of the above) | 19 ms |
| a value per bin | 16 ms | 7 ms |
| **one radius change** | **240 ms** | **57 ms** |

The first row is the pure form of it: nothing shared, so it divides by the core count and
comes back nineteen times faster. The others are limited by memory rather than arithmetic and
gain less. Twenty five million points cost about 390 ms a change, against roughly two seconds
before.

What changed:

- the projection and bin lookup run on all cores
- bin ids are hashed by mixing two integers rather than by the default hasher, which is built
  for keys that can be attacked rather than for a pair of small integers
- grouping is four passes instead of one, so only the merge is serial: the distinct bins of
  each chunk in parallel, merged in chunk order so the result is identical to the serial pass,
  then every point's bin looked up against a map nobody is writing to, then member lists
  allocated to their exact size so no push reallocates
- each bin's value is computed on all cores

The layer's output is unchanged: the aggregation render tests compare pixels and still pass.

## How much data can be held at once

`cargo run --release --bin gen_bench_data -- points 100000000 /tmp/points100m.arrow` writes a
file and `cargo run --release --bin load_race -- /tmp/points100m.arrow` opens it. The same
files open in the window: `DECKGL_JSON=/tmp/points100m.arrow cargo run --release --bin window`.

Arrow IPC rather than Parquet, so the file is the buffers and the read is not a decode. Apple
M series laptop, release build, 1024 x 1024 headless frames read back to the CPU, so the frame
times are an upper bound and depend on how much of the screen the data covers.

| Rows | File | Arrow in memory | GPU buffers | Read | Build | Upload | Frame |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000,000 points | 41.9 MB | 41.9 MB | 220 MB | 8.8 ms | 0.2 ms | 107.6 ms | 8.2 ms |
| 25,000,000 points | 209.4 MB | 209.4 MB | 1.1 GB | 48.4 ms | 0.1 ms | 2210.7 ms | 127.5 ms |
| 50,000,000 points | 418.8 MB | 418.8 MB | 2.2 GB | 105.3 ms | 0.1 ms | 1226.8 ms | 94.4 ms |
| 100,000,000 points | 837.5 MB | 837.5 MB | 4.4 GB | 203.8 ms | 0.1 ms | 2521.9 ms | 202.7 ms |
| 500,000 polygons, 8 vertices each | 77.8 MB | 233.4 MB | 342 MB | 23.3 ms | 0.1 ms | 327.3 ms | 2.7 ms |

Two things to read out of it.

**Build is a tenth of a millisecond, whatever the row count.** A columnar file already holds
the coordinates in the layout the GPU wants, so a file becomes a layer by moving the record
batch in. There is no row by row conversion left to measure. This is the difference columnar
data makes and it is the reason the rest of the table is possible.

**A hundred million rows is 837 MB of Arrow and 4.4 GB of GPU buffers.** That is the working
set that fits without tiling the data first or throwing any of it away.

Restyling the resident data costs one frame: new colours on twenty five million points took
131.0 ms against a 127.5 ms frame, because nothing is uploaded again except the styling.

Reaching the second row needed a fix rather than a faster machine: the device was asking for
wgpu's default limits, whose 256 MB maximum buffer size a layer of a few million rows runs
into on hardware that has no such limit. `create_headless_context` now asks the adapter for
what it offers.

## Comparing with deck.gl JS

The same scenes can be built in deck.gl JS on WebGPU (`new Deck({deviceProps: {type:
'webgpu'}})`) with the layers above and the same accessors, timing `deck.redraw('force')`
around `device.queue.onSubmittedWorkDone()` for the frame and the first `redraw` after
`setProps({layers})` for the upload. Run both on the same machine and browser; the numbers
here are the native side of that table.

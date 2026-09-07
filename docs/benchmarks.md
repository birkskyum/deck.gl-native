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
| `ScatterplotLayer` (Arrow `Float32` positions, `UInt8` colours) | 1,000,000 points | 16.8 ms | 2.9 ms |
| `ScatterplotLayer` (Arrow positions, constant styling) | 1,000,000 points | 16.3 ms | 2.9 ms |
| `SolidPolygonLayer` (extruded) | 100,000 polygons | 54.7 ms | 1.6 ms |
| `PathLayer` (20 vertices each) | 10,000 paths | 12.8 ms | 1.8 ms |

Constant accessors upload a single element with a zero vertex stride instead of one value per
object, so the colour buffer of that third row is four bytes rather than four megabytes. The
upload time barely moves, since writing a buffer is cheap next to resolving the positions, but
the memory and the buffer creation go away: a layer whose styling is all constants keeps only
the buffers that vary per object.

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
Run to run noise on the laptop is around 10 percent.

## Comparing with deck.gl JS

The same scenes can be built in deck.gl JS on WebGPU (`new Deck({deviceProps: {type:
'webgpu'}})`) with the layers above and the same accessors, timing `deck.redraw('force')`
around `device.queue.onSubmittedWorkDone()` for the frame and the first `redraw` after
`setProps({layers})` for the upload. Run both on the same machine and browser; the numbers
here are the native side of that table.

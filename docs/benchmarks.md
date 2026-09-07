# Benchmarks

`crates/deck-gl-layers/benches/layers.rs` measures, for each of three large layers, the
**upload** (resolving accessors, tessellating, creating the GPU buffers and drawing the first
frame) and the **frame** (drawing again with everything resident). Both include reading the
1024 x 1024 frame back to the CPU through `Deck::snapshot`, so real frame times are lower.

```sh
cargo bench -p deck-gl-layers --bench layers
```

Results on an Apple M series laptop (Metal, release build, 2026-09-07):

| Layer | Objects | Upload | Frame |
| --- | ---: | ---: | ---: |
| `ScatterplotLayer` | 1,000,000 points | 21 ms | 2.6 ms |
| `SolidPolygonLayer` (extruded) | 100,000 polygons | 52 ms | 1.5 ms |
| `PathLayer` (20 vertices each) | 10,000 paths | 11 ms | 1.6 ms |

The upload numbers are dominated by CPU work: accessor evaluation through boxed closures,
`f64` splitting for the high precision positions, earcut for polygons and the path
tessellation. Per attribute diffing (#9) and buffer reuse (#71) target that column.

## Comparing with deck.gl JS

The same scenes can be built in deck.gl JS on WebGPU (`new Deck({deviceProps: {type:
'webgpu'}})`) with the layers above and the same accessors, timing `deck.redraw('force')`
around `device.queue.onSubmittedWorkDone()` for the frame and the first `redraw` after
`setProps({layers})` for the upload. Run both on the same machine and browser; the numbers
here are the native side of that table.

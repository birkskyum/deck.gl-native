# maplibre-native GLFW overlay hook

The files here are the deck.gl-native overlay for maplibre-native's GLFW app on Metal, kept in
sync with the checkout used by `scripts/run-maplibre-demo.sh`. They are a snapshot so the demo
can be reproduced from this repository; the upstream direction is a custom layer host in
maplibre-native core (issue #61).

- `deckgl_overlay.hpp`, `deckgl_overlay.mm`: copy into `platform/glfw/`.
- `glfw-overlay.patch`: `git apply` on maplibre-native (last applied on commit `e5d0fce32a98`).
  It adds the `MLN_DECKGL_OVERLAY` CMake option, calls `deckgl_overlay::setCamera` before each
  frame, draws the overlay in `MetalRenderableResource::swap()` between the map's command buffer
  and present, and keeps the depth attachment cleared at load and stored after the map's pass so
  deck can depth test against it.

Depth interleaving details that the hook and `deck-gl-ffi` agree on:

- maplibre draws fill extrusions with `PaintParameters::nearClippedProjMatrix`, whose near
  plane is a tenth of the camera distance in whole pixels; flat layers use a 1 px near plane.
  deck uses the near clipped planes (`maplibre_near_far_pixels`), so its depth values compare
  correctly with buildings, and everything deck draws at ground level lands in front of the flat
  map layers (which sit at the far end of the depth range on the 1 px curve).
- maplibre's shaders write OpenGL style clip depth (`[-w, w]`, the near half clipped) straight
  into the Metal depth buffer, while deck.gl's WGSL remaps to WebGPU's `[0, w]`. The FFI creates
  the deck with `ClipDepthRange::NegativeOneToOne` so both write the same values; a test in
  `deck-gl-ffi` checks GPU depth against the CPU projection for elevated points.
- Opaque fills such as water write depth for the ground plane, so the FFI also gives deck a base
  depth bias (`DeckProps::depth_bias_base`) in addition to deck.gl's per layer polygon offset.
- `DECKGL_DEBUG=1` prints the camera and planes; `DECKGL_DUMP_DEPTH=/tmp/prefix` writes the
  map's depth buffer as raw `f32` at frame `DECKGL_DUMP_DEPTH_FRAME` (default 300) and prints
  building feet against deck's ground depth. Add `DECKGL_DUMP_DEPTH_AFTER=1` to dump deck's own
  depth instead (rendered on a cleared buffer).
- `DECKGL_LOAD_DEPTH=0` turns interleaving off: deck clears depth first and draws on top.

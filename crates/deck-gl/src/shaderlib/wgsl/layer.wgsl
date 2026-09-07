// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/core/src/shaderlib/misc/layer-uniforms.ts (WGSL source)

struct LayerUniforms {
  opacity: f32,
  // deck.gl's polygon offset for the layer, in 24 bit depth units (see project.wgsl)
  depthBias: f32,
};

@group(0) @binding(auto)
var<uniform> layer: LayerUniforms;

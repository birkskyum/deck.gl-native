// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Uniforms of @deck.gl/extensions/src/path-style (the code lives in the injections)

struct PathStyleUniforms {
  dashAlignMode: f32,
  dashGapPickable: i32,
};

@group(0) @binding(auto) var<uniform> pathStyle: PathStyleUniforms;

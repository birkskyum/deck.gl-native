// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/fill-style/shader-module.ts

struct FillUniforms {
  patternTextureSize: vec2<f32>,
  uvCoordinateOrigin: vec2<f32>,
  uvCoordinateOrigin64Low: vec2<f32>,
  patternEnabled: i32,
  patternMask: i32,
};

@group(0) @binding(auto) var<uniform> fill: FillUniforms;
@group(0) @binding(auto) var fill_patternTexture: texture_2d<f32>;
@group(0) @binding(auto) var fill_patternSampler: sampler;

// Common space units per meter at the equator: a pattern of 24 pixels at scale 1 spans 24 meters.
const FILL_UV_SCALE: f32 = 512.0 / 40000000.0;

fn fill_mod2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b * floor(a / b);
}

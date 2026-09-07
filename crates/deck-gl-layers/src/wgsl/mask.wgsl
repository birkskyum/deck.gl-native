// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/mask/shader-module.ts: one texture per mask instead of
// one channel per mask, and texture rows start at the top as in every wgpu render target.

struct MaskUniforms {
  bounds: vec4<f32>,
  enabled: i32,
  inverted: i32,
  maskByInstance: i32,
};

@group(0) @binding(auto) var<uniform> mask: MaskUniforms;
@group(0) @binding(auto) var mask_texture: texture_2d<f32>;
@group(0) @binding(auto) var mask_sampler: sampler;

fn mask_getCoords(position: vec4<f32>) -> vec2<f32> {
  let uv = (position.xy - mask.bounds.xy) / (mask.bounds.zw - mask.bounds.xy);
  return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn mask_isInBounds(texCoords: vec2<f32>) -> bool {
  if (mask.enabled == 0) {
    return true;
  }
  let maskValue = textureSample(mask_texture, mask_sampler, texCoords).r;
  if (mask.inverted != 0) {
    return maskValue >= 0.5;
  }
  return maskValue < 0.5;
}

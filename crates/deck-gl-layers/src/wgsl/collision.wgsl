// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/collision-filter/shader-module.ts. Texture rows start at
// the top as in every wgpu render target, and the map is sampled from the vertex stage.

struct CollisionUniforms {
  sort: i32,
  enabled: i32,
};

@group(0) @binding(auto) var<uniform> collision: CollisionUniforms;
@group(0) @binding(auto) var collision_texture: texture_2d<f32>;
@group(0) @binding(auto) var collision_sampler: sampler;

fn collision_getCoords(position: vec4<f32>) -> vec2<f32> {
  let collision_clipspace = project_common_position_to_clipspace(position);
  let uv = (1.0 + collision_clipspace.xy / collision_clipspace.w) / 2.0;
  return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn collision_match(tex: vec2<f32>, pickingColor: vec3<f32>) -> f32 {
  let collision_pickingColor = textureSampleLevel(collision_texture, collision_sampler, tex, 0.0);
  let delta = dot(abs(collision_pickingColor.rgb - pickingColor), vec3<f32>(1.0));
  let e = 0.001;
  return step(delta, e);
}

// Visibility test over a 5 by 5 pixel area, so objects fade in and out instead of flickering.
fn collision_isVisible(texCoords: vec2<f32>, pickingColor: vec3<f32>) -> f32 {
  if (collision.enabled == 0) {
    return 1.0;
  }
  let N = 2;
  var accumulator = 0.0;
  let stepSize = vec2<f32>(1.0) / project.viewportSize;
  let floatN = f32(N);
  var delta = -floatN * stepSize;
  for (var i = -N; i <= N; i = i + 1) {
    delta.x = -stepSize.x * floatN;
    for (var j = -N; j <= N; j = j + 1) {
      accumulator = accumulator + collision_match(texCoords + delta, pickingColor);
      delta.x = delta.x + stepSize.x;
    }
    delta.y = delta.y + stepSize.y;
  }
  let W = 2.0 * floatN + 1.0;
  return pow(accumulator / (W * W), 2.2);
}

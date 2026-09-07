// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/clip/clip-extension.ts (shader part)

struct ClipUniforms {
  bounds: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> clip: ClipUniforms;

fn clip_isInBounds(position: vec2<f32>) -> bool {
  return position.x >= clip.bounds[0] && position.y >= clip.bounds[1] &&
    position.x < clip.bounds[2] && position.y < clip.bounds[3];
}

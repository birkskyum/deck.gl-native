// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/core/src/shaderlib/shadow/shadow.ts to WGSL. The light space position of
// every vertex travels to the fragment stage in the shadow_vPosition varyings; while a shadow
// map is drawn, the depth of the vertex in light space is packed into the colour instead.
// Depth is packed into RGB (24 bits) with alpha 1, so the layers' premultiplication of the
// colour by alpha leaves it untouched.

struct shadowUniforms {
  viewProjectionMatrix0: mat4x4<f32>,
  viewProjectionMatrix1: mat4x4<f32>,
  projectCenter0: vec4<f32>,
  projectCenter1: vec4<f32>,
  color: vec4<f32>,
  drawShadowMap: f32,
  useShadowMap: f32,
  lightCount: f32,
  lightId: i32,
};
@group(0) @binding(auto) var<uniform> shadow: shadowUniforms;
@group(0) @binding(auto) var shadow_uShadowMap0: texture_2d<f32>;
@group(0) @binding(auto) var shadow_uShadowMap0Sampler: sampler;
@group(0) @binding(auto) var shadow_uShadowMap1: texture_2d<f32>;
@group(0) @binding(auto) var shadow_uShadowMap1Sampler: sampler;

const shadow_bitPackShift = vec3<f32>(1.0, 255.0, 65025.0);
const shadow_bitUnpackShift = vec3<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0);
const shadow_bitMask = vec3<f32>(1.0 / 255.0, 1.0 / 255.0, 0.0);

// The light's clip space position of a common space position (OpenGL depth range)
fn shadow_lightClip(position_commonspace: vec4<f32>, light: i32) -> vec4<f32> {
  if (light == 0) {
    return shadow.viewProjectionMatrix0 * position_commonspace + shadow.projectCenter0;
  }
  return shadow.viewProjectionMatrix1 * position_commonspace + shadow.projectCenter1;
}

// While drawing a shadow map: the vertex position as the light sees it, and its depth
fn shadow_drawPosition(position_commonspace: vec4<f32>) -> vec4<f32> {
  let clip = shadow_lightClip(position_commonspace, shadow.lightId);
  shadow_vDepth = (clip.z / clip.w + 1.0) / 2.0;
  if (shadow.lightId == 0) {
    return project_common_position_to_clipspace_with_projection(
      position_commonspace, shadow.viewProjectionMatrix0, shadow.projectCenter0);
  }
  return project_common_position_to_clipspace_with_projection(
    position_commonspace, shadow.viewProjectionMatrix1, shadow.projectCenter1);
}

// While drawing the scene: where the vertex lands in every shadow map, in 0..1
fn shadow_setVaryings(position_commonspace: vec4<f32>) {
  let clip0 = shadow_lightClip(position_commonspace, 0);
  shadow_vPosition0 = (clip0.xyz / clip0.w + 1.0) / 2.0;
  if (shadow.lightCount > 1.0) {
    let clip1 = shadow_lightClip(position_commonspace, 1);
    shadow_vPosition1 = (clip1.xyz / clip1.w + 1.0) / 2.0;
  }
}

fn shadow_setVertexPosition(position_commonspace: vec4<f32>, position: vec4<f32>) -> vec4<f32> {
  if (shadow.drawShadowMap > 0.5) {
    return shadow_drawPosition(position_commonspace);
  }
  if (shadow.useShadowMap > 0.5) {
    shadow_setVaryings(position_commonspace);
  }
  return position;
}

fn shadow_getShadowWeight(position: vec3<f32>, map: texture_2d<f32>, mapSampler: sampler) -> f32 {
  // The map was drawn with clip space y up; texture rows start at the top
  let rgbaDepth = textureSampleLevel(map, mapSampler, vec2<f32>(position.x, 1.0 - position.y), 0.0);
  let z = dot(rgbaDepth.rgb, shadow_bitUnpackShift);
  return smoothstep(0.001, 0.01, position.z - z);
}

fn shadow_filterShadowColor(color: vec4<f32>) -> vec4<f32> {
  if (shadow.drawShadowMap > 0.5) {
    var rgbDepth = fract(shadow_vDepth * shadow_bitPackShift);
    rgbDepth -= rgbDepth.gbb * shadow_bitMask;
    return vec4<f32>(rgbDepth, 1.0);
  }
  if (shadow.useShadowMap > 0.5) {
    var shadowAlpha = shadow_getShadowWeight(shadow_vPosition0, shadow_uShadowMap0, shadow_uShadowMap0Sampler);
    if (shadow.lightCount > 1.0) {
      shadowAlpha += shadow_getShadowWeight(shadow_vPosition1, shadow_uShadowMap1, shadow_uShadowMap1Sampler);
    }
    shadowAlpha *= shadow.color.a / shadow.lightCount;
    let blendedAlpha = shadowAlpha + color.a * (1.0 - shadowAlpha);
    return vec4<f32>(
      mix(color.rgb, shadow.color.rgb, shadowAlpha / max(blendedAlpha, 0.0001)),
      blendedAlpha
    );
  }
  return color;
}

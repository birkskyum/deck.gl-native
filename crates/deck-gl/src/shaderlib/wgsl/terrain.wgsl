// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/terrain/shader-module.ts, the height map half of it: a
// layer whose operation is `terrain` draws its ground elevation into a map covering the
// view, and a layer with the terrain extension lifts its geometry onto that ground.
//
// The map holds metres rather than deck.gl's common space units, so the sixteen bit float
// texture keeps about a metre of precision at any elevation.

const TERRAIN_MODE_NONE: f32 = 0.0;
const TERRAIN_MODE_WRITE_HEIGHT_MAP: f32 = 1.0;
const TERRAIN_MODE_USE_HEIGHT_MAP: f32 = 2.0;

struct terrainUniforms {
  // Origin and size of the map in common space: x, y, width, height
  bounds: vec4<f32>,
  mode: f32,
};

@group(0) @binding(auto) var<uniform> terrain: terrainUniforms;
@group(0) @binding(auto) var terrain_map: texture_2d<f32>;
@group(0) @binding(auto) var terrain_mapSampler: sampler;

/// Where a common space position falls in the map, in texture coordinates. The bounds are
/// absolute, while positions are relative to the layer's origin in auto offset mode, so the
/// origin is added back first.
fn terrain_texCoords(position: vec2<f32>) -> vec2<f32> {
  let absolute = position + project.commonOrigin.xy;
  return (absolute - terrain.bounds.xy) / terrain.bounds.zw;
}

/// The ground elevation the map holds under a position, in metres, and whether it has one.
fn terrain_heightAt(position: vec2<f32>) -> vec2<f32> {
  let uv = terrain_texCoords(position);
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    return vec2<f32>(0.0, 0.0);
  }
  // The map was drawn with clip space y up; texture rows start at the top
  let sample = textureSampleLevel(terrain_map, terrain_mapSampler, vec2<f32>(uv.x, 1.0 - uv.y), 0.0);
  return vec2<f32>(sample.r, 1.0);
}

fn terrain_setVertexPosition(position: vec4<f32>) -> vec4<f32> {
  let commonPosition = geometry.position.xyz;
  // The elevation this vertex sits at, in metres
  let unitsPerMeter = max(project.commonUnitsPerMeter.z, 1e-12);
  terrain_height = (commonPosition.z + project.commonOrigin.z) / unitsPerMeter;

  if (terrain.mode == TERRAIN_MODE_WRITE_HEIGHT_MAP) {
    // Draw the ground into the map: the position is the place it covers, flattened
    let uv = terrain_texCoords(commonPosition.xy);
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
  }
  if (terrain.mode == TERRAIN_MODE_USE_HEIGHT_MAP) {
    // Lift the geometry onto the ground under its anchor
    var anchor = geometry.worldPosition;
    anchor.z = 0.0;
    let anchorCommon = project_position_vec3_f32(anchor);
    let height = terrain_heightAt(anchorCommon.xy);
    if (height.y > 0.5) {
      geometry.position.z += height.x * unitsPerMeter;
      return project_common_position_to_clipspace(geometry.position);
    }
  }
  return position;
}

fn terrain_filterColor(color: vec4<f32>) -> vec4<f32> {
  if (terrain.mode == TERRAIN_MODE_WRITE_HEIGHT_MAP) {
    // The map holds the elevation of the ground, not its colour
    return vec4<f32>(terrain_height, 0.0, 0.0, 1.0);
  }
  return color;
}

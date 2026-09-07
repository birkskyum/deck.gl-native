// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/brushing/shader-module.ts

struct BrushingUniforms {
  mousePos: vec2<f32>,
  radius: f32,
  enabled: i32,
  targetMode: i32,
};

@group(0) @binding(auto) var<uniform> brushing: BrushingUniforms;

fn brushing_isPointInRange(position: vec2<f32>) -> bool {
  if (brushing.enabled == 0) {
    return true;
  }
  let source_commonspace = project_position_vec2_f32(position);
  let target_commonspace = project_position_vec2_f32(brushing.mousePos);
  let distance = length((target_commonspace - source_commonspace) / project.commonUnitsPerMeter.xy);
  return distance <= brushing.radius;
}

fn brushing_arePointsInRange(sourcePos: vec2<f32>, targetPos: vec2<f32>) -> bool {
  return brushing_isPointInRange(sourcePos) || brushing_isPointInRange(targetPos);
}

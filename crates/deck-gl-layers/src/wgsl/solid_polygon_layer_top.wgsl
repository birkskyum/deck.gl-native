// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/layers/src/solid-polygon-layer/solid-polygon-layer.wgsl.ts (top shader)

struct Attributes {
  @location(0) vertexPositions: vec3<f32>,
  @location(1) vertexPositions64Low: vec3<f32>,
  @location(2) elevations: f32,
  @location(3) fillColors: vec4<f32>,
  @location(4) lineColors: vec4<f32>,
  @location(5) rowIndexes: u32,
};

@vertex
fn vertexMain(attributes: Attributes) -> Varyings {
  var outp: Varyings;
  deckgl_vertex_start(attributes);

  var pos = attributes.vertexPositions;
  if (solidPolygon.extruded > 0.5) {
    pos.z += attributes.elevations * solidPolygon.elevationScale;
  }

  geometry.worldPosition = attributes.vertexPositions;
  geometry.pickingColor = picking_getPickingColorFromIndex(attributes.rowIndexes);

  let projectedPosition = project_position_to_clipspace_and_commonspace(
    pos,
    attributes.vertexPositions64Low,
    vec3<f32>(0.0)
  );
  geometry.position = projectedPosition.commonPosition;
  outp.position = deckgl_filter_gl_position(projectedPosition.clipPosition, geometry);

  let normal = project_normal(vec3<f32>(0.0, 0.0, 1.0));
  geometry.normal = normal;

  let colors = select(
    attributes.fillColors,
    attributes.lineColors,
    solidPolygon.isWireframe > 0.5
  );
  outp.vColor = deckgl_filter_color(apply_polygon_color(colors, normal, geometry.position), geometry);
  outp.pickingColor = geometry.pickingColor;

  deckgl_vertex_end(&outp);
  return outp;
}

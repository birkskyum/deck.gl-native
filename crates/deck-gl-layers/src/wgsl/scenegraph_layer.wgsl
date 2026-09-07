// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/mesh-layers/src/scenegraph-layer/scenegraph-layer-vertex.glsl.ts,
// scenegraph-layer-fragment.glsl.ts and scenegraph-layer-uniforms.ts to WGSL, with the
// shader hooks called explicitly. One model per glTF primitive: its node transform, base
// colour and base colour texture are uniforms.

struct ScenegraphUniforms {
  sceneModelMatrix: mat4x4<f32>,
  baseColor: vec4<f32>,
  sizeScale: f32,
  sizeMinPixels: f32,
  sizeMaxPixels: f32,
  composeModelMatrix: f32,
  hasTexture: f32,
  // 0 flat (unlit), 1 lit by the deck's lights
  lighting: f32,
  hasNormals: f32,
};

@group(0) @binding(auto) var<uniform> scenegraph: ScenegraphUniforms;
@group(0) @binding(auto) var scenegraphTexture: texture_2d<f32>;
@group(0) @binding(auto) var scenegraphTextureSampler: sampler;

struct Attributes {
  @builtin(instance_index) instanceIndex: u32,
  @location(0) positions: vec3<f32>,
  @location(1) normals: vec3<f32>,
  @location(2) colors: vec3<f32>,
  @location(3) texCoords: vec2<f32>,
  @location(4) instancePositions: vec3<f32>,
  @location(5) instancePositions64Low: vec3<f32>,
  @location(6) instanceColors: vec4<f32>,
  @location(7) instanceModelMatrixCol0: vec3<f32>,
  @location(8) instanceModelMatrixCol1: vec3<f32>,
  @location(9) instanceModelMatrixCol2: vec3<f32>,
  @location(10) instanceTranslation: vec3<f32>,
};

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) texCoords: vec2<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) positionCommon: vec3<f32>,
  @location(4) pickingColor: vec3<f32>,
};

@vertex
fn vertexMain(attributes: Attributes) -> Varyings {
  var varyings: Varyings;
  deckgl_vertex_start(attributes);

  geometry.worldPosition = attributes.instancePositions;
  geometry.uv = attributes.texCoords;
  geometry.pickingColor = picking_getPickingColorFromIndex(attributes.instanceIndex);

  let instanceModelMatrix = mat3x3<f32>(
    attributes.instanceModelMatrixCol0,
    attributes.instanceModelMatrixCol1,
    attributes.instanceModelMatrixCol2
  );
  var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (scenegraph.hasNormals > 0.5) {
    normal = instanceModelMatrix * (scenegraph.sceneModelMatrix * vec4<f32>(attributes.normals, 0.0)).xyz;
  }

  // Clamp the size of one scene unit to the pixel range
  let originalSize = project_unit_size_to_pixel(scenegraph.sizeScale, UNIT_METERS);
  let clampedSize = clamp(originalSize, scenegraph.sizeMinPixels, scenegraph.sizeMaxPixels);
  var sizeRatio = 1.0;
  if (originalSize > 0.0) {
    sizeRatio = clampedSize / originalSize;
  }
  let scenePosition = (scenegraph.sceneModelMatrix * vec4<f32>(attributes.positions, 1.0)).xyz;
  var pos = (instanceModelMatrix * scenePosition) * scenegraph.sizeScale * sizeRatio +
    attributes.instanceTranslation;

  if (scenegraph.composeModelMatrix > 0.5) {
    pos = deckgl_filter_size(pos, geometry);
    geometry.normal = project_normal(normal);
    geometry.worldPosition += pos;
    let projected = project_position_to_clipspace_and_commonspace(
      attributes.instancePositions + pos,
      attributes.instancePositions64Low,
      vec3<f32>(0.0)
    );
    geometry.position = projected.commonPosition;
    varyings.position = projected.clipPosition;
  } else {
    var offset = project_size_vec3(pos);
    offset = deckgl_filter_size(offset, geometry);
    let projected = project_position_to_clipspace_and_commonspace(
      attributes.instancePositions,
      attributes.instancePositions64Low,
      offset
    );
    geometry.position = projected.commonPosition;
    geometry.normal = project_normal(normal);
    varyings.position = projected.clipPosition;
  }
  varyings.position = deckgl_filter_gl_position(varyings.position, geometry);

  varyings.color = attributes.instanceColors * scenegraph.baseColor * vec4<f32>(attributes.colors, 1.0);
  varyings.color = deckgl_filter_color(varyings.color, geometry);
  varyings.texCoords = attributes.texCoords;
  varyings.normal = geometry.normal;
  varyings.positionCommon = geometry.position.xyz;
  varyings.pickingColor = geometry.pickingColor;

  deckgl_vertex_end(&varyings);
  return varyings;
}

@fragment
fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> {
  deckgl_fragment_start(varyings);
  fragmentGeometry.uv = varyings.texCoords;

  if (picking.isActive > 0.5) {
    if (!picking_isColorValid(varyings.pickingColor)) {
      discard;
    }
    return vec4<f32>(varyings.pickingColor, 1.0);
  }

  var color = varyings.color;
  if (scenegraph.hasTexture > 0.5) {
    color = color * textureSample(scenegraphTexture, scenegraphTextureSampler, varyings.texCoords);
  }
  if (scenegraph.lighting > 0.5) {
    var normal = varyings.normal;
    if (scenegraph.hasNormals < 0.5) {
      normal = normalize(cross(dpdy(varyings.positionCommon), dpdx(varyings.positionCommon)));
    }
    color = vec4<f32>(
      lighting_getLightColor2(color.rgb, project.cameraPosition, varyings.positionCommon, normal),
      color.a
    );
  }
  color.a = color.a * layer.opacity;
  color = deckgl_filter_fragment_color(color, fragmentGeometry);

  if (picking.isHighlightActive > 0.5) {
    let highlightedColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(varyings.pickingColor - highlightedColor))) {
      let blendedAlpha = picking.highlightColor.a + color.a * (1.0 - picking.highlightColor.a);
      if (blendedAlpha > 0.0) {
        color = vec4<f32>(
          mix(color.rgb, picking.highlightColor.rgb, picking.highlightColor.a / blendedAlpha),
          blendedAlpha
        );
      }
    }
  }

  return deckgl_premultiplied_alpha(color);
}

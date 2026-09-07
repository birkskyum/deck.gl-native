// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
//
// WGSL port of `@deck.gl/layers` text-background-layer: a rounded, optionally stroked box
// behind each text of a TextLayer.

struct TextBackgroundUniforms {
  billboard: f32,
  sizeScale: f32,
  sizeMinPixels: f32,
  sizeMaxPixels: f32,
  borderRadius: vec4<f32>,
  padding: vec4<f32>,
  sizeUnits: i32,
  stroked: f32
};

struct TextUniforms {
  cutoffPixels: vec2<f32>,
  align: vec2<i32>,
  fontSize: f32,
  flipY: f32
};

@group(0) @binding(auto) var<uniform> textBackground: TextBackgroundUniforms;
@group(0) @binding(auto) var<uniform> text: TextUniforms;

fn rotate_by_angle(vertex: vec2<f32>, angle: f32) -> vec2<f32> {
  let angle_radian = radians(angle);
  let cos_angle = cos(angle_radian);
  let sin_angle = sin(angle_radian);
  let rotationMatrix = mat2x2<f32>(vec2<f32>(cos_angle, -sin_angle), vec2<f32>(sin_angle, cos_angle));
  return rotationMatrix * vertex;
}

struct Attributes {
  @location(0) positions: vec2<f32>,

  @location(1) instancePositions: vec3<f32>,
  @location(2) instancePositions64Low: vec3<f32>,
  @location(3) instanceRects: vec4<f32>,
  @location(4) instanceClipRect: vec4<f32>,
  @location(5) instanceSizes: f32,
  @location(6) instanceAngles: f32,
  @location(7) instancePixelOffsets: vec2<f32>,
  @location(8) instanceLineWidths: f32,
  @location(9) instanceFillColors: vec4<f32>,
  @location(10) instanceLineColors: vec4<f32>,
  @location(11) rowIndexes: u32,
};

struct Varyings {
  @builtin(position) position: vec4<f32>,

  @location(0) vFillColor: vec4<f32>,
  @location(1) vLineColor: vec4<f32>,
  @location(2) vLineWidth: f32,
  @location(3) uv: vec2<f32>,
  @location(4) dimensions: vec2<f32>,
  @location(5) pickingColor: vec3<f32>,
};

@vertex
fn vertexMain(inp: Attributes) -> Varyings {
  deckgl_vertex_start(inp);
  geometry.worldPosition = inp.instancePositions;
  geometry.uv = inp.positions;
  geometry.pickingColor = picking_getPickingColorFromIndex(inp.rowIndexes);

  var outp: Varyings;
  outp.uv = inp.positions;
  outp.vLineWidth = inp.instanceLineWidths;

  // project size to pixels and clamp to limits
  let sizePixels = clamp(
    project_unit_size_to_pixel(inp.instanceSizes * textBackground.sizeScale, textBackground.sizeUnits),
    textBackground.sizeMinPixels, textBackground.sizeMaxPixels
  );
  let instanceScale = sizePixels / text.fontSize;

  var dimensions = inp.instanceRects.zw * instanceScale + textBackground.padding.xy + textBackground.padding.zw;

  var pixelOffset = (inp.positions * inp.instanceRects.zw + inp.instanceRects.xy) * instanceScale
    + mix(-textBackground.padding.xy, textBackground.padding.zw, inp.positions);
  pixelOffset = rotate_by_angle(pixelOffset, inp.instanceAngles);
  pixelOffset = pixelOffset + inp.instancePixelOffsets;
  pixelOffset.y = pixelOffset.y * -1.0;

  // apply clipping
  var xy = project_size_vec2(inp.instanceClipRect.xy) * project.scale;
  var wh = project_size_vec2(inp.instanceClipRect.zw) * project.scale;
  if (text.flipY > 0.5) {
    xy.y = -xy.y - wh.y;
  }
  if (inp.instanceClipRect.z >= 0.0) {
    dimensions.x = wh.x;
    pixelOffset.x = xy.x + inp.positions.x * wh.x + mix(-textBackground.padding.x, textBackground.padding.z, inp.positions.x);
  }
  if (inp.instanceClipRect.w >= 0.0) {
    dimensions.y = wh.y;
    pixelOffset.y = xy.y + inp.positions.y * wh.y + mix(-textBackground.padding.y, textBackground.padding.w, inp.positions.y);
  }
  outp.dimensions = dimensions;

  var pos: vec4<f32>;
  if (textBackground.billboard > 0.5) {
    pos = project_position_to_clipspace(inp.instancePositions, inp.instancePositions64Low, vec3<f32>(0.0));
    var offset = vec3<f32>(pixelOffset, 0.0);
    offset = deckgl_filter_size(offset, geometry);
    let clipOffset = project_pixel_size_to_clipspace(offset.xy);
    pos = vec4<f32>(pos.x + clipOffset.x, pos.y + clipOffset.y, pos.z, pos.w);
  } else {
    var offsetCommon = vec3<f32>(project_pixel_size_vec2(pixelOffset), 0.0);
    if (text.flipY > 0.5) {
      offsetCommon.y = offsetCommon.y * -1.0;
    }
    offsetCommon = deckgl_filter_size(offsetCommon, geometry);
    pos = project_position_to_clipspace(inp.instancePositions, inp.instancePositions64Low, offsetCommon);
  }
  pos = deckgl_filter_gl_position(pos, geometry);
  outp.position = pos;

  // Apply opacity to instance color
  outp.vFillColor = vec4<f32>(inp.instanceFillColors.rgb, inp.instanceFillColors.a * layer.opacity);
  outp.vFillColor = deckgl_filter_color(outp.vFillColor, geometry);
  outp.vLineColor = vec4<f32>(inp.instanceLineColors.rgb, inp.instanceLineColors.a * layer.opacity);
  outp.vLineColor = deckgl_filter_color(outp.vLineColor, geometry);
  outp.pickingColor = geometry.pickingColor;
  deckgl_vertex_end(&outp);
  return outp;
}

fn round_rect(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
  // Convert p and size to center-based coordinates [-0.5, 0.5]
  let pixelPositionCB = (p - 0.5) * size;
  let sizeCB = size * 0.5;

  let maxBorderRadius = min(size.x, size.y) * 0.5;
  let borderRadius = min(radii, vec4<f32>(maxBorderRadius));

  // from https://www.shadertoy.com/view/4llXD7
  let side = select(borderRadius.zw, borderRadius.xy, pixelPositionCB.x > 0.0);
  let r = select(side.y, side.x, pixelPositionCB.y > 0.0);
  let q = abs(pixelPositionCB) - sizeCB + r;
  return -(min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r);
}

fn rect(p: vec2<f32>, size: vec2<f32>) -> f32 {
  let pixelPosition = p * size;
  return min(min(pixelPosition.x, size.x - pixelPosition.x),
             min(pixelPosition.y, size.y - pixelPosition.y));
}

fn get_stroked_color(dist: f32, fillColor: vec4<f32>, lineColor: vec4<f32>, lineWidth: f32) -> vec4<f32> {
  let isBorder = smoothedge(dist, lineWidth);
  return mix(fillColor, lineColor, isBorder);
}

@fragment
fn fragmentMain(inp: Varyings) -> @location(0) vec4<f32> {
  deckgl_fragment_start(inp);
  fragmentGeometry.uv = inp.uv;

  var fragColor: vec4<f32>;
  if (any(textBackground.borderRadius != vec4<f32>(0.0))) {
    let distToEdge = round_rect(inp.uv, inp.dimensions, textBackground.borderRadius);
    let shapeAlpha = smoothedge(-distToEdge, 0.0);
    if (shapeAlpha == 0.0) {
      discard;
    }
    if (textBackground.stroked > 0.5) {
      fragColor = get_stroked_color(distToEdge, inp.vFillColor, inp.vLineColor, inp.vLineWidth);
    } else {
      fragColor = inp.vFillColor;
    }
    fragColor.a = fragColor.a * shapeAlpha;
  } else {
    if (textBackground.stroked > 0.5) {
      let distToEdge = rect(inp.uv, inp.dimensions);
      fragColor = get_stroked_color(distToEdge, inp.vFillColor, inp.vLineColor, inp.vLineWidth);
    } else {
      fragColor = inp.vFillColor;
    }
  }

  fragColor = deckgl_filter_fragment_color(fragColor, fragmentGeometry);

  if (picking.isActive > 0.5) {
    if (!picking_isColorValid(inp.pickingColor)) {
      discard;
    }
    return vec4<f32>(inp.pickingColor, 1.0);
  }

  fragColor = deckgl_premultiplied_alpha(fragColor);

  if (picking.isHighlightActive > 0.5) {
    let highlightedObjectColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(inp.pickingColor - highlightedObjectColor))) {
      let highLightAlpha = picking.highlightColor.a;
      let blendedAlpha = highLightAlpha + fragColor.a * (1.0 - highLightAlpha);
      if (blendedAlpha > 0.0) {
        let highLightRatio = highLightAlpha / blendedAlpha;
        fragColor = vec4<f32>(
          mix(fragColor.rgb, picking.highlightColor.rgb, highLightRatio),
          blendedAlpha
        );
      } else {
        fragColor = vec4<f32>(fragColor.rgb, 0.0);
      }
    }
  }

  return fragColor;
}

// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
//
// Screen space cells of a ScreenGridLayer, after screen-grid-cell-layer.wgsl: each instance
// is one grid cell in logical pixels from the top left of the viewport.

struct ScreenGridUniforms {
  cellSizePixels: f32,
  cellMarginPixels: f32,
};

@group(0) @binding(auto) var<uniform> screenGrid: ScreenGridUniforms;

struct Attributes {
  @location(0) positions: vec2<f32>,
  @location(1) instanceCells: vec2<f32>,
  @location(2) instanceColors: vec4<f32>,
  @location(3) rowIndexes: u32,
};

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) vColor: vec4<f32>,
  @location(1) pickingColor: vec3<f32>,
};

@vertex
fn vertexMain(attributes: Attributes) -> Varyings {
  var varyings: Varyings;
  geometry.pickingColor = picking_getPickingColorFromIndex(attributes.rowIndexes);

  let logicalSize = project.viewportSize / project.devicePixelRatio;
  let margin = vec2<f32>(screenGrid.cellMarginPixels);
  let origin = attributes.instanceCells * screenGrid.cellSizePixels + margin;
  let size = vec2<f32>(max(screenGrid.cellSizePixels - 2.0 * screenGrid.cellMarginPixels, 0.0));
  let pixel = origin + attributes.positions * size;
  let ndc = vec2<f32>(pixel.x / logicalSize.x * 2.0 - 1.0, 1.0 - pixel.y / logicalSize.y * 2.0);
  varyings.position = vec4<f32>(ndc, 0.0, 1.0);
  varyings.vColor = vec4<f32>(attributes.instanceColors.rgb, attributes.instanceColors.a * layer.opacity);
  varyings.pickingColor = geometry.pickingColor;
  return varyings;
}

@fragment
fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> {
  if (picking.isActive > 0.5) {
    if (!picking_isColorValid(varyings.pickingColor)) {
      discard;
    }
    return vec4<f32>(varyings.pickingColor, 1.0);
  }
  var fragColor = deckgl_premultiplied_alpha(varyings.vColor);
  if (picking.isHighlightActive > 0.5) {
    let highlightedObjectColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(varyings.pickingColor - highlightedObjectColor))) {
      let highLightAlpha = picking.highlightColor.a;
      let blendedAlpha = highLightAlpha + fragColor.a * (1.0 - highLightAlpha);
      if (blendedAlpha > 0.0) {
        let highLightRatio = highLightAlpha / blendedAlpha;
        fragColor = vec4<f32>(mix(fragColor.rgb, picking.highlightColor.rgb, highLightRatio), blendedAlpha);
      } else {
        fragColor = vec4<f32>(fragColor.rgb, 0.0);
      }
    }
  }
  return fragColor;
}

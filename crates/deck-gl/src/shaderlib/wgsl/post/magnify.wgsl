struct magnifyUniforms {
  screenXY: vec2<f32>,
  radiusPixels: f32,
  zoom: f32,
  borderWidthPixels: f32,
  borderColor: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> magnify: magnifyUniforms;

fn magnify_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let pos = vec2<f32>(magnify.screenXY.x, 1.0 - magnify.screenXY.y);
  let dist = distance(texCoord * texSize, pos * texSize);
  if (dist < magnify.radiusPixels) {
    return screen_texture(source, sourceSampler, (texCoord - pos) / magnify.zoom + pos);
  }
  if (dist <= magnify.radiusPixels + magnify.borderWidthPixels) {
    return magnify.borderColor;
  }
  return screen_texture(source, sourceSampler, texCoord);
}

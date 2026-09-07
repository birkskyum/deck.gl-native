struct dotScreenUniforms {
  center: vec2<f32>,
  angle: f32,
  size: f32,
};
@group(0) @binding(auto) var<uniform> dotScreen: dotScreenUniforms;

fn dotScreen_pattern(texSize: vec2<f32>, texCoord: vec2<f32>) -> f32 {
  let scale = 3.1415 / dotScreen.size;
  let s = sin(dotScreen.angle);
  let c = cos(dotScreen.angle);
  let tex = texCoord * texSize - dotScreen.center * texSize;
  let point = vec2<f32>(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
  return (sin(point.x) * sin(point.y)) * 4.0;
}

fn dotScreen_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let average = (color.r + color.g + color.b) / 3.0;
  return vec4<f32>(vec3<f32>(average * 10.0 - 5.0 + dotScreen_pattern(texSize, texCoord)), color.a);
}

struct vibranceUniforms {
  amount: f32,
};
@group(0) @binding(auto) var<uniform> vibrance: vibranceUniforms;

fn vibrance_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let average = (color.r + color.g + color.b) / 3.0;
  let mx = max(color.r, max(color.g, color.b));
  let amt = (mx - average) * (-vibrance.amount * 3.0);
  return vec4<f32>(mix(color.rgb, vec3<f32>(mx), amt), color.a);
}

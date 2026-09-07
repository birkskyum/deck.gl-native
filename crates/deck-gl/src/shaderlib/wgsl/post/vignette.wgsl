struct vignetteUniforms {
  radius: f32,
  amount: f32,
};
@group(0) @binding(auto) var<uniform> vignette: vignetteUniforms;

fn vignette_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let dist = distance(texCoord, vec2<f32>(0.5, 0.5));
  let ratio = smoothstep(0.8, vignette.radius * 0.799, dist * (vignette.amount + vignette.radius));
  return color * ratio + (1.0 - ratio) * vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

struct noiseUniforms {
  amount: f32,
};
@group(0) @binding(auto) var<uniform> noise: noiseUniforms;

fn noise_rand(co: vec2<f32>) -> f32 {
  return fract(sin(dot(co.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn noise_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let diff = (noise_rand(texCoord) - 0.5) * noise.amount;
  return vec4<f32>(color.rgb + diff, color.a);
}

struct sepiaUniforms {
  amount: f32,
};
@group(0) @binding(auto) var<uniform> sepia: sepiaUniforms;

fn sepia_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let r = color.r;
  let g = color.g;
  let b = color.b;
  let a = sepia.amount;
  return vec4<f32>(
    min(1.0, (r * (1.0 - (0.607 * a))) + (g * (0.769 * a)) + (b * (0.189 * a))),
    min(1.0, (r * 0.349 * a) + (g * (1.0 - (0.314 * a))) + (b * 0.168 * a)),
    min(1.0, (r * 0.272 * a) + (g * 0.534 * a) + (b * (1.0 - (0.869 * a)))),
    color.a
  );
}

struct hueSaturationUniforms {
  hue: f32,
  saturation: f32,
};
@group(0) @binding(auto) var<uniform> hueSaturation: hueSaturationUniforms;

fn hueSaturation_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  // hue adjustment, wolfram alpha: RotationTransform[angle, {1, 1, 1}][{x, y, z}]
  let angle = hueSaturation.hue * 3.14159265;
  let s = sin(angle);
  let c = cos(angle);
  let weights = (vec3<f32>(2.0 * c, -sqrt(3.0) * s - c, sqrt(3.0) * s - c) + 1.0) / 3.0;
  var rgb = vec3<f32>(
    dot(color.rgb, weights.xyz),
    dot(color.rgb, weights.zxy),
    dot(color.rgb, weights.yzx)
  );
  // saturation adjustment
  let average = (rgb.r + rgb.g + rgb.b) / 3.0;
  if (hueSaturation.saturation > 0.0) {
    rgb += (average - rgb) * (1.0 - 1.0 / (1.001 - hueSaturation.saturation));
  } else {
    rgb += (average - rgb) * (-hueSaturation.saturation);
  }
  return vec4<f32>(rgb, color.a);
}

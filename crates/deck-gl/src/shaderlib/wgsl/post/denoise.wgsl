struct denoiseUniforms {
  strength: f32,
};
@group(0) @binding(auto) var<uniform> denoise: denoiseUniforms;

fn denoise_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let adjustedExponent = 3.0 + 200.0 * pow(1.0 - denoise.strength, 4.0);
  let center = screen_texture(source, sourceSampler, texCoord);
  var color = vec4<f32>(0.0);
  var total = 0.0;
  for (var x = -4.0; x <= 4.0; x += 1.0) {
    for (var y = -4.0; y <= 4.0; y += 1.0) {
      let offsetColor = screen_texture(source, sourceSampler, texCoord + vec2<f32>(x, y) / texSize);
      var weight = 1.0 - abs(dot(offsetColor.rgb - center.rgb, vec3<f32>(0.25)));
      weight = pow(weight, adjustedExponent);
      color += offsetColor * weight;
      total += weight;
    }
  }
  return color / total;
}

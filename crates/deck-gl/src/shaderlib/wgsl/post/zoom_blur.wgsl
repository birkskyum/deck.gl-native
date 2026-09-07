struct zoomBlurUniforms {
  center: vec2<f32>,
  strength: f32,
};
@group(0) @binding(auto) var<uniform> zoomBlur: zoomBlurUniforms;

fn zoomBlur_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  var color = vec4<f32>(0.0);
  var total = 0.0;
  let toCenter = zoomBlur.center * texSize - texCoord * texSize;
  // randomize the lookup values to hide the fixed number of samples
  let offset = random(vec3<f32>(12.9898, 78.233, 151.7182), 0.0);
  for (var t = 0.0; t <= 40.0; t += 1.0) {
    let percent = (t + offset) / 40.0;
    let weight = 4.0 * (percent - percent * percent);
    var offsetColor = screen_texture(source, sourceSampler, texCoord + toCenter * percent * zoomBlur.strength / texSize);
    // switch to pre-multiplied alpha to correctly blur transparent images
    offsetColor = vec4<f32>(offsetColor.rgb * offsetColor.a, offsetColor.a);
    color += offsetColor * weight;
    total += weight;
  }
  color = color / total;
  // switch back from pre-multiplied alpha
  return vec4<f32>(color.rgb / (color.a + 0.00001), color.a);
}

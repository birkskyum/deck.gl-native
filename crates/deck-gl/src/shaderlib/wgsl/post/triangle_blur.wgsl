struct triangleBlurUniforms {
  radius: f32,
  delta: vec2<f32>,
};
@group(0) @binding(auto) var<uniform> triangleBlur: triangleBlurUniforms;

fn triangleBlur_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let adjustedDelta = triangleBlur.delta * triangleBlur.radius / texSize;
  var color = vec4<f32>(0.0);
  var total = 0.0;
  // randomize the lookup values to hide the fixed number of samples
  let offset = random(vec3<f32>(12.9898, 78.233, 151.7182), 0.0);
  for (var t = -30.0; t <= 30.0; t += 1.0) {
    let percent = (t + offset - 0.5) / 30.0;
    let weight = 1.0 - abs(percent);
    var offsetColor = screen_texture(source, sourceSampler, texCoord + adjustedDelta * percent);
    // switch to pre-multiplied alpha to correctly blur transparent images
    offsetColor = vec4<f32>(offsetColor.rgb * offsetColor.a, offsetColor.a);
    color += offsetColor * weight;
    total += weight;
  }
  color = color / total;
  // switch back from pre-multiplied alpha
  return vec4<f32>(color.rgb / (color.a + 0.00001), color.a);
}

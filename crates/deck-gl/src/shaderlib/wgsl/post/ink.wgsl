struct inkUniforms {
  strength: f32,
};
@group(0) @binding(auto) var<uniform> ink: inkUniforms;

fn ink_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let dx = vec2<f32>(1.0 / texSize.x, 0.0);
  let dy = vec2<f32>(0.0, 1.0 / texSize.y);
  let color = screen_texture(source, sourceSampler, texCoord);
  var bigTotal = 0.0;
  var smallTotal = 0.0;
  var bigAverage = vec3<f32>(0.0);
  var smallAverage = vec3<f32>(0.0);
  for (var x = -2.0; x <= 2.0; x += 1.0) {
    for (var y = -2.0; y <= 2.0; y += 1.0) {
      let offsetColor = screen_texture(source, sourceSampler, texCoord + dx * x + dy * y).rgb;
      bigAverage += offsetColor;
      bigTotal += 1.0;
      if (abs(x) + abs(y) < 2.0) {
        smallAverage += offsetColor;
        smallTotal += 1.0;
      }
    }
  }
  let edge = max(vec3<f32>(0.0), bigAverage / bigTotal - smallAverage / smallTotal);
  let power = ink.strength * ink.strength * ink.strength * ink.strength * ink.strength;
  return vec4<f32>(color.rgb - dot(edge, edge) * power * 100000.0, color.a);
}

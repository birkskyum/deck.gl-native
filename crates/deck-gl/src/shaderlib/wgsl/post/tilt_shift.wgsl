struct tiltShiftUniforms {
  blurRadius: f32,
  gradientRadius: f32,
  start: vec2<f32>,
  end: vec2<f32>,
  invert: f32,
};
@group(0) @binding(auto) var<uniform> tiltShift: tiltShiftUniforms;

fn tiltShift_getDelta(texSize: vec2<f32>) -> vec2<f32> {
  let vector = normalize((tiltShift.end - tiltShift.start) * texSize);
  return select(vector, vec2<f32>(-vector.y, vector.x), tiltShift.invert > 0.5);
}

fn tiltShift_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  var color = vec4<f32>(0.0);
  var total = 0.0;
  // randomize the lookup values to hide the fixed number of samples
  let offset = random(vec3<f32>(12.9898, 78.233, 151.7182), 0.0);
  let normal = normalize(vec2<f32>(
    (tiltShift.start.y - tiltShift.end.y) * texSize.y,
    (tiltShift.end.x - tiltShift.start.x) * texSize.x
  ));
  let radius = smoothstep(0.0, 1.0,
    abs(dot(texCoord * texSize - tiltShift.start * texSize, normal)) / tiltShift.gradientRadius) * tiltShift.blurRadius;
  let delta = tiltShift_getDelta(texSize);
  for (var t = -30.0; t <= 30.0; t += 1.0) {
    let percent = (t + offset - 0.5) / 30.0;
    let weight = 1.0 - abs(percent);
    var offsetColor = screen_texture(source, sourceSampler, texCoord + delta / texSize * percent * radius);
    // switch to pre-multiplied alpha to correctly blur transparent images
    offsetColor = vec4<f32>(offsetColor.rgb * offsetColor.a, offsetColor.a);
    color += offsetColor * weight;
    total += weight;
  }
  color = color / total;
  // switch back from pre-multiplied alpha
  return vec4<f32>(color.rgb / (color.a + 0.00001), color.a);
}

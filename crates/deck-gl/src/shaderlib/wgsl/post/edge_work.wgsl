struct edgeWorkUniforms {
  radius: f32,
  mode: i32,
};
@group(0) @binding(auto) var<uniform> edgeWork: edgeWorkUniforms;

fn edgeWork_sampleColorRGB(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>, delta: vec2<f32>) -> vec4<f32> {
  let relativeDelta = edgeWork.radius * delta / texSize;
  var color = vec2<f32>(0.0);
  var total = vec2<f32>(0.0);
  // randomize the lookup values to hide the fixed number of samples
  let offset = random(vec3<f32>(12.9898, 78.233, 151.7182), 0.0);
  for (var t = -30.0; t <= 30.0; t += 1.0) {
    let percent = (t + offset - 0.5) / 30.0;
    var weight = 1.0 - abs(percent);
    let sampleColor = screen_texture(source, sourceSampler, texCoord + relativeDelta * percent).rgb;
    let average = (sampleColor.r + sampleColor.g + sampleColor.b) / 3.0;
    color.x += average * weight;
    total.x += weight;
    if (abs(t) < 15.0) {
      weight = weight * 2.0 - 1.0;
      color.y += average * weight;
      total.y += weight;
    }
  }
  return vec4<f32>(color / total, 0.0, 1.0);
}

fn edgeWork_sampleColorXY(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>, delta: vec2<f32>) -> vec4<f32> {
  let relativeDelta = edgeWork.radius * delta / texSize;
  var color = vec2<f32>(0.0);
  var total = vec2<f32>(0.0);
  // randomize the lookup values to hide the fixed number of samples
  let offset = random(vec3<f32>(12.9898, 78.233, 151.7182), 0.0);
  for (var t = -30.0; t <= 30.0; t += 1.0) {
    let percent = (t + offset - 0.5) / 30.0;
    var weight = 1.0 - abs(percent);
    let sampleColor = screen_texture(source, sourceSampler, texCoord + relativeDelta * percent).xy;
    color.x += sampleColor.x * weight;
    total.x += weight;
    if (abs(t) < 15.0) {
      weight = weight * 2.0 - 1.0;
      color.y += sampleColor.y * weight;
      total.y += weight;
    }
  }
  let c = clamp(10000.0 * (color.y / total.y - color.x / total.x) + 0.5, 0.0, 1.0);
  return vec4<f32>(c, c, c, 1.0);
}

fn edgeWork_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  if (edgeWork.mode == 0) {
    return edgeWork_sampleColorRGB(source, sourceSampler, texSize, texCoord, vec2<f32>(1.0, 0.0));
  }
  return edgeWork_sampleColorXY(source, sourceSampler, texSize, texCoord, vec2<f32>(0.0, 1.0));
}

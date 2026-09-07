struct swirlUniforms {
  radius: f32,
  angle: f32,
  center: vec2<f32>,
};
@group(0) @binding(auto) var<uniform> swirl: swirlUniforms;

fn swirl_warp(coordIn: vec2<f32>, texCenter: vec2<f32>) -> vec2<f32> {
  var coord = coordIn - texCenter;
  let dist = length(coord);
  if (dist < swirl.radius) {
    let percent = (swirl.radius - dist) / swirl.radius;
    let theta = percent * percent * swirl.angle;
    let s = sin(theta);
    let c = cos(theta);
    coord = vec2<f32>(coord.x * c - coord.y * s, coord.x * s + coord.y * c);
  }
  return coord + texCenter;
}

fn swirl_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let coord = swirl_warp(texCoord * texSize, swirl.center * texSize);
  return warp_sampleColor(source, sourceSampler, texSize, coord);
}

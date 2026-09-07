struct bulgePinchUniforms {
  radius: f32,
  strength: f32,
  center: vec2<f32>,
};
@group(0) @binding(auto) var<uniform> bulgePinch: bulgePinchUniforms;

fn bulgePinch_warp(coordIn: vec2<f32>, texCenter: vec2<f32>) -> vec2<f32> {
  var coord = coordIn - texCenter;
  let dist = length(coord);
  if (dist < bulgePinch.radius) {
    let percent = dist / bulgePinch.radius;
    if (bulgePinch.strength > 0.0) {
      coord *= mix(1.0, smoothstep(0.0, bulgePinch.radius / dist, percent), bulgePinch.strength * 0.75);
    } else {
      coord *= mix(1.0, pow(percent, 1.0 + bulgePinch.strength * 0.75) * bulgePinch.radius / dist, 1.0 - percent);
    }
  }
  return coord + texCenter;
}

fn bulgePinch_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let coord = bulgePinch_warp(texCoord * texSize, bulgePinch.center * texSize);
  return warp_sampleColor(source, sourceSampler, texSize, coord);
}

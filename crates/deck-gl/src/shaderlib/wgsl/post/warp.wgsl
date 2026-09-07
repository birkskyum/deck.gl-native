fn warp_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, coord: vec2<f32>) -> vec4<f32> {
  var color = screen_texture(source, sourceSampler, coord / texSize);
  let clampedCoord = clamp(coord, vec2<f32>(0.0), texSize);
  if (any(coord != clampedCoord)) {
    // fade to transparent if we are outside the image
    color.a *= max(0.0, 1.0 - length(coord - clampedCoord));
  }
  return color;
}

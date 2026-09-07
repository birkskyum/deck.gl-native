// Port of deck.gl's screen pass uniforms and luma.gl's `random` module for post-processing
// passes. Texture coordinates follow luma.gl: (0, 0) is the bottom left of the frame, so
// props written for deck.gl keep their meaning; sampling flips them for the top left
// origin of wgpu textures.

struct screenUniforms {
  texSize: vec2<f32>,
};
@group(0) @binding(auto) var<uniform> screen: screenUniforms;
@group(0) @binding(auto) var texSrc: texture_2d<f32>;
@group(0) @binding(auto) var texSrcSampler: sampler;

// The fragment position, set by the pass before the filter runs
var<private> screen_fragCoord: vec3<f32>;

fn screen_texture(source: texture_2d<f32>, sourceSampler: sampler, uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(source, sourceSampler, vec2<f32>(uv.x, 1.0 - uv.y), 0.0);
}

fn random(scale: vec3<f32>, seed: f32) -> f32 {
  // use the fragment position for a different seed per pixel
  return fract(sin(dot(screen_fragCoord + seed, scale)) * 43758.5453 + seed);
}

// Port of @deck.gl/aggregation-layers/src/heatmap-layer/weights.wgsl.ts: splats every point
// into the weights texture with a gaussian falloff, blended additively.

struct weightUniforms {
  commonBounds: vec4<f32>,
  radiusPixels: f32,
  textureWidth: f32,
  weightsScale: f32,
};
@group(0) @binding(auto) var<uniform> weight: weightUniforms;

struct WeightVaryings {
  @builtin(position) position: vec4<f32>,
  @location(0) pointCoordinates: vec2<f32>,
  @location(1) weights: vec4<f32>,
};

const WEIGHT_QUAD_POSITIONS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(1.0, 1.0)
);

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) instancePositions: vec3<f32>,
  @location(1) instancePositions64Low: vec3<f32>,
  @location(2) instanceWeights: f32
) -> WeightVaryings {
  var output: WeightVaryings;
  let commonPosition = project_position_vec3_f64(instancePositions, instancePositions64Low);
  let clipPosition =
    (commonPosition.xy - weight.commonBounds.xy) /
    (weight.commonBounds.zw - weight.commonBounds.xy) * 2.0 - vec2<f32>(1.0);
  let radiusTexels =
    project_pixel_size_float(weight.radiusPixels) * weight.textureWidth /
    (weight.commonBounds.z - weight.commonBounds.x);
  let offset = WEIGHT_QUAD_POSITIONS[vertexIndex];
  output.position = vec4<f32>(
    clipPosition + offset * radiusTexels * 2.0 / weight.textureWidth,
    0.0,
    1.0
  );
  output.pointCoordinates = (offset + vec2<f32>(1.0)) * 0.5;
  output.weights = vec4<f32>(instanceWeights * weight.weightsScale, 0.0, 0.0, 1.0);
  return output;
}

@fragment
fn fragmentMain(varyings: WeightVaryings) -> @location(0) vec4<f32> {
  let distanceFromCenter = length(varyings.pointCoordinates - vec2<f32>(0.5));
  if (distanceFromCenter > 0.5) {
    discard;
  }
  let gaussian = exp(-pow(2.0 * distanceFromCenter, 2.0) / 0.05555) /
    (1.77245385 * 0.166666);
  return varyings.weights * gaussian;
}

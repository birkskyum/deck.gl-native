struct brightnessContrastUniforms {
  brightness: f32,
  contrast: f32,
};
@group(0) @binding(auto) var<uniform> brightnessContrast: brightnessContrastUniforms;

fn brightnessContrast_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  var rgb = color.rgb + brightnessContrast.brightness;
  if (brightnessContrast.contrast > 0.0) {
    rgb = (rgb - 0.5) / (1.0 - brightnessContrast.contrast) + 0.5;
  } else {
    rgb = (rgb - 0.5) * (1.0 + brightnessContrast.contrast) + 0.5;
  }
  return vec4<f32>(rgb, color.a);
}

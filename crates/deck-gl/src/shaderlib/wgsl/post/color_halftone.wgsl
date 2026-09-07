struct colorHalftoneUniforms {
  center: vec2<f32>,
  angle: f32,
  size: f32,
};
@group(0) @binding(auto) var<uniform> colorHalftone: colorHalftoneUniforms;

fn colorHalftone_pattern(angle: f32, scale: f32, texSize: vec2<f32>, texCoord: vec2<f32>) -> f32 {
  let s = sin(angle);
  let c = cos(angle);
  let tex = texCoord * texSize - colorHalftone.center * texSize;
  let point = vec2<f32>(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
  return (sin(point.x) * sin(point.y)) * 4.0;
}

fn colorHalftone_filterColor_ext(color: vec4<f32>, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  let scale = 3.1514 / colorHalftone.size;
  var cmy = 1.0 - color.rgb;
  var k = min(cmy.x, min(cmy.y, cmy.z));
  cmy = (cmy - k) / (1.0 - k);
  cmy = clamp(
    cmy * 10.0 - 3.0 + vec3<f32>(
      colorHalftone_pattern(colorHalftone.angle + 0.26179, scale, texSize, texCoord),
      colorHalftone_pattern(colorHalftone.angle + 1.30899, scale, texSize, texCoord),
      colorHalftone_pattern(colorHalftone.angle, scale, texSize, texCoord)
    ),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
  k = clamp(k * 10.0 - 5.0 + colorHalftone_pattern(colorHalftone.angle + 0.78539, scale, texSize, texCoord), 0.0, 1.0);
  return vec4<f32>(1.0 - cmy - k, color.a);
}

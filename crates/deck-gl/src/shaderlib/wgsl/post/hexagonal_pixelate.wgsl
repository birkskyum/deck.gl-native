struct hexagonalPixelateUniforms {
  center: vec2<f32>,
  scale: f32,
};
@group(0) @binding(auto) var<uniform> hexagonalPixelate: hexagonalPixelateUniforms;

fn hexagonalPixelate_sampleColor(source: texture_2d<f32>, sourceSampler: sampler, texSize: vec2<f32>, texCoord: vec2<f32>) -> vec4<f32> {
  var tex = (texCoord * texSize - hexagonalPixelate.center * texSize) / hexagonalPixelate.scale;
  tex.y /= 0.866025404;
  tex.x -= tex.y * 0.5;

  var a: vec2<f32>;
  if (tex.x + tex.y - floor(tex.x) - floor(tex.y) < 1.0) {
    a = vec2<f32>(floor(tex.x), floor(tex.y));
  } else {
    a = vec2<f32>(ceil(tex.x), ceil(tex.y));
  }
  let b = vec2<f32>(ceil(tex.x), floor(tex.y));
  let c = vec2<f32>(floor(tex.x), ceil(tex.y));

  let TEX = vec3<f32>(tex.x, tex.y, 1.0 - tex.x - tex.y);
  let A = vec3<f32>(a.x, a.y, 1.0 - a.x - a.y);
  let B = vec3<f32>(b.x, b.y, 1.0 - b.x - b.y);
  let C = vec3<f32>(c.x, c.y, 1.0 - c.x - c.y);

  let alen = length(TEX - A);
  let blen = length(TEX - B);
  let clen = length(TEX - C);

  var choice: vec2<f32>;
  if (alen < blen) {
    choice = select(c, a, alen < clen);
  } else {
    choice = select(c, b, blen < clen);
  }

  choice.x += choice.y * 0.5;
  choice.y *= 0.866025404;
  choice *= hexagonalPixelate.scale / texSize;

  return screen_texture(source, sourceSampler, choice + hexagonalPixelate.center);
}

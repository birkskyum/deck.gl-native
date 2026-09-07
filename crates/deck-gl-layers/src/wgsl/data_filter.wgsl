// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/data-filter/shader-module.ts without the preprocessor:
// values are widened to vec4 and the channel count masks the unused components.

struct DataFilterUniforms {
  min: vec4<f32>,
  softMin: vec4<f32>,
  softMax: vec4<f32>,
  max: vec4<f32>,
  categoryBitMask: vec4<u32>,
  useSoftMargin: i32,
  enabled: i32,
  transformSize: i32,
  transformColor: i32,
};

@group(0) @binding(auto) var<uniform> dataFilter: DataFilterUniforms;

// 1 where a value is inside the range, fading over the soft margins, per channel.
fn dataFilter_inRange(value: vec4<f32>) -> vec4<f32> {
  if (dataFilter.useSoftMargin != 0) {
    // smoothstep results are undefined if edge0 >= edge1: fall back to a hard edge where the
    // soft range is truncated by the range
    let leftInRange = mix(
      smoothstep(dataFilter.min, dataFilter.softMin, value),
      step(dataFilter.min, value),
      step(dataFilter.softMin, dataFilter.min)
    );
    let rightInRange = mix(
      1.0 - smoothstep(dataFilter.softMax, dataFilter.max, value),
      step(value, dataFilter.max),
      step(dataFilter.max, dataFilter.softMax)
    );
    return leftInRange * rightInRange;
  }
  return step(dataFilter.min, value) * step(value, dataFilter.max);
}

// The filter value of an object from its first `channels` values: the smallest per channel
// result, 0 hidden, 1 fully shown.
fn dataFilter_rangeValue(value: vec4<f32>, channels: u32) -> f32 {
  let unused = vec4<u32>(0u, 1u, 2u, 3u) >= vec4<u32>(channels);
  let r = select(dataFilter_inRange(value), vec4<f32>(1.0), unused);
  return min(min(r.x, r.y), min(r.z, r.w));
}

// Whether each of the first `channels` category keys has its bit set in the mask: one channel
// uses all 128 bits, two channels 64 bits each, three or four channels 32 bits each.
fn dataFilter_categoryPasses(category: vec4<u32>, channels: u32) -> bool {
  var words: vec4<u32>;
  if (channels == 1u) {
    words = vec4<u32>(dataFilter.categoryBitMask[min(category.x / 32u, 3u)], 0u, 0u, 0u);
  } else if (channels == 2u) {
    words = vec4<u32>(
      dataFilter.categoryBitMask[min(category.x / 32u, 1u)],
      dataFilter.categoryBitMask[min(category.y / 32u, 1u) + 2u],
      0u,
      0u
    );
  } else {
    words = dataFilter.categoryBitMask;
  }
  let bits = (words >> (category & vec4<u32>(31u))) & vec4<u32>(1u);
  let unused = vec4<u32>(0u, 1u, 2u, 3u) >= vec4<u32>(channels);
  return all((bits == vec4<u32>(1u)) | unused);
}

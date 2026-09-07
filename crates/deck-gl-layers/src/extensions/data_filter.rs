//! Port of `@deck.gl/extensions/src/data-filter`: show or hide objects on the GPU by one to
//! four numeric values and one to four categories per object, with optional soft margins that
//! fade objects out through their size and opacity.

use std::any::Any;

use deck_gl::attribute_manager::AttributeSource;
use deck_gl::data::{resolve_f32, resolve_vec2, resolve_vec3, resolve_vec4};
use deck_gl::glam::Vec4;
use deck_gl::luma_gl::{Model, ShaderField, ShaderInjection, ShaderModuleSource};
use deck_gl::{
    same_extension, Accessor, ExtensionAttribute, ExtensionShaders, LayerContext, LayerData, LayerExtension,
    Result, Viewport,
};
use wgpu::VertexFormat;

const MODULE: ShaderModuleSource = ShaderModuleSource {
    name: "dataFilter",
    source: include_str!("../wgsl/data_filter.wgsl"),
};

/// The numeric values an object is filtered by, deck.gl's `filterSize` with `getFilterValue`.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterValues {
    One(Accessor<f32>),
    Two(Accessor<[f32; 2]>),
    Three(Accessor<[f32; 3]>),
    Four(Accessor<[f32; 4]>),
}

impl FilterValues {
    /// Number of channels, 1 to 4.
    pub fn size(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
        }
    }

    fn format(&self) -> VertexFormat {
        match self {
            Self::One(_) => VertexFormat::Float32,
            Self::Two(_) => VertexFormat::Float32x2,
            Self::Three(_) => VertexFormat::Float32x3,
            Self::Four(_) => VertexFormat::Float32x4,
        }
    }

    fn source(&self) -> AttributeSource {
        match self {
            Self::One(a) => AttributeSource::Floats(a.clone()),
            Self::Two(a) => AttributeSource::Vec2(a.clone()),
            Self::Three(a) => AttributeSource::Vec3(a.clone()),
            Self::Four(a) => AttributeSource::Vec4(a.clone()),
        }
    }

    /// The attribute widened to a `vec4<f32>` in WGSL.
    fn vec4_expression(&self) -> &'static str {
        match self {
            Self::One(_) => "vec4<f32>(filterValues, 0.0, 0.0, 0.0)",
            Self::Two(_) => "vec4<f32>(filterValues, 0.0, 0.0)",
            Self::Three(_) => "vec4<f32>(filterValues, 0.0)",
            Self::Four(_) => "filterValues",
        }
    }

    /// Every object's values, padded to four channels.
    pub fn resolve(&self, data: &LayerData) -> Result<Vec<[f32; 4]>> {
        Ok(match self {
            Self::One(a) => resolve_f32(data, a)?
                .into_iter()
                .map(|v| [v, 0.0, 0.0, 0.0])
                .collect(),
            Self::Two(a) => resolve_vec2(data, a)?
                .into_iter()
                .map(|v| [v[0], v[1], 0.0, 0.0])
                .collect(),
            Self::Three(a) => resolve_vec3(data, a)?
                .into_iter()
                .map(|v| [v[0], v[1], v[2], 0.0])
                .collect(),
            Self::Four(a) => resolve_vec4(data, a)?,
        })
    }
}

/// The categories an object is filtered by, deck.gl's `categorySize` with `getFilterCategory`.
///
/// deck.gl maps arbitrary category values to small integer keys as it meets them; here the
/// keys are given directly (the JSON layers do the mapping for strings). Keys must stay below
/// 128 with one channel, 64 with two and 32 with three or four; larger keys never match.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterCategories {
    One(Accessor<u32>),
    Two(Accessor<[u32; 2]>),
    Three(Accessor<[u32; 3]>),
    Four(Accessor<[u32; 4]>),
}

fn map_accessor<A, B>(accessor: &Accessor<A>, f: impl Fn(A) -> B + Send + Sync + 'static) -> Accessor<B>
where
    A: Clone + Send + Sync + 'static,
    B: Clone + Send + Sync + 'static,
{
    match accessor {
        Accessor::Constant(value) => Accessor::Constant(f(value.clone())),
        Accessor::Column(name) => Accessor::Column(name.clone()),
        Accessor::Func(func) => {
            let func = func.clone();
            Accessor::func(move |i| f(func(i)))
        }
    }
}

impl FilterCategories {
    /// Number of channels, 1 to 4.
    pub fn size(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
        }
    }

    /// The largest number of distinct keys per channel deck.gl's 128 bit mask holds.
    pub fn max_keys(&self) -> u32 {
        match self.size() {
            1 => 128,
            2 => 64,
            _ => 32,
        }
    }

    fn format(&self) -> VertexFormat {
        match self {
            Self::One(_) => VertexFormat::Float32,
            Self::Two(_) => VertexFormat::Float32x2,
            Self::Three(_) => VertexFormat::Float32x3,
            Self::Four(_) => VertexFormat::Float32x4,
        }
    }

    /// Keys travel as floats (exact below 2^24) so that the attribute manager's float paths,
    /// including integer Arrow columns, apply.
    fn source(&self) -> AttributeSource {
        match self {
            Self::One(a) => AttributeSource::Floats(map_accessor(a, |v| v as f32)),
            Self::Two(a) => AttributeSource::Vec2(map_accessor(a, |v| [v[0] as f32, v[1] as f32])),
            Self::Three(a) => {
                AttributeSource::Vec3(map_accessor(a, |v| [v[0] as f32, v[1] as f32, v[2] as f32]))
            }
            Self::Four(a) => AttributeSource::Vec4(map_accessor(a, |v| {
                [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]
            })),
        }
    }

    fn vec4_expression(&self) -> &'static str {
        match self {
            Self::One(_) => "vec4<u32>(u32(filterCategoryValues), 0u, 0u, 0u)",
            Self::Two(_) => "vec4<u32>(vec2<u32>(filterCategoryValues), 0u, 0u)",
            Self::Three(_) => "vec4<u32>(vec3<u32>(filterCategoryValues), 0u)",
            Self::Four(_) => "vec4<u32>(filterCategoryValues)",
        }
    }

    /// Every object's keys, padded to four channels.
    pub fn resolve(&self, data: &LayerData) -> Result<Vec<[u32; 4]>> {
        let floats = match self.source() {
            AttributeSource::Floats(a) => resolve_f32(data, &a)?
                .into_iter()
                .map(|v| [v, 0.0, 0.0, 0.0])
                .collect(),
            AttributeSource::Vec2(a) => resolve_vec2(data, &a)?
                .into_iter()
                .map(|v| [v[0], v[1], 0.0, 0.0])
                .collect(),
            AttributeSource::Vec3(a) => resolve_vec3(data, &a)?
                .into_iter()
                .map(|v| [v[0], v[1], v[2], 0.0])
                .collect(),
            AttributeSource::Vec4(a) => resolve_vec4(data, &a)?,
            _ => Vec::new(),
        };
        Ok(floats
            .into_iter()
            .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32, v[3] as u32])
            .collect())
    }
}

/// deck.gl's `DataFilterExtension`: objects whose filter value lies outside `filter_range`
/// are hidden on the GPU, without touching the layer's data. With `filter_soft_range` objects
/// fade in and out between the two ranges through their size and opacity. Categories hide
/// every object whose keys are not all in `filter_categories`.
#[derive(Clone, Debug, PartialEq)]
pub struct DataFilterExtension {
    pub get_filter_value: FilterValues,
    /// `[min, max]` per channel; a single pair applies to every channel.
    pub filter_range: Vec<[f32; 2]>,
    /// Inside `filter_range`, the values that are fully shown; objects between the two ranges fade.
    pub filter_soft_range: Option<Vec<[f32; 2]>>,
    pub filter_enabled: bool,
    /// Shrink fading objects
    pub filter_transform_size: bool,
    /// Make fading objects translucent
    pub filter_transform_color: bool,
    pub get_filter_category: Option<FilterCategories>,
    /// The keys shown, per channel.
    pub filter_categories: Vec<Vec<u32>>,
}

impl Default for DataFilterExtension {
    fn default() -> Self {
        Self {
            get_filter_value: FilterValues::One(Accessor::Constant(0.0)),
            filter_range: vec![[-1.0, 1.0]],
            filter_soft_range: None,
            filter_enabled: true,
            filter_transform_size: true,
            filter_transform_color: true,
            get_filter_category: None,
            filter_categories: vec![vec![0]],
        }
    }
}

impl DataFilterExtension {
    /// Filter one value per object by `range`.
    pub fn new(get_filter_value: Accessor<f32>, range: [f32; 2]) -> Self {
        Self {
            get_filter_value: FilterValues::One(get_filter_value),
            filter_range: vec![range],
            ..Default::default()
        }
    }

    fn channel_range(&self, channel: usize) -> [f32; 2] {
        self.filter_range
            .get(channel)
            .or_else(|| self.filter_range.first())
            .copied()
            .unwrap_or([-1.0, 1.0])
    }

    fn channel_soft_range(&self, channel: usize) -> [f32; 2] {
        self.filter_soft_range
            .as_ref()
            .and_then(|soft| soft.get(channel).or_else(|| soft.first()))
            .copied()
            .unwrap_or_else(|| self.channel_range(channel))
    }

    /// deck.gl's category bit mask: one bit per shown key, laid out per channel count.
    pub fn category_bit_mask(&self) -> [u32; 4] {
        let mut mask = [0u32; 4];
        let Some(categories) = &self.get_filter_category else {
            return mask;
        };
        let max = categories.max_keys();
        for (channel, keys) in self.filter_categories.iter().enumerate().take(categories.size()) {
            for &key in keys {
                if key < max {
                    let word = channel * (max as usize / 32) + (key / 32) as usize;
                    mask[word] |= 1 << (key % 32);
                } else {
                    tracing::warn!("data filter: category key {key} exceeds the maximum of {max} keys");
                }
            }
        }
        mask
    }

    /// Whether an object with these values and keys is shown at all (the hard ranges, the
    /// same test the shader makes before fading).
    pub fn passes(&self, values: [f32; 4], keys: [u32; 4]) -> bool {
        if !self.filter_enabled {
            return true;
        }
        for (channel, value) in values.iter().enumerate().take(self.get_filter_value.size()) {
            let [min, max] = self.channel_range(channel);
            if *value < min || *value > max {
                return false;
            }
        }
        if let Some(categories) = &self.get_filter_category {
            for (channel, key) in keys.iter().enumerate().take(categories.size()) {
                let shown = self.filter_categories.get(channel);
                if !shown.is_some_and(|keys_shown| keys_shown.contains(key)) {
                    return false;
                }
            }
        }
        true
    }

    /// The number of objects of `data` the filter shows, deck.gl's `onFilteredItemsChange`
    /// count computed on the CPU.
    pub fn count_filtered(&self, data: &LayerData) -> Result<usize> {
        if !self.filter_enabled {
            return Ok(data.len());
        }
        let values = self.get_filter_value.resolve(data)?;
        let keys = match &self.get_filter_category {
            Some(categories) => categories.resolve(data)?,
            None => Vec::new(),
        };
        Ok(values
            .iter()
            .enumerate()
            .filter(|(i, v)| self.passes(**v, keys.get(*i).copied().unwrap_or_default()))
            .count())
    }
}

impl LayerExtension for DataFilterExtension {
    fn name(&self) -> &'static str {
        "DataFilterExtension"
    }

    fn shaders(&self) -> ExtensionShaders {
        let values = &self.get_filter_value;
        let mut main_start = format!(
            "dataFilter_value = 1.0;\n  if (dataFilter.enabled != 0) {{\n    dataFilter_value = dataFilter_rangeValue({}, {}u);\n",
            values.vec4_expression(),
            values.size()
        );
        let mut attributes = vec![ExtensionAttribute {
            name: "filterValues",
            format: values.format(),
        }];
        if let Some(categories) = &self.get_filter_category {
            main_start.push_str(&format!(
                "    if (!dataFilter_categoryPasses({}, {}u)) {{\n      dataFilter_value = 0.0;\n    }}\n",
                categories.vec4_expression(),
                categories.size()
            ));
            attributes.push(ExtensionAttribute {
                name: "filterCategoryValues",
                format: categories.format(),
            });
        }
        main_start.push_str("  }");
        ExtensionShaders {
            modules: vec![MODULE],
            injections: vec![
                ShaderInjection::new("vs:#main-start", main_start),
                ShaderInjection::new(
                    "vs:DECKGL_FILTER_GL_POSITION",
                    "if (dataFilter_value == 0.0) {\n    position = vec4<f32>(0.0);\n  }",
                ),
                ShaderInjection::new(
                    "vs:DECKGL_FILTER_SIZE",
                    "if (dataFilter.transformSize != 0) {\n    size = size * dataFilter_value;\n  }",
                ),
                ShaderInjection::new(
                    "fs:DECKGL_FILTER_COLOR",
                    "if (dataFilter_value == 0.0) {\n    discard;\n  }\n  if (dataFilter.transformColor != 0) {\n    color.a *= dataFilter_value;\n  }",
                ),
            ],
            attributes,
            varyings: vec![ShaderField {
                name: "dataFilter_value",
                ty: "f32",
            }],
        }
    }

    fn attributes(&self) -> Vec<(&'static str, AttributeSource)> {
        let mut sources = vec![("filterValues", self.get_filter_value.source())];
        if let Some(categories) = &self.get_filter_category {
            sources.push(("filterCategoryValues", categories.source()));
        }
        sources
    }

    fn update_uniforms(&self, model: &mut Model, _ctx: &LayerContext, _viewport: &Viewport) -> Result<()> {
        let mut min = [0.0f32; 4];
        let mut soft_min = [0.0f32; 4];
        let mut soft_max = [0.0f32; 4];
        let mut max = [0.0f32; 4];
        for channel in 0..self.get_filter_value.size() {
            [min[channel], max[channel]] = self.channel_range(channel);
            [soft_min[channel], soft_max[channel]] = self.channel_soft_range(channel);
        }
        let mask: Vec<u8> = self
            .category_bit_mask()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let enabled = self.filter_enabled;
        let u = model.uniforms("dataFilter")?;
        u.set_vec4("min", Vec4::from(min))?;
        u.set_vec4("softMin", Vec4::from(soft_min))?;
        u.set_vec4("softMax", Vec4::from(soft_max))?;
        u.set_vec4("max", Vec4::from(max))?;
        u.set_bytes("categoryBitMask", &mask)?;
        u.set_i32("useSoftMargin", self.filter_soft_range.is_some() as i32)?;
        u.set_i32("enabled", enabled as i32)?;
        u.set_i32("transformSize", (enabled && self.filter_transform_size) as i32)?;
        u.set_i32("transformColor", (enabled && self.filter_transform_color) as i32)?;
        Ok(())
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_mask_follows_the_channel_layout() {
        let one = DataFilterExtension {
            get_filter_category: Some(FilterCategories::One(Accessor::Constant(0))),
            filter_categories: vec![vec![0, 33, 127]],
            ..Default::default()
        };
        assert_eq!(one.category_bit_mask(), [1, 2, 0, 1 << 31]);
        let two = DataFilterExtension {
            get_filter_category: Some(FilterCategories::Two(Accessor::Constant([0, 0]))),
            filter_categories: vec![vec![1, 40], vec![3]],
            ..Default::default()
        };
        assert_eq!(two.category_bit_mask(), [2, 1 << 8, 8, 0]);
        let four = DataFilterExtension {
            get_filter_category: Some(FilterCategories::Four(Accessor::Constant([0; 4]))),
            filter_categories: vec![vec![0], vec![1], vec![2], vec![31, 32]],
            ..Default::default()
        };
        assert_eq!(four.category_bit_mask(), [1, 2, 4, 1 << 31]);
    }

    #[test]
    fn cpu_count_matches_the_ranges_and_categories() {
        let data = LayerData::with_length(5);
        let filter = DataFilterExtension {
            get_filter_value: FilterValues::Two(Accessor::func(|i| [i as f32, 10.0 - i as f32])),
            filter_range: vec![[1.0, 3.0], [7.5, 10.0]],
            get_filter_category: Some(FilterCategories::One(Accessor::func(|i| i as u32 % 2))),
            filter_categories: vec![vec![0]],
            ..Default::default()
        };
        // rows 1..=3 pass the ranges (second channel 9, 8, 7 needs >= 7.5: rows 1 and 2),
        // category 0 keeps even rows: row 2
        assert_eq!(filter.count_filtered(&data).unwrap(), 1);
        let disabled = DataFilterExtension {
            filter_enabled: false,
            ..filter
        };
        assert_eq!(disabled.count_filtered(&data).unwrap(), 5);
    }
}

//! Value aggregation and the colour and elevation scales of deck.gl's aggregation layers
//! (`cpu-aggregator/aggregate.ts`, `scale-utils.ts` and the cell layers' domain logic).

use deck_gl::Color;

/// How the weights of the points in a bin combine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AggregationOperation {
    #[default]
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

impl AggregationOperation {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "SUM" => Some(Self::Sum),
            "MEAN" => Some(Self::Mean),
            "MIN" => Some(Self::Min),
            "MAX" => Some(Self::Max),
            "COUNT" => Some(Self::Count),
            _ => None,
        }
    }

    /// Aggregate the weights of one bin.
    pub fn apply(self, points: &[usize], weights: &[f32]) -> f32 {
        match self {
            Self::Count => points.len() as f32,
            Self::Sum => points.iter().map(|i| weights[*i]).sum(),
            Self::Mean => {
                if points.is_empty() {
                    f32::NAN
                } else {
                    points.iter().map(|i| weights[*i]).sum::<f32>() / points.len() as f32
                }
            }
            Self::Min => points.iter().map(|i| weights[*i]).fold(f32::INFINITY, f32::min),
            Self::Max => points
                .iter()
                .map(|i| weights[*i])
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

/// Aggregate every bin and return the values with their min and max.
pub fn aggregate(
    bins: &[Vec<usize>],
    weights: &[f32],
    operation: AggregationOperation,
) -> (Vec<f32>, [f32; 2]) {
    let mut domain = [f32::INFINITY, f32::NEG_INFINITY];
    let values: Vec<f32> = bins
        .iter()
        .map(|points| {
            let v = operation.apply(points, weights);
            if v < domain[0] {
                domain[0] = v;
            }
            if v > domain[1] {
                domain[1] = v;
            }
            v
        })
        .collect();
    (values, domain)
}

/// How aggregated values map onto a colour range or elevation range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScaleType {
    /// Equal value intervals, one colour each
    #[default]
    Quantize,
    /// Interpolate between the range's entries
    Linear,
    /// Equal count intervals (percentiles)
    Quantile,
    /// One entry per distinct value
    Ordinal,
}

impl ScaleType {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "quantize" => Some(Self::Quantize),
            "linear" => Some(Self::Linear),
            "quantile" => Some(Self::Quantile),
            "ordinal" => Some(Self::Ordinal),
            _ => None,
        }
    }
}

/// Percentile thresholds of the finite values (99 of them for 100 buckets).
fn quantile_thresholds(values: &[f32], buckets: usize) -> Vec<f32> {
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let n = buckets.max(1);
    (1..n).map(|i| threshold(&sorted, i as f64 / n as f64)).collect()
}

fn threshold(sorted: &[f32], fraction: f64) -> f32 {
    let len = sorted.len();
    if len == 0 {
        return f32::NAN;
    }
    if fraction <= 0.0 || len < 2 {
        return sorted[0];
    }
    if fraction >= 1.0 {
        return sorted[len - 1];
    }
    let position = (len - 1) as f64 * fraction;
    let low_index = position.floor() as usize;
    let low = sorted[low_index] as f64;
    let high = sorted[(low_index + 1).min(len - 1)] as f64;
    (low + (high - low) * (position - low_index as f64)) as f32
}

fn bisect_right(thresholds: &[f32], x: f32) -> usize {
    thresholds.partition_point(|t| *t <= x)
}

/// Aggregated values prepared for a scale: transformed for quantile and ordinal scales, with
/// the domain and the percentile cutoff that hides bins outside `lower..upper`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaledValues {
    pub values: Vec<f32>,
    pub domain: [f32; 2],
    pub cutoff: Option<[f32; 2]>,
}

impl ScaledValues {
    /// `domain` overrides the data's min and max for linear and quantize scales.
    pub fn new(
        values: Vec<f32>,
        data_domain: [f32; 2],
        domain: Option<[f32; 2]>,
        scale: ScaleType,
        lower_percentile: f32,
        upper_percentile: f32,
    ) -> Self {
        let thresholds = quantile_thresholds(&values, 100);
        let percentile_cutoff = |scaled: &dyn Fn(f32) -> f32| -> Option<[f32; 2]> {
            if lower_percentile > 0.0 || upper_percentile < 100.0 {
                let low = (lower_percentile.floor() as i64 - 1)
                    .try_into()
                    .ok()
                    .and_then(|i: usize| thresholds.get(i).copied())
                    .unwrap_or(f32::NEG_INFINITY);
                let high = (upper_percentile.floor() as i64 - 1)
                    .try_into()
                    .ok()
                    .and_then(|i: usize| thresholds.get(i).copied())
                    .unwrap_or(f32::INFINITY);
                Some([scaled(low), scaled(high)])
            } else {
                None
            }
        };
        match scale {
            ScaleType::Quantile => {
                let mapped = values
                    .iter()
                    .map(|v| {
                        if v.is_finite() {
                            bisect_right(&thresholds, *v) as f32
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                Self {
                    values: mapped,
                    domain: [0.0, 99.0],
                    cutoff: Some([lower_percentile, upper_percentile - 1.0]),
                }
            }
            ScaleType::Ordinal => {
                let mut unique: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
                unique.sort_by(f32::total_cmp);
                unique.dedup();
                let index_of = |v: f32| unique.partition_point(|u| *u < v) as f32;
                let mapped = values
                    .iter()
                    .map(|v| if v.is_finite() { index_of(*v) } else { f32::NAN })
                    .collect();
                let cutoff = percentile_cutoff(&|v: f32| {
                    if v == f32::NEG_INFINITY {
                        f32::NEG_INFINITY
                    } else if v == f32::INFINITY {
                        f32::INFINITY
                    } else {
                        index_of(v)
                    }
                });
                Self {
                    values: mapped,
                    domain: [0.0, unique.len().saturating_sub(1) as f32],
                    cutoff,
                }
            }
            ScaleType::Linear | ScaleType::Quantize => Self {
                values,
                domain: domain.unwrap_or(data_domain),
                cutoff: percentile_cutoff(&|v| v),
            },
        }
    }

    /// Whether the bin at `index` is shown at all (inside the cutoff and not NaN).
    pub fn visible(&self, index: usize) -> bool {
        let v = self.values[index];
        if !v.is_finite() && !(v.is_infinite()) {
            return false;
        }
        let [low, high] = self.cutoff.unwrap_or([f32::NEG_INFINITY, f32::INFINITY]);
        v >= low && v <= high
    }

    /// Position of the value within the domain, 0..1.
    pub fn ratio(&self, index: usize) -> f32 {
        let [d0, d1] = self.domain;
        if d1 == d0 {
            return 0.0;
        }
        ((self.values[index] - d0) / (d1 - d0)).clamp(0.0, 1.0)
    }
}

/// deck.gl's default colour range, 6 steps from pale yellow to dark red.
pub const DEFAULT_COLOR_RANGE: [Color; 6] = [
    [255, 255, 178, 255],
    [254, 217, 118, 255],
    [254, 178, 76, 255],
    [253, 141, 60, 255],
    [240, 59, 32, 255],
    [189, 0, 38, 255],
];

/// Colour for a 0..1 ratio: nearest entry for stepped scales, interpolated for linear, like
/// sampling deck.gl's colour range texture.
pub fn sample_color_range(ratio: f32, range: &[Color], scale: ScaleType) -> Color {
    if range.is_empty() {
        return [0, 0, 0, 255];
    }
    let n = range.len();
    let ratio = ratio.clamp(0.0, 1.0);
    match scale {
        ScaleType::Linear => {
            let t = (ratio * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
            let i = t.floor() as usize;
            let f = t - i as f32;
            let (a, b) = (range[i], range[(i + 1).min(n - 1)]);
            let mut c = [0u8; 4];
            for k in 0..4 {
                c[k] = (a[k] as f32 + (b[k] as f32 - a[k] as f32) * f).round() as u8;
            }
            c
        }
        _ => range[((ratio * n as f32) as usize).min(n - 1)],
    }
}

/// Elevation for a ratio, in the units of `range` scaled by `elevation_scale`.
pub fn interpolate_elevation(ratio: f32, range: [f32; 2], elevation_scale: f32) -> f32 {
    (range[0] + (range[1] - range[0]) * ratio) * elevation_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations_aggregate_bin_weights() {
        let weights = [1.0, 2.0, 3.0, 10.0];
        let bins = vec![vec![0, 1, 2], vec![3], vec![]];
        assert_eq!(
            aggregate(&bins, &weights, AggregationOperation::Sum).0,
            [6.0, 10.0, 0.0]
        );
        assert_eq!(
            aggregate(&bins, &weights, AggregationOperation::Count).0,
            [3.0, 1.0, 0.0]
        );
        let (mean, domain) = aggregate(&bins[..2], &weights, AggregationOperation::Mean);
        assert_eq!(mean, [2.0, 10.0]);
        assert_eq!(domain, [2.0, 10.0]);
        assert_eq!(
            aggregate(&bins[..2], &weights, AggregationOperation::Min).0,
            [1.0, 10.0]
        );
        assert_eq!(
            aggregate(&bins[..2], &weights, AggregationOperation::Max).0,
            [3.0, 10.0]
        );
    }

    #[test]
    fn quantize_and_linear_colors() {
        let range = DEFAULT_COLOR_RANGE;
        assert_eq!(sample_color_range(0.0, &range, ScaleType::Quantize), range[0]);
        assert_eq!(sample_color_range(0.99, &range, ScaleType::Quantize), range[5]);
        assert_eq!(sample_color_range(0.5, &range, ScaleType::Quantize), range[3]);
        assert_eq!(sample_color_range(0.0, &range, ScaleType::Linear), range[0]);
        assert_eq!(sample_color_range(1.0, &range, ScaleType::Linear), range[5]);
        let mid = sample_color_range(0.25, &range, ScaleType::Linear);
        assert_eq!(mid, range[1], "0.25 sits on the second texel centre");
    }

    #[test]
    fn quantile_scale_maps_to_percentile_buckets() {
        let values: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let s = ScaledValues::new(values, [0.0, 199.0], None, ScaleType::Quantile, 0.0, 100.0);
        assert_eq!(s.domain, [0.0, 99.0]);
        assert_eq!(s.values[0], 0.0);
        assert_eq!(s.values[199], 99.0);
        assert!((s.values[100] - 50.0).abs() <= 1.0);
        assert_eq!(s.cutoff, Some([0.0, 99.0]));
        assert!(s.visible(0) && s.visible(199));
    }

    #[test]
    fn percentile_cutoff_hides_the_tails() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let s = ScaledValues::new(values, [0.0, 99.0], None, ScaleType::Quantize, 10.0, 90.0);
        assert!(!s.visible(2));
        assert!(s.visible(50));
        assert!(!s.visible(97));
        let explicit = ScaledValues::new(
            vec![5.0, 50.0],
            [5.0, 50.0],
            Some([0.0, 100.0]),
            ScaleType::Linear,
            0.0,
            100.0,
        );
        assert_eq!(explicit.domain, [0.0, 100.0]);
        assert!((explicit.ratio(1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ordinal_scale_indexes_distinct_values() {
        let s = ScaledValues::new(
            vec![3.0, 1.0, 3.0, 7.0],
            [1.0, 7.0],
            None,
            ScaleType::Ordinal,
            0.0,
            100.0,
        );
        assert_eq!(s.values, [1.0, 0.0, 1.0, 2.0]);
        assert_eq!(s.domain, [0.0, 2.0]);
    }
}

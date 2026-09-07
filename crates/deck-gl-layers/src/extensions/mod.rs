//! Port of `@deck.gl/extensions`: layer extensions built on the shader hooks, see
//! `docs/extensions.md` and [`deck_gl::LayerExtension`].

pub mod brushing;
pub mod clip;
pub mod collision_filter;
pub mod data_filter;
pub mod fill_style;
pub mod mask;
pub mod path_style;

pub use brushing::{BrushingExtension, BrushingTarget};
pub use clip::ClipExtension;
pub use collision_filter::CollisionFilterExtension;
pub use data_filter::{DataFilterExtension, FilterCategories, FilterValues};
pub use fill_style::{FillPattern, FillPatternAtlas, FillStyleExtension};
pub use mask::MaskExtension;
pub use path_style::{PathStyleExtension, PathStyleTarget};

use deck_gl::Extensions;

/// The extensions a composite layer hands to one of its sub layers, named by the suffix of
/// the sub layer's id: fills do not get path styles, strokes do not get fill styles, and
/// points get neither the path style of the strokes nor a fill pattern meant for polygons.
pub(crate) fn sub_layer_extensions(base: &Extensions, suffix: &str) -> Extensions {
    match suffix {
        "fill" | "polygons" => base.without::<PathStyleExtension>(),
        "stroke" | "lines" => base.without::<FillStyleExtension>(),
        "points" => base
            .without::<PathStyleExtension>()
            .without::<FillStyleExtension>(),
        _ => base.clone(),
    }
}

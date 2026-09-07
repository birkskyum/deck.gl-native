//! Port of `@deck.gl/extensions`: layer extensions built on the shader hooks, see
//! `docs/extensions.md` and [`deck_gl::LayerExtension`].

pub mod data_filter;

pub use data_filter::{DataFilterExtension, FilterCategories, FilterValues};

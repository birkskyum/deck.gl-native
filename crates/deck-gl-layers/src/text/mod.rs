//! Building blocks of [`TextLayer`](crate::TextLayer): font atlases and paragraph layout.

pub mod font;
pub mod layout;

pub use font::{Character, CharacterSet, FontAtlas, FontSettings, FontSource, DEFAULT_FONT};
pub use layout::{transform_paragraph, Paragraph, WordBreak};

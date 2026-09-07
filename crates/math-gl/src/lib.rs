//! Rust port of the parts of [math.gl](https://math.gl) that deck.gl-native needs.
//!
//! Module layout mirrors the JavaScript monorepo: `web_mercator` corresponds to
//! `@math.gl/web-mercator`. Vector and matrix types come from `glam` in f64
//! precision, which matches the JavaScript implementation's use of doubles.

pub mod fly_to;
pub mod web_mercator;

pub use glam;

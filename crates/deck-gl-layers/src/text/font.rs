//! Font atlases for [`TextLayer`](crate::TextLayer): glyph rasterization with fontdue, atlas
//! packing as in deck.gl's `font-atlas-manager.ts`, and signed distance fields as in
//! `@mapbox/tiny-sdf`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use deck_gl::{DeckError, Result};

use crate::bitmap_layer::BitmapImage;

/// Roboto Mono Regular, bundled under the SIL Open Font License (see `fonts/OFL.txt`).
pub const DEFAULT_FONT: &[u8] = include_bytes!("../../fonts/RobotoMono-Regular.ttf");

const MAX_ATLAS_WIDTH: u32 = 1024;

/// Where glyphs come from.
#[derive(Clone, Debug, PartialEq)]
pub enum FontSource {
    /// The bundled Roboto Mono Regular.
    Default,
    /// The bytes of a TrueType or OpenType font.
    Bytes(Arc<Vec<u8>>),
    /// A font file on disk.
    File(PathBuf),
}

/// Which characters go into the atlas. deck.gl's default is printable ASCII.
#[derive(Clone, Debug, PartialEq)]
pub enum CharacterSet {
    /// Every character that appears in the data.
    Auto,
    /// A fixed set; characters outside it render as blank space.
    Chars(String),
}

impl Default for CharacterSet {
    fn default() -> Self {
        CharacterSet::Chars((32u8..128).map(char::from).collect())
    }
}

/// Font and rasterization settings, the counterpart of deck.gl's `fontSettings`.
#[derive(Clone, Debug, PartialEq)]
pub struct FontSettings {
    pub font: FontSource,
    pub character_set: CharacterSet,
    /// Size in pixels at which glyphs are drawn into the atlas
    pub font_size: f32,
    /// Empty pixels around each glyph
    pub buffer: u32,
    /// Generate a signed distance field instead of coverage (needed for outlines and crisp
    /// scaling)
    pub sdf: bool,
    /// SDF: distance, as a fraction of `radius`, mapped to the middle of the alpha range
    pub cutoff: f32,
    /// SDF: distance in pixels over which the field falls off
    pub radius: f32,
    /// Anti-aliasing width of SDF text in field units
    pub smoothing: f32,
}

impl Default for FontSettings {
    fn default() -> Self {
        Self {
            font: FontSource::Default,
            character_set: CharacterSet::default(),
            font_size: 64.0,
            buffer: 4,
            sdf: false,
            cutoff: 0.25,
            radius: 12.0,
            smoothing: 0.1,
        }
    }
}

/// Where a character sits in the atlas, in atlas pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Character {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Horizontal position of the pen relative to the cell's centre, so that the cell centre
    /// sits at `pen + anchor_x`
    pub anchor_x: f32,
    /// Distance from the cell top to the baseline
    pub anchor_y: f32,
    /// Pen advance to the next character
    pub advance: f32,
}

/// A rasterized character set.
#[derive(Clone, Debug)]
pub struct FontAtlas {
    pub image: BitmapImage,
    pub mapping: HashMap<char, Character>,
    /// Half the difference between the font's ascent and descent; deck.gl uses it to centre a
    /// line of text on its position
    pub baseline_offset: f32,
    pub settings: FontSettings,
}

impl FontAtlas {
    /// Rasterize `characters` (deduplicated) with the given settings.
    pub fn build(settings: &FontSettings, characters: impl IntoIterator<Item = char>) -> Result<FontAtlas> {
        let bytes: Arc<Vec<u8>> = match &settings.font {
            FontSource::Default => Arc::new(DEFAULT_FONT.to_vec()),
            FontSource::Bytes(bytes) => bytes.clone(),
            FontSource::File(path) => Arc::new(
                std::fs::read(path)
                    .map_err(|e| DeckError::Data(format!("could not read font {}: {e}", path.display())))?,
            ),
        };
        let font = fontdue::Font::from_bytes(
            bytes.as_slice(),
            fontdue::FontSettings {
                scale: settings.font_size,
                ..Default::default()
            },
        )
        .map_err(|e| DeckError::Data(format!("could not parse font: {e}")))?;
        let size = settings.font_size;
        let line = font.horizontal_line_metrics(size);
        let (ascent, descent) = match line {
            Some(m) => (m.ascent, -m.descent),
            None => (size * 0.9, size * 0.3),
        };
        let baseline_offset = (ascent - descent) / 2.0;

        // Measure and pack, one row after another, as in deck.gl's buildMapping.
        let buffer = settings.buffer;
        let pad = if settings.sdf { buffer } else { 0 };
        let mut chars: Vec<char> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for c in characters {
            if !c.is_control() && seen.insert(c) {
                chars.push(c);
            }
        }
        struct Placed {
            c: char,
            metrics: fontdue::Metrics,
            cell_width: u32,
            cell_height: u32,
        }
        let mut placed = Vec::with_capacity(chars.len());
        let mut mapping = HashMap::with_capacity(chars.len());
        let (mut x, mut y_min, mut y_max, mut max_x) = (0u32, 0u32, 0u32, 0u32);
        for c in chars {
            let metrics = font.metrics(c, size);
            let cell_width = metrics.width as u32 + 2 * pad;
            let cell_height = metrics.height as u32 + 2 * pad;
            if x + cell_width + 2 * buffer > MAX_ATLAS_WIDTH {
                x = 0;
                y_min = y_max;
            }
            let glyph_ascent = metrics.ymin as f32 + metrics.height as f32;
            mapping.insert(
                c,
                Character {
                    x: x + buffer,
                    y: y_min + buffer,
                    width: cell_width,
                    height: cell_height,
                    anchor_x: metrics.xmin as f32 + metrics.width as f32 / 2.0,
                    anchor_y: glyph_ascent + pad as f32,
                    advance: metrics.advance_width,
                },
            );
            placed.push(Placed {
                c,
                metrics,
                cell_width,
                cell_height,
            });
            x += cell_width + 2 * buffer;
            max_x = max_x.max(x);
            y_max = y_max.max(y_min + cell_height + 2 * buffer);
        }
        let width = max_x.max(1).next_power_of_two().clamp(1, MAX_ATLAS_WIDTH);
        let height = y_max.max(1).next_power_of_two();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for i in 0..(width * height) as usize {
            rgba[i * 4] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
        }

        for glyph in placed {
            let cell = mapping[&glyph.c];
            if glyph.metrics.width == 0 || glyph.metrics.height == 0 {
                continue;
            }
            let (metrics, coverage) = font.rasterize(glyph.c, size);
            let alpha = if settings.sdf {
                signed_distance_field(
                    &coverage,
                    metrics.width,
                    metrics.height,
                    pad as usize,
                    settings.radius,
                    settings.cutoff,
                )
            } else {
                coverage
            };
            let cell_w = glyph.cell_width as usize;
            for row in 0..glyph.cell_height as usize {
                for col in 0..cell_w {
                    let (ax, ay) = (cell.x as usize + col, cell.y as usize + row);
                    if ax >= width as usize || ay >= height as usize {
                        continue;
                    }
                    rgba[(ay * width as usize + ax) * 4 + 3] = alpha[row * cell_w + col];
                }
            }
        }

        Ok(FontAtlas {
            image: BitmapImage::new(width, height, rgba),
            mapping,
            baseline_offset,
            settings: settings.clone(),
        })
    }
}

const INF: f64 = 1e20;

/// Port of tiny-sdf: turns glyph coverage into a signed distance field padded by `buffer`
/// pixels on each side. Output alpha is 255 at `cutoff * radius` inside the outline and falls
/// to 0 over `radius` pixels.
pub fn signed_distance_field(
    coverage: &[u8],
    glyph_width: usize,
    glyph_height: usize,
    buffer: usize,
    radius: f32,
    cutoff: f32,
) -> Vec<u8> {
    let width = glyph_width + 2 * buffer;
    let height = glyph_height + 2 * buffer;
    let size = width * height;
    let mut grid_outer = vec![INF; size];
    let mut grid_inner = vec![0.0f64; size];
    for y in 0..glyph_height {
        for x in 0..glyph_width {
            let a = coverage[y * glyph_width + x] as f64 / 255.0;
            let i = (y + buffer) * width + x + buffer;
            grid_outer[i] = if a == 1.0 {
                0.0
            } else if a == 0.0 {
                INF
            } else {
                (0.5 - a).max(0.0).powi(2)
            };
            grid_inner[i] = if a == 1.0 {
                INF
            } else if a == 0.0 {
                0.0
            } else {
                (a - 0.5).max(0.0).powi(2)
            };
        }
    }
    let longest = width.max(height);
    let mut f = vec![0.0f64; longest];
    let mut v = vec![0usize; longest];
    let mut z = vec![0.0f64; longest + 1];
    edt(
        &mut grid_outer,
        0,
        0,
        width,
        height,
        width,
        &mut f,
        &mut v,
        &mut z,
    );
    edt(
        &mut grid_inner,
        buffer,
        buffer,
        glyph_width,
        glyph_height,
        width,
        &mut f,
        &mut v,
        &mut z,
    );
    (0..size)
        .map(|i| {
            let d = grid_outer[i].sqrt() - grid_inner[i].sqrt();
            let value = 255.0 - 255.0 * (d / radius as f64 + cutoff as f64);
            value.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// 2D Euclidean distance transform by Felzenszwalb & Huttenlocher, in place on squared values.
#[allow(clippy::too_many_arguments)]
fn edt(
    data: &mut [f64],
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    grid_size: usize,
    f: &mut [f64],
    v: &mut [usize],
    z: &mut [f64],
) {
    for x in x0..x0 + width {
        edt1d(data, y0 * grid_size + x, grid_size, height, f, v, z);
    }
    for y in y0..y0 + height {
        edt1d(data, y * grid_size + x0, 1, width, f, v, z);
    }
}

/// 1D squared distance transform along one row or column.
fn edt1d(
    grid: &mut [f64],
    offset: usize,
    stride: usize,
    length: usize,
    f: &mut [f64],
    v: &mut [usize],
    z: &mut [f64],
) {
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    f[0] = grid[offset];
    let mut k = 0usize;
    for q in 1..length {
        f[q] = grid[offset + q * stride];
        let q2 = (q * q) as f64;
        let mut s;
        loop {
            let r = v[k];
            s = (f[q] - f[r] + q2 - (r * r) as f64) / (q as f64 - r as f64) / 2.0;
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = INF;
    }
    k = 0;
    for q in 0..length {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let r = v[k];
        let qr = q as f64 - r as f64;
        grid[offset + q * stride] = f[r] + qr * qr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_an_ascii_atlas_from_the_bundled_font() {
        let atlas = FontAtlas::build(&FontSettings::default(), "Hello, World!".chars()).unwrap();
        let h = atlas.mapping[&'H'];
        assert!(h.width > 10 && h.height > 20, "{h:?}");
        assert!(h.advance > 20.0);
        let space = atlas.mapping[&' '];
        assert_eq!(space.width, 0);
        assert!(space.advance > 0.0, "spaces advance the pen");
        assert!(atlas.baseline_offset > 0.0);
        // The H cell has ink
        let ink = (0..h.height)
            .flat_map(|row| (0..h.width).map(move |col| (row, col)))
            .filter(|(row, col)| {
                let i = ((h.y + row) * atlas.image.width + h.x + col) as usize;
                atlas.image.rgba[i * 4 + 3] > 128
            })
            .count();
        assert!(ink > 100, "ink pixels {ink}");
    }

    #[test]
    fn sdf_cells_are_padded_and_fade_outwards() {
        let settings = FontSettings {
            sdf: true,
            ..Default::default()
        };
        let atlas = FontAtlas::build(&settings, "I".chars()).unwrap();
        let plain = FontAtlas::build(&FontSettings::default(), "I".chars()).unwrap();
        let (sdf, cov) = (atlas.mapping[&'I'], plain.mapping[&'I']);
        assert_eq!(sdf.width, cov.width + 8);
        assert_eq!(sdf.height, cov.height + 8);
        assert!((sdf.anchor_y - cov.anchor_y - 4.0).abs() < 1e-6);
        let alpha = |c: Character, col: u32, row: u32| {
            atlas.image.rgba[(((c.y + row) * atlas.image.width + c.x + col) * 4 + 3) as usize]
        };
        let middle = alpha(sdf, sdf.width / 2, sdf.height / 2);
        let edge = alpha(sdf, 0, 0);
        assert!(middle > 200, "inside the stem {middle}");
        assert!(edge < middle, "corner {edge} fades from {middle}");
    }

    #[test]
    fn distance_field_of_a_square() {
        // 4x4 solid square, 2px buffer, 8px radius: the field falls off from the outline
        let coverage = vec![255u8; 16];
        let field = signed_distance_field(&coverage, 4, 4, 2, 8.0, 0.25);
        assert_eq!(field.len(), 64);
        let at = |x: usize, y: usize| field[y * 8 + x];
        assert!(
            at(3, 3) >= at(2, 2),
            "centre is at least as deep inside as the edge"
        );
        assert!(at(2, 2) > at(1, 1), "the edge is inside, the buffer outside");
        assert!(at(1, 1) > at(0, 0), "further out is fainter");
        // A wide field far from any ink reaches zero
        let far = signed_distance_field(&[255u8], 1, 1, 12, 4.0, 0.25);
        assert_eq!(far[0], 0);
    }
}

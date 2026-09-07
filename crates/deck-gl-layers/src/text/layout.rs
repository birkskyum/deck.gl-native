//! Paragraph layout, a port of deck.gl's `text-layer/utils.ts`: line breaking, word wrapping
//! and per character positions in atlas pixel units.

use std::collections::HashMap;

use super::font::Character;

/// Width used for characters missing from the atlas.
const MISSING_CHAR_WIDTH: f32 = 32.0;

/// How lines wrap when `max_width` is set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WordBreak {
    /// Break between words, and inside a word only when it is wider than the line
    #[default]
    BreakWord,
    /// Break anywhere
    BreakAll,
}

/// Layout of one text: per character positions plus the size of the block, in the units of
/// the font atlas (`font_size` pixels per em).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Paragraph {
    /// Horizontal centre of each character's cell, from the start of its row
    pub x: Vec<f32>,
    /// Baseline of each character's row, from the top of the block
    pub y: Vec<f32>,
    /// Width of the row each character is on
    pub row_width: Vec<f32>,
    /// Width and height of the whole block
    pub size: [f32; 2],
}

fn text_width(text: &[char], start: usize, end: usize, mapping: &HashMap<char, Character>) -> f32 {
    text[start..end]
        .iter()
        .map(|c| mapping.get(c).map(|m| m.advance).unwrap_or(0.0))
        .sum()
}

fn break_all(
    text: &[char],
    start: usize,
    end: usize,
    max_width: f32,
    mapping: &HashMap<char, Character>,
    target: &mut Vec<usize>,
) -> f32 {
    let mut row_start = start;
    let mut row_offset_left = 0.0;
    for i in start..end {
        let width = text_width(text, i, i + 1, mapping);
        if row_offset_left + width > max_width {
            if row_start < i {
                target.push(i);
            }
            row_start = i;
            row_offset_left = 0.0;
        }
        row_offset_left += width;
    }
    row_offset_left
}

fn break_word(
    text: &[char],
    start: usize,
    end: usize,
    max_width: f32,
    mapping: &HashMap<char, Character>,
    target: &mut Vec<usize>,
) -> f32 {
    let mut row_start = start;
    let mut group_start = start;
    let mut group_end = start;
    let mut row_offset_left = 0.0;
    for i in start..end {
        if text[i] == ' ' || text.get(i + 1) == Some(&' ') || i + 1 == end {
            group_end = i + 1;
        }
        if group_end > group_start {
            let mut group_width = text_width(text, group_start, group_end, mapping);
            if row_offset_left + group_width > max_width {
                if row_start < group_start {
                    target.push(group_start);
                    row_start = group_start;
                    row_offset_left = 0.0;
                }
                if group_width > max_width {
                    group_width = break_all(text, group_start, group_end, max_width, mapping, target);
                    row_start = *target.last().unwrap_or(&row_start);
                }
            }
            group_start = group_end;
            row_offset_left += group_width;
        }
    }
    let _ = row_start;
    row_offset_left
}

/// Indices at which a line of `text` wraps.
pub fn auto_wrapping(
    text: &[char],
    word_break: WordBreak,
    max_width: f32,
    mapping: &HashMap<char, Character>,
    start: usize,
    end: usize,
) -> Vec<usize> {
    let mut result = Vec::new();
    match word_break {
        WordBreak::BreakAll => {
            break_all(text, start, end, max_width, mapping, &mut result);
        }
        WordBreak::BreakWord => {
            break_word(text, start, end, max_width, mapping, &mut result);
        }
    }
    result
}

fn transform_row(
    line: &[char],
    start: usize,
    end: usize,
    mapping: &HashMap<char, Character>,
    left_offsets: &mut [f32],
) -> [f32; 2] {
    let mut x = 0.0f32;
    let mut row_height = 0.0f32;
    for c in &line[start..end] {
        if let Some(frame) = mapping.get(c) {
            row_height = row_height.max(frame.height as f32);
        }
    }
    for i in start..end {
        match mapping.get(&line[i]) {
            Some(frame) => {
                left_offsets[i] = x + frame.anchor_x;
                x += frame.advance;
            }
            None => {
                left_offsets[i] = x;
                x += MISSING_CHAR_WIDTH;
            }
        }
    }
    [x, row_height]
}

/// Lay out a paragraph. `line_height` and `max_width` are in atlas pixels; `max_width <= 0`
/// disables wrapping. Newlines start new lines.
pub fn transform_paragraph(
    paragraph: &str,
    baseline_offset: f32,
    line_height: f32,
    word_break: WordBreak,
    max_width: f32,
    mapping: &HashMap<char, Character>,
) -> Paragraph {
    let characters: Vec<char> = paragraph.chars().collect();
    let n = characters.len();
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut row_width = vec![0.0f32; n];
    let wrapping = max_width.is_finite() && max_width > 0.0;
    let mut size = [0.0f32, 0.0f32];
    let mut row_count = 0;
    // places the top of the first row at 0
    let mut row_offset_top = baseline_offset + line_height / 2.0;
    let mut line_start = 0;
    let mut line_end = 0;
    for i in 0..=n {
        let c = characters.get(i).copied();
        if c == Some('\n') || i == n {
            line_end = i;
        }
        if line_end > line_start {
            let rows = if wrapping {
                auto_wrapping(&characters, word_break, max_width, mapping, line_start, line_end)
            } else {
                Vec::new()
            };
            for row_index in 0..=rows.len() {
                let row_start = if row_index == 0 {
                    line_start
                } else {
                    rows[row_index - 1]
                };
                let row_end = if row_index < rows.len() {
                    rows[row_index]
                } else {
                    line_end
                };
                let row_size = transform_row(&characters, row_start, row_end, mapping, &mut x);
                for j in row_start..row_end {
                    y[j] = row_offset_top;
                    row_width[j] = row_size[0];
                }
                row_count += 1;
                row_offset_top += line_height;
                size[0] = size[0].max(row_size[0]);
            }
            line_start = line_end;
        }
        if c == Some('\n') {
            x[line_start] = 0.0;
            y[line_start] = 0.0;
            row_width[line_start] = 0.0;
            line_start += 1;
        }
    }
    size[1] = row_count as f32 * line_height;
    Paragraph {
        x,
        y,
        row_width,
        size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping() -> HashMap<char, Character> {
        let mut m = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz ".chars().enumerate() {
            m.insert(
                c,
                Character {
                    x: 0,
                    y: 0,
                    width: 10,
                    height: 20,
                    anchor_x: 5.0,
                    anchor_y: 16.0,
                    advance: if c == ' ' { 6.0 } else { 10.0 + i as f32 % 3.0 },
                },
            );
        }
        m
    }

    #[test]
    fn lays_out_a_single_line() {
        let m = mapping();
        let p = transform_paragraph("abc", 4.0, 32.0, WordBreak::BreakWord, -1.0, &m);
        assert_eq!(p.x, vec![5.0, 15.0, 26.0]); // advances 10, 11
        assert_eq!(p.y, vec![20.0; 3]);
        assert_eq!(p.row_width, vec![33.0; 3]);
        assert_eq!(p.size, [33.0, 32.0]);
    }

    #[test]
    fn newlines_start_new_rows() {
        let m = mapping();
        let p = transform_paragraph("ab\ncd", 0.0, 30.0, WordBreak::BreakWord, -1.0, &m);
        assert_eq!(p.y[0], 15.0);
        assert_eq!(p.y[3], 45.0);
        assert_eq!(p.x[2], 0.0, "the newline itself has no position");
        assert_eq!(p.size[1], 60.0);
    }

    #[test]
    fn wraps_words_within_max_width() {
        let m = mapping();
        // "aaa bbb ccc" with every glyph 10 wide plus spaces 6: 30 + 6 + 30 + 6 + 30 = 102
        let text = "aaa bbb ccc";
        let p = transform_paragraph(text, 0.0, 30.0, WordBreak::BreakWord, 70.0, &m);
        assert_eq!(p.y[0], 15.0);
        assert_eq!(p.y[8], 45.0, "third word wraps to the second row");
        assert!(p.size[0] <= 70.0 + 6.0);
        assert_eq!(p.size[1], 60.0);
        let all = transform_paragraph("aaaaaaaaaa", 0.0, 30.0, WordBreak::BreakAll, 35.0, &m);
        assert_eq!(all.size[1], 120.0, "10 glyphs of width 10 break every 3");
    }

    #[test]
    fn missing_characters_take_a_default_width() {
        let m = mapping();
        let p = transform_paragraph("a?b", 0.0, 30.0, WordBreak::BreakWord, -1.0, &m);
        assert_eq!(p.x[1], 10.0);
        assert_eq!(p.x[2], 10.0 + MISSING_CHAR_WIDTH + 5.0);
    }
}

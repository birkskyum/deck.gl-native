//! Port of `@deck.gl/layers/src/text-layer`: texts drawn as glyphs from a font atlas, with an
//! optional background box and SDF outlines.

use std::collections::BTreeSet;
use std::sync::Arc;

use deck_gl::data::{
    resolve_colors, resolve_f32, resolve_positions, resolve_strings, resolve_vec2, resolve_with,
};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, DeckError, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use glam::Vec2;
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, create_rgba8_texture, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::text::{transform_paragraph, CharacterSet, FontAtlas, FontSettings, WordBreak};

const CHARACTERS_SHADER: &str = include_str!("wgsl/multi_icon_layer.wgsl");
const BACKGROUND_SHADER: &str = include_str!("wgsl/text_background_layer.wgsl");
/// deck.gl's `DEFAULT_BUFFER`: the SDF value at the glyph outline.
const SDF_BUFFER: f32 = 192.0 / 256.0;

/// Horizontal alignment of a text relative to its position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextAnchor {
    Start,
    #[default]
    Middle,
    End,
}

impl TextAnchor {
    fn value(self) -> f32 {
        match self {
            TextAnchor::Start => 1.0,
            TextAnchor::Middle => 0.0,
            TextAnchor::End => -1.0,
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "start" => Some(TextAnchor::Start),
            "middle" => Some(TextAnchor::Middle),
            "end" => Some(TextAnchor::End),
            _ => None,
        }
    }
}

/// Vertical alignment of a text relative to its position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlignmentBaseline {
    Top,
    #[default]
    Center,
    Bottom,
}

impl AlignmentBaseline {
    fn value(self) -> f32 {
        match self {
            AlignmentBaseline::Top => 1.0,
            AlignmentBaseline::Center => 0.0,
            AlignmentBaseline::Bottom => -1.0,
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "top" => Some(AlignmentBaseline::Top),
            "center" => Some(AlignmentBaseline::Center),
            "bottom" => Some(AlignmentBaseline::Bottom),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstancePositions {
    position: [f32; 3],
    position_low: [f32; 3],
}

impl From<Position> for InstancePositions {
    fn from(p: Position) -> Self {
        let hi = [p[0] as f32, p[1] as f32, p[2] as f32];
        Self {
            position: hi,
            position_low: [
                (p[0] - hi[0] as f64) as f32,
                (p[1] - hi[1] as f64) as f32,
                (p[2] - hi[2] as f64) as f32,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CharacterInstance {
    size: f32,
    angle: f32,
    color: [u8; 4],
    frame: [f32; 4],
    color_mode: f32,
    offset: [f32; 2],
    pixel_offset: [f32; 2],
    row_index: u32,
    clip_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BackgroundInstance {
    rect: [f32; 4],
    clip_rect: [f32; 4],
    size: f32,
    angle: f32,
    pixel_offset: [f32; 2],
    line_width: f32,
    fill_color: [u8; 4],
    line_color: [u8; 4],
    row_index: u32,
}

/// Properties of a [`TextLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub billboard: bool,
    pub size_scale: f32,
    pub size_units: Unit,
    pub size_min_pixels: f32,
    pub size_max_pixels: f32,
    /// Draw a box behind each text
    pub background: bool,
    pub get_background_color: Accessor<Color>,
    pub get_border_color: Accessor<Color>,
    pub get_border_width: Accessor<f32>,
    /// Corner radii of the background, in pixels: top left, top right, bottom right, bottom left
    pub background_border_radius: [f32; 4],
    /// Background padding in pixels: left, top, right, bottom
    pub background_padding: [f32; 4],
    /// Font, character set and rasterization settings
    pub font: FontSettings,
    /// Line height as a multiple of the font size
    pub line_height: f32,
    /// Outline width as a fraction of the font size; needs `font.sdf`
    pub outline_width: f32,
    pub outline_color: Color,
    pub word_break: WordBreak,
    /// Maximum line width as a multiple of the font size; wrapping is off when negative
    pub max_width: f32,
    pub get_text: Accessor<String>,
    pub get_position: Accessor<Position>,
    pub get_color: Accessor<Color>,
    /// Font size in `size_units`
    pub get_size: Accessor<f32>,
    /// Rotation in degrees
    pub get_angle: Accessor<f32>,
    pub get_text_anchor: Accessor<TextAnchor>,
    pub get_alignment_baseline: Accessor<AlignmentBaseline>,
    pub get_pixel_offset: Accessor<[f32; 2]>,
}

impl Default for TextLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("TextLayer"),
            data: LayerData::default(),
            billboard: true,
            size_scale: 1.0,
            size_units: Unit::Pixels,
            size_min_pixels: 0.0,
            size_max_pixels: f32::MAX,
            background: false,
            get_background_color: Accessor::Constant([255, 255, 255, 255]),
            get_border_color: Accessor::Constant([0, 0, 0, 255]),
            get_border_width: Accessor::Constant(0.0),
            background_border_radius: [0.0; 4],
            background_padding: [0.0; 4],
            font: FontSettings::default(),
            line_height: 1.0,
            outline_width: 0.0,
            outline_color: [0, 0, 0, 255],
            word_break: WordBreak::BreakWord,
            max_width: -1.0,
            get_text: Accessor::column("text"),
            get_position: Accessor::column("position"),
            get_color: Accessor::Constant([0, 0, 0, 255]),
            get_size: Accessor::Constant(32.0),
            get_angle: Accessor::Constant(0.0),
            get_text_anchor: Accessor::Constant(TextAnchor::Middle),
            get_alignment_baseline: Accessor::Constant(AlignmentBaseline::Center),
            get_pixel_offset: Accessor::Constant([0.0, 0.0]),
        }
    }
}

/// Renders text labels at given coordinates.
pub struct TextLayer {
    props: TextLayerProps,
    /// First character pass: fill, or outline and fill mixed when an outline is set
    characters: Option<Model>,
    /// Second character pass that redraws the fill crisply over the outline
    fill_pass: Option<Model>,
    background: Option<Model>,
    atlas: Option<Arc<FontAtlas>>,
    atlas_key: Option<(FontSettings, String)>,
    texture_size: Vec2,
    data_dirty: bool,
    stroked: bool,
    has_background: bool,
}

fn resolve_enum<T: Clone + Send + Sync + 'static>(
    data: &LayerData,
    accessor: &Accessor<T>,
    parse: fn(&str) -> Option<T>,
    what: &str,
) -> Result<Vec<T>> {
    match accessor {
        Accessor::Column(name) => resolve_strings(data, &Accessor::column(name.clone()))?
            .iter()
            .map(|s| parse(s).ok_or_else(|| DeckError::Data(format!("unknown {what} `{s}`"))))
            .collect(),
        other => resolve_with(data, other, |_| unreachable!("constant or function accessor")),
    }
}

impl TextLayer {
    pub fn new(props: TextLayerProps) -> Self {
        Self {
            props,
            characters: None,
            fill_pass: None,
            background: None,
            atlas: None,
            atlas_key: None,
            texture_size: Vec2::ONE,
            data_dirty: true,
            stroked: false,
            has_background: false,
        }
    }

    pub fn props(&self) -> &TextLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: TextLayerProps) {
        if self.props != props {
            self.props = props;
            self.data_dirty = true;
        }
    }

    /// The font atlas built for the current data, once updated.
    pub fn atlas(&self) -> Option<&Arc<FontAtlas>> {
        self.atlas.as_ref()
    }

    fn outline_buffer(&self) -> f32 {
        let props = &self.props;
        if props.font.sdf && props.outline_width > 0.0 {
            props
                .font
                .smoothing
                .max(SDF_BUFFER * (1.0 - props.outline_width / props.font.radius))
        } else {
            -1.0
        }
    }

    fn uses_outline_pass(&self) -> bool {
        self.props.font.sdf && self.props.outline_width > 0.0
    }

    fn update_atlas(&mut self, ctx: &LayerContext, texts: &[String]) -> Result<()> {
        let props = &self.props;
        let chars: String = match &props.font.character_set {
            CharacterSet::Auto => texts
                .iter()
                .flat_map(|t| t.chars())
                .collect::<BTreeSet<char>>()
                .into_iter()
                .collect(),
            CharacterSet::Chars(set) => set.clone(),
        };
        let key = (props.font.clone(), chars);
        if self.atlas_key.as_ref() == Some(&key) {
            return Ok(());
        }
        let atlas = FontAtlas::build(&key.0, key.1.chars())?;
        let texture = create_rgba8_texture(
            &ctx.device,
            &ctx.queue,
            &props.base.id,
            atlas.image.width,
            atlas.image.height,
            &atlas.image.rgba,
        );
        for model in [&mut self.characters, &mut self.fill_pass].into_iter().flatten() {
            model.set_texture("iconsTexture", texture.create_view(&Default::default()))?;
        }
        self.texture_size = Vec2::new(atlas.image.width as f32, atlas.image.height as f32);
        self.atlas = Some(Arc::new(atlas));
        self.atlas_key = Some(key);
        Ok(())
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let texts = resolve_strings(&self.props.data, &self.props.get_text)?;
        self.update_atlas(ctx, &texts)?;
        let props = &self.props;
        let data = &props.data;
        let device = &ctx.device;
        let atlas = self.atlas.as_ref().expect("atlas built");

        let positions = resolve_positions(data, &props.get_position)?;
        let colors = resolve_colors(data, &props.get_color)?;
        let sizes = resolve_f32(data, &props.get_size)?;
        let angles = resolve_f32(data, &props.get_angle)?;
        let anchors = resolve_enum(data, &props.get_text_anchor, TextAnchor::parse, "text anchor")?;
        let baselines = resolve_enum(
            data,
            &props.get_alignment_baseline,
            AlignmentBaseline::parse,
            "alignment baseline",
        )?;
        let pixel_offsets = resolve_vec2(data, &props.get_pixel_offset)?;
        let background_colors = resolve_colors(data, &props.get_background_color)?;
        let border_colors = resolve_colors(data, &props.get_border_color)?;
        let border_widths = resolve_f32(data, &props.get_border_width)?;

        let font_size = props.font.font_size;
        let mut char_positions = Vec::new();
        let mut char_instances = Vec::new();
        let mut bg_positions = Vec::with_capacity(data.len());
        let mut bg_instances = Vec::with_capacity(data.len());
        for (i, text) in texts.iter().enumerate() {
            let paragraph = transform_paragraph(
                text,
                atlas.baseline_offset,
                props.line_height * font_size,
                props.word_break,
                props.max_width * font_size,
                &atlas.mapping,
            );
            let anchor_x = anchors[i].value();
            let anchor_y = baselines[i].value();
            let [width, height] = paragraph.size;
            let position = InstancePositions::from(positions[i]);
            let row_index = data.source_row(i);
            for (j, c) in text.chars().enumerate() {
                let Some(frame) = atlas.mapping.get(&c) else {
                    continue;
                };
                let offset_x = (anchor_x - 1.0) * paragraph.row_width[j] / 2.0 + paragraph.x[j];
                let offset_y = (anchor_y - 1.0) * height / 2.0 + paragraph.y[j];
                char_positions.push(position);
                char_instances.push(CharacterInstance {
                    size: sizes[i],
                    angle: angles[i],
                    color: colors[i],
                    frame: [
                        frame.x as f32,
                        frame.y as f32,
                        frame.width as f32,
                        frame.height as f32,
                    ],
                    color_mode: 1.0,
                    offset: [offset_x, frame.height as f32 / 2.0 - frame.anchor_y + offset_y],
                    pixel_offset: pixel_offsets[i],
                    row_index,
                    clip_rect: [0.0, 0.0, -1.0, -1.0],
                });
            }
            if props.background {
                bg_positions.push(position);
                bg_instances.push(BackgroundInstance {
                    rect: [
                        (anchor_x - 1.0) * width / 2.0,
                        (anchor_y - 1.0) * height / 2.0,
                        width,
                        height,
                    ],
                    clip_rect: [0.0, 0.0, -1.0, -1.0],
                    size: sizes[i],
                    angle: angles[i],
                    pixel_offset: pixel_offsets[i],
                    line_width: border_widths[i],
                    fill_color: background_colors[i],
                    line_color: border_colors[i],
                    row_index,
                });
            }
        }
        self.stroked = border_widths.iter().any(|w| *w > 0.0);
        self.has_background = props.background && !bg_instances.is_empty();

        let positions_buffer = create_vertex_buffer_from(device, "instancePositions", &char_positions);
        let data_buffer = create_vertex_buffer_from(device, "instanceData", &char_instances);
        for model in [&mut self.characters, &mut self.fill_pass].into_iter().flatten() {
            model.set_vertex_buffer("instancePositions", positions_buffer.clone())?;
            model.set_vertex_buffer("instanceData", data_buffer.clone())?;
            model.set_instance_count(char_instances.len() as u32);
        }
        if let Some(model) = &mut self.background {
            model.set_vertex_buffer(
                "instancePositions",
                create_vertex_buffer_from(device, "instancePositions", &bg_positions),
            )?;
            model.set_vertex_buffer(
                "instanceData",
                create_vertex_buffer_from(device, "instanceData", &bg_instances),
            )?;
            model.set_instance_count(bg_instances.len() as u32);
        }
        Ok(())
    }

    fn create_character_model(&self, ctx: &LayerContext, id: &str, extra_bias: i32) -> Result<Model> {
        let shader = assemble_shader(id, &STANDARD_MODULES, CHARACTERS_SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x2),
            VertexBufferLayout::interleaved(
                "instancePositions",
                std::mem::size_of::<InstancePositions>() as u64,
                wgpu::VertexStepMode::Instance,
                &[(1, VertexFormat::Float32x3, 0), (2, VertexFormat::Float32x3, 12)],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<CharacterInstance>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (3, VertexFormat::Float32, 0),
                    (4, VertexFormat::Float32, 4),
                    (5, VertexFormat::Unorm8x4, 8),
                    (6, VertexFormat::Float32x4, 12),
                    (7, VertexFormat::Float32, 28),
                    (8, VertexFormat::Float32x2, 32),
                    (9, VertexFormat::Float32x2, 40),
                    (10, VertexFormat::Uint32, 48),
                    (11, VertexFormat::Float32x4, 52),
                ],
            ),
        ];
        let mut desc = ModelDescriptor::new(
            id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.depth_bias.constant += extra_bias;
        desc.pickable = self.props.base.pickable;
        let mut model = Model::new(&ctx.device, &desc)?;
        let positions: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(4);
        Ok(model)
    }

    fn create_background_model(&self, ctx: &LayerContext) -> Result<Model> {
        let id = format!("{}-background", self.props.base.id);
        let shader = assemble_shader(&id, &STANDARD_MODULES, BACKGROUND_SHADER)?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x2),
            VertexBufferLayout::interleaved(
                "instancePositions",
                std::mem::size_of::<InstancePositions>() as u64,
                wgpu::VertexStepMode::Instance,
                &[(1, VertexFormat::Float32x3, 0), (2, VertexFormat::Float32x3, 12)],
            ),
            VertexBufferLayout::interleaved(
                "instanceData",
                std::mem::size_of::<BackgroundInstance>() as u64,
                wgpu::VertexStepMode::Instance,
                &[
                    (3, VertexFormat::Float32x4, 0),
                    (4, VertexFormat::Float32x4, 16),
                    (5, VertexFormat::Float32, 32),
                    (6, VertexFormat::Float32, 36),
                    (7, VertexFormat::Float32x2, 40),
                    (8, VertexFormat::Float32, 48),
                    (9, VertexFormat::Unorm8x4, 52),
                    (10, VertexFormat::Unorm8x4, 56),
                    (11, VertexFormat::Uint32, 60),
                ],
            ),
        ];
        let mut desc = ModelDescriptor::new(
            &id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        let mut model = Model::new(&ctx.device, &desc)?;
        let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(4);
        Ok(model)
    }

    fn write_text_uniforms(model: &mut Model, font_size: f32) -> Result<()> {
        let u = model.uniforms("text")?;
        u.set_vec2("cutoffPixels", Vec2::ZERO)?;
        u.set_bytes("align", bytemuck::bytes_of(&[0i32, 0i32]))?;
        u.set_f32("fontSize", font_size)?;
        u.set_f32("flipY", 0.0)?;
        Ok(())
    }

    fn write_character_uniforms(&self, model: &mut Model, outline_buffer: f32) -> Result<()> {
        let props = &self.props;
        let u = model.uniforms("icon")?;
        u.set_f32("sizeScale", props.size_scale)?;
        u.set_vec2("iconsTextureDim", self.texture_size)?;
        u.set_f32("sizeBasis", 0.0)?;
        u.set_f32("sizeMinPixels", props.size_min_pixels)?;
        u.set_f32("sizeMaxPixels", props.size_max_pixels)?;
        u.set_i32("billboard", props.billboard as i32)?;
        u.set_i32("sizeUnits", props.size_units.shader_value())?;
        u.set_f32("alphaCutoff", 0.001)?;
        Self::write_text_uniforms(model, props.font.font_size)?;
        let c = props.outline_color;
        let u = model.uniforms("sdf")?;
        u.set_f32("gamma", props.font.smoothing)?;
        u.set_f32("enabled", if props.font.sdf { 1.0 } else { 0.0 })?;
        u.set_f32("buffer", SDF_BUFFER)?;
        u.set_f32("outlineBuffer", outline_buffer)?;
        u.set_vec4(
            "outlineColor",
            glam::Vec4::new(
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
                c[3] as f32 / 255.0,
            ),
        )?;
        Ok(())
    }
}

impl Layer for TextLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let id = self.props.base.id.clone();
        // The background is a sub layer below the characters, like deck.gl's two sub layers.
        self.background = Some(self.create_background_model(ctx)?);
        self.characters = Some(self.create_character_model(ctx, &format!("{id}-characters"), -100)?);
        self.fill_pass = Some(self.create_character_model(ctx, &format!("{id}-characters-fill"), -100)?);
        self.atlas_key = None;
        self.data_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let outline_buffer = self.outline_buffer();
        let base = self.props.base.clone();
        if let Some(mut model) = self.characters.take() {
            update_standard_uniforms(&mut model, ctx, viewport, &base)?;
            self.write_character_uniforms(&mut model, outline_buffer)?;
            model.upload_uniforms(&ctx.queue);
            self.characters = Some(model);
        }
        if let Some(mut model) = self.fill_pass.take() {
            update_standard_uniforms(&mut model, ctx, viewport, &base)?;
            self.write_character_uniforms(&mut model, SDF_BUFFER)?;
            model.upload_uniforms(&ctx.queue);
            self.fill_pass = Some(model);
        }
        if let Some(mut model) = self.background.take() {
            update_standard_uniforms(&mut model, ctx, viewport, &base)?;
            let props = &self.props;
            let u = model.uniforms("textBackground")?;
            u.set_f32("billboard", if props.billboard { 1.0 } else { 0.0 })?;
            u.set_f32("sizeScale", props.size_scale)?;
            u.set_f32("sizeMinPixels", props.size_min_pixels)?;
            u.set_f32("sizeMaxPixels", props.size_max_pixels)?;
            u.set_vec4("borderRadius", glam::Vec4::from(props.background_border_radius))?;
            u.set_vec4("padding", glam::Vec4::from(props.background_padding))?;
            u.set_i32("sizeUnits", props.size_units.shader_value())?;
            u.set_f32("stroked", if self.stroked { 1.0 } else { 0.0 })?;
            Self::write_text_uniforms(&mut model, props.font.font_size)?;
            model.upload_uniforms(&ctx.queue);
            self.background = Some(model);
        }
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if self.has_background {
            if let Some(model) = &self.background {
                model.draw(pass)?;
            }
        }
        if let Some(model) = &self.characters {
            model.draw(pass)?;
        }
        if self.uses_outline_pass() {
            if let Some(model) = &self.fill_pass {
                model.draw(pass)?;
            }
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for model in [&mut self.background, &mut self.characters, &mut self.fill_pass]
            .into_iter()
            .flatten()
        {
            set_model_picking_active(model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if self.has_background {
            if let Some(model) = &self.background {
                model.draw_picking(pass)?;
            }
        }
        if let Some(model) = &self.characters {
            model.draw_picking(pass)?;
        }
        Ok(())
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn update_from(&mut self, incoming: &mut dyn Layer) -> bool {
        match incoming.as_any_mut().downcast_mut::<Self>() {
            Some(other) => {
                self.set_props(std::mem::take(&mut other.props));
                true
            }
            None => false,
        }
    }
}

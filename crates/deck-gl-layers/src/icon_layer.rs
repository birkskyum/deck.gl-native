//! Port of `@deck.gl/layers/src/icon-layer/icon-layer.ts` for pre-packed icon atlases.

use std::collections::HashMap;
use std::sync::Arc;

use deck_gl::data::{resolve_colors, resolve_f32, resolve_positions, resolve_strings, resolve_vec2};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, DeckError, Layer, LayerContext, LayerData, LayerProps, Position, Result, Unit, Viewport,
};
use glam::Vec2;
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{assemble_shader, create_rgba8_texture, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::BitmapImage;

const SHADER: &str = include_str!("wgsl/icon_layer.wgsl");

/// Where one icon sits in the atlas. Mirrors deck.gl's `iconMapping` entries.
#[derive(Clone, Debug, PartialEq)]
pub struct IconMapping {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Horizontal anchor in icon pixels. Default: half the width.
    pub anchor_x: Option<f32>,
    /// Vertical anchor in icon pixels. Default: half the height.
    pub anchor_y: Option<f32>,
    /// Treat the icon as a transparency mask colored by `get_color`.
    pub mask: bool,
}

impl IconMapping {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            anchor_x: None,
            anchor_y: None,
            mask: false,
        }
    }

    pub fn mask(mut self) -> Self {
        self.mask = true;
        self
    }

    pub fn anchor(mut self, x: f32, y: f32) -> Self {
        self.anchor_x = Some(x);
        self.anchor_y = Some(y);
        self
    }
}

/// An atlas image and the icons it contains.
#[derive(Clone, Debug, PartialEq)]
pub struct IconAtlas {
    pub image: BitmapImage,
    pub mapping: HashMap<String, IconMapping>,
}

impl IconAtlas {
    /// Parse a deck.gl `iconMapping` JSON object, for example
    /// `{"marker": {"x": 0, "y": 0, "width": 128, "height": 128, "anchorY": 128, "mask": true}}`.
    pub fn mapping_from_json(text: &str) -> Result<HashMap<String, IconMapping>> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| DeckError::Data(format!("invalid icon mapping JSON: {e}")))?;
        let object = value
            .as_object()
            .ok_or_else(|| DeckError::Data("icon mapping must be a JSON object".into()))?;
        let mut mapping = HashMap::new();
        for (name, entry) in object {
            let number = |key: &str| entry.get(key).and_then(serde_json::Value::as_f64);
            let required = |key: &str| {
                number(key).ok_or_else(|| DeckError::Data(format!("icon `{name}` is missing `{key}`")))
            };
            mapping.insert(
                name.clone(),
                IconMapping {
                    x: required("x")? as u32,
                    y: required("y")? as u32,
                    width: required("width")? as u32,
                    height: required("height")? as u32,
                    anchor_x: number("anchorX").map(|v| v as f32),
                    anchor_y: number("anchorY").map(|v| v as f32),
                    mask: entry
                        .get("mask")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                },
            );
        }
        Ok(mapping)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstancePositions {
    position: [f32; 3],
    position_low: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    size: f32,
    angle: f32,
    color: [u8; 4],
    frame: [f32; 4],
    color_mode: f32,
    offset: [f32; 2],
    pixel_offset: [f32; 2],
    row_index: u32,
}

/// Properties of an [`IconLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct IconLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub atlas: Option<Arc<IconAtlas>>,
    pub size_units: Unit,
    pub size_scale: f32,
    pub size_min_pixels: f32,
    pub size_max_pixels: f32,
    /// Scale icons so their height (true) or width (false) matches `get_size`
    pub size_by_height: bool,
    pub billboard: bool,
    /// Fragments with alpha below this are discarded
    pub alpha_cutoff: f32,
    pub get_position: Accessor<Position>,
    /// Icon name in the atlas mapping
    pub get_icon: Accessor<String>,
    pub get_color: Accessor<Color>,
    pub get_size: Accessor<f32>,
    /// Rotation in degrees
    pub get_angle: Accessor<f32>,
    pub get_pixel_offset: Accessor<[f32; 2]>,
}

impl Default for IconLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("IconLayer"),
            data: LayerData::default(),
            atlas: None,
            size_units: Unit::Pixels,
            size_scale: 1.0,
            size_min_pixels: 0.0,
            size_max_pixels: f32::MAX,
            size_by_height: true,
            billboard: true,
            alpha_cutoff: 0.05,
            get_position: Accessor::column("position"),
            get_icon: Accessor::column("icon"),
            get_color: Accessor::Constant([0, 0, 0, 255]),
            get_size: Accessor::Constant(1.0),
            get_angle: Accessor::Constant(0.0),
            get_pixel_offset: Accessor::Constant([0.0, 0.0]),
        }
    }
}

/// Renders icons from a texture atlas at given coordinates.
pub struct IconLayer {
    props: IconLayerProps,
    model: Option<Model>,
    data_dirty: bool,
    texture_size: Vec2,
}

impl IconLayer {
    pub fn new(props: IconLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
            texture_size: Vec2::ONE,
        }
    }

    pub fn props(&self) -> &IconLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: IconLayerProps) {
        if self.props == props {
            return;
        }
        if Self::attributes_changed(&self.props, &props) {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms (sizes, units, flags and the base props).
    fn attributes_changed(old: &IconLayerProps, new: &IconLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.size_units = old.size_units;
        probe.size_scale = old.size_scale;
        probe.size_min_pixels = old.size_min_pixels;
        probe.size_max_pixels = old.size_max_pixels;
        probe.billboard = old.billboard;
        probe.alpha_cutoff = old.alpha_cutoff;
        probe != *old
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        let Some(atlas) = &props.atlas else {
            model.set_instance_count(0);
            return Ok(());
        };

        let texture = create_rgba8_texture(
            &ctx.device,
            &ctx.queue,
            &props.base.id,
            atlas.image.width,
            atlas.image.height,
            &atlas.image.rgba,
        );
        model.set_texture("iconsTexture", texture.create_view(&Default::default()))?;
        self.texture_size = Vec2::new(atlas.image.width as f32, atlas.image.height as f32);

        let positions = resolve_positions(data, &props.get_position)?;
        let icons = resolve_strings(data, &props.get_icon)?;
        let colors = resolve_colors(data, &props.get_color)?;
        let sizes = resolve_f32(data, &props.get_size)?;
        let angles = resolve_f32(data, &props.get_angle)?;
        let pixel_offsets = resolve_vec2(data, &props.get_pixel_offset)?;

        let instance_positions: Vec<InstancePositions> = positions
            .iter()
            .map(|p| {
                let hi = [p[0] as f32, p[1] as f32, p[2] as f32];
                InstancePositions {
                    position: hi,
                    position_low: [
                        (p[0] - hi[0] as f64) as f32,
                        (p[1] - hi[1] as f64) as f32,
                        (p[2] - hi[2] as f64) as f32,
                    ],
                }
            })
            .collect();
        let instance_data: Vec<InstanceData> = (0..data.len())
            .map(|i| {
                let mapping = atlas.mapping.get(&icons[i]);
                let (frame, color_mode, offset) = match mapping {
                    Some(m) => {
                        let (w, h) = (m.width as f32, m.height as f32);
                        let anchor_x = m.anchor_x.unwrap_or(w / 2.0);
                        let anchor_y = m.anchor_y.unwrap_or(h / 2.0);
                        (
                            [m.x as f32, m.y as f32, w, h],
                            if m.mask { 1.0 } else { 0.0 },
                            [w / 2.0 - anchor_x, h / 2.0 - anchor_y],
                        )
                    }
                    // Unknown icons get an empty frame and are not drawn
                    None => ([0.0; 4], 0.0, [0.0; 2]),
                };
                InstanceData {
                    size: sizes[i],
                    angle: angles[i],
                    color: colors[i],
                    frame,
                    color_mode,
                    offset,
                    pixel_offset: pixel_offsets[i],
                    row_index: data.source_row(i),
                }
            })
            .collect();
        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(&ctx.device, "instancePositions", &instance_positions),
        )?;
        model.set_vertex_buffer(
            "instanceData",
            create_vertex_buffer_from(&ctx.device, "instanceData", &instance_data),
        )?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }
}

impl Layer for IconLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = assemble_shader(&self.props.base.id, &STANDARD_MODULES, SHADER)?;
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
                std::mem::size_of::<InstanceData>() as u64,
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
                ],
            ),
        ];
        let mut desc = ModelDescriptor::new(
            &self.props.base.id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        self.props.base.parameters.apply(&mut desc);
        let mut model = Model::new(&ctx.device, &desc)?;
        let positions: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_count(4);
        self.model = Some(model);
        self.data_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        let props = &self.props;
        let texture_size = self.texture_size;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;
        let u = model.uniforms("icon")?;
        u.set_f32("sizeScale", props.size_scale)?;
        u.set_vec2("iconsTextureDim", texture_size)?;
        u.set_f32("sizeBasis", if props.size_by_height { 1.0 } else { 0.0 })?;
        u.set_f32("sizeMinPixels", props.size_min_pixels)?;
        u.set_f32("sizeMaxPixels", props.size_max_pixels)?;
        u.set_i32("billboard", props.billboard as i32)?;
        u.set_i32("sizeUnits", props.size_units.shader_value())?;
        u.set_f32("alphaCutoff", props.alpha_cutoff)?;
        model.upload_uniforms(&ctx.queue);
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(model) = &self.model {
            model.draw(pass)?;
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        if let Some(model) = &mut self.model {
            set_model_picking_active(model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(model) = &self.model {
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

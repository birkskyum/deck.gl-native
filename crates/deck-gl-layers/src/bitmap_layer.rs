//! Port of `@deck.gl/layers/src/bitmap-layer/bitmap-layer.ts`: an image draped over
//! geographic bounds.

use std::sync::Arc;

use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{Layer, LayerContext, LayerProps, Result, Viewport};
use glam::{Vec3, Vec4};
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from, split_f64};
use luma_gl::{create_rgba8_texture, Model, ModelDescriptor, VertexBufferLayout};
use wgpu::VertexFormat;

const SHADER: &str = include_str!("wgsl/bitmap_layer.wgsl");

/// An RGBA8 image, row major, top row first.
#[derive(Clone, Debug, PartialEq)]
pub struct BitmapImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Arc<Vec<u8>>,
}

impl BitmapImage {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        debug_assert_eq!(rgba.len(), (width * height * 4) as usize);
        Self {
            width,
            height,
            rgba: Arc::new(rgba),
        }
    }
}

/// Properties of a [`BitmapLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct BitmapLayerProps {
    pub base: LayerProps,
    pub image: Option<BitmapImage>,
    /// `[min_lng, min_lat, max_lng, max_lat]` the image is stretched over
    pub bounds: [f64; 4],
    /// 0 keeps the image colors, 1 makes it grayscale
    pub desaturate: f32,
    /// RGBA in 0..255, blended in where the image is transparent
    pub transparent_color: [u8; 4],
    /// RGB in 0..255, multiplied with the image
    pub tint_color: [u8; 3],
}

impl Default for BitmapLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("BitmapLayer"),
            image: None,
            bounds: [1.0, 0.0, 0.0, 1.0],
            desaturate: 0.0,
            transparent_color: [0, 0, 0, 0],
            tint_color: [255, 255, 255],
        }
    }
}

/// Renders an image over a rectangle of longitude and latitude bounds.
pub struct BitmapLayer {
    props: BitmapLayerProps,
    model: Option<Model>,
    dirty: bool,
}

impl BitmapLayer {
    pub fn new(props: BitmapLayerProps) -> Self {
        Self {
            props,
            model: None,
            dirty: true,
        }
    }

    pub fn props(&self) -> &BitmapLayerProps {
        &self.props
    }

    /// Replace the props. Geometry is rebuilt on the next update when they changed.
    pub fn set_props(&mut self, props: BitmapLayerProps) {
        if self.props != props {
            self.props = props;
            self.dirty = true;
        }
    }

    fn update_mesh_and_texture(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        let [min_x, min_y, max_x, max_y] = props.bounds;
        // [[minX, minY], [minX, maxY], [maxX, maxY], [maxX, minY]]
        let corners: [f64; 12] = [
            min_x, min_y, 0.0, min_x, max_y, 0.0, max_x, max_y, 0.0, max_x, min_y, 0.0,
        ];
        let (hi, lo) = split_f64(&corners);
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &hi),
        )?;
        model.set_vertex_buffer(
            "positions64Low",
            create_vertex_buffer_from(&ctx.device, "positions64Low", &lo),
        )?;

        if let Some(image) = &props.image {
            let texture = create_rgba8_texture(
                &ctx.device,
                &ctx.queue,
                &props.base.id,
                image.width,
                image.height,
                &image.rgba,
            );
            model.set_texture("bitmapTexture", texture.create_view(&Default::default()))?;
        }
        Ok(())
    }
}

impl Layer for BitmapLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = self.props.base.extensions.assemble_without_attributes(
            &self.props.base.id,
            &STANDARD_MODULES,
            SHADER,
        )?;
        let layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::vertex("positions64Low", 1, VertexFormat::Float32x3),
            VertexBufferLayout::vertex("texCoords", 2, VertexFormat::Float32x2),
        ];
        let mut desc = ModelDescriptor::new(
            &self.props.base.id,
            &shader,
            &layouts,
            wgpu::PrimitiveTopology::TriangleList,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.pickable = self.props.base.pickable;
        self.props.base.parameters.apply(&mut desc);
        let mut model = Model::new(&ctx.device, &desc)?;
        let tex_coords: [f32; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let indices: [u32; 6] = [0, 2, 1, 0, 3, 2];
        model.set_vertex_buffer(
            "texCoords",
            create_vertex_buffer_from(&ctx.device, "texCoords", &tex_coords),
        )?;
        model.set_index_buffer(
            create_index_buffer(&ctx.device, "indices", &indices),
            wgpu::IndexFormat::Uint32,
            6,
        );
        model.set_instance_count(1);
        self.model = Some(model);
        self.dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.dirty {
            self.update_mesh_and_texture(ctx)?;
            self.dirty = false;
        }
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &self.props.base.id)?;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;

        let u = model.uniforms("bitmap")?;
        let b = props.bounds;
        u.set_vec4(
            "bounds",
            Vec4::new(b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32),
        )?;
        u.set_f32("coordinateConversion", 0.0)?;
        u.set_f32("desaturate", props.desaturate)?;
        let t = props.tint_color;
        u.set_vec3(
            "tintColor",
            Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32) / 255.0,
        )?;
        let c = props.transparent_color;
        u.set_vec4(
            "transparentColor",
            Vec4::new(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32) / 255.0,
        )?;
        model.upload_uniforms(&ctx.queue);
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(model) = &self.model {
            if self.props.image.is_some() {
                model.draw(pass)?;
            }
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
            if self.props.image.is_some() {
                model.draw_picking(pass)?;
            }
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

//! Port of `@deck.gl/aggregation-layers/src/heatmap-layer/heatmap-layer.ts`: GPU aggregation of
//! weighted points into a texture that covers the visible area, coloured through a colour range.
//!
//! Each frame on the main viewport the layer checks whether the view still lies inside the
//! aggregated bounds; if not (or when the zoom or the data changed) it splats every point into a
//! float weights texture with additive blending, reduces the texture to its maximum with a max
//! blend, and then draws the visible part of the texture through the colour ramp.

use deck_gl::data::{resolve_f32, resolve_positions};
use deck_gl::glam::{DVec2, DVec3};
use deck_gl::layer::update_standard_uniforms;
use deck_gl::luma_gl::device::create_render_texture;
use deck_gl::luma_gl::model::create_rgba8_texture;
use deck_gl::shaderlib::STANDARD_MODULES;
use deck_gl::{
    Accessor, Color, Layer, LayerContext, LayerData, LayerProps, Position, ProjectionMode, RenderParameters,
    Result, Viewport,
};
use luma_gl::buffer::{create_vertex_buffer_from, split_f64};
use luma_gl::{assemble_shader, Model, ModelDescriptor, RenderTarget, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::aggregation::DEFAULT_COLOR_RANGE;

const WEIGHTS_SHADER: &str = include_str!("wgsl/heatmap_weights.wgsl");
const MAX_SHADER: &str = include_str!("wgsl/heatmap_max.wgsl");
const TRIANGLE_SHADER: &str = include_str!("wgsl/heatmap_triangle.wgsl");

/// Common space units per texel of the weights texture
const RESOLUTION: f64 = 2.0;
const MAX_WEIGHT_REDUCTION_SIZE: u32 = 16;
const WEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How the weights of overlapping points combine, deck.gl's `aggregation`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeatmapAggregation {
    #[default]
    Sum,
    Mean,
}

impl HeatmapAggregation {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "SUM" => Some(Self::Sum),
            "MEAN" => Some(Self::Mean),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeatmapLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    pub get_position: Accessor<Position>,
    pub get_weight: Accessor<f32>,
    /// Radius in pixels over which each point's weight is spread
    pub radius_pixels: f32,
    /// Colours from the lowest to the highest weight
    pub color_range: Vec<Color>,
    /// Multiplier on the aggregated weight
    pub intensity: f32,
    /// Ratio of the fading weight to the maximum weight (0 to 1); ignored with `color_domain`
    pub threshold: f32,
    /// Weights mapped to the first and last colour, instead of 0 and the maximum
    pub color_domain: Option<[f32; 2]>,
    pub aggregation: HeatmapAggregation,
    /// Side of the square weights texture in texels (128 to 2048)
    pub weights_texture_size: u32,
}

impl Default for HeatmapLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("heatmap"),
            data: LayerData::default(),
            get_position: Accessor::Constant([0.0, 0.0, 0.0]),
            get_weight: Accessor::Constant(1.0),
            radius_pixels: 50.0,
            color_range: DEFAULT_COLOR_RANGE.to_vec(),
            intensity: 1.0,
            threshold: 0.05,
            color_domain: None,
            aggregation: HeatmapAggregation::Sum,
            weights_texture_size: 2048,
        }
    }
}

struct Resources {
    weights: Model,
    max: Model,
    triangle: Model,
    weights_texture: wgpu::Texture,
    max_texture: wgpu::Texture,
    texture_size: u32,
}

pub struct HeatmapLayer {
    props: HeatmapLayerProps,
    resources: Option<Resources>,
    data_dirty: bool,
    color_range_dirty: bool,
    instance_count: u32,
    /// Longitude and latitude bounds the weights texture covers
    world_bounds: Option<[f64; 4]>,
    normalized_common_bounds: [f64; 4],
    /// The four viewport corners, unprojected onto the ground
    viewport_corners: [[f64; 2]; 4],
    zoom: Option<f64>,
    /// Colour domain in weight units, see `update_weightmap`
    color_domain: [f32; 2],
}

impl HeatmapLayer {
    pub fn new(props: HeatmapLayerProps) -> Self {
        Self {
            props,
            resources: None,
            data_dirty: true,
            color_range_dirty: true,
            instance_count: 0,
            world_bounds: None,
            normalized_common_bounds: [0.0; 4],
            viewport_corners: [[0.0; 2]; 4],
            zoom: None,
            color_domain: [0.0; 2],
        }
    }

    pub fn props(&self) -> &HeatmapLayerProps {
        &self.props
    }

    pub fn set_props(&mut self, props: HeatmapLayerProps) {
        if self.props != props {
            if self.props.color_range != props.color_range {
                self.color_range_dirty = true;
            }
            self.props = props;
            self.data_dirty = true;
        }
    }

    fn texture_size(&self, ctx: &LayerContext) -> u32 {
        self.props
            .weights_texture_size
            .clamp(128, ctx.device.limits().max_texture_dimension_2d)
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let positions = resolve_positions(&props.data, &props.get_position)?;
        let weights = resolve_f32(&props.data, &props.get_weight)?;
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let (hi, lo) = split_f64(&flat);
        let model = &mut self.resources.as_mut().expect("initialized").weights;
        model.set_vertex_buffer(
            "instancePositions",
            create_vertex_buffer_from(&ctx.device, "instancePositions", &hi),
        )?;
        model.set_vertex_buffer(
            "instancePositions64Low",
            create_vertex_buffer_from(&ctx.device, "instancePositions64Low", &lo),
        )?;
        model.set_vertex_buffer(
            "instanceWeights",
            create_vertex_buffer_from(&ctx.device, "instanceWeights", &weights),
        )?;
        self.instance_count = props.data.len() as u32;
        model.set_instance_count(self.instance_count);
        model.set_vertex_count(6);
        Ok(())
    }

    /// The area the weights texture covers: the visible bounds, expanded to a square of at
    /// least `texture_size * RESOLUTION` common units. Returns true when it changed.
    fn update_bounds(&mut self, viewport: &Viewport, force: bool) -> bool {
        let corner = |x: f64, y: f64| {
            let p = viewport.unproject(DVec2::new(x, y), None, true, Some(0.0));
            [p.x as f32 as f64, p.y as f32 as f64]
        };
        self.viewport_corners = [
            corner(0.0, 0.0),
            corner(viewport.width, 0.0),
            corner(0.0, viewport.height),
            corner(viewport.width, viewport.height),
        ];
        let visible = bounds_of(&self.viewport_corners);
        let contained = self
            .world_bounds
            .is_some_and(|current| bounds_contain(current, visible));
        if !force && contained {
            return false;
        }
        let texture_size = self.resources.as_ref().map_or(128, |r| r.texture_size);
        let scaled_common = world_to_common_bounds(visible, viewport, texture_size);
        let mut world = common_to_world_bounds(scaled_common, viewport);
        // Clip to the web mercator limits
        world[1] = world[1].max(-85.051129);
        world[3] = world[3].min(85.051129);
        world[0] = world[0].max(-360.0);
        world[2] = world[2].min(360.0);
        self.normalized_common_bounds = world_to_common_bounds(world, viewport, texture_size);
        self.world_bounds = Some(world);
        true
    }

    /// The quad that shows the visible part of the texture: viewport corners in world
    /// coordinates with their texture coordinates.
    fn update_texture_rendering_bounds(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        let bounds = self.normalized_common_bounds;
        let mut positions = Vec::with_capacity(12);
        let mut tex_coords = Vec::with_capacity(8);
        for corner in self.viewport_corners {
            positions.extend_from_slice(&[corner[0] as f32, corner[1] as f32, 0.0]);
            let common = viewport.project_position(DVec3::new(corner[0], corner[1], 0.0));
            tex_coords.push(((common.x - bounds[0]) / (bounds[2] - bounds[0])) as f32);
            tex_coords.push(((common.y - bounds[1]) / (bounds[3] - bounds[1])) as f32);
        }
        let model = &mut self.resources.as_mut().expect("initialized").triangle;
        model.set_vertex_buffer(
            "positions",
            create_vertex_buffer_from(&ctx.device, "positions", &positions),
        )?;
        model.set_vertex_buffer(
            "texCoords",
            create_vertex_buffer_from(&ctx.device, "texCoords", &tex_coords),
        )?;
        model.set_vertex_count(4);
        Ok(())
    }

    fn update_color_texture(&mut self, ctx: &LayerContext) -> Result<()> {
        let colors: Vec<u8> = self.props.color_range.iter().flatten().copied().collect();
        let width = self.props.color_range.len().max(1) as u32;
        let texture = create_rgba8_texture(&ctx.device, &ctx.queue, "heatmap colors", width, 1, &colors);
        let model = &mut self.resources.as_mut().expect("initialized").triangle;
        model.set_texture("colorTexture", texture.create_view(&Default::default()))?;
        self.color_range_dirty = false;
        Ok(())
    }

    /// Splat the points into the weights texture and reduce it to its maximum.
    fn update_weightmap(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        let props = &self.props;
        let resources = self.resources.as_mut().expect("initialized");
        let texture_size = resources.texture_size;
        let world = self.world_bounds.expect("bounds");
        // The weights shader projects through the layer's project module, whose positions are
        // relative to the viewport centre in auto offset mode (zoom 12 and up): shift the
        // bounds the same way, as deck.gl's layer level projectPosition does.
        let mut common = world_to_common_bounds(world, viewport, texture_size);
        if viewport.projection_mode() == ProjectionMode::WebMercatorAutoOffset {
            let origin = viewport.project_position(viewport.geospatial_origin_f32());
            common = [
                common[0] - origin.x,
                common[1] - origin.y,
                common[2] - origin.x,
                common[3] - origin.y,
            ];
        }
        self.color_domain = match (props.color_domain, props.aggregation) {
            (Some(domain), HeatmapAggregation::Sum) => {
                // scale the colour domain to weight per texel
                let meters_per_texel = viewport.distance_scales.meters_per_unit.z * (common[2] - common[0])
                    / texture_size as f64;
                [
                    domain[0] * meters_per_texel as f32,
                    domain[1] * meters_per_texel as f32,
                ]
            }
            (Some(domain), HeatmapAggregation::Mean) => domain,
            (None, _) => [0.0, 0.0],
        };

        let weights = &mut resources.weights;
        update_standard_uniforms(weights, ctx, viewport, &props.base)?;
        let u = weights.uniforms("weight")?;
        u.set_vec4(
            "commonBounds",
            deck_gl::glam::Vec4::new(
                common[0] as f32,
                common[1] as f32,
                common[2] as f32,
                common[3] as f32,
            ),
        )?;
        u.set_f32("radiusPixels", props.radius_pixels)?;
        u.set_f32("textureWidth", texture_size as f32)?;
        u.set_f32("weightsScale", 1.0)?;
        weights.upload_uniforms(&ctx.queue);
        resources.max.set_uniform_slot(0);
        resources
            .max
            .uniforms("maxWeight")?
            .set_f32("textureSize", texture_size as f32)?;
        resources.max.upload_uniforms(&ctx.queue);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("heatmap aggregation"),
            });
        let weights_view = resources.weights_texture.create_view(&Default::default());
        let max_view = resources.max_texture.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("heatmap weights"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &weights_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count > 0 {
                resources.weights.draw(&mut pass)?;
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("heatmap max weight"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &max_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            resources.max.draw(&mut pass)?;
        }
        ctx.queue.submit([encoder.finish()]);
        Ok(())
    }
}

fn bounds_of(points: &[[f64; 2]]) -> [f64; 4] {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

fn bounds_contain(current: [f64; 4], target: [f64; 4]) -> bool {
    target[0] >= current[0] && target[2] <= current[2] && target[1] >= current[1] && target[3] <= current[3]
}

/// Expand a bounding box to the aspect ratio of `width` x `height`, and to at least that size.
fn scale_to_aspect_ratio(bounds: [f64; 4], width: f64, height: f64) -> [f64; 4] {
    let [x_min, y_min, x_max, y_max] = bounds;
    let current_width = x_max - x_min;
    let current_height = y_max - y_min;
    let mut new_width = current_width;
    let mut new_height = current_height;
    if current_width / current_height < width / height {
        new_width = (width / height) * current_height;
    } else {
        new_height = (height / width) * current_width;
    }
    if new_width < width {
        new_width = width;
        new_height = height;
    }
    let x_center = (x_max + x_min) / 2.0;
    let y_center = (y_max + y_min) / 2.0;
    [
        x_center - new_width / 2.0,
        y_center - new_height / 2.0,
        x_center + new_width / 2.0,
        y_center + new_height / 2.0,
    ]
}

/// Longitude and latitude bounds to common space, expanded to the texture's square.
fn world_to_common_bounds(world: [f64; 4], viewport: &Viewport, texture_size: u32) -> [f64; 4] {
    let size = texture_size as f64 * RESOLUTION / viewport.scale;
    let bottom_left = viewport.project_position(DVec3::new(world[0], world[1], 0.0));
    let top_right = viewport.project_position(DVec3::new(world[2], world[3], 0.0));
    scale_to_aspect_ratio(
        [bottom_left.x, bottom_left.y, top_right.x, top_right.y],
        size,
        size,
    )
}

fn common_to_world_bounds(common: [f64; 4], viewport: &Viewport) -> [f64; 4] {
    let bottom_left = viewport.unproject_position(DVec3::new(common[0], common[1], 0.0));
    let top_right = viewport.unproject_position(DVec3::new(common[2], common[3], 0.0));
    [bottom_left.x, bottom_left.y, top_right.x, top_right.y]
}

fn one_one(operation: wgpu::BlendOperation) -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

impl Layer for HeatmapLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let id = &self.props.base.id;
        let texture_size = self.texture_size(ctx);
        let offscreen = RenderTarget {
            color_format: WEIGHTS_FORMAT,
            depth_format: None,
            sample_count: 1,
        };

        let weights_label = format!("{id}-weights");
        let max_label = format!("{id}-max");
        let weights_shader = assemble_shader(&weights_label, &STANDARD_MODULES, WEIGHTS_SHADER)?;
        let weights_layouts = [
            VertexBufferLayout::instance("instancePositions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instancePositions64Low", 1, VertexFormat::Float32x3),
            VertexBufferLayout::instance("instanceWeights", 2, VertexFormat::Float32),
        ];
        let mut desc = ModelDescriptor::new(
            &weights_label,
            &weights_shader,
            &weights_layouts,
            wgpu::PrimitiveTopology::TriangleList,
            offscreen,
        );
        desc.blend = Some(one_one(wgpu::BlendOperation::Add));
        let weights = Model::new(&ctx.device, &desc)?;

        let max_shader = assemble_shader(&max_label, &[], MAX_SHADER)?;
        let mut desc = ModelDescriptor::new(
            &max_label,
            &max_shader,
            &[],
            wgpu::PrimitiveTopology::PointList,
            offscreen,
        );
        desc.blend = Some(one_one(wgpu::BlendOperation::Max));
        let mut max = Model::new(&ctx.device, &desc)?;
        max.set_vertex_count(texture_size.div_ceil(MAX_WEIGHT_REDUCTION_SIZE).pow(2));

        let triangle_shader = assemble_shader(&format!("{id}-triangle"), &STANDARD_MODULES, TRIANGLE_SHADER)?;
        let triangle_layouts = [
            VertexBufferLayout::vertex("positions", 0, VertexFormat::Float32x3),
            VertexBufferLayout::vertex("texCoords", 1, VertexFormat::Float32x2),
        ];
        let mut desc = ModelDescriptor::new(
            id,
            &triangle_shader,
            &triangle_layouts,
            wgpu::PrimitiveTopology::TriangleStrip,
            ctx.target,
        );
        desc.depth_bias = ctx.depth_bias();
        desc.depth_write_enabled = false;
        RenderParameters::apply(&self.props.base.parameters, &mut desc);
        let mut triangle = Model::new(&ctx.device, &desc)?;

        let weights_texture = create_render_texture(
            &ctx.device,
            "heatmap weights",
            texture_size,
            texture_size,
            WEIGHTS_FORMAT,
        );
        let max_texture = create_render_texture(&ctx.device, "heatmap max weight", 1, 1, WEIGHTS_FORMAT);
        max.set_texture("inTexture", weights_texture.create_view(&Default::default()))?;
        triangle.set_texture("weightsTexture", weights_texture.create_view(&Default::default()))?;
        triangle.set_texture("maxTexture", max_texture.create_view(&Default::default()))?;

        self.resources = Some(Resources {
            weights,
            max,
            triangle,
            weights_texture,
            max_texture,
            texture_size,
        });
        self.data_dirty = true;
        self.color_range_dirty = true;
        self.world_bounds = None;
        self.zoom = None;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        // Aggregation follows the main viewport only; repeated world copies reuse it, like
        // deck.gl, since the quad is in world coordinates.
        if ctx.uniform_slot == 0 {
            let mut force = false;
            if self.data_dirty {
                self.update_attributes(ctx)?;
                self.data_dirty = false;
                force = true;
            }
            let bounds_changed = self.update_bounds(viewport, force);
            self.update_texture_rendering_bounds(ctx, viewport)?;
            if self.color_range_dirty {
                self.update_color_texture(ctx)?;
            }
            let zoom_changed = self.zoom != Some(viewport.zoom);
            if force || bounds_changed || zoom_changed {
                self.update_weightmap(ctx, viewport)?;
                self.zoom = Some(viewport.zoom);
            }
        }
        let props = &self.props;
        let color_domain = self.color_domain;
        let model = &mut self.resources.as_mut().expect("initialized").triangle;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;
        let u = model.uniforms("triangle")?;
        u.set_f32(
            "aggregationMode",
            match props.aggregation {
                HeatmapAggregation::Sum => 0.0,
                HeatmapAggregation::Mean => 1.0,
            },
        )?;
        u.set_vec2(
            "colorDomain",
            deck_gl::glam::Vec2::new(color_domain[0], color_domain[1]),
        )?;
        u.set_f32("intensity", props.intensity)?;
        u.set_f32("threshold", props.threshold)?;
        model.upload_uniforms(&ctx.queue);
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        if let Some(resources) = &self.resources {
            resources.triangle.draw(pass)?;
        }
        Ok(())
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

impl Default for HeatmapLayer {
    fn default() -> Self {
        Self::new(HeatmapLayerProps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_expands_to_the_texture_square() {
        let b = scale_to_aspect_ratio([0.0, 0.0, 10.0, 5.0], 100.0, 100.0);
        assert_eq!(b, [-45.0, -47.5, 55.0, 52.5]);
        let b = scale_to_aspect_ratio([0.0, 0.0, 400.0, 100.0], 100.0, 100.0);
        assert_eq!(b, [0.0, -150.0, 400.0, 250.0]);
        assert!(bounds_contain([0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]));
        assert!(!bounds_contain([0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 9.0]));
    }
}

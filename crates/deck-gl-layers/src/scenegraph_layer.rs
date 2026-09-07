//! Port of `@deck.gl/mesh-layers/src/scenegraph-layer/scenegraph-layer.ts`: instances of a
//! glTF scene at given coordinates. Every primitive of the scene is a model that shares the
//! instance attributes; its node transform, base colour and base colour texture are its own.

use std::sync::Arc;

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::data::{resolve_vec3, resolve_with};
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, DeckError, Layer, LayerContext, LayerData, LayerProps, Position, Result, Viewport,
};
use glam::Vec4;
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from};
use luma_gl::model::create_rgba8_texture;
use luma_gl::{Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::mesh::MeshVertex;
use crate::scenegraph::Scenegraph;
use crate::simple_mesh_layer::transform_matrix;

const SHADER: &str = include_str!("wgsl/scenegraph_layer.wgsl");

/// deck.gl's `_lighting`: `Flat` draws the material colours as they are, `Pbr` lights them
/// with the deck's lights (through the same shading as the other lit layers, not a full PBR
/// model).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScenegraphLighting {
    #[default]
    Flat,
    Pbr,
}

impl ScenegraphLighting {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "flat" => Some(Self::Flat),
            "pbr" => Some(Self::Pbr),
            _ => None,
        }
    }
}

/// Properties of a [`ScenegraphLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenegraphLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// The scene every instance draws; nothing is drawn without one
    pub scenegraph: Option<Arc<Scenegraph>>,
    /// Multiplier applied to the scene's units
    pub size_scale: f32,
    /// The minimum size in pixels of one scene unit
    pub size_min_pixels: f32,
    /// The maximum size in pixels of one scene unit
    pub size_max_pixels: f32,
    pub lighting: ScenegraphLighting,
    pub get_position: Accessor<Position>,
    /// Multiplied with the material colours; white keeps them
    pub get_color: Accessor<Color>,
    /// `[pitch, yaw, roll]` in degrees
    pub get_orientation: Accessor<[f32; 3]>,
    pub get_scale: Accessor<[f32; 3]>,
    /// Offset from the anchor in meters
    pub get_translation: Accessor<[f32; 3]>,
    /// A column major 4x4 matrix per instance that replaces orientation, scale and
    /// translation when given
    pub get_transform_matrix: Option<Accessor<[f32; 16]>>,
}

impl Default for ScenegraphLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("ScenegraphLayer"),
            data: LayerData::default(),
            scenegraph: None,
            size_scale: 1.0,
            size_min_pixels: 0.0,
            size_max_pixels: f32::MAX,
            lighting: ScenegraphLighting::Flat,
            get_position: Accessor::column("position"),
            get_color: Accessor::Constant([255, 255, 255, 255]),
            get_orientation: Accessor::Constant([0.0, 0.0, 0.0]),
            get_scale: Accessor::Constant([1.0, 1.0, 1.0]),
            get_translation: Accessor::Constant([0.0, 0.0, 0.0]),
            get_transform_matrix: None,
        }
    }
}

/// The instance buffers, shared by every primitive's model.
fn scenegraph_attributes() -> AttributeManager {
    AttributeManager::new(vec![
        BufferSpec::interleaved(
            "instancePositions",
            24,
            vec![
                Field::new("position", 4, VertexFormat::Float32x3, 0),
                Field::low("position", 5, 12),
            ],
        ),
        BufferSpec::instance("instanceColors", "color", 6, VertexFormat::Unorm8x4),
        BufferSpec::interleaved(
            "instanceModelMatrix",
            48,
            vec![
                Field::new("modelMatrixCol0", 7, VertexFormat::Float32x3, 0),
                Field::new("modelMatrixCol1", 8, VertexFormat::Float32x3, 12),
                Field::new("modelMatrixCol2", 9, VertexFormat::Float32x3, 24),
                Field::new("translation", 10, VertexFormat::Float32x3, 36),
            ],
        ),
    ])
}

/// One model per primitive of the scene.
struct PrimitiveModel {
    model: Model,
    primitive: usize,
}

/// Renders instances of a glTF scene at given coordinates.
pub struct ScenegraphLayer {
    props: ScenegraphLayerProps,
    models: Vec<PrimitiveModel>,
    attributes: AttributeManager,
    data_dirty: bool,
    models_dirty: bool,
}

impl ScenegraphLayer {
    pub fn new(props: ScenegraphLayerProps) -> Self {
        Self {
            props,
            models: Vec::new(),
            attributes: scenegraph_attributes(),
            data_dirty: true,
            models_dirty: false,
        }
    }

    pub fn props(&self) -> &ScenegraphLayerProps {
        &self.props
    }

    /// Replace the props. The models are rebuilt when the scene or the pipeline state
    /// changed, the attributes when accessors or data changed.
    pub fn set_props(&mut self, props: ScenegraphLayerProps) {
        if self.props == props {
            return;
        }
        if self.props.base.needs_new_model(&props.base)
            || !same_scene(&self.props.scenegraph, &props.scenegraph)
        {
            self.models_dirty = true;
        }
        if self.props.base.extensions != props.base.extensions
            || Self::attributes_changed(&self.props, &props)
        {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms and the scene.
    fn attributes_changed(old: &ScenegraphLayerProps, new: &ScenegraphLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.scenegraph = old.scenegraph.clone();
        probe.size_scale = old.size_scale;
        probe.size_min_pixels = old.size_min_pixels;
        probe.size_max_pixels = old.size_max_pixels;
        probe.lighting = old.lighting;
        probe != *old
    }

    /// The per instance model matrix columns and translations, as deck.gl's
    /// `MATRIX_ATTRIBUTES` computes them.
    fn instance_matrices(props: &ScenegraphLayerProps) -> Result<Arc<Vec<[[f32; 3]; 4]>>> {
        let data = &props.data;
        if let Some(matrix) = &props.get_transform_matrix {
            let matrices = resolve_with(data, matrix, |_| {
                Err(DeckError::Data(
                    "getTransformMatrix columns are not supported; use a constant or a function".into(),
                ))
            })?;
            return Ok(Arc::new(
                matrices
                    .iter()
                    .map(|m| {
                        [
                            [m[0], m[1], m[2]],
                            [m[4], m[5], m[6]],
                            [m[8], m[9], m[10]],
                            [m[12], m[13], m[14]],
                        ]
                    })
                    .collect(),
            ));
        }
        let orientations = resolve_vec3(data, &props.get_orientation)?;
        let scales = resolve_vec3(data, &props.get_scale)?;
        let translations = resolve_vec3(data, &props.get_translation)?;
        Ok(Arc::new(
            (0..data.len())
                .map(|i| {
                    let [c0, c1, c2] = transform_matrix(orientations[i], scales[i]);
                    [c0, c1, c2, translations[i]]
                })
                .collect(),
        ))
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let data = &props.data;
        let matrices = Self::instance_matrices(props)?;
        let column = |k: usize| {
            let matrices = matrices.clone();
            AttributeSource::Vec3(Accessor::func(move |i| matrices[i][k]))
        };
        let mut sources = vec![
            ("position", AttributeSource::Positions(props.get_position.clone())),
            ("color", AttributeSource::Colors(props.get_color.clone())),
            ("modelMatrixCol0", column(0)),
            ("modelMatrixCol1", column(1)),
            ("modelMatrixCol2", column(2)),
            ("translation", column(3)),
        ];
        sources.extend(props.base.extensions.sources(data)?);
        let mut models: Vec<&mut Model> = self.models.iter_mut().map(|m| &mut m.model).collect();
        self.attributes
            .update_many(&ctx.device, &ctx.queue, &mut models, data, &sources)?;
        for model in models {
            model.set_instance_count(data.len() as u32);
        }
        Ok(())
    }
}

/// Whether two scene props are the same scene, by identity first and by value otherwise.
fn same_scene(a: &Option<Arc<Scenegraph>>, b: &Option<Arc<Scenegraph>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
        _ => false,
    }
}

impl Layer for ScenegraphLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        self.models.clear();
        self.models_dirty = false;
        let props = &self.props;
        let Some(scene) = &props.scenegraph else {
            return Ok(());
        };
        let modules: Vec<ShaderModuleSource> = STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect();
        let extensions = &props.base.extensions;
        let shader = extensions.assemble(&props.base.id, &modules, SHADER)?;
        self.attributes = scenegraph_attributes();
        self.attributes.extend(extensions.buffer_specs(&shader)?);
        let mut layouts = vec![VertexBufferLayout::interleaved(
            "geometry",
            std::mem::size_of::<MeshVertex>() as u64,
            wgpu::VertexStepMode::Vertex,
            &[
                (0, VertexFormat::Float32x3, 0),
                (1, VertexFormat::Float32x3, 12),
                (2, VertexFormat::Float32x3, 24),
                (3, VertexFormat::Float32x2, 36),
            ],
        )];
        layouts.extend(self.attributes.layouts());
        let textures: Vec<wgpu::TextureView> = scene
            .textures
            .iter()
            .enumerate()
            .map(|(i, image)| {
                create_rgba8_texture(
                    &ctx.device,
                    &ctx.queue,
                    &format!("{}-texture-{i}", props.base.id),
                    image.width,
                    image.height,
                    &image.rgba,
                )
                .create_view(&Default::default())
            })
            .collect();
        for (index, primitive) in scene.primitives.iter().enumerate() {
            let label = format!("{}-primitive-{index}", props.base.id);
            let mut desc = ModelDescriptor::new(
                &label,
                &shader,
                &layouts,
                wgpu::PrimitiveTopology::TriangleList,
                ctx.target,
            );
            ctx.configure(&mut desc, &props.base);
            let mut model = Model::new(&ctx.device, &desc)?;
            let vertices = primitive.mesh.vertices();
            model.set_vertex_buffer(
                "geometry",
                create_vertex_buffer_from(&ctx.device, &format!("{label}-geometry"), &vertices),
            )?;
            model.set_vertex_count(vertices.len() as u32);
            if let Some(indices) = &primitive.mesh.indices {
                model.set_index_buffer(
                    create_index_buffer(&ctx.device, &format!("{label}-indices"), indices),
                    wgpu::IndexFormat::Uint32,
                    indices.len() as u32,
                );
            }
            if let Some(view) = primitive.base_color_texture.and_then(|i| textures.get(i)) {
                model.set_texture("scenegraphTexture", view.clone())?;
            }
            self.models.push(PrimitiveModel {
                model,
                primitive: index,
            });
        }
        self.data_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.models_dirty {
            self.initialize(ctx)?;
        }
        if self.models.is_empty() {
            return Ok(());
        }
        self.attributes.set_time(ctx.time);
        self.attributes.set_transitions(&self.props.base.transitions);
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        {
            let mut models: Vec<&mut Model> = self.models.iter_mut().map(|m| &mut m.model).collect();
            self.attributes
                .animate(&ctx.device, &ctx.queue, &mut models, &self.props.data, ctx.time)?;
        }
        let props = &self.props;
        let Some(scene) = &props.scenegraph else {
            return Ok(());
        };
        // deck.gl's `shouldComposeModelMatrix`: only off a map are scene coordinates composed
        // into world coordinates; on a map they stay meter offsets around the anchor
        let compose = !viewport.is_geospatial;
        for entry in &mut self.models {
            let Some(primitive) = scene.primitives.get(entry.primitive) else {
                continue;
            };
            let model = &mut entry.model;
            update_standard_uniforms(model, ctx, viewport, &props.base)?;
            let u = model.uniforms("scenegraph")?;
            u.set_mat4("sceneModelMatrix", primitive.model_matrix)?;
            u.set_vec4("baseColor", Vec4::from(primitive.base_color))?;
            u.set_f32("sizeScale", props.size_scale)?;
            u.set_f32("sizeMinPixels", props.size_min_pixels)?;
            u.set_f32("sizeMaxPixels", props.size_max_pixels)?;
            u.set_f32("composeModelMatrix", if compose { 1.0 } else { 0.0 })?;
            u.set_f32(
                "hasTexture",
                if primitive
                    .base_color_texture
                    .is_some_and(|i| i < scene.textures.len())
                {
                    1.0
                } else {
                    0.0
                },
            )?;
            u.set_f32(
                "lighting",
                if props.lighting == ScenegraphLighting::Pbr {
                    1.0
                } else {
                    0.0
                },
            )?;
            u.set_f32("hasNormals", if primitive.mesh.has_normals() { 1.0 } else { 0.0 })?;
            model.upload_uniforms(&ctx.queue);
        }
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for entry in &self.models {
            entry.model.draw(pass)?;
        }
        Ok(())
    }

    fn set_picking_active(&mut self, ctx: &LayerContext, active: bool) -> Result<()> {
        for entry in &mut self.models {
            set_model_picking_active(&mut entry.model, &ctx.queue, active)?;
        }
        Ok(())
    }

    fn draw_picking(&mut self, _ctx: &LayerContext, pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        for entry in &self.models {
            entry.model.draw_picking(pass)?;
        }
        Ok(())
    }

    fn set_highlighted_object(&mut self, index: Option<u32>) {
        self.props.base.highlighted_object_index = index;
    }

    fn bounds(&self) -> Option<[f64; 4]> {
        self.attributes.bounds()
    }

    fn in_transition(&self) -> bool {
        self.attributes.in_transition() || self.models.iter().any(|m| m.model.in_transition())
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

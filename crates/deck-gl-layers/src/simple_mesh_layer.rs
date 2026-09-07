//! Port of `@deck.gl/mesh-layers/src/simple-mesh-layer/simple-mesh-layer.ts`: instances of
//! one triangle mesh at given coordinates, each with its own orientation, scale, translation
//! and colour, lit by the deck's lights and optionally textured.

use std::sync::Arc;

use deck_gl::attribute_manager::{AttributeManager, AttributeSource, BufferSpec, Field};
use deck_gl::data::{resolve_vec3, resolve_with};
use deck_gl::layer::{initialized, set_model_picking_active, update_standard_uniforms};
use deck_gl::shaderlib::{LIGHTING_MODULES, STANDARD_MODULES};
use deck_gl::{
    Accessor, Color, DeckError, Layer, LayerContext, LayerData, LayerProps, Position, Result, Viewport,
};
use luma_gl::buffer::{create_index_buffer, create_vertex_buffer_from};
use luma_gl::model::create_rgba8_texture;
use luma_gl::{Model, ModelDescriptor, ShaderModuleSource, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::mesh::{Mesh, MeshVertex};
use crate::BitmapImage;

const SHADER: &str = include_str!("wgsl/simple_mesh_layer.wgsl");

/// Properties of a [`SimpleMeshLayer`]. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleMeshLayerProps {
    pub base: LayerProps,
    pub data: LayerData,
    /// The mesh every instance draws; nothing is drawn without one
    pub mesh: Option<Arc<Mesh>>,
    /// Sampled with the mesh's texture coordinates; replaces the colours when given
    pub texture: Option<BitmapImage>,
    /// Multiplier applied to every mesh
    pub size_scale: f32,
    /// Draw the mesh as a line strip through its vertices, as deck.gl's quick wireframe does
    pub wireframe: bool,
    /// deck.gl's `_instanced`: with `false`, mesh positions are offsets in the layer's
    /// coordinates (degrees on a map) instead of meters around the anchor
    pub instanced: bool,
    pub get_position: Accessor<Position>,
    /// Multiplied with the mesh's vertex colours; `[255, 255, 255, 255]` keeps them
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

impl Default for SimpleMeshLayerProps {
    fn default() -> Self {
        Self {
            base: LayerProps::new("SimpleMeshLayer"),
            data: LayerData::default(),
            mesh: None,
            texture: None,
            size_scale: 1.0,
            wireframe: false,
            instanced: true,
            get_position: Accessor::column("position"),
            get_color: Accessor::Constant([0, 0, 0, 255]),
            get_orientation: Accessor::Constant([0.0, 0.0, 0.0]),
            get_scale: Accessor::Constant([1.0, 1.0, 1.0]),
            get_translation: Accessor::Constant([0.0, 0.0, 0.0]),
            get_transform_matrix: None,
        }
    }
}

/// deck.gl's `calculateTransformMatrix`: the rotation from `[pitch, yaw, roll]` degrees
/// scaled per axis, as three columns.
pub fn transform_matrix(orientation: [f32; 3], scale: [f32; 3]) -> [[f32; 3]; 3] {
    let [pitch, yaw, roll] = orientation.map(f32::to_radians);
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sw, cw) = yaw.sin_cos();
    let [scx, scy, scz] = scale;
    [
        [scx * cw * cp, scx * sw * cp, scx * -sp],
        [
            scy * (-sw * cr + cw * sp * sr),
            scy * (cw * cr + sw * sp * sr),
            scy * cp * sr,
        ],
        [
            scz * (sw * sr + cw * sp * cr),
            scz * (-cw * sr + sw * sp * cr),
            scz * cp * cr,
        ],
    ]
}

/// The instance buffers: position with its low part, colour, and the model matrix columns
/// with the translation.
fn mesh_attributes() -> AttributeManager {
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

/// Renders instances of a mesh at given coordinates.
pub struct SimpleMeshLayer {
    props: SimpleMeshLayerProps,
    model: Option<Model>,
    attributes: AttributeManager,
    data_dirty: bool,
    texture_dirty: bool,
    models_dirty: bool,
}

impl SimpleMeshLayer {
    pub fn new(props: SimpleMeshLayerProps) -> Self {
        Self {
            props,
            model: None,
            attributes: mesh_attributes(),
            data_dirty: true,
            texture_dirty: true,
            models_dirty: false,
        }
    }

    pub fn props(&self) -> &SimpleMeshLayerProps {
        &self.props
    }

    /// Replace the props. The model is rebuilt when the mesh or the pipeline state changed,
    /// the attributes when accessors or data changed.
    pub fn set_props(&mut self, props: SimpleMeshLayerProps) {
        if self.props == props {
            return;
        }
        if self.props.base.needs_new_model(&props.base)
            || !same_mesh(&self.props.mesh, &props.mesh)
            || self.props.wireframe != props.wireframe
        {
            self.models_dirty = true;
        }
        if self.props.texture != props.texture {
            self.texture_dirty = true;
        }
        if self.props.base.extensions != props.base.extensions
            || Self::attributes_changed(&self.props, &props)
        {
            self.data_dirty = true;
        }
        self.props = props;
    }

    /// Whether the new props need the attributes rebuilt: everything except the props that
    /// only feed uniforms, the mesh and the texture.
    fn attributes_changed(old: &SimpleMeshLayerProps, new: &SimpleMeshLayerProps) -> bool {
        let mut probe = new.clone();
        probe.base = old.base.clone();
        probe.mesh = old.mesh.clone();
        probe.texture = old.texture.clone();
        probe.size_scale = old.size_scale;
        probe.wireframe = old.wireframe;
        probe.instanced = old.instanced;
        probe != *old
    }

    /// The per instance model matrix columns and translations, as deck.gl's
    /// `MATRIX_ATTRIBUTES` computes them.
    fn instance_matrices(props: &SimpleMeshLayerProps) -> Result<Arc<Vec<[[f32; 3]; 4]>>> {
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
        let model = initialized(self.model.as_mut(), &props.base.id)?;
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
        self.attributes
            .update(&ctx.device, &ctx.queue, model, data, &sources)?;
        model.set_instance_count(data.len() as u32);
        Ok(())
    }

    fn update_texture(&mut self, ctx: &LayerContext) -> Result<()> {
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        if let Some(image) = &props.texture {
            let texture = create_rgba8_texture(
                &ctx.device,
                &ctx.queue,
                &format!("{}-texture", props.base.id),
                image.width,
                image.height,
                &image.rgba,
            );
            model.set_texture("simpleMeshTexture", texture.create_view(&Default::default()))?;
        }
        Ok(())
    }
}

/// Whether two mesh props are the same mesh, by identity first and by value otherwise.
fn same_mesh(a: &Option<Arc<Mesh>>, b: &Option<Arc<Mesh>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
        _ => false,
    }
}

impl Layer for SimpleMeshLayer {
    fn props(&self) -> &LayerProps {
        &self.props.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        self.model = None;
        self.models_dirty = false;
        let props = &self.props;
        let Some(mesh) = &props.mesh else {
            return Ok(());
        };
        mesh.validate().map_err(DeckError::Data)?;
        let modules: Vec<ShaderModuleSource> = STANDARD_MODULES
            .iter()
            .chain(LIGHTING_MODULES.iter())
            .copied()
            .collect();
        let extensions = &props.base.extensions;
        let shader = extensions.assemble(&props.base.id, &modules, SHADER)?;
        self.attributes = mesh_attributes();
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
        let topology = if props.wireframe {
            wgpu::PrimitiveTopology::LineStrip
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };
        let mut desc = ModelDescriptor::new(&props.base.id, &shader, &layouts, topology, ctx.target);
        ctx.configure(&mut desc, &props.base);
        let mut model = Model::new(&ctx.device, &desc)?;
        let vertices = mesh.vertices();
        model.set_vertex_buffer(
            "geometry",
            create_vertex_buffer_from(&ctx.device, "geometry", &vertices),
        )?;
        model.set_vertex_count(vertices.len() as u32);
        if let Some(indices) = &mesh.indices {
            model.set_index_buffer(
                create_index_buffer(&ctx.device, "indices", indices),
                wgpu::IndexFormat::Uint32,
                indices.len() as u32,
            );
        }
        self.model = Some(model);
        self.data_dirty = true;
        self.texture_dirty = true;
        Ok(())
    }

    fn update(&mut self, ctx: &LayerContext, viewport: &Viewport) -> Result<()> {
        if self.models_dirty {
            self.initialize(ctx)?;
        }
        if self.model.is_none() {
            return Ok(());
        }
        self.attributes.set_time(ctx.time);
        self.attributes.set_transitions(&self.props.base.transitions);
        if self.data_dirty {
            self.update_attributes(ctx)?;
            self.data_dirty = false;
        }
        if self.texture_dirty {
            self.update_texture(ctx)?;
            self.texture_dirty = false;
        }
        if let Some(model) = self.model.as_mut() {
            self.attributes
                .animate(&ctx.device, &ctx.queue, &mut [model], &self.props.data, ctx.time)?;
        }
        let props = &self.props;
        let model = initialized(self.model.as_mut(), &props.base.id)?;
        update_standard_uniforms(model, ctx, viewport, &props.base)?;
        let flat = !props.mesh.as_ref().is_some_and(|m| m.has_normals());
        // deck.gl's `shouldComposeModelMatrix`: only off a map are mesh coordinates composed
        // into world coordinates; on a map they stay meter offsets around the anchor
        let compose = !props.instanced || !viewport.is_geospatial;
        let u = model.uniforms("simpleMesh")?;
        u.set_f32("sizeScale", props.size_scale)?;
        u.set_f32("composeModelMatrix", if compose { 1.0 } else { 0.0 })?;
        u.set_f32("hasTexture", if props.texture.is_some() { 1.0 } else { 0.0 })?;
        u.set_f32("flatShading", if flat { 1.0 } else { 0.0 })?;
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

    fn bounds(&self) -> Option<[f64; 4]> {
        self.attributes.bounds()
    }

    fn in_transition(&self) -> bool {
        self.attributes.in_transition() || self.model.as_ref().is_some_and(Model::in_transition)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_matrix_matches_deck_gl() {
        let identity = transform_matrix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(identity, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        // A yaw of 90 degrees turns x into y
        let yaw = transform_matrix([0.0, 90.0, 0.0], [2.0, 3.0, 4.0]);
        let x = [yaw[0][0], yaw[0][1], yaw[0][2]];
        assert!(
            (x[0]).abs() < 1e-6 && (x[1] - 2.0).abs() < 1e-6 && x[2].abs() < 1e-6,
            "{x:?}"
        );
        assert!((yaw[1][0] + 3.0).abs() < 1e-6, "{:?}", yaw[1]);
        assert!((yaw[2][2] - 4.0).abs() < 1e-6);
        // A pitch of 90 degrees tips x down along z
        let pitch = transform_matrix([90.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((pitch[0][2] + 1.0).abs() < 1e-6, "{:?}", pitch[0]);
    }
}

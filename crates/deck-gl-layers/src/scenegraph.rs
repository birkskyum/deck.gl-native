//! glTF scenes for the [`ScenegraphLayer`](crate::ScenegraphLayer): the default scene is
//! flattened into triangle primitives with their node transforms, base colours and base
//! colour textures. Binary glTF (`.glb`) and JSON glTF with embedded or external buffers
//! and images are read; external URIs are resolved by a caller supplied function.

use std::sync::Arc;

use base64::Engine;
use glam::Mat4;

use crate::mesh::Mesh;
use crate::BitmapImage;

/// One glTF primitive, ready to draw.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenePrimitive {
    pub mesh: Mesh,
    /// The world transform of the primitive's node in the scene
    pub model_matrix: Mat4,
    /// The material's `baseColorFactor`
    pub base_color: [f32; 4],
    /// Index into [`Scenegraph::textures`] of the material's base colour texture
    pub base_color_texture: Option<usize>,
    /// Name of the node the primitive belongs to
    pub node_name: Option<String>,
}

/// A flattened glTF scene.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scenegraph {
    pub primitives: Vec<ScenePrimitive>,
    pub textures: Vec<Arc<BitmapImage>>,
}

impl Scenegraph {
    /// Read a `.glb` or `.gltf` whose buffers and images are in the file or data URIs.
    pub fn from_gltf(bytes: &[u8]) -> Result<Self, String> {
        Self::from_gltf_with(bytes, |uri| {
            Err(format!("external resource `{uri}` cannot be resolved"))
        })
    }

    /// Read a `.glb` or `.gltf`, fetching external buffers and images through `resolve`,
    /// which gets the URI as written in the file (relative to the file's location).
    pub fn from_gltf_with(
        bytes: &[u8],
        resolve: impl Fn(&str) -> Result<Vec<u8>, String>,
    ) -> Result<Self, String> {
        let gltf::Gltf { document, blob } =
            gltf::Gltf::from_slice(bytes).map_err(|e| format!("invalid glTF: {e}"))?;
        let mut blob = blob;
        let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(document.buffers().len());
        for buffer in document.buffers() {
            let data = match buffer.source() {
                gltf::buffer::Source::Bin => blob
                    .take()
                    .ok_or_else(|| "glTF refers to a BIN chunk the file does not have".to_string())?,
                gltf::buffer::Source::Uri(uri) => load_uri(uri, &resolve)?,
            };
            if data.len() < buffer.length() {
                return Err(format!(
                    "glTF buffer {} has {} bytes, expected {}",
                    buffer.index(),
                    data.len(),
                    buffer.length()
                ));
            }
            buffers.push(data);
        }
        let mut textures: Vec<Arc<BitmapImage>> = Vec::with_capacity(document.images().len());
        for image in document.images() {
            let encoded: Vec<u8> = match image.source() {
                gltf::image::Source::View { view, .. } => {
                    let buffer = buffers
                        .get(view.buffer().index())
                        .ok_or_else(|| format!("image {} refers to a missing buffer", image.index()))?;
                    buffer
                        .get(view.offset()..view.offset() + view.length())
                        .ok_or_else(|| format!("image {} lies outside its buffer", image.index()))?
                        .to_vec()
                }
                gltf::image::Source::Uri { uri, .. } => load_uri(uri, &resolve)?,
            };
            let decoded = image::load_from_memory(&encoded)
                .map_err(|e| format!("image {}: {e}", image.index()))?
                .to_rgba8();
            textures.push(Arc::new(BitmapImage {
                width: decoded.width(),
                height: decoded.height(),
                rgba: Arc::new(decoded.into_raw()),
            }));
        }
        let scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .ok_or_else(|| "glTF has no scene".to_string())?;
        let mut primitives = Vec::new();
        for node in scene.nodes() {
            collect_node(&node, Mat4::IDENTITY, &buffers, &mut primitives)?;
        }
        if primitives.is_empty() {
            return Err("glTF scene has no triangle primitives".into());
        }
        Ok(Self { primitives, textures })
    }

    /// `[min, max]` of every primitive's positions, in scene units after node transforms.
    pub fn bounds(&self) -> Option<[[f32; 3]; 2]> {
        let mut bounds: Option<[[f32; 3]; 2]> = None;
        for primitive in &self.primitives {
            for p in &primitive.mesh.positions {
                let p = primitive.model_matrix.transform_point3(glam::Vec3::from(*p));
                let p = [p.x, p.y, p.z];
                bounds = Some(match bounds {
                    None => [p, p],
                    Some([min, max]) => [
                        [min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])],
                        [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])],
                    ],
                });
            }
        }
        bounds
    }
}

/// Bytes of a URI: a base64 `data:` URI decoded here, anything else through `resolve`.
fn load_uri(uri: &str, resolve: &impl Fn(&str) -> Result<Vec<u8>, String>) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, payload) = rest
            .split_once(";base64,")
            .ok_or_else(|| "data URI without base64 payload".to_string())?;
        return base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|e| format!("invalid base64 data URI: {e}"));
    }
    resolve(uri)
}

fn collect_node(
    node: &gltf::Node<'_>,
    parent: Mat4,
    buffers: &[Vec<u8>],
    out: &mut Vec<ScenePrimitive>,
) -> Result<(), String> {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let mut geometry = Mesh::new(positions.collect());
            geometry.normals = reader.read_normals().map(Iterator::collect);
            geometry.tex_coords = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            geometry.colors = reader.read_colors(0).map(|c| c.into_rgb_f32().collect());
            geometry.indices = reader.read_indices().map(|i| i.into_u32().collect());
            geometry
                .validate()
                .map_err(|e| format!("mesh {} primitive {}: {e}", mesh.index(), primitive.index()))?;
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            out.push(ScenePrimitive {
                mesh: geometry,
                model_matrix: world,
                base_color: pbr.base_color_factor(),
                base_color_texture: pbr
                    .base_color_texture()
                    .map(|info| info.texture().source().index()),
                node_name: node.name().map(str::to_string),
            });
        }
    }
    for child in node.children() {
        collect_node(&child, world, buffers, out)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A binary glTF with one triangle (positions and normals) in a node scaled by
    /// `scale`, coloured by `base_color`, with an optional 2x1 red and green texture.
    pub(crate) fn triangle_glb(scale: f32, base_color: [f32; 4], textured: bool) -> Vec<u8> {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals: [[f32; 3]; 3] = [[0.0, 0.0, 1.0]; 3];
        let tex_coords: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let indices: [u16; 3] = [0, 1, 2];
        let mut bin: Vec<u8> = Vec::new();
        bin.extend(bytemuck::cast_slice(&positions));
        bin.extend(bytemuck::cast_slice(&normals));
        bin.extend(bytemuck::cast_slice(&tex_coords));
        bin.extend(bytemuck::cast_slice(&indices));
        bin.extend([0u8; 2]);
        let mut png = Vec::new();
        if textured {
            let image = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]).unwrap();
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
        }
        let image_offset = bin.len();
        bin.extend(&png);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let texture_json = if textured {
            format!(
                r#","images":[{{"bufferView":4,"mimeType":"image/png"}}],"textures":[{{"source":0}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[{},{},{},{}],"baseColorTexture":{{"index":0}}}}}}]"#,
                base_color[0], base_color[1], base_color[2], base_color[3]
            )
        } else {
            format!(
                r#","materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[{},{},{},{}]}}}}]"#,
                base_color[0], base_color[1], base_color[2], base_color[3]
            )
        };
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"name":"root","scale":[{scale},{scale},{scale}],"children":[1]}},{{"name":"tri","mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":0}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"}},{{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":36}},{{"buffer":0,"byteOffset":72,"byteLength":24}},{{"buffer":0,"byteOffset":96,"byteLength":6}},{{"buffer":0,"byteOffset":{image_offset},"byteLength":{}}}],"buffers":[{{"byteLength":{}}}]{texture_json}}}"#,
            png.len(),
            bin.len()
        );
        let mut json = json.into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut glb = Vec::new();
        glb.extend(b"glTF");
        glb.extend(2u32.to_le_bytes());
        glb.extend(((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
        glb.extend((json.len() as u32).to_le_bytes());
        glb.extend(b"JSON");
        glb.extend(&json);
        glb.extend((bin.len() as u32).to_le_bytes());
        glb.extend(b"BIN\0");
        glb.extend(&bin);
        glb
    }

    #[test]
    fn reads_a_binary_gltf_with_transforms_materials_and_textures() {
        let scene = Scenegraph::from_gltf(&triangle_glb(2.0, [0.5, 0.25, 1.0, 1.0], true)).unwrap();
        assert_eq!(scene.primitives.len(), 1);
        let primitive = &scene.primitives[0];
        assert_eq!(
            primitive.mesh.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert!(primitive.mesh.has_normals());
        assert_eq!(primitive.mesh.tex_coords.as_ref().unwrap()[2], [0.0, 1.0]);
        assert_eq!(primitive.mesh.indices.as_ref().unwrap(), &[0, 1, 2]);
        assert_eq!(primitive.model_matrix, Mat4::from_scale(glam::Vec3::splat(2.0)));
        assert_eq!(primitive.base_color, [0.5, 0.25, 1.0, 1.0]);
        assert_eq!(primitive.base_color_texture, Some(0));
        assert_eq!(primitive.node_name.as_deref(), Some("tri"));
        assert_eq!(scene.textures.len(), 1);
        assert_eq!((scene.textures[0].width, scene.textures[0].height), (2, 1));
        assert_eq!(&scene.textures[0].rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(scene.bounds(), Some([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]));
        let plain = Scenegraph::from_gltf(&triangle_glb(1.0, [1.0; 4], false)).unwrap();
        assert!(plain.textures.is_empty());
        assert_eq!(plain.primitives[0].base_color_texture, None);
        assert!(Scenegraph::from_gltf(b"not gltf").is_err());
    }

    #[test]
    fn resolves_external_and_data_uris() {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let bin: Vec<u8> = bytemuck::cast_slice(&positions).to_vec();
        let json = |uri: &str| {
            format!(
                r#"{{"asset":{{"version":"2.0"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}}],"bufferViews":[{{"buffer":0,"byteLength":36}}],"buffers":[{{"byteLength":36,"uri":"{uri}"}}]}}"#
            )
        };
        let external = json("model.bin");
        let scene = Scenegraph::from_gltf_with(external.as_bytes(), |uri| {
            assert_eq!(uri, "model.bin");
            Ok(bin.clone())
        })
        .unwrap();
        assert_eq!(scene.primitives[0].mesh.vertex_count(), 3);
        assert!(scene.primitives[0].mesh.indices.is_none());
        assert!(Scenegraph::from_gltf(external.as_bytes()).is_err(), "no resolver");
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bin);
        let embedded = json(&format!("data:application/octet-stream;base64,{encoded}"));
        let scene = Scenegraph::from_gltf(embedded.as_bytes()).unwrap();
        assert_eq!(scene.primitives[0].mesh.positions[1], [1.0, 0.0, 0.0]);
    }
}

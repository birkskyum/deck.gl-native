//! Triangle meshes for the mesh layers: deck.gl's mesh attributes (`positions`, `normals`,
//! `colors`, `texCoords` and `indices`), a Wavefront OBJ reader for the common case of a
//! model file, and a unit cube.

use std::collections::HashMap;

/// A triangle mesh. Attributes other than positions are optional, as in deck.gl: missing
/// colours are white, missing texture coordinates zero, and a mesh without normals is flat
/// shaded from screen space derivatives.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    /// Vertex colours in 0..1, multiplied with the instance colour
    pub colors: Option<Vec<[f32; 3]>>,
    pub tex_coords: Option<Vec<[f32; 2]>>,
    /// Triangle list indices; without them every three positions form a triangle
    pub indices: Option<Vec<u32>>,
}

/// One vertex as the mesh layer's geometry buffer holds it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub tex_coord: [f32; 2],
}

impl Mesh {
    pub fn new(positions: Vec<[f32; 3]>) -> Self {
        Self {
            positions,
            ..Default::default()
        }
    }

    pub fn with_normals(mut self, normals: Vec<[f32; 3]>) -> Self {
        self.normals = Some(normals);
        self
    }

    pub fn with_colors(mut self, colors: Vec<[f32; 3]>) -> Self {
        self.colors = Some(colors);
        self
    }

    pub fn with_tex_coords(mut self, tex_coords: Vec<[f32; 2]>) -> Self {
        self.tex_coords = Some(tex_coords);
        self
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Self {
        self.indices = Some(indices);
        self
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Whether the mesh has normals for smooth shading.
    pub fn has_normals(&self) -> bool {
        self.normals
            .as_ref()
            .is_some_and(|n| n.len() == self.positions.len())
    }

    /// Whether every attribute and index is consistent with the positions.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.positions.len();
        if n == 0 {
            return Err("mesh has no positions".into());
        }
        let check = |name: &str, len: Option<usize>| match len {
            Some(len) if len != n => Err(format!("mesh has {n} positions but {len} {name}")),
            _ => Ok(()),
        };
        check("normals", self.normals.as_ref().map(Vec::len))?;
        check("colors", self.colors.as_ref().map(Vec::len))?;
        check("texCoords", self.tex_coords.as_ref().map(Vec::len))?;
        if let Some(indices) = &self.indices {
            if let Some(bad) = indices.iter().find(|i| **i as usize >= n) {
                return Err(format!("mesh index {bad} is out of range for {n} vertices"));
            }
        }
        Ok(())
    }

    /// `[min, max]` of the positions.
    pub fn bounds(&self) -> Option<[[f32; 3]; 2]> {
        let mut iter = self.positions.iter();
        let first = *iter.next()?;
        Some(iter.fold([first, first], |[min, max], p| {
            [
                [min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])],
                [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])],
            ]
        }))
    }

    /// The interleaved vertices of the geometry buffer.
    pub fn vertices(&self) -> Vec<MeshVertex> {
        (0..self.positions.len())
            .map(|i| MeshVertex {
                position: self.positions[i],
                normal: self.normals.as_ref().map_or([0.0; 3], |n| n[i]),
                color: self.colors.as_ref().map_or([1.0; 3], |c| c[i]),
                tex_coord: self.tex_coords.as_ref().map_or([0.0; 2], |t| t[i]),
            })
            .collect()
    }

    /// A cube of side 1 centred at the origin, with face normals and one texture square per
    /// face.
    pub fn cube() -> Self {
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            // normal, tangent (u), bitangent (v)
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let mut positions = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut tex_coords = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for (normal, u, v) in faces {
            let base = positions.len() as u32;
            for (su, sv, tu, tv) in [
                (-1.0, -1.0, 0.0, 1.0),
                (1.0, -1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0, 0.0),
            ] {
                positions.push([
                    0.5 * (normal[0] + su * u[0] + sv * v[0]),
                    0.5 * (normal[1] + su * u[1] + sv * v[1]),
                    0.5 * (normal[2] + su * u[2] + sv * v[2]),
                ]);
                normals.push(normal);
                tex_coords.push([tu, tv]);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self {
            positions,
            normals: Some(normals),
            colors: None,
            tex_coords: Some(tex_coords),
            indices: Some(indices),
        }
    }

    /// Read a Wavefront OBJ file: `v`, `vn`, `vt` and `f` records, with polygons split into
    /// triangle fans. Each distinct position, texture coordinate and normal combination
    /// becomes one vertex. Materials, groups and other records are ignored.
    pub fn from_obj(text: &str) -> Result<Self, String> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut tex_coords: Vec<[f32; 2]> = Vec::new();
        let mut vertices: Vec<(usize, Option<usize>, Option<usize>)> = Vec::new();
        let mut lookup: HashMap<(usize, Option<usize>, Option<usize>), u32> = HashMap::new();
        let mut indices: Vec<u32> = Vec::new();
        let number = |s: &str, line: usize| {
            s.parse::<f32>()
                .map_err(|_| format!("line {line}: `{s}` is not a number"))
        };
        for (i, line) in text.lines().enumerate() {
            let line_no = i + 1;
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let mut xyz = [0.0f32; 3];
                    for value in &mut xyz {
                        *value = number(
                            parts.next().ok_or(format!("line {line_no}: `v` needs x y z"))?,
                            line_no,
                        )?;
                    }
                    positions.push(xyz);
                }
                Some("vn") => {
                    let mut xyz = [0.0f32; 3];
                    for value in &mut xyz {
                        *value = number(
                            parts.next().ok_or(format!("line {line_no}: `vn` needs x y z"))?,
                            line_no,
                        )?;
                    }
                    normals.push(xyz);
                }
                Some("vt") => {
                    let mut uv = [0.0f32; 2];
                    for value in &mut uv {
                        *value = number(
                            parts.next().ok_or(format!("line {line_no}: `vt` needs u v"))?,
                            line_no,
                        )?;
                    }
                    tex_coords.push(uv);
                }
                Some("f") => {
                    let mut corners: Vec<u32> = Vec::new();
                    for corner in parts {
                        let mut fields = corner.split('/');
                        let resolve = |field: Option<&str>, count: usize| -> Result<Option<usize>, String> {
                            match field {
                                None | Some("") => Ok(None),
                                Some(s) => {
                                    let index: i64 = s
                                        .parse()
                                        .map_err(|_| format!("line {line_no}: `{s}` is not an index"))?;
                                    let resolved = if index < 0 {
                                        count as i64 + index
                                    } else {
                                        index - 1
                                    };
                                    if resolved < 0 || resolved as usize >= count {
                                        return Err(format!("line {line_no}: index {index} is out of range"));
                                    }
                                    Ok(Some(resolved as usize))
                                }
                            }
                        };
                        let position = resolve(fields.next(), positions.len())?
                            .ok_or(format!("line {line_no}: face corner without a position"))?;
                        let tex_coord = resolve(fields.next(), tex_coords.len())?;
                        let normal = resolve(fields.next(), normals.len())?;
                        let key = (position, tex_coord, normal);
                        let index = *lookup.entry(key).or_insert_with(|| {
                            vertices.push(key);
                            (vertices.len() - 1) as u32
                        });
                        corners.push(index);
                    }
                    if corners.len() < 3 {
                        return Err(format!("line {line_no}: a face needs at least three corners"));
                    }
                    for k in 1..corners.len() - 1 {
                        indices.extend_from_slice(&[corners[0], corners[k], corners[k + 1]]);
                    }
                }
                _ => {}
            }
        }
        if vertices.is_empty() {
            return Err("OBJ has no faces".into());
        }
        let has_normals = vertices.iter().all(|(_, _, n)| n.is_some());
        let has_tex_coords = vertices.iter().all(|(_, t, _)| t.is_some());
        Ok(Self {
            positions: vertices.iter().map(|(p, _, _)| positions[*p]).collect(),
            normals: has_normals.then(|| {
                vertices
                    .iter()
                    .map(|(_, _, n)| n.map_or([0.0; 3], |n| normals[n]))
                    .collect()
            }),
            colors: None,
            tex_coords: has_tex_coords.then(|| {
                vertices
                    .iter()
                    .map(|(_, t, _)| t.map_or([0.0; 2], |t| tex_coords[t]))
                    .collect()
            }),
            indices: Some(indices),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_is_closed_and_unit_sized() {
        let cube = Mesh::cube();
        cube.validate().unwrap();
        assert_eq!(cube.vertex_count(), 24);
        assert_eq!(cube.indices.as_ref().unwrap().len(), 36);
        assert_eq!(cube.bounds(), Some([[-0.5; 3], [0.5; 3]]));
        assert!(cube.has_normals());
        // Every normal points away from the centre
        for (p, n) in cube.positions.iter().zip(cube.normals.as_ref().unwrap()) {
            assert!(p[0] * n[0] + p[1] * n[1] + p[2] * n[2] > 0.0);
        }
    }

    #[test]
    fn reads_obj_faces_with_shared_vertices_and_quads() {
        let obj = "# a square\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\ng other\nf -4//1 -2//1 -1//1\n";
        let mesh = Mesh::from_obj(obj).unwrap();
        mesh.validate().unwrap();
        // Four corners with texture coordinates plus three without: distinct vertices
        assert_eq!(mesh.vertex_count(), 7);
        assert_eq!(mesh.indices.as_ref().unwrap(), &[0, 1, 2, 0, 2, 3, 4, 5, 6]);
        assert!(mesh.has_normals());
        assert!(
            mesh.tex_coords.is_none(),
            "not every corner had texture coordinates"
        );
        assert_eq!(mesh.positions[4], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[5], [1.0, 1.0, 0.0]);
        assert!(
            Mesh::from_obj("v 0 0 0\nf 1 2 3\n").is_err(),
            "index out of range"
        );
        assert!(Mesh::from_obj("v 0 0 0\n").is_err(), "no faces");
    }
}

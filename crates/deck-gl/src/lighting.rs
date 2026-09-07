//! Port of deck.gl's default `LightingEffect` and luma.gl's `lighting` uniform packing.

use glam::Vec3;
use luma_gl::UniformBlock;

use crate::Result;

/// Ambient light contribution shared across the entire scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientLight {
    /// RGB in 0..255
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Directional light defined by its incoming world space direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub direction: [f32; 3],
}

/// Omnidirectional point light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    /// Position in the common space of the viewport
    pub position: [f32; 3],
    /// Constant, linear and quadratic attenuation
    pub attenuation: [f32; 3],
}

/// Light sources for lit layers. The default matches deck.gl's default lighting effect.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingEffect {
    pub ambient: AmbientLight,
    pub directional: Vec<DirectionalLight>,
    pub point: Vec<PointLight>,
}

impl Default for LightingEffect {
    fn default() -> Self {
        Self {
            ambient: AmbientLight {
                color: [255.0, 255.0, 255.0],
                intensity: 1.0,
            },
            directional: vec![
                DirectionalLight {
                    color: [255.0, 255.0, 255.0],
                    intensity: 1.0,
                    direction: [-1.0, 3.0, -1.0],
                },
                DirectionalLight {
                    color: [255.0, 255.0, 255.0],
                    intensity: 0.9,
                    direction: [1.0, -8.0, -2.5],
                },
            ],
            point: Vec::new(),
        }
    }
}

const MAX_LIGHTS: usize = 5;
/// Size of one `UniformLight` in the WGSL uniform layout.
const LIGHT_STRIDE: usize = 80;

fn convert_color(color: [f32; 3], intensity: f32) -> Vec3 {
    Vec3::from(color) / 255.0 * intensity
}

fn write_vec3(buf: &mut [u8], offset: usize, v: Vec3) {
    buf[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&v.to_array()));
}

impl LightingEffect {
    /// Write the `lighting` uniform block (luma.gl `lightingUniforms`).
    pub fn write(&self, block: &mut UniformBlock) -> Result<()> {
        let mut lights = vec![0u8; LIGHT_STRIDE * MAX_LIGHTS];
        let mut current = 0usize;
        let mut point_count = 0i32;
        let mut directional_count = 0i32;

        for light in &self.point {
            if current >= MAX_LIGHTS {
                break;
            }
            let base = current * LIGHT_STRIDE;
            write_vec3(&mut lights, base, convert_color(light.color, light.intensity));
            write_vec3(&mut lights, base + 16, Vec3::from(light.position));
            write_vec3(&mut lights, base + 48, Vec3::from(light.attenuation));
            current += 1;
            point_count += 1;
        }
        for light in &self.directional {
            if current >= MAX_LIGHTS {
                break;
            }
            let base = current * LIGHT_STRIDE;
            write_vec3(&mut lights, base, convert_color(light.color, light.intensity));
            write_vec3(&mut lights, base + 32, Vec3::from(light.direction));
            current += 1;
            directional_count += 1;
        }

        block.set_i32("enabled", 1)?;
        block.set_i32("directionalLightCount", directional_count)?;
        block.set_i32("pointLightCount", point_count)?;
        block.set_i32("spotLightCount", 0)?;
        block.set_vec3(
            "ambientColor",
            convert_color(self.ambient.color, self.ambient.intensity),
        )?;
        block.set_bytes("lights", &lights)?;
        Ok(())
    }

    /// Write luma.gl's `gouraudMaterial` defaults.
    pub fn write_default_material(block: &mut UniformBlock) -> Result<()> {
        Material::default().write(block)
    }
}

/// Surface reflectance of a lit layer, deck.gl's `material` prop (luma.gl's Gouraud
/// material). Defaults match deck.gl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Draw the raw colours without lighting
    pub unlit: bool,
    pub ambient: f32,
    pub diffuse: f32,
    pub shininess: f32,
    /// 0 to 255 per channel
    pub specular_color: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            unlit: false,
            ambient: 0.35,
            diffuse: 0.6,
            shininess: 32.0,
            specular_color: [38.25, 38.25, 38.25],
        }
    }
}

impl Material {
    /// deck.gl's `material: false`: colours as given.
    pub fn unlit() -> Self {
        Self {
            unlit: true,
            ..Default::default()
        }
    }

    pub fn write(&self, block: &mut UniformBlock) -> Result<()> {
        block.set_u32("unlit", self.unlit as u32)?;
        block.set_f32("ambient", self.ambient)?;
        block.set_f32("diffuse", self.diffuse)?;
        block.set_f32("shininess", self.shininess)?;
        block.set_vec3("specularColor", Vec3::from(self.specular_color))?;
        Ok(())
    }
}

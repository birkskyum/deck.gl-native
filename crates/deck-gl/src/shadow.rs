//! Port of deck.gl's shadow pass and `shadow` shader module: directional lights with
//! `shadow` set render every layer into a shadow map from the light's point of view, and the
//! layers darken the fragments the maps say are behind something.
//!
//! The deck adds the module to every layer's shader through the default extension shaders
//! (see [`crate::extension::set_default_shaders`]) while a light casts shadows, so layers
//! need nothing of their own. [`write_shadow_uniforms`] fills the module's uniforms from the
//! layer context.

use std::sync::Arc;

use glam::{DMat4, DVec2, DVec3, DVec4, Mat4, Vec4};
use luma_gl::device::create_render_texture;
use luma_gl::{
    Model, ShaderField, ShaderInjection, ShaderModuleSource, SHADOW_DEPTH_FORMAT, SHADOW_MAP_FORMAT,
};
use math_gl::web_mercator::pixels_to_world;

use crate::extension::ExtensionShaders;
use crate::layer::LayerContext;
use crate::lighting::LightingEffect;
use crate::shaderlib::project::ProjectUniforms;
use crate::viewport::Viewport;
use crate::Result;

/// The `shadow` shader module.
pub const SHADOW: ShaderModuleSource = ShaderModuleSource {
    name: "shadow",
    source: include_str!("shaderlib/wgsl/shadow.wgsl"),
};

/// Shadow maps per light and how the lights see the scene.
pub const MAX_SHADOW_LIGHTS: usize = 2;

/// The largest side of a shadow map.
pub const MAX_SHADOW_MAP_SIZE: u32 = 2048;

/// The shader code the deck adds to every layer while shadows are on: the module, the light
/// space varyings and the hook injections deck.gl's module declares.
pub fn shadow_shaders() -> ExtensionShaders {
    ExtensionShaders {
        modules: vec![SHADOW],
        injections: vec![
            ShaderInjection::new(
                "vs:DECKGL_FILTER_GL_POSITION",
                "position = shadow_setVertexPosition(geometry.position, position);",
            ),
            ShaderInjection::new(
                "fs:DECKGL_FILTER_COLOR",
                "color = shadow_filterShadowColor(color);",
            ),
        ],
        attributes: Vec::new(),
        varyings: vec![
            ShaderField {
                name: "shadow_vPosition0",
                ty: "vec3<f32>",
            },
            ShaderField {
                name: "shadow_vPosition1",
                ty: "vec3<f32>",
            },
            ShaderField {
                name: "shadow_vDepth",
                ty: "f32",
            },
        ],
    }
}

/// The colour and depth attachments of one light's shadow map.
pub struct ShadowTarget {
    pub color: wgpu::Texture,
    pub depth: wgpu::Texture,
}

impl ShadowTarget {
    pub fn new(device: &wgpu::Device, light: usize, width: u32, height: u32) -> Self {
        Self {
            color: create_render_texture(
                device,
                &format!("shadow map {light}"),
                width,
                height,
                SHADOW_MAP_FORMAT,
            ),
            depth: create_render_texture(
                device,
                &format!("shadow depth {light}"),
                width,
                height,
                SHADOW_DEPTH_FORMAT,
            ),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.color.width(), self.color.height())
    }
}

/// What the layers need to draw shadow maps and to sample them: one view projection matrix
/// per light, relative to the viewport centre, the maps themselves and the shadow colour.
#[derive(Clone, Debug)]
pub struct ShadowState {
    /// Per light: the light's orthographic view projection of common space positions
    /// relative to the viewport centre
    pub light_matrices: Vec<DMat4>,
    /// The viewport centre in common space the matrices are relative to
    pub viewport_center: DVec3,
    /// The shadow map of each light, once drawn
    pub maps: Vec<wgpu::TextureView>,
    /// Bound where no map exists
    pub dummy: wgpu::TextureView,
    /// RGBA in 0..1
    pub color: [f32; 4],
}

/// Whether the lighting asks for shadows.
pub fn shadows_enabled(lighting: &LightingEffect) -> bool {
    lighting.directional.iter().any(|light| light.shadow)
}

/// deck.gl's `_calculateMatrices` and `getViewProjectionMatrices`: for every light casting
/// shadows, a view matrix looking along the light and an orthographic projection that fits
/// the view frustum, together as one matrix of common space positions relative to the
/// viewport centre.
pub fn light_matrices(lighting: &LightingEffect, viewport: &Viewport) -> Vec<DMat4> {
    let corners = frustum_corners(viewport);
    lighting
        .directional
        .iter()
        .filter(|light| light.shadow)
        .take(MAX_SHADOW_LIGHTS)
        .map(|light| {
            let direction = DVec3::from(light.direction.map(f64::from));
            let eye = -direction;
            let up = if direction.x.abs() < 1e-9 && direction.z.abs() < 1e-9 {
                DVec3::Z
            } else {
                DVec3::Y
            };
            let view = look_at(eye, DVec3::ZERO, up);
            let mut min = DVec3::splat(f64::INFINITY);
            let mut max = DVec3::splat(f64::NEG_INFINITY);
            for corner in &corners {
                let p = view.transform_point3(*corner - viewport.center);
                min = min.min(p);
                max = max.max(p);
            }
            let projection = ortho_gl(min.x, max.x, min.y, max.y, -max.z, -min.z);
            projection * view
        })
        .collect()
}

/// gl-matrix's `lookAt`, as the viewport uses it: a right handed view matrix.
fn look_at(eye: DVec3, center: DVec3, up: DVec3) -> DMat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    DMat4::from_cols(
        DVec4::new(s.x, u.x, -f.x, 0.0),
        DVec4::new(s.y, u.y, -f.y, 0.0),
        DVec4::new(s.z, u.z, -f.z, 0.0),
        DVec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// gl-matrix's `ortho`: an orthographic projection with OpenGL clip space depth.
fn ortho_gl(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> DMat4 {
    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);
    DMat4::from_cols(
        DVec4::new(-2.0 * lr, 0.0, 0.0, 0.0),
        DVec4::new(0.0, -2.0 * bt, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 2.0 * nf, 0.0),
        DVec4::new((left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1.0),
    )
}

/// The eight corners of the view frustum in common space: the ground plane (or the far
/// plane off a map) and the near plane.
fn frustum_corners(viewport: &Viewport) -> [DVec3; 8] {
    let (w, h) = (viewport.width, viewport.height);
    let pixels = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
    let far = if viewport.is_geospatial { None } else { Some(1.0) };
    let mut corners = [DVec3::ZERO; 8];
    for (i, (x, y)) in pixels.iter().enumerate() {
        corners[i] = pixels_to_world(DVec2::new(*x, *y), far, &viewport.pixel_unprojection_matrix, 0.0);
        corners[i + 4] = pixels_to_world(
            DVec2::new(*x, *y),
            Some(-1.0),
            &viewport.pixel_unprojection_matrix,
            0.0,
        );
    }
    corners
}

/// The size of the shadow maps for a deck of `width` x `height` logical pixels.
pub fn shadow_map_size(width: u32, height: u32, device_pixel_ratio: f32) -> (u32, u32) {
    let scale = |v: u32| ((v as f32 * device_pixel_ratio).round() as u32).clamp(1, MAX_SHADOW_MAP_SIZE);
    (scale(width), scale(height))
}

/// Fill a model's `shadow` uniforms and bind its shadow maps from the layer context, for the
/// layer whose `project` uniforms are `project`. Does nothing for models without the module.
pub fn write_shadow_uniforms(
    model: &mut Model,
    ctx: &LayerContext,
    viewport: &Viewport,
    project: &ProjectUniforms,
) -> Result<()> {
    if !model.has_uniforms("shadow") {
        model.set_shadow_mode(false);
        return Ok(());
    }
    let drawing = ctx.shadow_pass.is_some();
    model.set_shadow_mode(drawing);
    let Some(state) = ctx.shadow.as_ref() else {
        let block = model.uniforms("shadow")?;
        block.set_f32("drawShadowMap", 0.0)?;
        block.set_f32("useShadowMap", 0.0)?;
        return Ok(());
    };
    // deck.gl's `createShadowUniforms`: with positions relative to the layer's origin
    // (auto offset), the matrix acts on vectors and the origin's light space position is
    // added; with absolute positions the matrix takes them relative to the viewport centre.
    let center = project.center;
    let absolute = center == Vec4::ZERO;
    let origin = if absolute {
        DVec3::ZERO
    } else {
        let clip = DVec4::new(center.x as f64, center.y as f64, center.z as f64, center.w as f64);
        let common = viewport.view_projection_matrix.inverse() * clip;
        common.truncate() / common.w
    };
    let block = model.uniforms("shadow")?;
    for (i, matrix) in state.light_matrices.iter().enumerate().take(MAX_SHADOW_LIGHTS) {
        let (view_projection, project_center) = if absolute {
            (
                *matrix * DMat4::from_translation(-state.viewport_center),
                DVec4::ZERO,
            )
        } else {
            let vector_to_point = DMat4::from_cols(DVec4::X, DVec4::Y, DVec4::Z, DVec4::ZERO);
            (
                *matrix * vector_to_point,
                *matrix * (origin - state.viewport_center).extend(1.0),
            )
        };
        block.set_mat4(
            &format!("viewProjectionMatrix{i}"),
            Mat4::from_cols_array(&view_projection.to_cols_array().map(|v| v as f32)),
        )?;
        block.set_vec4(&format!("projectCenter{i}"), project_center.as_vec4())?;
    }
    block.set_vec4("color", Vec4::from(state.color))?;
    block.set_f32("drawShadowMap", if drawing { 1.0 } else { 0.0 })?;
    block.set_f32(
        "useShadowMap",
        if !drawing && !state.maps.is_empty() {
            1.0
        } else {
            0.0
        },
    )?;
    block.set_f32("lightCount", state.light_matrices.len() as f32)?;
    block.set_i32("lightId", ctx.shadow_pass.unwrap_or(0) as i32)?;
    for i in 0..MAX_SHADOW_LIGHTS {
        let view = state.maps.get(i).unwrap_or(&state.dummy).clone();
        model.set_texture(&format!("shadow_uShadowMap{i}"), view)?;
    }
    Ok(())
}

/// A shared, empty shadow state: no maps, nothing drawn.
pub fn empty_state(dummy: wgpu::TextureView) -> Arc<ShadowState> {
    Arc::new(ShadowState {
        light_matrices: Vec::new(),
        viewport_center: DVec3::ZERO,
        maps: Vec::new(),
        dummy,
        color: [0.0, 0.0, 0.0, 1.0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::DirectionalLight;
    use crate::viewport::WebMercatorViewportOptions;

    #[test]
    fn light_matrices_fit_the_view_frustum() {
        let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 200.0,
            height: 100.0,
            longitude: 10.0,
            latitude: 20.0,
            zoom: 12.0,
            pitch: 30.0,
            bearing: 0.0,
            ..Default::default()
        });
        let mut lighting = LightingEffect {
            directional: vec![
                DirectionalLight {
                    color: [255.0; 3],
                    intensity: 1.0,
                    direction: [1.0, 1.0, -2.0],
                    shadow: true,
                },
                DirectionalLight {
                    color: [255.0; 3],
                    intensity: 1.0,
                    direction: [0.0, 0.0, -1.0],
                    shadow: false,
                },
            ],
            ..Default::default()
        };
        let matrices = light_matrices(&lighting, &viewport);
        assert_eq!(matrices.len(), 1, "only lights with shadow get a matrix");
        // Every frustum corner projects inside the light's clip box
        for corner in frustum_corners(&viewport) {
            let clip = matrices[0] * (corner - viewport.center).extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(ndc.abs().max_element() <= 1.0 + 1e-6, "{ndc:?}");
        }
        // A straight down light still gets a view (its up vector is not degenerate)
        lighting.directional[1].shadow = true;
        assert_eq!(light_matrices(&lighting, &viewport).len(), 2);
        assert!(!shadows_enabled(&LightingEffect::default()));
        assert!(shadows_enabled(&lighting));
    }
}

//! Port of the height map half of `@deck.gl/extensions/src/terrain`: layers whose operation
//! is `terrain` draw their ground elevation into a map covering the view, and layers with the
//! terrain extension read it to sit on that ground.
//!
//! deck.gl's other terrain mode, draping a layer's pixels onto the terrain through a cover
//! texture, is not here yet.

use std::sync::Arc;

use glam::Vec4;
use luma_gl::device::create_render_texture;
use luma_gl::{Model, RenderTarget, ShaderField, ShaderInjection, ShaderModuleSource};

use crate::extension::ExtensionShaders;
use crate::layer::LayerContext;
use crate::viewport::Viewport;
use crate::Result;

/// The `terrain` shader module.
pub const TERRAIN: ShaderModuleSource = ShaderModuleSource {
    name: "terrain",
    source: include_str!("shaderlib/wgsl/terrain.wgsl"),
};

/// Side of the height map in pixels.
pub const HEIGHT_MAP_SIZE: u32 = 1024;

/// Format of the height map: metres in the red channel, filterable so the ground reads
/// smoothly between texels.
pub const HEIGHT_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Attachments of the height map pass.
pub const HEIGHT_MAP_TARGET: RenderTarget = RenderTarget {
    color_format: HEIGHT_MAP_FORMAT,
    depth_format: None,
    sample_count: 1,
};

/// What every layer needs to take part: the module and the hooks it injects.
pub fn terrain_shaders() -> ExtensionShaders {
    ExtensionShaders {
        modules: vec![TERRAIN],
        injections: vec![
            ShaderInjection::new(
                "vs:DECKGL_FILTER_GL_POSITION",
                "position = terrain_setVertexPosition(position);",
            ),
            ShaderInjection::new("fs:DECKGL_FILTER_COLOR", "color = terrain_filterColor(color);"),
        ],
        attributes: Vec::new(),
        varyings: vec![ShaderField {
            name: "terrain_height",
            ty: "f32",
        }],
    }
}

/// The ground of the current frame: a map of elevations in metres over a part of the world.
#[derive(Clone, Debug)]
pub struct TerrainMap {
    /// The map, once it has been drawn
    pub view: Option<wgpu::TextureView>,
    /// Bound where there is none
    pub dummy: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Origin and size of the map in common space: `[x, y, width, height]`
    pub bounds: [f32; 4],
}

impl TerrainMap {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let dummy = luma_gl::create_rgba8_texture(device, queue, "terrain fallback", 1, 1, &[0; 4]);
        Self {
            view: None,
            dummy: dummy.create_view(&Default::default()),
            sampler: luma_gl::default_sampler(device),
            bounds: [0.0; 4],
        }
    }

    /// The map to bind, the dummy when there is none.
    pub fn texture(&self) -> &wgpu::TextureView {
        self.view.as_ref().unwrap_or(&self.dummy)
    }
}

/// A height map texture of the standard size.
pub fn create_height_map(device: &wgpu::Device) -> wgpu::Texture {
    create_render_texture(
        device,
        "terrain height map",
        HEIGHT_MAP_SIZE,
        HEIGHT_MAP_SIZE,
        HEIGHT_MAP_FORMAT,
    )
}

/// The part of the world the map covers, in common space: the viewport's bounds with a
/// margin, so geometry a little outside the view still finds ground under it.
pub fn height_map_bounds(viewport: &Viewport) -> [f32; 4] {
    let bounds = viewport.get_bounds(0.0);
    let bottom_left = viewport.project_position(glam::DVec3::new(bounds[0], bounds[1], 0.0));
    let top_right = viewport.project_position(glam::DVec3::new(bounds[2], bounds[3], 0.0));
    let (min_x, max_x) = (bottom_left.x.min(top_right.x), bottom_left.x.max(top_right.x));
    let (min_y, max_y) = (bottom_left.y.min(top_right.y), bottom_left.y.max(top_right.y));
    let (width, height) = (
        (max_x - min_x).max(f64::EPSILON),
        (max_y - min_y).max(f64::EPSILON),
    );
    // An eighth of the view on every side
    let margin = 0.125;
    [
        (min_x - width * margin) as f32,
        (min_y - height * margin) as f32,
        (width * (1.0 + 2.0 * margin)) as f32,
        (height * (1.0 + 2.0 * margin)) as f32,
    ]
}

/// What a model does with the terrain module this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMode {
    /// Draw as usual
    None,
    /// Draw the ground elevation into the map
    WriteHeightMap,
    /// Sit on the ground the map holds
    UseHeightMap,
}

impl TerrainMode {
    fn shader_value(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::WriteHeightMap => 1.0,
            Self::UseHeightMap => 2.0,
        }
    }
}

/// Give a model the terrain uniforms for `mode`. Models without the module are left alone,
/// so this is safe to call for every layer.
pub fn write_terrain_uniforms(model: &mut Model, ctx: &LayerContext, mode: TerrainMode) -> Result<()> {
    if !model.has_uniforms("terrain") {
        model.set_terrain_mode(false);
        return Ok(());
    }
    model.set_terrain_mode(mode == TerrainMode::WriteHeightMap);
    let terrain = ctx.terrain.clone();
    let (bounds, has_map) = match &terrain {
        Some(map) => (map.bounds, map.view.is_some()),
        None => ([0.0; 4], false),
    };
    // Without a map there is no ground to sit on
    let mode = match mode {
        TerrainMode::UseHeightMap if !has_map => TerrainMode::None,
        other => other,
    };
    if let Some(map) = &terrain {
        model.set_texture("terrain_map", map.texture().clone())?;
        model.set_sampler("terrain_mapSampler", map.sampler.clone())?;
    }
    let block = model.uniforms("terrain")?;
    block.set_vec4("bounds", Vec4::from(bounds))?;
    block.set_f32("mode", mode.shader_value())?;
    Ok(())
}

/// The mode a layer's own models draw with, before its extensions have their say.
pub fn layer_terrain_mode(ctx: &LayerContext, is_terrain_layer: bool) -> TerrainMode {
    match (ctx.terrain_pass, is_terrain_layer) {
        (true, true) => TerrainMode::WriteHeightMap,
        _ => TerrainMode::None,
    }
}

/// An empty map, for decks without terrain.
pub fn empty_map(device: &wgpu::Device, queue: &wgpu::Queue) -> Arc<TerrainMap> {
    Arc::new(TerrainMap::new(device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewport::WebMercatorViewportOptions;

    #[test]
    fn the_map_covers_the_view_with_a_margin() {
        let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.4,
            latitude: 37.8,
            zoom: 12.0,
            ..Default::default()
        });
        let [x, y, width, height] = height_map_bounds(&viewport);
        // The viewport centre falls inside the map
        let centre = viewport.center;
        assert!(
            (centre.x as f32) > x && (centre.x as f32) < x + width,
            "centre {} in {x}..{}",
            centre.x,
            x + width
        );
        assert!((centre.y as f32) > y && (centre.y as f32) < y + height);
        // And the map is wider than the view itself
        let bounds = viewport.get_bounds(0.0);
        let bl = viewport.project_position(glam::DVec3::new(bounds[0], bounds[1], 0.0));
        let tr = viewport.project_position(glam::DVec3::new(bounds[2], bounds[3], 0.0));
        assert!(width as f64 > (tr.x - bl.x).abs());
        assert!(height as f64 > (tr.y - bl.y).abs());
    }
}

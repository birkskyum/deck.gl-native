//! Port of `@deck.gl/geo-layers` TripsLayer: paths with a timestamp per vertex, showing the
//! part travelled before `current_time` with a fading trail.

use deck_gl::data::resolve_f32_lists;
use deck_gl::layer::{set_model_picking_active, update_standard_uniforms};
use deck_gl::{Accessor, DeckError, Layer, LayerContext, LayerProps, Result, Viewport};
use luma_gl::buffer::create_vertex_buffer_from;
use luma_gl::{Model, VertexBufferLayout};
use wgpu::VertexFormat;

use crate::path_layer::{path_model, upload_path_attributes, write_path_uniforms, PathLayerProps};

const PATH_SHADER: &str = include_str!("wgsl/path_layer.wgsl");

/// deck.gl's WGSL injections for the trips layer, applied to the path shader source.
const INJECTIONS: [(&str, &str); 5] = [
    (
        "@group(0) @binding(auto)",
        "struct TripsUniforms {\n  fadeTrail: f32,\n  trailLength: f32,\n  currentTime: f32,\n};\n\n@group(0) @binding(auto) var<uniform> trips: TripsUniforms;\n\n@group(0) @binding(auto)",
    ),
    (
        "  @location(12) rowIndexes: u32,",
        "  @location(12) rowIndexes: u32,\n  @location(13) instanceTimestamps: vec2<f32>,",
    ),
    (
        "  @location(6) pickingColor: vec3<f32>,",
        "  @location(6) pickingColor: vec3<f32>,\n  @location(7) vTime: f32,",
    ),
    (
        "    attributes.instanceColors.a * layer.opacity\n  );",
        "    attributes.instanceColors.a * layer.opacity\n  );\n\n  varyings.vTime = mix(\n    attributes.instanceTimestamps.x,\n    attributes.instanceTimestamps.y,\n    varyings.vPathPosition.y / varyings.vPathLength\n  );\n\n  if (trips.fadeTrail > 0.5) {\n    varyings.vColor.a *=\n      1.0 - (trips.currentTime - varyings.vTime) / trips.trailLength;\n  }",
    ),
    (
        "  geometry.uv = varyings.vPathPosition;",
        "  geometry.uv = varyings.vPathPosition;\n\n  if (\n    varyings.vTime > trips.currentTime ||\n    (trips.fadeTrail > 0.5 && varyings.vTime < trips.currentTime - trips.trailLength)\n  ) {\n    discard;\n  }",
    ),
];

/// The path shader with the trips injections applied.
pub fn trips_shader() -> Result<String> {
    // Windows checkouts may carry CRLF line endings; anchors are written with LF.
    let mut source = PATH_SHADER.replace("\r\n", "\n");
    for (anchor, replacement) in INJECTIONS {
        if !source.contains(anchor) {
            return Err(DeckError::Layer {
                layer: "TripsLayer".into(),
                message: format!("path shader anchor not found: {anchor:?}"),
            });
        }
        source = source.replacen(anchor, replacement, 1);
    }
    Ok(source)
}

/// Pack timestamps in the padded instance order the path tesselator uses (after deck.gl's
/// `packTripTimestamps`): the timestamp of the instance's vertex and of the next one. A
/// closed path repeats its first segment at the end, so those instances wrap back to the
/// first timestamps; unlike deck.gl the end time of the last real segment is the arrival
/// time rather than the departure time, so time never runs backwards along an edge.
pub fn pack_trip_timestamps(timestamps: &[f32], instance_count: usize, is_loop: bool) -> Vec<[f32; 2]> {
    let mut packed = vec![[0.0f32; 2]; instance_count];
    if timestamps.is_empty() {
        return packed;
    }
    let last = timestamps.len() - 1;
    let is_closed = instance_count > timestamps.len();
    let cycle = if is_loop { timestamps.len() } else { last.max(1) };
    for (index, slot) in packed.iter_mut().enumerate() {
        let i = if is_closed { index % cycle } else { index.min(last) };
        let next = if is_loop {
            (i + 1) % cycle
        } else {
            (i + 1).min(last)
        };
        *slot = [timestamps[i], timestamps[next]];
    }
    packed
}

/// Properties of a [`TripsLayer`]: a path layer plus time. Defaults match deck.gl.
#[derive(Clone, Debug, PartialEq)]
pub struct TripsLayerProps {
    pub path: PathLayerProps,
    /// Fade the trail out over `trail_length` behind the current time
    pub fade_trail: bool,
    /// How far back in time the trail reaches
    pub trail_length: f32,
    pub current_time: f32,
    /// One timestamp per path vertex
    pub get_timestamps: Accessor<Vec<f32>>,
}

impl Default for TripsLayerProps {
    fn default() -> Self {
        Self {
            path: PathLayerProps {
                base: LayerProps::new("TripsLayer"),
                ..Default::default()
            },
            fade_trail: true,
            trail_length: 120.0,
            current_time: 0.0,
            get_timestamps: Accessor::column("timestamps"),
        }
    }
}

/// Renders the travelled part of timed paths.
pub struct TripsLayer {
    props: TripsLayerProps,
    model: Option<Model>,
    data_dirty: bool,
}

impl TripsLayer {
    pub fn new(props: TripsLayerProps) -> Self {
        Self {
            props,
            model: None,
            data_dirty: true,
        }
    }

    pub fn props(&self) -> &TripsLayerProps {
        &self.props
    }

    /// Replace the props. Attributes are rebuilt on the next update when they changed; moving
    /// only `current_time` is free.
    pub fn set_props(&mut self, props: TripsLayerProps) {
        let attributes_changed =
            self.props.path != props.path || self.props.get_timestamps != props.get_timestamps;
        self.props = props;
        if attributes_changed {
            self.data_dirty = true;
        }
    }

    /// Advance the animation.
    pub fn set_current_time(&mut self, time: f32) {
        self.props.current_time = time;
    }

    fn update_attributes(&mut self, ctx: &LayerContext) -> Result<()> {
        let model = self.model.as_mut().expect("initialized");
        let tesselated = upload_path_attributes(model, ctx, &self.props.path)?;
        let timestamps = resolve_f32_lists(&self.props.path.data, &self.props.get_timestamps)?;
        let mut packed = Vec::with_capacity(tesselated.instance_count());
        let mut start = 0;
        while start < tesselated.row_index.len() {
            let row = tesselated.row_index[start] as usize;
            let mut end = start;
            while end < tesselated.row_index.len() && tesselated.row_index[end] as usize == row {
                end += 1;
            }
            let stamps = timestamps.get(row).map(Vec::as_slice).unwrap_or(&[]);
            packed.extend(pack_trip_timestamps(stamps, end - start, false));
            start = end;
        }
        model.set_vertex_buffer(
            "instanceTimestamps",
            create_vertex_buffer_from(&ctx.device, "instanceTimestamps", &packed),
        )?;
        Ok(())
    }
}

impl Layer for TripsLayer {
    fn props(&self) -> &LayerProps {
        &self.props.path.base
    }

    fn initialize(&mut self, ctx: &LayerContext) -> Result<()> {
        let shader = trips_shader()?;
        let timestamps = VertexBufferLayout::instance("instanceTimestamps", 13, VertexFormat::Float32x2);
        let model = path_model(
            ctx,
            &self.props.path.base.id,
            &shader,
            &[timestamps],
            &self.props.path.base,
        )?;
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
        let model = self.model.as_mut().expect("initialized");
        update_standard_uniforms(model, ctx, viewport, &props.path.base)?;
        write_path_uniforms(model, &props.path)?;
        let u = model.uniforms("trips")?;
        u.set_f32("fadeTrail", if props.fade_trail { 1.0 } else { 0.0 })?;
        u.set_f32("trailLength", props.trail_length)?;
        u.set_f32("currentTime", props.current_time)?;
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
        self.props.path.base.highlighted_object_index = index;
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
    fn packs_timestamps_per_instance() {
        let packed = pack_trip_timestamps(&[0.0, 10.0, 20.0], 3, false);
        assert_eq!(packed, [[0.0, 10.0], [10.0, 20.0], [20.0, 20.0]]);
        // a closed path has two extra instances that wrap around
        let closed = pack_trip_timestamps(&[0.0, 10.0, 20.0], 5, false);
        assert_eq!(
            closed,
            [[0.0, 10.0], [10.0, 20.0], [0.0, 10.0], [10.0, 20.0], [0.0, 10.0]]
        );
        assert_eq!(pack_trip_timestamps(&[], 2, false), [[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn shader_injections_apply() {
        let source = trips_shader().unwrap();
        assert!(source.contains("var<uniform> trips: TripsUniforms"));
        assert!(source.contains("@location(13) instanceTimestamps"));
        assert!(source.contains("varyings.vTime = mix("));
        assert!(source.contains("discard;"));
    }
}

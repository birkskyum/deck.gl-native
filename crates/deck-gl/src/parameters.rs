//! deck.gl's `parameters` prop: per-layer overrides of the GPU pipeline state.

use luma_gl::model::ModelDescriptor;

/// Which triangle faces to skip, `cullMode` in deck.gl.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

impl From<CullMode> for Option<wgpu::Face> {
    fn from(mode: CullMode) -> Self {
        match mode {
            CullMode::None => None,
            CullMode::Front => Some(wgpu::Face::Front),
            CullMode::Back => Some(wgpu::Face::Back),
        }
    }
}

/// Overrides for the pipeline state a layer builds its models with. Every field defaults to
/// `None`, meaning "keep what the layer chooses"; this matches deck.gl, where `parameters` only
/// lists the settings that differ from the defaults (premultiplied alpha blending, depth test
/// less-equal with writes, no culling).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderParameters {
    /// `Some(false)` turns blending off: the fragment colour replaces the target as is.
    pub blend: Option<bool>,
    /// A full blend state to use instead of the layer's own when blending is on.
    pub blend_state: Option<wgpu::BlendState>,
    /// `Some(false)` disables depth testing and depth writes, like disabling `DEPTH_TEST` in
    /// WebGL. deck.gl's `depthCompare: "always"` is the same test without touching writes.
    pub depth_test: Option<bool>,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<wgpu::CompareFunction>,
    pub cull_mode: Option<CullMode>,
}

impl RenderParameters {
    /// Apply the overrides to a model descriptor whose layer defaults are already set.
    pub fn apply(&self, desc: &mut ModelDescriptor<'_>) {
        if let Some(state) = self.blend_state {
            desc.blend = Some(state);
        }
        if self.blend == Some(false) {
            desc.blend = None;
        }
        if let Some(compare) = self.depth_compare {
            desc.depth_compare = compare;
        }
        if let Some(write) = self.depth_write_enabled {
            desc.depth_write_enabled = write;
        }
        if self.depth_test == Some(false) {
            desc.depth_compare = wgpu::CompareFunction::Always;
            desc.depth_write_enabled = false;
        }
        if let Some(cull) = self.cull_mode {
            desc.cull_mode = cull.into();
        }
    }

    /// Whether two parameter sets build different pipelines.
    pub fn differs(&self, other: &Self) -> bool {
        self != other
    }
}

/// Additive blending, a common `parameters` choice for glowing overlays.
pub fn additive_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

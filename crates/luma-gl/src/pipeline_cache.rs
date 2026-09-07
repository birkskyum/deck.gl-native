//! Sharing of shader modules, bind group layouts and render pipelines between models that
//! build the same ones (luma.gl's `PipelineFactory`): a deck with many layers of one type
//! compiles each shader and pipeline once.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Everything a render pipeline depends on, so equal descriptors share one pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub wgsl: u64,
    pub vertex_layouts: Vec<(u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)>,
    pub bind_group_layout: Vec<wgpu::BindGroupLayoutEntry>,
    pub topology: wgpu::PrimitiveTopology,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub sample_count: u32,
    pub blend: Option<wgpu::BlendState>,
    pub depth_write_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
    /// Constant, slope scale bits and clamp bits of the depth bias
    pub depth_bias: (i32, u32, u32),
    pub cull_mode: Option<wgpu::Face>,
}

#[derive(Default)]
struct Inner {
    modules: HashMap<u64, wgpu::ShaderModule>,
    layouts: HashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    builds: usize,
}

/// A shared cache; clones point at the same cache.
#[derive(Clone, Default)]
pub struct PipelineCache(Arc<Mutex<Inner>>);

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PipelineCache")
    }
}

/// A stable hash of a WGSL source, the cache key of its shader module.
pub fn wgsl_hash(wgsl: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    wgsl.hash(&mut hasher);
    hasher.finish()
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The shader module of `wgsl`, compiled once.
    pub fn shader_module(&self, device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ShaderModule {
        let key = wgsl_hash(wgsl);
        let mut inner = self.lock();
        inner
            .modules
            .entry(key)
            .or_insert_with(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
                })
            })
            .clone()
    }

    /// The bind group layout of `entries` and a pipeline layout with just that group.
    pub fn layouts(
        &self,
        device: &wgpu::Device,
        label: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let mut inner = self.lock();
        inner
            .layouts
            .entry(entries.to_vec())
            .or_insert_with(|| {
                let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries,
                });
                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
                (bind_group_layout, pipeline_layout)
            })
            .clone()
    }

    /// The pipeline for `key`, built with `build` the first time it is asked for.
    pub fn render_pipeline(
        &self,
        key: PipelineKey,
        build: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> wgpu::RenderPipeline {
        let mut inner = self.lock();
        if let Some(pipeline) = inner.pipelines.get(&key) {
            return pipeline.clone();
        }
        let pipeline = build();
        inner.builds += 1;
        inner.pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// How many pipelines were built (as opposed to shared) so far.
    pub fn builds(&self) -> usize {
        self.lock().builds
    }

    /// How many distinct pipelines the cache holds.
    pub fn len(&self) -> usize {
        self.lock().pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // a panic while holding the lock leaves plain data behind, still usable
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

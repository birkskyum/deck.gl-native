//! The collision pass behind deck.gl's `CollisionFilterExtension` (`CollisionFilterEffect`
//! and `CollisionFilterPass` in `@deck.gl/extensions`): the layers of a collision group are
//! drawn into a half resolution texture with their picking colours, sorted by collision
//! priority, and the extension hides every object whose anchor is covered by another one.

use std::collections::HashMap;

use luma_gl::model::create_rgba8_texture;

/// The collision pass is rendered at this fraction of the frame's resolution.
pub const COLLISION_DOWNSCALE: u32 = 2;
/// Pixels left clear around the collision map.
pub const COLLISION_PADDING: u32 = 1;

/// The collision maps of the current frame by collision group, with the sampler, a fallback
/// texture, and whether layers are being drawn into the maps right now (deck.gl's
/// `drawToCollisionMap`).
#[derive(Debug)]
pub struct CollisionMaps {
    pub sampler: wgpu::Sampler,
    pub dummy: wgpu::TextureView,
    pub groups: HashMap<String, wgpu::TextureView>,
    pub drawing_to_map: bool,
}

impl CollisionMaps {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let dummy = create_rgba8_texture(device, queue, "collision fallback", 1, 1, &[0, 0, 0, 0]);
        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("collision nearest clamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            dummy: dummy.create_view(&Default::default()),
            groups: HashMap::new(),
            drawing_to_map: false,
        }
    }
}

/// The colour and depth textures of one collision group's map.
#[derive(Debug)]
pub struct CollisionTarget {
    pub color: wgpu::Texture,
    pub depth: Option<wgpu::Texture>,
}

impl CollisionTarget {
    pub fn new(
        device: &wgpu::Device,
        group: &str,
        width: u32,
        height: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("collision {group}")),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: luma_gl::PICKING_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = depth_format.map(|format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("collision {group} depth")),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        Self { color, depth }
    }

    pub fn size(&self) -> (u32, u32) {
        let size = self.color.size();
        (size.width, size.height)
    }
}

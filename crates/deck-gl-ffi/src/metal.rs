//! Metal host integration: adopt the host's device and queue, wrap its attachments.

use std::ffi::c_void;

use deck_gl::luma_gl::RenderTarget;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTexture, MTLTextureType};
use wgpu::hal::api::Metal;

use crate::DeckglHandle;

fn wgpu_format(format: MTLPixelFormat) -> Option<wgpu::TextureFormat> {
    Some(match format {
        MTLPixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
        MTLPixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
        MTLPixelFormat::RGBA16Float => wgpu::TextureFormat::Rgba16Float,
        MTLPixelFormat::RGB10A2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        MTLPixelFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        MTLPixelFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        MTLPixelFormat::Depth32Float_Stencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
        MTLPixelFormat::Depth24Unorm_Stencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        _ => return None,
    })
}

/// Create a wgpu device on the host's `MTLDevice` that submits on the host's `MTLCommandQueue`.
fn create_device(
    raw_device: Retained<ProtocolObject<dyn MTLDevice>>,
    raw_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
) -> Result<(wgpu::Device, wgpu::Queue), String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL));
    let registry_id = raw_device.registryID();
    let adapter = adapters
        .into_iter()
        .find(|adapter| {
            // SAFETY: the adapter was created by the Metal backend.
            unsafe { adapter.as_hal::<Metal>() }
                .map(|hal| hal.raw_device().registryID() == registry_id)
                .unwrap_or(false)
        })
        .ok_or_else(|| "no wgpu adapter matches the host MTLDevice".to_string())?;

    let features = wgpu::Features::empty();
    let limits = wgpu::Limits::default();
    // SAFETY: the raw handles are retained and belong to the same device as the adapter.
    let hal_device = unsafe { wgpu::hal::metal::Device::device_from_raw(raw_device, features, &limits) };
    let hal_queue = unsafe { wgpu::hal::metal::Queue::queue_from_raw(raw_queue, 1.0) };
    let open = wgpu::hal::OpenDevice {
        device: hal_device,
        queue: hal_queue,
    };
    unsafe {
        adapter.create_device_from_hal::<Metal>(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("deck.gl-native (host Metal device)"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            },
        )
    }
    .map_err(|e| format!("create_device_from_hal failed: {e}"))
}

/// Wrap a host `MTLTexture` as a wgpu texture without copying.
unsafe fn wrap_texture(
    device: &wgpu::Device,
    raw: Retained<ProtocolObject<dyn MTLTexture>>,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    let width = raw.width() as u32;
    let height = raw.height() as u32;
    let is_depth = format.is_depth_stencil_format();
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            raw,
            format,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
            None,
        )
    };
    unsafe {
        device.create_texture_from_hal::<Metal>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
            if is_depth {
                wgpu::TextureUses::DEPTH_STENCIL_WRITE
            } else {
                wgpu::TextureUses::COLOR_TARGET
            },
        )
    }
}

/// # Safety
/// `mtl_device` and `mtl_command_queue` must be valid `id<MTLDevice>` and `id<MTLCommandQueue>`.
#[no_mangle]
pub unsafe extern "C" fn deckgl_metal_create(
    mtl_device: *mut c_void,
    mtl_command_queue: *mut c_void,
) -> *mut DeckglHandle {
    let raw_device = unsafe { Retained::retain(mtl_device as *mut ProtocolObject<dyn MTLDevice>) };
    let raw_queue =
        unsafe { Retained::retain(mtl_command_queue as *mut ProtocolObject<dyn MTLCommandQueue>) };
    let (Some(raw_device), Some(raw_queue)) = (raw_device, raw_queue) else {
        eprintln!("deck.gl-native: null Metal device or queue");
        return std::ptr::null_mut();
    };
    match create_device(raw_device, raw_queue) {
        Ok((device, queue)) => Box::into_raw(Box::new(DeckglHandle::new(device, queue))),
        Err(e) => {
            eprintln!("deck.gl-native: {e}");
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `deck` must be a valid handle; the textures must be valid `id<MTLTexture>` render targets of
/// the same size. `mtl_depth_texture` may be null.
#[no_mangle]
pub unsafe extern "C" fn deckgl_metal_render(
    deck: *mut DeckglHandle,
    mtl_color_texture: *mut c_void,
    mtl_depth_texture: *mut c_void,
    clear_depth: i32,
) -> i32 {
    let Some(handle) = (unsafe { deck.as_mut() }) else {
        return 1;
    };
    let Some(color) = (unsafe { Retained::retain(mtl_color_texture as *mut ProtocolObject<dyn MTLTexture>) })
    else {
        return handle.set_error("null color texture");
    };
    let depth = unsafe { Retained::retain(mtl_depth_texture as *mut ProtocolObject<dyn MTLTexture>) };

    let Some(color_format) = wgpu_format(color.pixelFormat()) else {
        return handle.set_error(format!("unsupported color format {:?}", color.pixelFormat()));
    };
    let depth_format = match &depth {
        Some(depth) => match wgpu_format(depth.pixelFormat()) {
            Some(format) => Some(format),
            None => return handle.set_error(format!("unsupported depth format {:?}", depth.pixelFormat())),
        },
        None => None,
    };
    let target = RenderTarget {
        color_format,
        depth_format,
        sample_count: 1,
    };
    if let Err(e) = handle.ensure_deck(target) {
        return handle.set_error(e);
    }

    let color_texture = unsafe { wrap_texture(&handle.device, color, color_format, "host color") };
    let depth_texture = match (depth, depth_format) {
        (Some(depth), Some(format)) => {
            Some(unsafe { wrap_texture(&handle.device, depth, format, "host depth") })
        }
        _ => None,
    };
    let color_view = color_texture.create_view(&Default::default());
    let depth_view = depth_texture.as_ref().map(|t| t.create_view(&Default::default()));
    if let (Some(depth), Some(deck)) = (&depth_texture, handle.deck.as_ref()) {
        crate::debug::maybe_dump(
            &handle.device,
            &handle.queue,
            depth,
            deck.viewport(),
            handle.frame,
            false,
        );
    }

    let mut encoder = handle
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deck.gl overlay"),
        });
    let Some(deck) = handle.deck.as_mut() else {
        return handle.set_error("deck was not created");
    };
    let result = deck.render_with(
        &mut encoder,
        &color_view,
        depth_view.as_ref(),
        wgpu::LoadOp::Load,
        // The after pass depth dump wants deck's depth alone, on the shared planes.
        if clear_depth != 0 || std::env::var_os("DECKGL_DUMP_DEPTH_AFTER").is_some() {
            wgpu::LoadOp::Clear(1.0)
        } else {
            wgpu::LoadOp::Load
        },
    );
    if let Err(e) = result {
        return handle.set_error(format!("render failed: {e}"));
    }
    handle.queue.submit([encoder.finish()]);
    if let (Some(depth), Some(deck)) = (&depth_texture, handle.deck.as_ref()) {
        crate::debug::maybe_dump(
            &handle.device,
            &handle.queue,
            depth,
            deck.viewport(),
            handle.frame,
            true,
        );
    }
    handle.frame += 1;
    crate::screenshot::maybe_capture(handle, &color_texture, color_format);
    0
}

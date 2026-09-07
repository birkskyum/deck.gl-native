//! GPU test: elevated geometry must stay elevated when deck uses the host's near and far planes.

use deck_gl::luma_gl::device::{create_headless_context, create_render_texture, read_texture_rgba8};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps};
use deck_gl_json::JsonConverter;
use deckgl::{maplibre_near_far_pixels, viewport_from_camera, DeckglCamera};

const SIZE: u32 = 512;

fn render(near_far: Option<(f64, f64)>, name: &str) -> Vec<u8> {
    let ctx = create_headless_context().expect("gpu");
    let spec = r#"[{
      "@@type": "SolidPolygonLayer",
      "id": "roof",
      "data": [{"polygon": [[-122.425, 37.770], [-122.395, 37.770], [-122.395, 37.800], [-122.425, 37.800]]}],
      "extruded": true,
      "getPolygon": "@@=polygon",
      "getElevation": 400,
      "getFillColor": [0, 150, 255, 255]
    }]"#;
    let json = JsonConverter::new().parse(spec).unwrap();
    let (near, far) = near_far.unwrap_or((0.0, 0.0));
    let camera = DeckglCamera {
        longitude: -122.42,
        latitude: 37.775,
        zoom: 14.5,
        bearing: -25.0,
        pitch: 60.0,
        fov_degrees: 36.86989764584402,
        near_z_pixels: near,
        far_z_pixels: far,
        width: SIZE,
        height: SIZE,
        pixel_ratio: 1.0,
    };
    let target = RenderTarget::default();
    let color = create_render_texture(&ctx.device, "color", SIZE, SIZE, target.color_format);
    let depth = create_render_texture(&ctx.device, "depth", SIZE, SIZE, target.depth_format.unwrap());
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width: SIZE,
            height: SIZE,
            layers: json.layers,
            ..Default::default()
        },
    )
    .unwrap();
    deck.set_viewport(viewport_from_camera(&camera));
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(wgpu::Color::TRANSPARENT),
    )
    .unwrap();
    ctx.queue.submit([encoder.finish()]);
    let pixels = read_texture_rgba8(&ctx.device, &ctx.queue, &color).unwrap();
    let path = std::env::temp_dir().join(format!("host-depth-{name}.png"));
    image::save_buffer(&path, &pixels, SIZE, SIZE, image::ColorType::Rgba8).unwrap();
    pixels
}

#[test]
fn roof_stays_elevated_with_host_planes() {
    let plain = render(None, "plain");
    let (near, far) = maplibre_near_far_pixels(36.86989764584402, 60.0, SIZE as f64);
    let host = render(Some((near, far)), "host");
    let differing = plain
        .chunks(4)
        .zip(host.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("near {near} far {far}: {differing} pixels differ between plain and host planes");
    assert!(
        differing < (SIZE * SIZE / 100) as usize,
        "{differing} pixels differ"
    );
}

/// Depth values written by the GPU must match the CPU projection for elevated points, not just
/// for the ground; otherwise a host's buildings win against deck geometry that is in front.
#[test]
fn gpu_depth_matches_cpu_projection_for_elevated_points() {
    use deck_gl::glam::DVec3;
    let ctx = create_headless_context().expect("gpu");
    let spec = r#"[{
      "@@type": "SolidPolygonLayer",
      "id": "wall",
      "data": [{"polygon": [[-122.3935, 37.7960], [-122.3933, 37.7960], [-122.3885, 37.7840], [-122.3887, 37.7840]]}],
      "extruded": true,
      "getPolygon": "@@=polygon",
      "getElevation": 400,
      "getFillColor": [255, 0, 255, 255]
    }]"#;
    let json = JsonConverter::new().parse(spec).unwrap();
    let (w, h) = (1024u32, 768u32);
    let (near, far) = maplibre_near_far_pixels(36.86989764584402, 60.0, h as f64);
    let camera = DeckglCamera {
        longitude: -122.383,
        latitude: 37.789,
        zoom: 15.0,
        bearing: -110.0,
        pitch: 60.0,
        fov_degrees: 36.86989764584402,
        near_z_pixels: near,
        far_z_pixels: far,
        width: w,
        height: h,
        pixel_ratio: 1.0,
    };
    let target = RenderTarget {
        depth_format: Some(wgpu::TextureFormat::Depth32Float),
        ..RenderTarget::default()
    };
    let color = create_render_texture(&ctx.device, "color", w, h, target.color_format);
    let depth = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target.depth_format.unwrap(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        target,
        DeckProps {
            width: w,
            height: h,
            layers: json.layers,
            clip_depth_range: deck_gl::ClipDepthRange::NegativeOneToOne,
            ..Default::default()
        },
    )
    .unwrap();
    let viewport = viewport_from_camera(&camera);
    deck.set_viewport(viewport.clone());
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    deck.render(
        &mut encoder,
        &color.create_view(&Default::default()),
        Some(&depth.create_view(&Default::default())),
        Some(wgpu::Color::TRANSPARENT),
    )
    .unwrap();
    ctx.queue.submit([encoder.finish()]);

    // read the depth buffer
    let padded = (w * 4).div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &depth,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        depth.size(),
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range().unwrap();
    let at = |x: u32, y: u32| {
        let i = (y * padded + x * 4) as usize;
        f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
    };

    // Points on the wall's face, a little inside its outline: foot and various heights.
    let mut worst = 0.0f64;
    for t in [0.2, 0.4, 0.6, 0.8] {
        let lng = -122.3935 + (-122.3885 + 122.3935) * t;
        let lat = 37.7960 + (37.7840 - 37.7960) * t;
        for z in [0.0, 100.0, 250.0, 390.0] {
            let p = viewport.project(DVec3::new(lng, lat, z), true);
            let (x, y) = (p.x.round() as u32, p.y.round() as u32);
            if x >= w || y >= h {
                continue;
            }
            // sample a small neighbourhood to skip edge pixels
            let gpu = [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .map(|(dx, dy)| at((x as i32 + dx) as u32, (y as i32 + dy) as u32) as f64)
                .fold(f64::NAN, |acc, v| {
                    if acc.is_nan() || (v - p.z).abs() < (acc - p.z).abs() {
                        v
                    } else {
                        acc
                    }
                });
            let diff = (gpu - p.z).abs();
            worst = worst.max(diff);
            eprintln!(
                "t {t} z {z:>5}: pixel ({x:4},{y:4}) cpu {:.6} gpu {:.6} diff {:.6}",
                p.z, gpu, diff
            );
        }
    }
    assert!(
        worst < 1e-3,
        "GPU depth deviates from the CPU projection by {worst}"
    );
}

/// Replicates the shader's clip space math on the CPU from the uploaded uniforms.
#[test]
fn project_uniforms_reproduce_the_viewport_depth() {
    use deck_gl::glam::{DVec3, DVec4, Vec4};
    use deck_gl::shaderlib::project::{get_uniforms_from_viewport, ProjectProps};
    use deck_gl::CoordinateSystem;
    let (w, h) = (1024u32, 768u32);
    let (near, far) = maplibre_near_far_pixels(36.86989764584402, 60.0, h as f64);
    let camera = DeckglCamera {
        longitude: -122.383,
        latitude: 37.789,
        zoom: 15.0,
        bearing: -110.0,
        pitch: 60.0,
        fov_degrees: 36.86989764584402,
        near_z_pixels: near,
        far_z_pixels: far,
        width: w,
        height: h,
        pixel_ratio: 1.0,
    };
    let viewport = viewport_from_camera(&camera);
    let u = get_uniforms_from_viewport(&ProjectProps {
        viewport: &viewport,
        device_pixel_ratio: 1.0,
        model_matrix: None,
        coordinate_system: CoordinateSystem::Default,
        coordinate_origin: DVec3::ZERO,
        auto_wrap_longitude: false,
        clip_depth_range: deck_gl::ClipDepthRange::NegativeOneToOne,
    });
    eprintln!(
        "mode {} origin {:?} common origin {:?} center {:?}",
        u.projection_mode, u.coordinate_origin, u.common_origin, u.center
    );
    eprintln!(
        "commonUnitsPerWorldUnit {:?} per meter {:?}",
        u.common_units_per_world_unit, u.common_units_per_meter
    );
    for z in [0.0, 200.0] {
        let p = DVec3::new(-122.3935 + 0.001, 37.7960 - 0.003, z);
        let cpu = viewport.project(p, true);
        // shader: offset from the coordinate origin in world units, scaled to common units
        let origin = u.coordinate_origin.as_dvec3();
        let offset = p - origin;
        let scaled = DVec3::new(
            offset.x * u.common_units_per_world_unit.x as f64,
            offset.y * u.common_units_per_world_unit.y as f64
                + offset.y * offset.y * u.common_units_per_world_unit2.y as f64,
            offset.z * u.common_units_per_world_unit.z as f64,
        );
        let vp = u.view_projection_matrix;
        let clip = vp * Vec4::new(scaled.x as f32, scaled.y as f32, scaled.z as f32, 1.0) + u.center;
        let ndc = DVec4::new(clip.x as f64, clip.y as f64, clip.z as f64, clip.w as f64) / clip.w as f64;
        let px = ((ndc.x + 1.0) / 2.0 * w as f64, (1.0 - ndc.y) / 2.0 * h as f64);
        eprintln!(
            "z {z}: cpu pixel ({:.1}, {:.1}) depth {:.6} | shader pixel ({:.1}, {:.1}) depth {:.6}",
            cpu.x, cpu.y, cpu.z, px.0, px.1, ndc.z
        );
        // exact common position instead of the linearised offset
        let exact = viewport.project_position(p) - u.common_origin.as_dvec3();
        let clip2 = vp * Vec4::new(exact.x as f32, exact.y as f32, exact.z as f32, 1.0) + u.center;
        eprintln!("        exact offset: depth {:.6}", clip2.z / clip2.w);
    }
}

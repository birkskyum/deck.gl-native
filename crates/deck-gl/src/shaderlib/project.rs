//! Port of `@deck.gl/core/src/shaderlib/project/viewport-uniforms.ts`: computes the values
//! of the `project` uniform block from a viewport and a layer's coordinate settings.

use glam::{DMat4, DVec3, DVec4, Mat4, Vec2, Vec3, Vec4};
use luma_gl::UniformBlock;

use crate::constants::{CoordinateSystem, ProjectionMode};
use crate::viewport::Viewport;
use crate::Result;

/// 4x4 matrix that drops the 4th component of a vector
const VECTOR_TO_POINT_MATRIX: DMat4 = DMat4::from_cols(
    DVec4::new(1.0, 0.0, 0.0, 0.0),
    DVec4::new(0.0, 1.0, 0.0, 0.0),
    DVec4::new(0.0, 0.0, 1.0, 0.0),
    DVec4::new(0.0, 0.0, 0.0, 0.0),
);

fn fround(v: f64) -> f64 {
    v as f32 as f64
}

fn fround3(v: DVec3) -> DVec3 {
    DVec3::new(fround(v.x), fround(v.y), fround(v.z))
}

/// Inputs to [`get_uniforms_from_viewport`]. Mirrors `ProjectProps`.
#[derive(Clone, Copy, Debug)]
pub struct ProjectProps<'a> {
    pub viewport: &'a Viewport,
    pub device_pixel_ratio: f32,
    pub model_matrix: Option<DMat4>,
    pub coordinate_system: CoordinateSystem,
    pub coordinate_origin: DVec3,
    pub auto_wrap_longitude: bool,
}

/// Values of the `project` uniform block. Mirrors `ProjectUniforms`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectUniforms {
    pub coordinate_system: i32,
    pub projection_mode: i32,
    pub coordinate_origin: Vec3,
    pub common_origin: Vec3,
    pub center: Vec4,
    pub pseudo_meters: bool,
    pub viewport_size: Vec2,
    pub device_pixel_ratio: f32,
    pub focal_distance: f32,
    pub common_units_per_meter: Vec3,
    pub common_units_per_world_unit: Vec3,
    pub common_units_per_world_unit2: Vec3,
    /// 2^zoom
    pub scale: f32,
    pub wrap_longitude: bool,
    pub view_projection_matrix: Mat4,
    pub model_matrix: Mat4,
    /// For lighting calculations
    pub camera_position: Vec3,
}

impl ProjectUniforms {
    /// Write every field into a `project` uniform block.
    pub fn write(&self, block: &mut UniformBlock) -> Result<()> {
        block.set_bool("wrapLongitude", self.wrap_longitude)?;
        block.set_i32("coordinateSystem", self.coordinate_system)?;
        block.set_vec3("commonUnitsPerMeter", self.common_units_per_meter)?;
        block.set_i32("projectionMode", self.projection_mode)?;
        block.set_f32("scale", self.scale)?;
        block.set_vec3("commonUnitsPerWorldUnit", self.common_units_per_world_unit)?;
        block.set_vec3("commonUnitsPerWorldUnit2", self.common_units_per_world_unit2)?;
        block.set_vec4("center", self.center)?;
        block.set_mat4("modelMatrix", self.model_matrix)?;
        block.set_mat4("viewProjectionMatrix", self.view_projection_matrix)?;
        block.set_vec2("viewportSize", self.viewport_size)?;
        block.set_f32("devicePixelRatio", self.device_pixel_ratio)?;
        block.set_f32("focalDistance", self.focal_distance)?;
        block.set_vec3("cameraPosition", self.camera_position)?;
        block.set_vec3("coordinateOrigin", self.coordinate_origin)?;
        block.set_vec3("commonOrigin", self.common_origin)?;
        block.set_bool("pseudoMeters", self.pseudo_meters)?;
        Ok(())
    }
}

struct OffsetOrigin {
    geospatial_origin: Option<DVec3>,
    shader_coordinate_origin: DVec3,
    offset_mode: bool,
}

fn get_offset_origin(
    viewport: &Viewport,
    coordinate_system: CoordinateSystem,
    coordinate_origin: DVec3,
) -> OffsetOrigin {
    let mut shader_coordinate_origin = coordinate_origin;
    let mut offset_mode = true;

    let mut geospatial_origin = if matches!(
        coordinate_system,
        CoordinateSystem::LngLatOffsets | CoordinateSystem::MeterOffsets
    ) {
        Some(coordinate_origin)
    } else if viewport.is_geospatial {
        Some(viewport.geospatial_origin_f32())
    } else {
        None
    };

    match viewport.projection_mode() {
        ProjectionMode::WebMercator => {
            if matches!(
                coordinate_system,
                CoordinateSystem::LngLat | CoordinateSystem::Cartesian
            ) {
                geospatial_origin = Some(DVec3::ZERO);
                offset_mode = false;
            }
        }
        ProjectionMode::WebMercatorAutoOffset => {
            if coordinate_system == CoordinateSystem::LngLat {
                // viewport center in world space
                shader_coordinate_origin = geospatial_origin.unwrap_or(DVec3::ZERO);
            } else if coordinate_system == CoordinateSystem::Cartesian {
                // viewport center in common space
                shader_coordinate_origin =
                    DVec3::new(fround(viewport.center.x), fround(viewport.center.y), 0.0);
                // Geospatial origin (wgs84) must match shaderCoordinateOrigin (common)
                geospatial_origin = Some(viewport.unproject_position(shader_coordinate_origin));
                shader_coordinate_origin -= coordinate_origin;
            }
        }
        ProjectionMode::Identity => {
            shader_coordinate_origin = fround3(viewport.position);
        }
        ProjectionMode::Globe => {
            offset_mode = false;
            geospatial_origin = None;
        }
    }

    OffsetOrigin {
        geospatial_origin,
        shader_coordinate_origin,
        offset_mode,
    }
}

struct MatrixAndOffset {
    view_projection_matrix: DMat4,
    projection_center: DVec4,
    origin_common: DVec4,
    camera_pos_common: DVec3,
    shader_coordinate_origin: DVec3,
    geospatial_origin: Option<DVec3>,
}

fn calculate_matrix_and_offset(
    viewport: &Viewport,
    coordinate_system: CoordinateSystem,
    coordinate_origin: DVec3,
) -> MatrixAndOffset {
    let mut view_projection_matrix = viewport.view_projection_matrix;
    let mut projection_center = DVec4::ZERO;
    let mut origin_common = DVec4::ZERO;
    let mut camera_pos_common = viewport.camera_position;

    let OffsetOrigin {
        geospatial_origin,
        shader_coordinate_origin,
        offset_mode,
    } = get_offset_origin(viewport, coordinate_system, coordinate_origin);

    if offset_mode {
        // Calculate transformed projectionCenter (using 64 bit precision)
        // This is the key to offset mode precision
        // (avoids doing this addition in 32 bit precision in the shader)
        let origin = viewport.project_position(geospatial_origin.unwrap_or(shader_coordinate_origin));
        camera_pos_common -= origin;
        origin_common = origin.extend(1.0);

        projection_center = view_projection_matrix * origin_common;

        // Always apply uncentered projection matrix if available (shader adds center)
        // Zero out 4th coordinate ("after" model matrix) - avoids further translations
        view_projection_matrix =
            viewport.projection_matrix * viewport.view_matrix_uncentered * VECTOR_TO_POINT_MATRIX;
    }

    MatrixAndOffset {
        view_projection_matrix,
        projection_center,
        origin_common,
        camera_pos_common,
        shader_coordinate_origin,
        geospatial_origin,
    }
}

/// Returns uniforms for shaders based on the current projection.
pub fn get_uniforms_from_viewport(props: &ProjectProps<'_>) -> ProjectUniforms {
    let viewport = props.viewport;
    let coordinate_system = match props.coordinate_system {
        CoordinateSystem::Default => {
            if viewport.is_geospatial {
                CoordinateSystem::LngLat
            } else {
                CoordinateSystem::Cartesian
            }
        }
        other => other,
    };

    let mut uniforms = calculate_viewport_uniforms(
        viewport,
        props.device_pixel_ratio,
        coordinate_system,
        props.coordinate_origin,
    );
    uniforms.wrap_longitude = props.auto_wrap_longitude;
    uniforms.model_matrix = props.model_matrix.map(|m| m.as_mat4()).unwrap_or(Mat4::IDENTITY);
    uniforms
}

fn calculate_viewport_uniforms(
    viewport: &Viewport,
    device_pixel_ratio: f32,
    coordinate_system: CoordinateSystem,
    coordinate_origin: DVec3,
) -> ProjectUniforms {
    let MatrixAndOffset {
        view_projection_matrix,
        projection_center,
        origin_common,
        camera_pos_common,
        shader_coordinate_origin,
        geospatial_origin,
    } = calculate_matrix_and_offset(viewport, coordinate_system, coordinate_origin);

    // Calculate projection pixels per unit
    let distance_scales = viewport.get_distance_scales(None);

    let viewport_size = Vec2::new(
        viewport.width as f32 * device_pixel_ratio,
        viewport.height as f32 * device_pixel_ratio,
    );

    // Distance at which screen pixels are projected.
    // Used to scale sizes in clipspace to match screen pixels.
    let focal = viewport
        .transform_vector(DVec4::new(0.0, 0.0, -viewport.focal_distance, 1.0))
        .w;
    let focal_distance = if focal == 0.0 { 1.0 } else { focal };

    let mut uniforms = ProjectUniforms {
        coordinate_system: coordinate_system.shader_value(),
        projection_mode: viewport.projection_mode().shader_value(),
        coordinate_origin: shader_coordinate_origin.as_vec3(),
        common_origin: origin_common.truncate().as_vec3(),
        center: projection_center.as_vec4(),
        pseudo_meters: false,
        viewport_size,
        device_pixel_ratio,
        focal_distance: focal_distance as f32,
        common_units_per_meter: distance_scales.units_per_meter.as_vec3(),
        common_units_per_world_unit: distance_scales.units_per_meter.as_vec3(),
        common_units_per_world_unit2: Vec3::ZERO,
        scale: viewport.scale as f32,
        wrap_longitude: false,
        view_projection_matrix: view_projection_matrix.as_mat4(),
        model_matrix: Mat4::IDENTITY,
        camera_position: camera_pos_common.as_vec3(),
    };

    if let Some(geospatial_origin) = geospatial_origin {
        // Get high-precision DistanceScales from geospatial viewport
        let at_origin = viewport.get_distance_scales(Some(geospatial_origin));
        match coordinate_system {
            CoordinateSystem::MeterOffsets => {
                uniforms.common_units_per_world_unit = at_origin.units_per_meter.as_vec3();
                uniforms.common_units_per_world_unit2 = at_origin.units_per_meter2.as_vec3();
            }
            CoordinateSystem::LngLat | CoordinateSystem::LngLatOffsets => {
                uniforms.common_units_per_meter = at_origin.units_per_meter.as_vec3();
                uniforms.common_units_per_world_unit = at_origin.units_per_degree.as_vec3();
                uniforms.common_units_per_world_unit2 = at_origin.units_per_degree2.as_vec3();
            }
            // a.k.a "preprojected" positions
            CoordinateSystem::Cartesian => {
                uniforms.common_units_per_world_unit =
                    Vec3::new(1.0, 1.0, at_origin.units_per_meter.z as f32);
                uniforms.common_units_per_world_unit2 =
                    Vec3::new(0.0, 0.0, at_origin.units_per_meter2.z as f32);
            }
            _ => {}
        }
    }

    uniforms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewport::WebMercatorViewportOptions;

    #[test]
    fn web_mercator_mode_uses_centered_matrix() {
        let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.45,
            latitude: 37.78,
            zoom: 11.0,
            ..Default::default()
        });
        let u = get_uniforms_from_viewport(&ProjectProps {
            viewport: &viewport,
            device_pixel_ratio: 1.0,
            model_matrix: None,
            coordinate_system: CoordinateSystem::Default,
            coordinate_origin: DVec3::ZERO,
            auto_wrap_longitude: false,
        });
        assert_eq!(u.coordinate_system, CoordinateSystem::LngLat.shader_value());
        assert_eq!(u.projection_mode, ProjectionMode::WebMercator.shader_value());
        assert_eq!(u.center, Vec4::ZERO);
        assert_eq!(
            u.view_projection_matrix,
            viewport.view_projection_matrix.as_mat4()
        );
        assert!((u.scale - 2048.0).abs() < 1e-6);
        assert!((u.focal_distance - 1.5).abs() < 1e-6);
        // Degrees to common units at the equator (origin [0, 0, 0])
        assert!((u.common_units_per_world_unit.x - 512.0 / 360.0).abs() < 1e-5);
    }

    #[test]
    fn auto_offset_mode_moves_origin_to_viewport_center() {
        let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.45,
            latitude: 37.78,
            zoom: 14.0,
            ..Default::default()
        });
        let u = get_uniforms_from_viewport(&ProjectProps {
            viewport: &viewport,
            device_pixel_ratio: 1.0,
            model_matrix: None,
            coordinate_system: CoordinateSystem::LngLat,
            coordinate_origin: DVec3::ZERO,
            auto_wrap_longitude: false,
        });
        assert_eq!(
            u.projection_mode,
            ProjectionMode::WebMercatorAutoOffset.shader_value()
        );
        assert!((u.coordinate_origin.x - -122.45f32).abs() < 1e-6);
        assert!((u.coordinate_origin.y - 37.78f32).abs() < 1e-6);
        // The projected center is the viewport center in clip space: x and y are 0.
        assert!(u.center.x.abs() < 1e-3, "{:?}", u.center);
        assert!(u.center.y.abs() < 1e-3, "{:?}", u.center);
        assert!(u.center.w > 0.0);
        // Common origin is the mercator projection of the viewport center.
        let expected = viewport.project_position(DVec3::new(-122.45, 37.78, 0.0));
        assert!((u.common_origin.x - expected.x as f32).abs() < 1e-3);
        assert!((u.common_origin.y - expected.y as f32).abs() < 1e-3);
    }
}

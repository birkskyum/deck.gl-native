//! Port of `@math.gl/web-mercator/src/web-mercator-utils.ts`.
//!
//! All angles in the public API are in degrees unless the name says otherwise.
//! World coordinates are on the 512 x 512 Web Mercator zoom 0 tile, y up.

use glam::{DMat4, DVec2, DVec3, DVec4};

pub const PI: f64 = std::f64::consts::PI;
const PI_4: f64 = PI / 4.0;
const DEGREES_TO_RADIANS: f64 = PI / 180.0;
const RADIANS_TO_DEGREES: f64 = 180.0 / PI;
pub const TILE_SIZE: f64 = 512.0;
/// Average circumference (40075 km equatorial, 40007 km meridional)
pub const EARTH_CIRCUMFERENCE: f64 = 40.03e6;
/// Latitude that makes a square world, 2 * atan(E ** PI) - PI / 2
pub const MAX_LATITUDE: f64 = 85.051129;
/// Mapbox default altitude
pub const DEFAULT_ALTITUDE: f64 = 1.5;

/// Scale factors between world space and meters or degrees around a given lng/lat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceScales {
    pub units_per_meter: DVec3,
    pub meters_per_unit: DVec3,
    pub units_per_degree: DVec3,
    pub degrees_per_unit: DVec3,
    /// Second order terms, only populated with `high_precision`.
    pub units_per_meter2: DVec3,
    pub units_per_degree2: DVec3,
}

impl Default for DistanceScales {
    fn default() -> Self {
        Self {
            units_per_meter: DVec3::ONE,
            meters_per_unit: DVec3::ONE,
            units_per_degree: DVec3::ONE,
            degrees_per_unit: DVec3::ONE,
            units_per_meter2: DVec3::ZERO,
            units_per_degree2: DVec3::ZERO,
        }
    }
}

impl DistanceScales {
    /// One unit per meter on every axis, the scales of non geospatial viewports.
    pub fn identity() -> Self {
        Self {
            units_per_meter: DVec3::ONE,
            meters_per_unit: DVec3::ONE,
            ..Default::default()
        }
    }

    /// Custom first order scales, for viewports with independent axis zooms.
    pub fn scaled(units_per_meter: DVec3, meters_per_unit: DVec3) -> Self {
        Self {
            units_per_meter,
            meters_per_unit,
            ..Default::default()
        }
    }
}

/// Projection matrix parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionParameters {
    /// Field of view in radians. Varies with pitch and altitude.
    pub fov: f64,
    /// width / height
    pub aspect: f64,
    /// Distance at which the visual scale factor is 1.
    pub focal_distance: f64,
    pub near: f64,
    pub far: f64,
}

/// Logarithmic zoom to linear scale.
pub fn zoom_to_scale(zoom: f64) -> f64 {
    2f64.powf(zoom)
}

/// Linear scale to logarithmic zoom.
pub fn scale_to_zoom(scale: f64) -> f64 {
    scale.log2()
}

/// Project [lng, lat] on the sphere onto [x, y] on the 512 x 512 Mercator zoom 0 tile.
///
/// Performs the nonlinear part of the web mercator projection. The remaining projection
/// is done with 4x4 matrices which also handle perspective.
pub fn lng_lat_to_world(lng_lat: [f64; 2]) -> [f64; 2] {
    let [lng, lat] = lng_lat;
    debug_assert!(lng.is_finite());
    debug_assert!(
        lat.is_finite() && (-90.0..=90.0).contains(&lat),
        "invalid latitude"
    );

    let lambda2 = lng * DEGREES_TO_RADIANS;
    let phi2 = lat * DEGREES_TO_RADIANS;
    let x = (TILE_SIZE * (lambda2 + PI)) / (2.0 * PI);
    let y = (TILE_SIZE * (PI + (PI_4 + phi2 * 0.5).tan().ln())) / (2.0 * PI);
    [x, y]
}

/// Unproject a world point [x, y] on the map onto [lng, lat] on the sphere.
pub fn world_to_lng_lat(xy: [f64; 2]) -> [f64; 2] {
    let [x, y] = xy;
    let lambda2 = (x / TILE_SIZE) * (2.0 * PI) - PI;
    let phi2 = 2.0 * (((y / TILE_SIZE) * (2.0 * PI) - PI).exp().atan() - PI_4);
    [lambda2 * RADIANS_TO_DEGREES, phi2 * RADIANS_TO_DEGREES]
}

/// Returns the zoom level that gives a 1 meter pixel at a certain latitude.
/// 1 = C*cos(y)/2^z/TILE_SIZE = C*cos(y)/2^(z+9)
pub fn get_meter_zoom(latitude: f64) -> f64 {
    let lat_cosine = (latitude * DEGREES_TO_RADIANS).cos();
    scale_to_zoom(EARTH_CIRCUMFERENCE * lat_cosine) - 9.0
}

/// Calculate the conversion from meters to common units at a given latitude.
/// This is a cheaper version of `get_distance_scales`.
pub fn units_per_meter(latitude: f64) -> f64 {
    let lat_cosine = (latitude * DEGREES_TO_RADIANS).cos();
    TILE_SIZE / EARTH_CIRCUMFERENCE / lat_cosine
}

/// Calculate distance scales in meters around the current lat/lon, both for degrees and pixels.
/// In mercator projection mode, the distance scales vary significantly with latitude.
pub fn get_distance_scales(longitude: f64, latitude: f64, high_precision: bool) -> DistanceScales {
    debug_assert!(latitude.is_finite() && longitude.is_finite());
    let world_size = TILE_SIZE;
    let lat_cosine = (latitude * DEGREES_TO_RADIANS).cos();

    // Number of pixels occupied by one degree longitude around current lat/lon:
    // unitsPerDegreeX = d(lngLatToWorld([lng, lat])[0])/d(lng)
    //   = scale * TILE_SIZE * DEGREES_TO_RADIANS / (2 * PI)
    // unitsPerDegreeY = d(lngLatToWorld([lng, lat])[1])/d(lat)
    //   = -scale * TILE_SIZE * DEGREES_TO_RADIANS / cos(lat * DEGREES_TO_RADIANS)  / (2 * PI)
    let units_per_degree_x = world_size / 360.0;
    let units_per_degree_y = units_per_degree_x / lat_cosine;

    // Number of pixels occupied by one meter around current lat/lon:
    let alt_units_per_meter = world_size / EARTH_CIRCUMFERENCE / lat_cosine;

    let mut result = DistanceScales {
        units_per_meter: DVec3::splat(alt_units_per_meter),
        meters_per_unit: DVec3::splat(1.0 / alt_units_per_meter),
        units_per_degree: DVec3::new(units_per_degree_x, units_per_degree_y, alt_units_per_meter),
        degrees_per_unit: DVec3::new(
            1.0 / units_per_degree_x,
            1.0 / units_per_degree_y,
            1.0 / alt_units_per_meter,
        ),
        units_per_meter2: DVec3::ZERO,
        units_per_degree2: DVec3::ZERO,
    };

    // Taylor series 2nd order for 1/latCosine
    //   f'(a) * (x - a)
    //     = d(1/cos(lat * DEGREES_TO_RADIANS))/d(lat) * dLat
    //     = DEGREES_TO_RADIANS * tan(lat * DEGREES_TO_RADIANS) / cos(lat * DEGREES_TO_RADIANS) * dLat
    if high_precision {
        let lat_cosine2 = (DEGREES_TO_RADIANS * (latitude * DEGREES_TO_RADIANS).tan()) / lat_cosine;
        let units_per_degree_y2 = (units_per_degree_x * lat_cosine2) / 2.0;
        let alt_units_per_degree2 = (world_size / EARTH_CIRCUMFERENCE) * lat_cosine2;
        let alt_units_per_meter2 = (alt_units_per_degree2 / units_per_degree_y) * alt_units_per_meter;

        result.units_per_degree2 = DVec3::new(0.0, units_per_degree_y2, alt_units_per_degree2);
        result.units_per_meter2 = DVec3::new(alt_units_per_meter2, 0.0, alt_units_per_meter2);
    }

    result
}

/// Offset a lng/lat position by a meter offset (easting, northing, up).
pub fn add_meters_to_lng_lat(lng_lat_z: [f64; 3], xyz: [f64; 3]) -> [f64; 3] {
    let [longitude, latitude, z0] = lng_lat_z;
    let [x, y, z] = xyz;

    let scales = get_distance_scales(longitude, latitude, true);
    let upm = scales.units_per_meter;
    let upm2 = scales.units_per_meter2;

    let mut worldspace = lng_lat_to_world([longitude, latitude]);
    worldspace[0] += x * (upm.x + upm2.x * y);
    worldspace[1] += y * (upm.y + upm2.y * y);

    let new_lng_lat = world_to_lng_lat(worldspace);
    [new_lng_lat[0], new_lng_lat[1], z0 + z]
}

/// Options for [`get_view_matrix`].
#[derive(Clone, Copy, Debug)]
pub struct ViewMatrixOptions {
    pub height: f64,
    pub pitch: f64,
    pub bearing: f64,
    pub altitude: f64,
    /// Pre-calculated scale (2 ^ zoom)
    pub scale: f64,
    pub center: Option<DVec3>,
}

/// View matrix creation is intentionally kept compatible with mapbox-gl's implementation to
/// ensure seamless interoperation with mapbox and maplibre.
pub fn get_view_matrix(options: &ViewMatrixOptions) -> DMat4 {
    // VIEW MATRIX: PROJECTS MERCATOR WORLD COORDINATES
    // Note that mercator world coordinates typically need to be flipped
    //
    // Note: As usual, matrix operation orders should be read in reverse
    // since vectors will be multiplied from the right during transformation
    let mut vm = DMat4::IDENTITY;

    // Move camera to altitude (along the pitch & bearing direction)
    vm *= DMat4::from_translation(DVec3::new(0.0, 0.0, -options.altitude));

    // Rotate by bearing, and then by pitch (which tilts the view)
    vm *= DMat4::from_rotation_x(-options.pitch * DEGREES_TO_RADIANS);
    vm *= DMat4::from_rotation_z(options.bearing * DEGREES_TO_RADIANS);

    let relative_scale = options.scale / options.height;
    vm *= DMat4::from_scale(DVec3::splat(relative_scale));

    if let Some(center) = options.center {
        vm *= DMat4::from_translation(-center);
    }

    vm
}

/// Options for [`get_projection_parameters`].
#[derive(Clone, Copy, Debug)]
pub struct ProjectionOptions {
    pub width: f64,
    pub height: f64,
    /// Scale at the current zoom
    pub scale: f64,
    /// Offset of the target, in world space
    pub center: Option<DVec3>,
    /// Offset of the focal point, in screen space
    pub offset: Option<DVec2>,
    /// Field of view in degrees. If `None`, derived from `altitude`.
    pub fovy: Option<f64>,
    /// If provided, field of view is calculated using `altitude_to_fovy`
    pub altitude: Option<f64>,
    /// Camera angle in degrees (0 is straight down)
    pub pitch: f64,
    pub near_z_multiplier: f64,
    pub far_z_multiplier: f64,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            scale: 1.0,
            center: None,
            offset: None,
            fovy: None,
            altitude: None,
            pitch: 0.0,
            near_z_multiplier: 1.0,
            far_z_multiplier: 1.0,
        }
    }
}

/// Calculates mapbox compatible projection parameters.
pub fn get_projection_parameters(options: &ProjectionOptions) -> ProjectionParameters {
    let mut fovy = options.fovy.unwrap_or_else(|| altitude_to_fovy(DEFAULT_ALTITUDE));
    // For back-compatibility allow field of view to be derived from altitude
    if let Some(altitude) = options.altitude {
        fovy = altitude_to_fovy(altitude);
    }

    let fov_radians = fovy * DEGREES_TO_RADIANS;
    let pitch_radians = options.pitch * DEGREES_TO_RADIANS;

    // Distance from camera to the target
    let focal_distance = fovy_to_altitude(fovy);

    let mut camera_to_sea_level_distance = focal_distance;
    if let Some(center) = options.center {
        camera_to_sea_level_distance += (center.z * options.scale) / pitch_radians.cos() / options.height;
    }

    let offset_y = options.offset.map(|o| o.y).unwrap_or(0.0);
    let fov_above_center = fov_radians * (0.5 + offset_y / options.height);

    // Find the distance from the center point to the center top
    // in focal distance units using law of sines.
    let top_half_surface_distance = (fov_above_center.sin() * camera_to_sea_level_distance)
        / (PI / 2.0 - pitch_radians - fov_above_center)
            .clamp(0.01, PI - 0.01)
            .sin();

    // Calculate z distance of the farthest fragment that should be rendered.
    let furthest_distance = pitch_radians.sin() * top_half_surface_distance + camera_to_sea_level_distance;
    // Matches mapbox limit
    let horizon_distance = camera_to_sea_level_distance * 10.0;

    // Calculate z value of the farthest fragment that should be rendered.
    let far_z = (furthest_distance * options.far_z_multiplier).min(horizon_distance);

    ProjectionParameters {
        fov: fov_radians,
        aspect: options.width / options.height,
        focal_distance,
        near: options.near_z_multiplier,
        far: far_z,
    }
}

/// Projection matrix from camera (view) space to clip space, using the OpenGL
/// `[-1, 1]` depth convention that deck.gl's shaders expect.
pub fn get_projection_matrix(options: &ProjectionOptions) -> DMat4 {
    let p = get_projection_parameters(options);
    perspective(p.fov, p.aspect, p.near, p.far)
}

/// gl-matrix compatible perspective matrix (OpenGL clip space).
pub fn perspective(fovy_radians: f64, aspect: f64, near: f64, far: f64) -> DMat4 {
    let f = 1.0 / (fovy_radians / 2.0).tan();
    let nf = 1.0 / (near - far);
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, (far + near) * nf, -1.0),
        DVec4::new(0.0, 0.0, 2.0 * far * near * nf, 0.0),
    )
}

/// Convert an altitude to field of view such that the focal distance is equal to the altitude.
/// Returns fovy in degrees.
pub fn altitude_to_fovy(altitude: f64) -> f64 {
    2.0 * (0.5 / altitude).atan() * RADIANS_TO_DEGREES
}

/// Convert a field of view (degrees) such that the focal distance is equal to the altitude.
pub fn fovy_to_altitude(fovy: f64) -> f64 {
    0.5 / (0.5 * fovy * DEGREES_TO_RADIANS).tan()
}

/// Transform a vector by a matrix and divide by w.
pub fn transform_vector(matrix: &DMat4, vector: DVec4) -> DVec4 {
    let result = *matrix * vector;
    result * (1.0 / result.w)
}

/// Project flat coordinates to pixels on screen.
pub fn world_to_pixels(xyz: DVec3, pixel_projection_matrix: &DMat4) -> DVec3 {
    transform_vector(pixel_projection_matrix, xyz.extend(1.0)).truncate()
}

/// Unproject pixels on screen to flat coordinates.
///
/// If `z` is `None`, `target_z` is used as the elevation plane to unproject onto.
pub fn pixels_to_world(xy: DVec2, z: Option<f64>, pixel_unprojection_matrix: &DMat4, target_z: f64) -> DVec3 {
    if let Some(z) = z {
        // Has depth component
        return transform_vector(pixel_unprojection_matrix, DVec4::new(xy.x, xy.y, z, 1.0)).truncate();
    }

    // since we don't know the correct projected z value for the point,
    // unproject two points to get a line and then find the point on that line with z=0
    let coord0 = transform_vector(pixel_unprojection_matrix, DVec4::new(xy.x, xy.y, 0.0, 1.0)).truncate();
    let coord1 = transform_vector(pixel_unprojection_matrix, DVec4::new(xy.x, xy.y, 1.0, 1.0)).truncate();

    let z0 = coord0.z;
    let z1 = coord1.z;

    let t = if z0 == z1 {
        0.0
    } else {
        (target_z - z0) / (z1 - z0)
    };
    coord0.lerp(coord1, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} != {b} (eps {eps})");
    }

    // Golden values generated with @math.gl/web-mercator in Node.
    #[test]
    fn lng_lat_round_trip() {
        let world = lng_lat_to_world([-122.45, 37.78]);
        assert_close(world[0], 81.84888888888887, 1e-9);
        assert_close(world[1], 314.1104572310264, 1e-9);

        let ll = world_to_lng_lat([100.0, 200.0]);
        assert_close(ll[0], -109.6875, 1e-9);
        assert_close(ll[1], -36.59788913307019, 1e-9);

        let back = world_to_lng_lat(world);
        assert_close(back[0], -122.45, 1e-9);
        assert_close(back[1], 37.78, 1e-9);
    }

    #[test]
    fn distance_scales_match_math_gl() {
        let ds = get_distance_scales(-122.45, 37.78, true);
        assert_close(ds.units_per_meter.x, 0.000016182831898449418, 1e-18);
        assert_close(ds.meters_per_unit.x, 61793.88170594644, 1e-6);
        assert_close(ds.units_per_degree.x, 1.4222222222222223, 1e-12);
        assert_close(ds.units_per_degree.y, 1.7994410024859173, 1e-12);
        assert_close(ds.units_per_degree2.y, 0.01217178428526735, 1e-12);
        assert_close(ds.units_per_degree2.z, 2.1892792119391683e-7, 1e-18);
        assert_close(ds.units_per_meter2.x, 1.9688746347691744e-12, 1e-22);
    }

    #[test]
    fn projection_parameters_match_math_gl() {
        let p = get_projection_parameters(&ProjectionOptions {
            width: 800.0,
            height: 600.0,
            scale: 2048.0,
            pitch: 30.0,
            altitude: Some(1.5),
            near_z_multiplier: 0.1,
            far_z_multiplier: 1.01,
            ..Default::default()
        });
        assert_close(p.fov, 0.6435011087932844, 1e-12);
        assert_close(p.aspect, 4.0 / 3.0, 1e-12);
        assert_close(p.focal_distance, 1.5, 1e-12);
        assert_close(p.near, 0.1, 1e-12);
        assert_close(p.far, 1.876045035400021, 1e-12);
        assert_close(altitude_to_fovy(1.5), 36.86989764584402, 1e-12);
        assert_close(fovy_to_altitude(36.86989764584402), 1.5, 1e-12);
    }

    #[test]
    fn view_matrix_matches_math_gl() {
        // math.gl WebMercatorViewport({width: 800, height: 600, longitude: -122.45,
        //   latitude: 37.78, zoom: 11, pitch: 30, bearing: 20}).viewMatrix (centered)
        let center = lng_lat_to_world([-122.45, 37.78]);
        let vm = get_view_matrix(&ViewMatrixOptions {
            height: 600.0,
            pitch: 30.0,
            bearing: 20.0,
            altitude: 1.5,
            scale: 2048.0,
            center: Some(DVec3::new(center[0], center[1], 0.0)),
        });
        let expected = [
            3.207484145615901,
            1.0110229597048281,
            -0.5837143779424745,
            0.0,
            -1.1674287558849492,
            2.777762752339196,
            -1.6037420728079503,
            0.0,
            0.0,
            1.7066666666666666,
            2.9560333782508845,
            0.0,
            104.17256684828112,
            -955.2754341095751,
            550.0285290333998,
            1.0,
        ];
        let actual = vm.to_cols_array();
        for i in 0..16 {
            assert_close(actual[i], expected[i], 1e-9);
        }
    }

    #[test]
    fn pixels_to_world_finds_ground_plane() {
        let m = DMat4::IDENTITY;
        let p = pixels_to_world(DVec2::new(1.0, 2.0), None, &m, 0.0);
        assert_close(p.x, 1.0, 1e-12);
        assert_close(p.y, 2.0, 1e-12);
        assert_close(p.z, 0.0, 1e-12);
    }
}

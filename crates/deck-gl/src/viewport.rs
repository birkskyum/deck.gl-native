//! Port of `@deck.gl/core/src/viewports/viewport.ts` and `web-mercator-viewport.ts`.
//!
//! Only the Web Mercator (perspective) viewport is implemented for now. The struct is
//! immutable: build a new one whenever a parameter changes.

use glam::{DMat4, DVec2, DVec3, DVec4};
use math_gl::web_mercator::{
    self as wm, get_distance_scales, get_projection_parameters, get_view_matrix, lng_lat_to_world,
    pixels_to_world, units_per_meter, world_to_lng_lat, world_to_pixels, DistanceScales, ProjectionOptions,
    ViewMatrixOptions,
};

use crate::constants::ProjectionMode;

/// Options for [`Viewport::web_mercator`]. Field names follow deck.gl's `WebMercatorViewport`.
#[derive(Clone, Debug)]
pub struct WebMercatorViewportOptions {
    pub id: String,
    /// Left offset from the canvas edge, in pixels
    pub x: f64,
    /// Top offset from the canvas edge, in pixels
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
    /// Tilt of the camera in degrees
    pub pitch: f64,
    /// Heading of the camera in degrees
    pub bearing: f64,
    /// Camera altitude relative to the viewport height, legacy property used to control the FOV.
    pub altitude: f64,
    /// Camera fovy in degrees. If provided, overrides `altitude`
    pub fovy: Option<f64>,
    /// Viewport center as meter offsets from lng, lat, elevation
    pub position: Option<DVec3>,
    /// Scaler for the near plane, 1 unit equals to the height of the viewport.
    pub near_z_multiplier: f64,
    /// Scaler for the far plane, 1 unit equals to the distance from the camera to the edge of the screen.
    pub far_z_multiplier: f64,
    /// Optionally override the near plane position.
    pub near_z: Option<f64>,
    /// Optionally override the far plane position.
    pub far_z: Option<f64>,
}

impl Default for WebMercatorViewportOptions {
    fn default() -> Self {
        Self {
            id: "WebMercatorViewport".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            longitude: 0.0,
            latitude: 0.0,
            zoom: 0.0,
            pitch: 0.0,
            bearing: 0.0,
            altitude: 1.5,
            fovy: None,
            position: None,
            near_z_multiplier: 0.1,
            far_z_multiplier: 1.01,
            near_z: None,
            far_z: None,
        }
    }
}

/// Manages coordinate system transformations between world, common and screen space.
#[derive(Clone, Debug)]
pub struct Viewport {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub is_geospatial: bool,
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
    pub pitch: f64,
    pub bearing: f64,
    pub altitude: f64,
    /// Field of view in degrees
    pub fovy: f64,
    pub focal_distance: f64,
    pub position: DVec3,
    pub model_matrix: Option<DMat4>,

    /// Scale factors between world space and common space
    pub distance_scales: DistanceScales,
    /// 2^zoom
    pub scale: f64,
    /// Viewport center in common space
    pub center: DVec3,
    /// Camera position in common space
    pub camera_position: DVec3,
    pub projection_matrix: DMat4,
    pub view_matrix: DMat4,
    pub view_matrix_uncentered: DMat4,
    pub view_matrix_inverse: DMat4,
    pub view_projection_matrix: DMat4,
    pub pixel_projection_matrix: DMat4,
    pub pixel_unprojection_matrix: DMat4,
    pub near: f64,
    pub far: f64,
}

fn fround(v: f64) -> f64 {
    v as f32 as f64
}

impl Viewport {
    /// Create a Web Mercator viewport. Port of `new WebMercatorViewport(opts)`.
    pub fn web_mercator(opts: &WebMercatorViewportOptions) -> Self {
        let width = if opts.width > 0.0 { opts.width } else { 1.0 };
        let height = if opts.height > 0.0 { opts.height } else { 1.0 };
        let scale = wm::zoom_to_scale(opts.zoom);

        let (fovy, altitude) = match opts.fovy {
            Some(fovy) => (fovy, wm::fovy_to_altitude(fovy)),
            None => (wm::altitude_to_fovy(opts.altitude), opts.altitude),
        };

        let mut projection_parameters = get_projection_parameters(&ProjectionOptions {
            width,
            height,
            scale,
            center: opts
                .position
                .map(|p| DVec3::new(0.0, 0.0, p.z * units_per_meter(opts.latitude))),
            offset: None,
            fovy: Some(fovy),
            altitude: None,
            pitch: opts.pitch,
            near_z_multiplier: opts.near_z_multiplier,
            far_z_multiplier: opts.far_z_multiplier,
        });
        if let Some(near) = opts.near_z {
            projection_parameters.near = near;
        }
        if let Some(far) = opts.far_z {
            projection_parameters.far = far;
        }

        // The uncentered matrix allows us to move the center addition to the shader (cheap)
        // which gives a coordinate system that has its center in the layer's center position.
        let view_matrix_uncentered = get_view_matrix(&ViewMatrixOptions {
            height,
            pitch: opts.pitch,
            bearing: opts.bearing,
            scale,
            altitude,
            center: None,
        });

        // Base Viewport initialization (`_initProps` and `_initMatrices`)
        let distance_scales = get_distance_scales(opts.longitude, opts.latitude, false);
        let meter_offset = opts.position.unwrap_or(DVec3::ZERO);
        let center_ll = Self::project_flat_geospatial([opts.longitude, opts.latitude]);
        let center =
            DVec3::new(center_ll[0], center_ll[1], 0.0) + meter_offset * distance_scales.units_per_meter;

        let view_matrix = view_matrix_uncentered * DMat4::from_translation(-center);
        let projection_matrix = wm::perspective(
            projection_parameters.fov,
            projection_parameters.aspect,
            projection_parameters.near,
            projection_parameters.far,
        );
        let view_projection_matrix = projection_matrix * view_matrix;
        let view_matrix_inverse = view_matrix.inverse();
        let camera_position = view_matrix_inverse.w_axis.truncate();

        // matrix for conversion from world location to screen (pixel) coordinates
        let viewport_matrix = DMat4::from_scale(DVec3::new(width / 2.0, -height / 2.0, 1.0))
            * DMat4::from_translation(DVec3::new(1.0, -1.0, 0.0));
        let pixel_projection_matrix = viewport_matrix * view_projection_matrix;
        let pixel_unprojection_matrix = pixel_projection_matrix.inverse();

        Self {
            id: opts.id.clone(),
            x: opts.x,
            y: opts.y,
            width,
            height,
            is_geospatial: true,
            longitude: opts.longitude,
            latitude: opts.latitude,
            zoom: opts.zoom,
            pitch: opts.pitch,
            bearing: opts.bearing,
            altitude,
            fovy,
            focal_distance: altitude,
            position: meter_offset,
            model_matrix: None,
            distance_scales,
            scale,
            center,
            camera_position,
            projection_matrix,
            view_matrix,
            view_matrix_uncentered,
            view_matrix_inverse,
            view_projection_matrix,
            pixel_projection_matrix,
            pixel_unprojection_matrix,
            near: projection_parameters.near,
            far: projection_parameters.far,
        }
    }

    pub fn meters_per_pixel(&self) -> f64 {
        self.distance_scales.meters_per_unit.z / self.scale
    }

    pub fn projection_mode(&self) -> ProjectionMode {
        if self.is_geospatial {
            if self.zoom < 12.0 {
                ProjectionMode::WebMercator
            } else {
                ProjectionMode::WebMercatorAutoOffset
            }
        } else {
            ProjectionMode::Identity
        }
    }

    /// Two viewports are equal if width and height are identical, and if their view and
    /// projection matrices are (approximately) equal.
    pub fn equals(&self, other: &Viewport) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.scale == other.scale
            && self.projection_matrix.abs_diff_eq(other.projection_matrix, 1e-9)
            && self.view_matrix.abs_diff_eq(other.view_matrix, 1e-9)
    }

    fn project_flat_geospatial(lng_lat: [f64; 2]) -> [f64; 2] {
        // Shader clamps latitude to +-89.9, see project.wgsl
        // lngLatToWorld([0, -89.9])[1] = -317.9934163758329
        // lngLatToWorld([0, 89.9])[1] = 829.9934163758271
        let lat = lng_lat[1].clamp(-90.0, 90.0);
        let mut result = lng_lat_to_world([lng_lat[0], lat]);
        result[1] = result[1].clamp(-318.0, 830.0);
        result
    }

    /// Project [lng, lat] on sphere onto [x, y] on the 512 x 512 Mercator zoom 0 tile.
    pub fn project_flat(&self, xy: [f64; 2]) -> [f64; 2] {
        if self.is_geospatial {
            Self::project_flat_geospatial(xy)
        } else {
            xy
        }
    }

    /// Unproject world point [x, y] on map onto [lng, lat] on sphere.
    pub fn unproject_flat(&self, xy: [f64; 2]) -> [f64; 2] {
        if self.is_geospatial {
            world_to_lng_lat(xy)
        } else {
            xy
        }
    }

    /// Project a position in the viewport's coordinate system (lng, lat, meters above sea
    /// level) into common space.
    pub fn project_position(&self, xyz: DVec3) -> DVec3 {
        let [x, y] = self.project_flat([xyz.x, xyz.y]);
        let z = if self.is_geospatial {
            xyz.z * units_per_meter(xyz.y)
        } else {
            xyz.z * self.distance_scales.units_per_meter.z
        };
        DVec3::new(x, y, z)
    }

    /// Inverse of [`Viewport::project_position`].
    pub fn unproject_position(&self, xyz: DVec3) -> DVec3 {
        let [x, y] = self.unproject_flat([xyz.x, xyz.y]);
        let z = if self.is_geospatial {
            xyz.z / units_per_meter(y)
        } else {
            xyz.z * self.distance_scales.meters_per_unit.z
        };
        DVec3::new(x, y, z)
    }

    /// Projects xyz (possibly latitude and longitude) to pixel coordinates in window.
    /// Returns top-left coordinates unless `top_left` is false.
    pub fn project(&self, xyz: DVec3, top_left: bool) -> DVec3 {
        let world_position = self.project_position(xyz);
        let coord = world_to_pixels(world_position, &self.pixel_projection_matrix);
        let y2 = if top_left { coord.y } else { self.height - coord.y };
        DVec3::new(coord.x, y2, coord.z)
    }

    /// Unproject pixel coordinates on screen onto world coordinates.
    ///
    /// `z` is the depth buffer value if known. Otherwise the point is unprojected onto the
    /// plane at `target_z` meters.
    pub fn unproject(&self, xy: DVec2, z: Option<f64>, top_left: bool, target_z: Option<f64>) -> DVec3 {
        let y2 = if top_left { xy.y } else { self.height - xy.y };
        let target_z_world = target_z.unwrap_or(0.0) * self.distance_scales.units_per_meter.z;
        let coord = pixels_to_world(
            DVec2::new(xy.x, y2),
            z,
            &self.pixel_unprojection_matrix,
            target_z_world,
        );
        let mut result = self.unproject_position(coord);
        if z.is_none() {
            result.z = target_z.unwrap_or(0.0);
        }
        result
    }

    /// Bounds of the current viewport as [min_x, min_y, max_x, max_y] in lng/lat.
    pub fn get_bounds(&self, z: f64) -> [f64; 4] {
        let corners = [
            self.unproject(DVec2::new(0.0, 0.0), None, true, Some(z)),
            self.unproject(DVec2::new(self.width, 0.0), None, true, Some(z)),
            self.unproject(DVec2::new(0.0, self.height), None, true, Some(z)),
            self.unproject(DVec2::new(self.width, self.height), None, true, Some(z)),
        ];
        let mut bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for c in corners {
            bounds[0] = bounds[0].min(c.x);
            bounds[1] = bounds[1].min(c.y);
            bounds[2] = bounds[2].max(c.x);
            bounds[3] = bounds[3].max(c.y);
        }
        bounds
    }

    /// Distance scales, optionally high precision around a coordinate origin.
    pub fn get_distance_scales(&self, coordinate_origin: Option<DVec3>) -> DistanceScales {
        match coordinate_origin {
            Some(origin) if self.is_geospatial => get_distance_scales(origin.x, origin.y, true),
            _ => self.distance_scales,
        }
    }

    /// Viewport center with longitude/latitude rounded to f32, as the shader will see it.
    pub(crate) fn geospatial_origin_f32(&self) -> DVec3 {
        DVec3::new(fround(self.longitude), fround(self.latitude), 0.0)
    }

    pub(crate) fn transform_vector(&self, v: DVec4) -> DVec4 {
        self.projection_matrix * v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} != {b} (eps {eps})");
    }

    fn sf() -> Viewport {
        Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.45,
            latitude: 37.78,
            zoom: 11.0,
            pitch: 30.0,
            bearing: 20.0,
            ..Default::default()
        })
    }

    // Golden values from @math.gl/web-mercator WebMercatorViewport with nearZMultiplier 0.1
    // and farZMultiplier 1.01 (deck.gl's defaults).
    #[test]
    fn matrices_match_math_gl() {
        let vp = sf();
        assert_close(vp.center.x, 81.84888888888887, 1e-9);
        assert_close(vp.center.y, 314.1104572310264, 1e-9);
        assert_close(vp.fovy, 36.86989764584402, 1e-9);
        assert_close(vp.near, 0.1, 1e-12);
        assert_close(vp.far, 1.876045035400021, 1e-9);

        let expected_projection = [
            2.25,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.112609757080261,
            -1.0,
            0.0,
            0.0,
            -0.21126097570802613,
            0.0,
        ];
        let actual = vp.projection_matrix.to_cols_array();
        for i in 0..16 {
            assert_close(actual[i], expected_projection[i], 1e-9);
        }

        let expected_vpm = [
            7.216839327635777,
            3.0330688791144844,
            0.6494463122468322,
            0.5837143779424745,
            -2.6267147007411356,
            8.333288257017587,
            1.784339078046248,
            1.6037420728079503,
            0.0,
            5.119999999999999,
            -3.2889115788968604,
            -2.9560333782508845,
            234.38827540863252,
            -2865.8263023287254,
            -612.1783690507723,
            -550.0285290333998,
        ];
        let actual = vp.view_projection_matrix.to_cols_array();
        for i in 0..16 {
            assert_close(actual[i], expected_vpm[i], 1e-8);
        }
    }

    #[test]
    fn project_and_unproject_round_trip() {
        let vp = sf();
        let px = vp.project(DVec3::new(-122.4, 37.8, 0.0), true);
        assert_close(px.x, 504.71326247477714, 1e-6);
        assert_close(px.y, 203.27288655051697, 1e-6);
        assert_close(px.z, 0.9805083354762708, 1e-6);

        // deck.gl scales z by units_per_meter at the point's latitude; math.gl's viewport uses
        // the viewport center latitude, hence the looser tolerance.
        let px3 = vp.project(DVec3::new(-122.4, 37.8, 100.0), true);
        assert_close(px3.x, 505.0274252940352, 1e-3);
        assert_close(px3.y, 201.42372391804233, 1e-3);

        let ll2 = vp.unproject(DVec2::new(100.0, 50.0), None, true, None);
        assert_close(ll2.x, -122.52490031491318, 1e-9);
        assert_close(ll2.y, 37.900729875960344, 1e-9);

        let ll = vp.unproject(DVec2::new(400.0, 300.0), None, true, None);
        assert_close(ll.x, -122.45, 1e-9);
        assert_close(ll.y, 37.78, 1e-9);

        let back = vp.unproject(DVec2::new(px.x, px.y), None, true, None);
        assert_close(back.x, -122.4, 1e-9);
        assert_close(back.y, 37.8, 1e-9);
    }

    #[test]
    fn projection_mode_switches_at_zoom_12() {
        let mut opts = WebMercatorViewportOptions {
            zoom: 11.9,
            ..Default::default()
        };
        assert_eq!(
            Viewport::web_mercator(&opts).projection_mode(),
            ProjectionMode::WebMercator
        );
        opts.zoom = 12.0;
        assert_eq!(
            Viewport::web_mercator(&opts).projection_mode(),
            ProjectionMode::WebMercatorAutoOffset
        );
    }
}

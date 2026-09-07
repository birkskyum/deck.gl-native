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
    /// Whole worlds this viewport is shifted by along longitude, see
    /// [`Viewport::sub_viewports`]. Zero for the viewport itself.
    pub world_offset: i32,
    /// Depth of the viewport centre in pixel space, the default depth `unproject` uses when
    /// none is given (orbit viewports, deck.gl's `projectedCenter`)
    pub projected_center_depth: Option<f64>,
    /// A globe viewport: positions project onto a sphere of [`GLOBE_RADIUS`] common units
    pub globe: bool,
    /// Degrees per mesh segment when flat geometry is turned into 3D on the globe
    pub resolution: f64,
}

/// Radius of the globe in common units.
pub const GLOBE_RADIUS: f64 = 256.0;
/// Earth radius used by the globe projection, in meters.
pub const GLOBE_EARTH_RADIUS: f64 = 6370972.0;
/// Latitude beyond which globe scales are held fixed to avoid the singularity at the poles.
pub const MAX_LATITUDE: f64 = 85.051129;

/// deck.gl's `zoomAdjust`: the zoom offset that makes the globe and the Web Mercator map
/// converge at high zoom (the map's scale grows with 1 / cos(latitude)).
pub fn globe_zoom_adjust(latitude: f64, clamp_to_poles: bool) -> f64 {
    let latitude = if clamp_to_poles {
        latitude.clamp(-MAX_LATITUDE, MAX_LATITUDE)
    } else {
        latitude
    };
    (std::f64::consts::PI * latitude.to_radians().cos()).log2()
}

/// Options of [`Viewport::globe`], deck.gl's `GlobeViewport`.
#[derive(Clone, Debug)]
pub struct GlobeViewportOptions {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub zoom: f64,
    pub bearing: f64,
    pub pitch: f64,
    /// Camera altitude relative to the viewport height, controls the field of view
    pub altitude: f64,
    /// Field of view in degrees; overrides `altitude` when set
    pub fovy: Option<f64>,
    pub position: DVec3,
    pub orthographic: bool,
    /// Near plane in viewport heights (0.5, as maplibre)
    pub near_z_multiplier: f64,
    /// Far plane multiplier
    pub far_z_multiplier: f64,
    pub near_z: Option<f64>,
    pub far_z: Option<f64>,
    /// Degrees per mesh segment for flat geometry
    pub resolution: f64,
}

impl Default for GlobeViewportOptions {
    fn default() -> Self {
        Self {
            id: "globe".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            longitude: 0.0,
            latitude: 0.0,
            zoom: 0.0,
            bearing: 0.0,
            pitch: 0.0,
            altitude: 1.5,
            fovy: None,
            position: DVec3::ZERO,
            orthographic: false,
            near_z_multiplier: 0.5,
            far_z_multiplier: 1.0,
            near_z: None,
            far_z: None,
            resolution: 10.0,
        }
    }
}

fn globe_distance_scales() -> DistanceScales {
    let units_per_meter = GLOBE_RADIUS / GLOBE_EARTH_RADIUS;
    let units_per_degree = std::f64::consts::PI / 180.0 * GLOBE_RADIUS;
    DistanceScales {
        units_per_meter: DVec3::splat(units_per_meter),
        meters_per_unit: DVec3::splat(1.0 / units_per_meter),
        units_per_degree: DVec3::new(units_per_degree, units_per_degree, units_per_meter),
        degrees_per_unit: DVec3::new(
            1.0 / units_per_degree,
            1.0 / units_per_degree,
            1.0 / units_per_meter,
        ),
        ..Default::default()
    }
}

/// Longitude, latitude and meters above the surface to the globe's common space.
pub fn globe_project_position(lng_lat_z: DVec3) -> DVec3 {
    let lambda = lng_lat_z.x.to_radians();
    let phi = lng_lat_z.y.to_radians();
    let cos_phi = phi.cos();
    let d = (lng_lat_z.z / GLOBE_EARTH_RADIUS + 1.0) * GLOBE_RADIUS;
    DVec3::new(
        lambda.sin() * cos_phi * d,
        -lambda.cos() * cos_phi * d,
        phi.sin() * d,
    )
}

/// The inverse of [`globe_project_position`].
pub fn globe_unproject_position(xyz: DVec3) -> DVec3 {
    let d = xyz.length();
    let phi = (xyz.z / d).asin();
    let lambda = xyz.x.atan2(-xyz.y);
    DVec3::new(
        lambda.to_degrees(),
        phi.to_degrees(),
        (d / GLOBE_RADIUS - 1.0) * GLOBE_EARTH_RADIUS,
    )
}

/// Options of the generic constructor, deck.gl's `Viewport` base class. The camera is given
/// as an uncentered view matrix and either a projection matrix or its parameters.
#[derive(Clone, Debug)]
pub struct ViewportOptions {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// Both set for a geospatial viewport
    pub longitude: Option<f64>,
    pub latitude: Option<f64>,
    /// Viewport centre in world space, meter offsets from the anchor when geospatial
    pub position: DVec3,
    pub zoom: f64,
    pub distance_scales: Option<DistanceScales>,
    /// The uncentered view matrix
    pub view_matrix: DMat4,
    pub projection_matrix: Option<DMat4>,
    pub orthographic: bool,
    /// Field of view in degrees
    pub fovy: f64,
    pub near: f64,
    pub far: f64,
    /// Pixels per common unit at zoom 0
    pub focal_distance: f64,
}

impl Default for ViewportOptions {
    fn default() -> Self {
        Self {
            id: "viewport".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            longitude: None,
            latitude: None,
            position: DVec3::ZERO,
            zoom: 0.0,
            distance_scales: None,
            view_matrix: DMat4::IDENTITY,
            projection_matrix: None,
            orthographic: false,
            fovy: 75.0,
            near: 0.1,
            far: 1000.0,
            focal_distance: 1.0,
        }
    }
}

/// Options of [`Viewport::orthographic`], deck.gl's `OrthographicViewport`.
#[derive(Clone, Debug)]
pub struct OrthographicViewportOptions {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// World position at the centre of the viewport
    pub target: DVec3,
    /// `zoom: 0` maps one world unit to one pixel; each level doubles the size
    pub zoom: f64,
    /// Independent zoom along X and Y, overriding `zoom`
    pub zoom_x: Option<f64>,
    pub zoom_y: Option<f64>,
    pub near: f64,
    pub far: f64,
    /// Top left screen coordinates (`true`) or bottom left
    pub flip_y: bool,
}

impl Default for OrthographicViewportOptions {
    fn default() -> Self {
        Self {
            id: "orthographic".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            target: DVec3::ZERO,
            zoom: 0.0,
            zoom_x: None,
            zoom_y: None,
            near: 0.1,
            far: 1000.0,
            flip_y: true,
        }
    }
}

/// Options of [`Viewport::orbit`], deck.gl's `OrbitViewport`.
#[derive(Clone, Debug)]
pub struct OrbitViewportOptions {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub orbit_axis: crate::views::OrbitAxis,
    pub target: DVec3,
    pub zoom: f64,
    /// Rotation around the orbit axis in degrees
    pub rotation_orbit: f64,
    /// Rotation around the X axis in degrees
    pub rotation_x: f64,
    /// Field of view in degrees
    pub fovy: f64,
    pub near: f64,
    pub far: f64,
    pub orthographic: bool,
}

impl Default for OrbitViewportOptions {
    fn default() -> Self {
        Self {
            id: "orbit".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            orbit_axis: crate::views::OrbitAxis::Z,
            target: DVec3::ZERO,
            zoom: 0.0,
            rotation_orbit: 0.0,
            rotation_x: 0.0,
            fovy: 50.0,
            near: 0.1,
            far: 1000.0,
            orthographic: false,
        }
    }
}

/// Options of [`Viewport::first_person`], deck.gl's `FirstPersonViewport`.
#[derive(Clone, Debug)]
pub struct FirstPersonViewportOptions {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// Anchor of the camera when geospatial
    pub longitude: Option<f64>,
    pub latitude: Option<f64>,
    /// Meter offsets of the camera from the anchor, or its world position
    pub position: DVec3,
    pub bearing: f64,
    pub pitch: f64,
    pub up: DVec3,
    pub fovy: f64,
    pub near: f64,
    pub far: f64,
    /// Pixels per meter
    pub focal_distance: f64,
}

impl Default for FirstPersonViewportOptions {
    fn default() -> Self {
        Self {
            id: "first-person".to_string(),
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            longitude: None,
            latitude: None,
            position: DVec3::ZERO,
            bearing: 0.0,
            pitch: 0.0,
            up: DVec3::Z,
            fovy: 75.0,
            near: 0.1,
            far: 1000.0,
            focal_distance: 1.0,
        }
    }
}

/// gl-matrix's `lookAt`: a right handed view matrix looking from `eye` at `center`.
fn look_at(eye: DVec3, center: DVec3, up: DVec3) -> DMat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    DMat4::from_cols(
        DVec4::new(s.x, u.x, -f.x, 0.0),
        DVec4::new(s.y, u.y, -f.y, 0.0),
        DVec4::new(s.z, u.z, -f.z, 0.0),
        DVec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// gl-matrix's `ortho`: an orthographic projection with OpenGL clip space depth.
fn ortho_gl(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> DMat4 {
    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);
    DMat4::from_cols(
        DVec4::new(-2.0 * lr, 0.0, 0.0, 0.0),
        DVec4::new(0.0, -2.0 * bt, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 2.0 * nf, 0.0),
        DVec4::new((left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1.0),
    )
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
            world_offset: 0,
            projected_center_depth: None,
            globe: false,
            resolution: 0.0,
        }
    }

    /// The generic constructor: deck.gl's `Viewport` base class, used by the non map views.
    pub fn from_options(opts: &ViewportOptions) -> Self {
        let width = if opts.width > 0.0 { opts.width } else { 1.0 };
        let height = if opts.height > 0.0 { opts.height } else { 1.0 };
        let is_geospatial = opts.longitude.is_some() && opts.latitude.is_some();
        let (longitude, latitude) = (opts.longitude.unwrap_or(0.0), opts.latitude.unwrap_or(0.0));
        let distance_scales = match (&opts.distance_scales, is_geospatial) {
            (Some(scales), _) => *scales,
            (None, true) => get_distance_scales(longitude, latitude, false),
            (None, false) => DistanceScales::identity(),
        };
        let scale = wm::zoom_to_scale(opts.zoom);
        let center = if is_geospatial {
            let center_ll = Self::project_flat_geospatial([longitude, latitude]);
            DVec3::new(center_ll[0], center_ll[1], 0.0) + opts.position * distance_scales.units_per_meter
        } else {
            DVec3::new(
                opts.position.x * distance_scales.units_per_meter.x,
                opts.position.y * distance_scales.units_per_meter.y,
                opts.position.z * distance_scales.units_per_meter.z,
            )
        };
        let view_matrix_uncentered = opts.view_matrix;
        let view_matrix = view_matrix_uncentered * DMat4::from_translation(-center);
        let fovy_radians = opts.fovy.to_radians();
        let aspect = width / height;
        let projection_matrix = opts.projection_matrix.unwrap_or_else(|| {
            if opts.orthographic {
                let top = opts.focal_distance * (fovy_radians / 2.0).tan();
                let right = top * aspect;
                ortho_gl(-right, right, -top, top, opts.near, opts.far)
            } else {
                wm::perspective(fovy_radians, aspect, opts.near, opts.far)
            }
        });
        let view_projection_matrix = projection_matrix * view_matrix;
        let view_matrix_inverse = view_matrix.inverse();
        let camera_position = view_matrix_inverse.w_axis.truncate();
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
            is_geospatial,
            longitude,
            latitude,
            zoom: opts.zoom,
            pitch: 0.0,
            bearing: 0.0,
            altitude: opts.focal_distance,
            fovy: opts.fovy,
            focal_distance: opts.focal_distance,
            position: opts.position,
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
            near: opts.near,
            far: opts.far,
            world_offset: 0,
            projected_center_depth: None,
            globe: false,
            resolution: 0.0,
        }
    }

    /// A globe, deck.gl's `GlobeViewport`: the world is a sphere of [`GLOBE_RADIUS`] common
    /// units and the camera looks at `longitude`, `latitude` from `altitude` viewport heights.
    pub fn globe(opts: &GlobeViewportOptions) -> Self {
        let height = if opts.height > 0.0 { opts.height } else { 1.0 };
        let latitude = opts.latitude.clamp(-90.0, 90.0);
        let (fovy, altitude) = match opts.fovy {
            Some(fovy) => (fovy, wm::fovy_to_altitude(fovy)),
            None => (wm::altitude_to_fovy(opts.altitude), opts.altitude),
        };
        // Exaggerate the distance by latitude to match the Web Mercator distortion, so the
        // globe and the map converge at high zoom
        let scale_latitude = latitude.clamp(-MAX_LATITUDE, MAX_LATITUDE);
        let scale = wm::zoom_to_scale(opts.zoom - globe_zoom_adjust(scale_latitude, false));
        let pitch_radians = opts.pitch.to_radians();
        let near = opts.near_z.unwrap_or(opts.near_z_multiplier);
        let far = opts.far_z.unwrap_or(
            (altitude + (GLOBE_RADIUS * 2.0 * scale) / height / pitch_radians.cos().max(0.1))
                * opts.far_z_multiplier,
        );
        // The camera sits on -Y looking at the origin; after the globe rotation the surface at
        // the target faces it with east along +X and north along +Z
        let view_matrix = look_at(DVec3::new(0.0, -altitude, 0.0), DVec3::ZERO, DVec3::Z)
            * DMat4::from_rotation_x(-pitch_radians)
            * DMat4::from_rotation_y(-opts.bearing.to_radians())
            * DMat4::from_rotation_x(latitude.to_radians())
            * DMat4::from_rotation_z(-opts.longitude.to_radians())
            * DMat4::from_scale(DVec3::splat(scale / height));
        let distance_scales = globe_distance_scales();
        // The base constructor projects the centre through the flat helpers; do it here with
        // the sphere instead
        let center = globe_project_position(DVec3::new(opts.longitude, latitude, 0.0))
            + opts.position * distance_scales.units_per_meter;
        let mut viewport = Self::from_options(&ViewportOptions {
            id: opts.id.clone(),
            x: opts.x,
            y: opts.y,
            width: opts.width,
            height,
            longitude: Some(opts.longitude),
            latitude: Some(latitude),
            position: DVec3::ZERO,
            zoom: opts.zoom,
            distance_scales: Some(distance_scales),
            view_matrix,
            orthographic: opts.orthographic,
            fovy,
            near,
            far,
            focal_distance: altitude,
            ..Default::default()
        });
        viewport.globe = true;
        viewport.resolution = opts.resolution;
        viewport.scale = scale;
        viewport.pitch = opts.pitch;
        viewport.bearing = opts.bearing;
        viewport.altitude = altitude;
        viewport.position = opts.position;
        viewport.center = center;
        viewport.view_matrix = view_matrix * DMat4::from_translation(-center);
        viewport.view_projection_matrix = viewport.projection_matrix * viewport.view_matrix;
        viewport.view_matrix_inverse = viewport.view_matrix.inverse();
        viewport.camera_position = viewport.view_matrix_inverse.w_axis.truncate();
        let viewport_matrix =
            DMat4::from_scale(DVec3::new(viewport.width / 2.0, -viewport.height / 2.0, 1.0))
                * DMat4::from_translation(DVec3::new(1.0, -1.0, 0.0));
        viewport.pixel_projection_matrix = viewport_matrix * viewport.view_projection_matrix;
        viewport.pixel_unprojection_matrix = viewport.pixel_projection_matrix.inverse();
        viewport
    }

    /// The longitude, latitude and zoom that keep the globe under a dragged pointer: port of
    /// `GlobeViewport.panByPosition`. `start` is the longitude, latitude and zoom when the
    /// drag started at `start_pixel`.
    pub fn globe_pan_by_position(&self, start: [f64; 3], pixel: [f64; 2], start_pixel: [f64; 2]) -> [f64; 3] {
        // Rotation speed falls with zoom, for a roughly constant panning speed on screen
        let scale = wm::zoom_to_scale(self.zoom - globe_zoom_adjust(self.latitude, false));
        let rotation_speed = 0.25 / scale;
        let longitude = start[0] + rotation_speed * (start_pixel[0] - pixel[0]);
        let latitude = (start[1] - rotation_speed * (start_pixel[1] - pixel[1])).clamp(-90.0, 90.0);
        let zoom = start[2] - globe_zoom_adjust(start[1], false) + globe_zoom_adjust(latitude, false);
        [longitude, latitude, zoom]
    }

    /// A 2D view of cartesian coordinates, deck.gl's `OrthographicViewport`.
    pub fn orthographic(opts: &OrthographicViewportOptions) -> Self {
        let width = if opts.width > 0.0 { opts.width } else { 1.0 };
        let height = if opts.height > 0.0 { opts.height } else { 1.0 };
        let zoom_x = opts.zoom_x.unwrap_or(opts.zoom);
        let zoom_y = opts.zoom_y.unwrap_or(opts.zoom);
        let scale = wm::zoom_to_scale(opts.zoom);
        // Axis specific zooms override the scalar one independently on each axis
        let distance_scales = if zoom_x != opts.zoom || zoom_y != opts.zoom {
            let (scale_x, scale_y) = (wm::zoom_to_scale(zoom_x), wm::zoom_to_scale(zoom_y));
            Some(DistanceScales::scaled(
                DVec3::new(scale_x / scale, scale_y / scale, 1.0),
                DVec3::new(scale / scale_x, scale / scale_y, 1.0),
            ))
        } else {
            None
        };
        let flip = if opts.flip_y { -1.0 } else { 1.0 };
        let view_matrix = look_at(DVec3::Z, DVec3::ZERO, DVec3::Y)
            * DMat4::from_scale(DVec3::new(scale, scale * flip, scale));
        let projection_matrix = ortho_gl(
            -width / 2.0,
            width / 2.0,
            -height / 2.0,
            height / 2.0,
            opts.near,
            opts.far,
        );
        Self::from_options(&ViewportOptions {
            id: opts.id.clone(),
            x: opts.x,
            y: opts.y,
            width,
            height,
            position: opts.target,
            zoom: opts.zoom,
            distance_scales,
            view_matrix,
            projection_matrix: Some(projection_matrix),
            near: opts.near,
            far: opts.far,
            ..Default::default()
        })
    }

    /// A 3D view orbiting a target in cartesian coordinates, deck.gl's `OrbitViewport`: one
    /// common unit at the target maps to one pixel, as in the map view.
    pub fn orbit(opts: &OrbitViewportOptions) -> Self {
        use crate::views::OrbitAxis;
        let height = if opts.height > 0.0 { opts.height } else { 1.0 };
        let focal_distance = wm::fovy_to_altitude(opts.fovy);
        let (up, eye) = match opts.orbit_axis {
            OrbitAxis::Z => (DVec3::Z, DVec3::new(0.0, -focal_distance, 0.0)),
            OrbitAxis::Y => (DVec3::Y, DVec3::new(0.0, 0.0, focal_distance)),
        };
        let orbit = match opts.orbit_axis {
            OrbitAxis::Z => DMat4::from_rotation_z(opts.rotation_orbit.to_radians()),
            OrbitAxis::Y => DMat4::from_rotation_y(opts.rotation_orbit.to_radians()),
        };
        // Scale the common space down instead of moving the camera away, so the depth field
        // keeps the default near and far planes
        let projection_scale = wm::zoom_to_scale(opts.zoom) / height;
        let view_matrix = look_at(eye, DVec3::ZERO, up)
            * DMat4::from_rotation_x(opts.rotation_x.to_radians())
            * orbit
            * DMat4::from_scale(DVec3::splat(projection_scale));
        let mut viewport = Self::from_options(&ViewportOptions {
            id: opts.id.clone(),
            x: opts.x,
            y: opts.y,
            width: opts.width,
            height,
            position: opts.target,
            zoom: opts.zoom,
            view_matrix,
            orthographic: opts.orthographic,
            fovy: opts.fovy,
            near: opts.near,
            far: opts.far,
            focal_distance,
            ..Default::default()
        });
        viewport.pitch = opts.rotation_x;
        viewport.bearing = opts.rotation_orbit;
        let projected_center = viewport.project(viewport.center, true);
        viewport.projected_center_depth = Some(projected_center.z);
        viewport
    }

    /// A camera at a position looking along a bearing and pitch, deck.gl's
    /// `FirstPersonViewport`. Geospatial when longitude and latitude are given.
    pub fn first_person(opts: &FirstPersonViewportOptions) -> Self {
        // Avoid a non invertible pixel projection matrix when looking straight up
        let pitch = if opts.pitch == -90.0 {
            0.0001
        } else {
            90.0 + opts.pitch
        };
        // math.gl's SphericalCoordinates: bearing 0 looks north, pitch 0 is horizontal
        let direction = DMat4::from_rotation_z(std::f64::consts::PI - opts.bearing.to_radians())
            * DMat4::from_rotation_x(pitch.to_radians())
            * DVec4::new(0.0, 0.0, 1.0, 0.0);
        let center = direction.truncate().normalize();
        let zoom = match opts.latitude {
            Some(latitude) => wm::get_meter_zoom(latitude),
            None => 0.0,
        };
        let scale = wm::zoom_to_scale(zoom);
        let view_matrix = look_at(DVec3::ZERO, center, opts.up) * DMat4::from_scale(DVec3::splat(scale));
        let mut viewport = Self::from_options(&ViewportOptions {
            id: opts.id.clone(),
            x: opts.x,
            y: opts.y,
            width: opts.width,
            height: opts.height,
            longitude: opts.longitude,
            latitude: opts.latitude,
            position: opts.position,
            zoom,
            view_matrix,
            fovy: opts.fovy,
            near: opts.near,
            far: opts.far,
            focal_distance: opts.focal_distance,
            ..Default::default()
        });
        viewport.pitch = opts.pitch;
        viewport.bearing = opts.bearing;
        viewport
    }

    /// The target that puts the world position `coords` under `pixel`, for the orthographic and
    /// orbit viewports: port of their `panByPosition`. Map viewports use
    /// [`Viewport::pan_by_position`].
    pub fn pan_target_by_position(&self, coords: DVec3, pixel: DVec2) -> DVec3 {
        match self.projected_center_depth {
            Some(depth) => {
                // Orbit: shift the centre in pixel space, scaled by the relative depth of the
                // grabbed point, and unproject it back at the centre's depth
                let p0 = self.project(coords, true);
                let (near, far) = (self.near, self.far);
                let pz = (near * far) / (far - p0.z * (far - near));
                let center_z = (near * far) / (far - depth * (far - near));
                let shift_scale = pz / center_z;
                let next_center = DVec2::new(
                    self.width / 2.0 + (p0.x - pixel.x) * shift_scale,
                    self.height / 2.0 + (p0.y - pixel.y) * shift_scale,
                );
                self.unproject(next_center, Some(depth), true, None)
            }
            None => {
                // Orthographic: move the centre by the offset between the grabbed point and the
                // point currently under the pointer, in common space
                let from = pixels_to_world(pixel, None, &self.pixel_unprojection_matrix, 0.0);
                let to = self.project_flat([coords.x, coords.y]);
                let new_center = [self.center.x + to[0] - from.x, self.center.y + to[1] - from.y];
                let flat = self.unproject_flat(new_center);
                DVec3::new(flat[0], flat[1], coords.z)
            }
        }
    }

    /// A copy of this viewport looking at the world shifted by `offset` whole worlds (512
    /// common units each) along longitude: port of `WebMercatorViewport`'s `worldOffset`.
    pub fn with_world_offset(&self, offset: i32) -> Viewport {
        let mut v = self.clone();
        v.world_offset = offset;
        v.view_matrix_uncentered = self.view_matrix_uncentered
            * DMat4::from_translation(DVec3::new(512.0 * offset as f64, 0.0, 0.0));
        v.view_matrix = v.view_matrix_uncentered * DMat4::from_translation(-self.center);
        v.view_projection_matrix = self.projection_matrix * v.view_matrix;
        v.view_matrix_inverse = v.view_matrix.inverse();
        v.camera_position = v.view_matrix_inverse.w_axis.truncate();
        let viewport_matrix = DMat4::from_scale(DVec3::new(self.width / 2.0, -self.height / 2.0, 1.0))
            * DMat4::from_translation(DVec3::new(1.0, -1.0, 0.0));
        v.pixel_projection_matrix = viewport_matrix * v.view_projection_matrix;
        v.pixel_unprojection_matrix = v.pixel_projection_matrix.inverse();
        v
    }

    /// The viewports needed to fill the screen when it shows more than one copy of the world
    /// (deck.gl's `MapView({repeat: true})`): this viewport followed by one per extra world
    /// copy visible across the antimeridian, at most three to each side.
    pub fn sub_viewports(&self) -> Vec<Viewport> {
        if !self.is_geospatial || self.globe {
            return vec![self.clone()];
        }
        let bounds = self.get_bounds(0.0);
        let min_offset = (((bounds[0] + 180.0) / 360.0).floor() as i32).clamp(-3, 0);
        let max_offset = (((bounds[2] - 180.0) / 360.0).ceil() as i32).clamp(0, 3);
        let mut viewports = vec![self.clone()];
        for offset in min_offset..=max_offset {
            if offset != 0 {
                viewports.push(self.with_world_offset(offset));
            }
        }
        viewports
    }

    pub fn meters_per_pixel(&self) -> f64 {
        self.distance_scales.meters_per_unit.z / self.scale
    }

    pub fn projection_mode(&self) -> ProjectionMode {
        if self.globe {
            return ProjectionMode::Globe;
        }
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
        if self.globe {
            return xy;
        }
        if self.is_geospatial {
            Self::project_flat_geospatial(xy)
        } else {
            let u = self.distance_scales.units_per_meter;
            [xy[0] * u.x, xy[1] * u.y]
        }
    }

    /// Unproject world point [x, y] on map onto [lng, lat] on sphere.
    /// The longitude and latitude that put the geographic point `lng_lat` under `pixel`
    /// (logical pixels from the top left), keeping zoom, pitch and bearing. Port of
    /// `WebMercatorViewport.panByPosition`.
    pub fn pan_by_position(&self, lng_lat: [f64; 2], pixel: [f64; 2]) -> [f64; 2] {
        let from = self.unproject(DVec2::new(pixel[0], pixel[1]), None, true, None);
        let from_world = self.project_flat([from.x, from.y]);
        let to_world = self.project_flat(lng_lat);
        let center = [
            self.center.x + to_world[0] - from_world[0],
            self.center.y + to_world[1] - from_world[1],
        ];
        self.unproject_flat(center)
    }

    pub fn unproject_flat(&self, xy: [f64; 2]) -> [f64; 2] {
        if self.globe {
            return xy;
        }
        if self.is_geospatial {
            world_to_lng_lat(xy)
        } else {
            let m = self.distance_scales.meters_per_unit;
            [xy[0] * m.x, xy[1] * m.y]
        }
    }

    /// Project a position in the viewport's coordinate system (lng, lat, meters above sea
    /// level) into common space.
    pub fn project_position(&self, xyz: DVec3) -> DVec3 {
        if self.globe {
            return globe_project_position(xyz);
        }
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
        if self.globe {
            return globe_unproject_position(xyz);
        }
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
        if self.globe {
            return self.unproject_globe(DVec2::new(xy.x, y2), z, target_z);
        }
        // Orbit viewports unproject onto the plane through the target by default
        let z = z.or(if target_z.is_none() {
            self.projected_center_depth
        } else {
            None
        });
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

    /// Globe unprojection: with a depth the pixel maps straight back; without one the pixel's
    /// view ray is intersected with the sphere at `target_z` meters above the surface.
    fn unproject_globe(&self, pixel: DVec2, z: Option<f64>, target_z: Option<f64>) -> DVec3 {
        let transform = |v: DVec4| {
            let r = self.pixel_unprojection_matrix * v;
            r.truncate() / r.w
        };
        let coord = match z {
            Some(z) => transform(DVec4::new(pixel.x, pixel.y, z, 1.0)),
            None => {
                let coord0 = transform(DVec4::new(pixel.x, pixel.y, -1.0, 1.0));
                let coord1 = transform(DVec4::new(pixel.x, pixel.y, 1.0, 1.0));
                let lt = (target_z.unwrap_or(0.0) / GLOBE_EARTH_RADIUS + 1.0) * GLOBE_RADIUS;
                let l_sqr = (coord0 - coord1).length_squared();
                let l0_sqr = coord0.length_squared();
                let l1_sqr = coord1.length_squared();
                let s_sqr = (4.0 * l0_sqr * l1_sqr - (l_sqr - l0_sqr - l1_sqr).powi(2)) / 16.0;
                let d_sqr = 4.0 * s_sqr / l_sqr;
                let r0 = (l0_sqr - d_sqr).max(0.0).sqrt();
                let dr = (lt * lt - d_sqr).max(0.0).sqrt();
                let t = (r0 - dr) / l_sqr.sqrt();
                coord0.lerp(coord1, t)
            }
        };
        let mut result = globe_unproject_position(coord);
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
        if self.globe {
            return self.distance_scales;
        }
        match coordinate_origin {
            Some(origin) if self.is_geospatial => get_distance_scales(origin.x, origin.y, true),
            _ => self.distance_scales,
        }
    }

    /// Viewport center with longitude/latitude rounded to f32, as the shader will see it.
    /// The viewport centre rounded to f32, the origin of the shader\'s auto offset mode.
    pub fn geospatial_origin_f32(&self) -> DVec3 {
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

    #[test]
    fn world_offset_shifts_by_whole_worlds() {
        let v = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 512.0,
            height: 256.0,
            longitude: 90.0,
            latitude: 0.0,
            zoom: 0.0,
            ..Default::default()
        });
        // Half the screen lies past the antimeridian: one extra world copy to the east
        let copies = v.sub_viewports();
        assert_eq!(copies.iter().map(|c| c.world_offset).collect::<Vec<_>>(), [0, 1]);
        let shifted = &copies[1];
        let a = shifted.project(DVec3::new(-170.0, 10.0, 0.0), true);
        let b = v.project(DVec3::new(190.0, 10.0, 0.0), true);
        assert!((a - b).length() < 1e-6, "{a:?} vs {b:?}");
        // A view inside one world has no copies
        let inside = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 64.0,
            height: 64.0,
            longitude: 0.0,
            latitude: 0.0,
            zoom: 4.0,
            ..Default::default()
        });
        assert_eq!(inside.sub_viewports().len(), 1);
    }

    #[test]
    fn orthographic_viewport_maps_units_to_pixels() {
        let v = Viewport::orthographic(&OrthographicViewportOptions {
            width: 200.0,
            height: 100.0,
            target: DVec3::new(10.0, 20.0, 0.0),
            zoom: 1.0,
            ..Default::default()
        });
        assert!(!v.is_geospatial);
        let c = v.project(DVec3::new(10.0, 20.0, 0.0), true);
        assert!((c.x - 100.0).abs() < 1e-9 && (c.y - 50.0).abs() < 1e-9, "{c:?}");
        // Zoom 1: one unit is two pixels; flipY puts +y downwards on screen
        let p = v.project(DVec3::new(15.0, 25.0, 0.0), true);
        assert!((p.x - 110.0).abs() < 1e-9 && (p.y - 60.0).abs() < 1e-9, "{p:?}");
        let back = v.unproject(DVec2::new(110.0, 60.0), None, true, None);
        assert!(
            (back.x - 15.0).abs() < 1e-9 && (back.y - 25.0).abs() < 1e-9,
            "{back:?}"
        );
        // Independent axis zooms
        let v = Viewport::orthographic(&OrthographicViewportOptions {
            width: 200.0,
            height: 100.0,
            zoom: 0.0,
            zoom_x: Some(2.0),
            zoom_y: Some(0.0),
            ..Default::default()
        });
        let p = v.project(DVec3::new(5.0, 5.0, 0.0), true);
        assert!((p.x - 120.0).abs() < 1e-9 && (p.y - 55.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn orbit_viewport_looks_at_the_target() {
        let target = DVec3::new(3.0, -2.0, 1.0);
        let v = Viewport::orbit(&OrbitViewportOptions {
            width: 300.0,
            height: 200.0,
            target,
            zoom: 2.0,
            rotation_x: 30.0,
            rotation_orbit: 45.0,
            ..Default::default()
        });
        let c = v.project(target, true);
        assert!((c.x - 150.0).abs() < 1e-6 && (c.y - 100.0).abs() < 1e-6, "{c:?}");
        // Unprojecting the centre without a depth lands back on the target
        let back = v.unproject(DVec2::new(150.0, 100.0), None, true, None);
        assert!((back - target).length() < 1e-6, "{back:?}");
        // One unit at the target is 2^zoom pixels, regardless of the viewport height
        let side = v.project(target + DVec3::new(0.0, 0.0, 1.0), true);
        let flat = Viewport::orbit(&OrbitViewportOptions {
            width: 300.0,
            height: 200.0,
            target,
            zoom: 2.0,
            ..Default::default()
        });
        let up = flat.project(target + DVec3::new(1.0, 0.0, 0.0), true);
        assert!((up.x - 154.0).abs() < 1e-6, "{up:?}");
        assert!(side.y < c.y, "z goes up on screen: {side:?}");
    }

    #[test]
    fn first_person_viewport_looks_along_the_bearing() {
        let v = Viewport::first_person(&FirstPersonViewportOptions {
            width: 100.0,
            height: 100.0,
            position: DVec3::new(0.0, 0.0, 2.0),
            bearing: 0.0,
            pitch: 0.0,
            ..Default::default()
        });
        // A point straight ahead (north) at eye height sits at the centre
        let p = v.project(DVec3::new(0.0, 10.0, 2.0), true);
        assert!((p.x - 50.0).abs() < 1e-6 && (p.y - 50.0).abs() < 1e-6, "{p:?}");
        // East is to the right, up is up
        let e = v.project(DVec3::new(1.0, 10.0, 2.0), true);
        assert!(e.x > 50.0, "{e:?}");
        let u = v.project(DVec3::new(0.0, 10.0, 3.0), true);
        assert!(u.y < 50.0, "{u:?}");
        // Facing east instead
        let v = Viewport::first_person(&FirstPersonViewportOptions {
            width: 100.0,
            height: 100.0,
            bearing: 90.0,
            ..Default::default()
        });
        let p = v.project(DVec3::new(10.0, 0.0, 0.0), true);
        assert!((p.x - 50.0).abs() < 1e-6 && (p.y - 50.0).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn globe_viewport_projects_onto_the_sphere() {
        let v = Viewport::globe(&GlobeViewportOptions {
            width: 400.0,
            height: 300.0,
            longitude: 10.0,
            latitude: 50.0,
            zoom: 1.0,
            ..Default::default()
        });
        assert_eq!(v.projection_mode(), ProjectionMode::Globe);
        // The target is at the centre of the screen and on the sphere
        let c = v.project(DVec3::new(10.0, 50.0, 0.0), true);
        assert!((c.x - 200.0).abs() < 1e-6 && (c.y - 150.0).abs() < 1e-6, "{c:?}");
        let common = v.project_position(DVec3::new(10.0, 50.0, 0.0));
        assert!((common.length() - GLOBE_RADIUS).abs() < 1e-9);
        let back = v.unproject_position(common);
        assert!(
            (back.x - 10.0).abs() < 1e-9 && (back.y - 50.0).abs() < 1e-9,
            "{back:?}"
        );
        // Unprojecting a pixel hits the sphere
        let hit = v.unproject(DVec2::new(230.0, 140.0), None, true, None);
        let round_trip = v.project(DVec3::new(hit.x, hit.y, 0.0), true);
        assert!(
            (round_trip.x - 230.0).abs() < 1e-6 && (round_trip.y - 140.0).abs() < 1e-6,
            "{round_trip:?}"
        );
        // North is up and east is right
        let north = v.project(DVec3::new(10.0, 55.0, 0.0), true);
        let east = v.project(DVec3::new(15.0, 50.0, 0.0), true);
        assert!(north.y < c.y && east.x > c.x, "{north:?} {east:?}");
        // Panning keeps a constant speed on screen and adjusts the zoom with latitude
        let [lng, lat, zoom] = v.globe_pan_by_position([10.0, 50.0, 1.0], [210.0, 150.0], [200.0, 150.0]);
        assert!(
            lng < 10.0 && (lat - 50.0).abs() < 1e-9 && (zoom - 1.0).abs() < 1e-9,
            "{lng} {lat} {zoom}"
        );
        let [_, lat2, zoom2] = v.globe_pan_by_position([10.0, 50.0, 1.0], [200.0, 100.0], [200.0, 150.0]);
        assert!(
            lat2 < 50.0 && zoom2 > 1.0,
            "moving south towards the equator raises the zoom: {lat2} {zoom2}"
        );
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

//! Port of `@deck.gl/core/src/lib/constants.ts`.

/// How positions in layer data are interpreted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoordinateSystem {
    /// `lnglat` for geospatial viewports, `cartesian` otherwise.
    #[default]
    Default,
    /// Longitude/latitude in degrees, elevation in meters. Dimensions are in meters.
    LngLat,
    /// [x, y, z] in meter offsets from the coordinate origin. Dimensions are in meters.
    MeterOffsets,
    /// deltaLng/deltaLat in degrees, elevation in meters. Dimensions are in meters.
    LngLatOffsets,
    /// Positions and dimensions are in the common units of the viewport.
    Cartesian,
}

impl CoordinateSystem {
    /// The integer the shader uses for this coordinate system.
    pub fn shader_value(self) -> i32 {
        match self {
            CoordinateSystem::Default => -1,
            CoordinateSystem::Cartesian => 0,
            CoordinateSystem::LngLat => 1,
            CoordinateSystem::MeterOffsets => 2,
            CoordinateSystem::LngLatOffsets => 3,
        }
    }
}

/// How coordinates are transformed from the world space into the common space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ProjectionMode {
    Identity = 0,
    WebMercator = 1,
    Globe = 2,
    /// Web Mercator with a coordinate origin near the viewport center for f32 precision.
    WebMercatorAutoOffset = 4,
}

impl ProjectionMode {
    pub fn shader_value(self) -> i32 {
        self as i32
    }
}

/// Units in which sizes (radius, width) are specified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Unit {
    Common = 0,
    #[default]
    Meters = 1,
    Pixels = 2,
}

impl Unit {
    pub fn shader_value(self) -> i32 {
        self as i32
    }
}

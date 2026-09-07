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

/// Depth range deck writes to the depth buffer.
///
/// deck.gl's projection matrices follow OpenGL, so the shaders remap clip z from `[-w, w]` to
/// WebGPU's `[0, w]`. A host that shares its depth buffer and writes OpenGL style depth values,
/// such as maplibre-native on Metal, needs the remap turned off so both sides compare equal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClipDepthRange {
    /// WebGPU convention, depth in `[0, 1]`.
    #[default]
    ZeroToOne,
    /// OpenGL convention, clip z in `[-w, w]`; geometry nearer than about twice the near
    /// plane is clipped, as it is in hosts that use this convention.
    NegativeOneToOne,
}

/// Where the host's framebuffer has its origin.
///
/// Every graphics API but OpenGL puts pixel (0, 0) at the top left of a render target, and
/// deck draws for that. An OpenGL default framebuffer has it at the bottom left, so a host
/// that hands deck one of those, a maplibre-gl-js custom layer for instance, wants the
/// layers the other way up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClipOrigin {
    /// WebGPU, Metal, Vulkan and D3D12, and any texture wgpu owns.
    #[default]
    TopLeft,
    /// An OpenGL or WebGL2 default framebuffer.
    BottomLeft,
}

impl ClipOrigin {
    pub fn shader_value(self) -> i32 {
        match self {
            ClipOrigin::TopLeft => 0,
            ClipOrigin::BottomLeft => 1,
        }
    }
}

impl ClipDepthRange {
    pub fn shader_value(self) -> i32 {
        match self {
            ClipDepthRange::ZeroToOne => 0,
            ClipDepthRange::NegativeOneToOne => 1,
        }
    }
}

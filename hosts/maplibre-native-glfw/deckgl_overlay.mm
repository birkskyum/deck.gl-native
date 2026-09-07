#include "deckgl_overlay.hpp"

#include <mln/math/angles.hpp>
#include <mln/util/constants.hpp>

#include <deckgl.h>

#include <Metal/Metal.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace deckgl_overlay {

namespace {

DeckglHandle* handle = nullptr;
bool creationFailed = false;
std::optional<DeckglCamera> pendingCamera;

bool envFlag(const char* name, bool defaultValue) {
    const char* value = std::getenv(name);
    if (!value || !*value) return defaultValue;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

} // namespace

bool enabled() {
    static const bool value = envFlag("DECKGL_OVERLAY", true);
    return value;
}

// Interleaved by default: deck depth tests against the map's depth buffer, so layers on the
// ground sit under 3D buildings, which requires matching the map's near and far planes.
// DECKGL_LOAD_DEPTH=0 clears depth first instead and lets deck keep its own planes, drawing
// everything on top of the map.
bool loadDepth() {
    static const bool value = envFlag("DECKGL_LOAD_DEPTH", true);
    return value;
}

void setCamera(const mln::CameraOptions& camera, uint32_t width, uint32_t height, float pixelRatio) {
    if (!enabled()) return;

    const double fov = camera.fov.value_or(mln::util::rad2deg(mln::util::DEFAULT_FOV));
    const double pitch = camera.pitch.value_or(0.0);

    // Same planes as the projection maplibre uses for its 3D layers (PaintParameters'
    // nearClippedProjMatrix: near plane a tenth of the camera distance, truncated to whole
    // pixels, far plane from TransformState::getProjMatrix), so that deck.gl's depth values are
    // comparable with the buildings'. Flat map layers use a 1 px near plane instead, which puts
    // their depth further away than anything deck draws at ground level.
    const double fovRadians = fov * M_PI / 180.0;
    const double pitchRadians = std::clamp(pitch, 0.0, 89.0) * M_PI / 180.0;
    const double cameraToCenterDistance = 0.5 * height / std::tan(fovRadians / 2.0);
    const double tanFovAboveCenter = std::tan(fovRadians / 2.0);
    const double tanMultiple = std::clamp(tanFovAboveCenter * std::tan(pitchRadians), 0.0, 0.99);
    const double furthestDistance = cameraToCenterDistance / (1.0 - tanMultiple);
    const double farZ = furthestDistance * 1.01;
    const double nearZ = std::max(1.0, std::floor(0.1 * cameraToCenterDistance));

    if (envFlag("DECKGL_DEBUG", false)) {
        static double lastNear = -1.0;
        if (nearZ != lastNear) {
            lastNear = nearZ;
            fprintf(stderr, "deck.gl overlay: size %ux%u ratio %.2f fov %.3f pitch %.2f near %.1f far %.1f px\n",
                    width, height, pixelRatio, fov, pitch, nearZ, farZ);
        }
    }

    DeckglCamera c{};
    c.longitude = camera.center ? camera.center->longitude() : 0.0;
    c.latitude = camera.center ? camera.center->latitude() : 0.0;
    c.zoom = camera.zoom.value_or(0.0);
    c.bearing = camera.bearing.value_or(0.0);
    c.pitch = pitch;
    c.fov_degrees = fov;
    c.near_z_pixels = loadDepth() ? nearZ : 0.0;
    c.far_z_pixels = loadDepth() ? farZ : 0.0;
    c.width = width;
    c.height = height;
    c.pixel_ratio = pixelRatio;

    pendingCamera = c;
    if (handle) {
        deckgl_set_camera(handle, &c);
    }
}

void render(MTL::Device* device,
            MTL::CommandQueue* queue,
            MTL::Texture* color,
            MTL::Texture* depth,
            uint32_t /*drawableWidth*/,
            uint32_t /*drawableHeight*/) {
    if (!enabled() || creationFailed || !color) return;

    if (!handle) {
        handle = deckgl_metal_create(device, queue);
        if (!handle) {
            creationFailed = true;
            return;
        }
        const char* spec = std::getenv("DECKGL_JSON");
        if (spec && *spec) {
            if (deckgl_load_json_file(handle, spec) != 0) {
                fprintf(stderr, "deck.gl overlay: %s\n", deckgl_last_error(handle));
            }
        } else {
            deckgl_load_demo_scene(handle);
        }
        if (pendingCamera) {
            deckgl_set_camera(handle, &*pendingCamera);
        }
    }

    if (deckgl_metal_render(handle, color, depth, loadDepth() ? 0 : 1) != 0) {
        // Errors are already logged by the library; keep going so the map still presents.
    }
}

} // namespace deckgl_overlay

#pragma once

// deck.gl-native overlay for the GLFW demo app (Metal backend).
//
// The map frame is committed first, then deck.gl layers are drawn into the same drawable and
// depth texture on the same command queue, then the drawable is presented.

#include <mln/map/camera.hpp>

#include <cstdint>

namespace MTL {
class Device;
class CommandQueue;
class Texture;
} // namespace MTL

namespace deckgl_overlay {

/// False when the DECKGL_OVERLAY environment variable is set to 0.
bool enabled();

/// Record the map camera for the next frame. `width` and `height` are the map size in points.
void setCamera(const mln::CameraOptions& camera, uint32_t width, uint32_t height, float pixelRatio);

/// Whether deck.gl is still loading data or tiles; the view keeps repainting while it is.
bool isLoading();

/// Draw the overlay. Creates the deck on first use. `depth` may be null.
void render(MTL::Device* device,
            MTL::CommandQueue* queue,
            MTL::Texture* color,
            MTL::Texture* depth,
            uint32_t drawableWidth,
            uint32_t drawableHeight);

} // namespace deckgl_overlay

/* deck.gl-native C API
 *
 * A host renderer creates a deck on its own GPU device, feeds it the camera every frame, and
 * asks it to draw into the host's color and depth attachments after the host has drawn the
 * basemap. Work is submitted on the host's command queue, so ordering follows commit order.
 */
#ifndef DECKGL_H
#define DECKGL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeckglHandle DeckglHandle;

/** Camera of the host map, in maplibre conventions. */
typedef struct DeckglCamera {
    double longitude;
    double latitude;
    double zoom;
    /** Degrees, clockwise from north */
    double bearing;
    /** Degrees */
    double pitch;
    /** Vertical field of view in degrees */
    double fov_degrees;
    /** Distances of the host's near and far planes in pixels, as used by its projection matrix.
     *  Set both to 0 to let deck choose its own planes. */
    double near_z_pixels;
    double far_z_pixels;
    /** Viewport size in logical (CSS) pixels */
    uint32_t width;
    uint32_t height;
    /** Physical pixels per logical pixel */
    float pixel_ratio;
} DeckglCamera;

/** Create a deck on an existing Metal device and command queue (id<MTLDevice>, id<MTLCommandQueue>).
 *  Returns NULL on failure. */
DeckglHandle* deckgl_metal_create(void* mtl_device, void* mtl_command_queue);

void deckgl_destroy(DeckglHandle* deck);

/** Replace the layers with the built-in demo scene (San Francisco). */
int32_t deckgl_load_demo_scene(DeckglHandle* deck);

void deckgl_set_camera(DeckglHandle* deck, const DeckglCamera* camera);

/** Draw all layers into the given id<MTLTexture> color and depth attachments. The color
 *  contents are loaded; depth is cleared when `clear_depth` is nonzero and loaded otherwise.
 *  Submits a command buffer on the queue passed at creation. Returns 0 on success. */
int32_t deckgl_metal_render(DeckglHandle* deck,
                            void* mtl_color_texture,
                            void* mtl_depth_texture,
                            int32_t clear_depth);

/** Last error message for this deck, or an empty string. Valid until the next call. */
const char* deckgl_last_error(DeckglHandle* deck);

#ifdef __cplusplus
}
#endif

#endif /* DECKGL_H */

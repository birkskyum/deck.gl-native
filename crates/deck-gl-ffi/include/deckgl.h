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
    /** Distances of the host's near and far planes in pixels, as used by the projection matrix
     *  of its 3D layers (maplibre-native: a tenth of the camera distance and the far plane of
     *  TransformState::getProjMatrix). Set both to 0 to let deck choose its own planes. */
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

/** Replace the layers with a JSON description in the @deck.gl/json (pydeck) format: either a
 *  description object with `layers`, or a bare array of layers. Relative `data`, `image` and
 *  `iconAtlas` paths resolve against `base_dir` when it is not NULL. Returns 0 on success;
 *  see deckgl_last_error otherwise. Unsupported props are reported on stderr and skipped. */
int32_t deckgl_set_layers_json(DeckglHandle* deck, const char* json, const char* base_dir);

/** Load a JSON description from a file. Relative paths inside resolve against its directory. */
int32_t deckgl_load_json_file(DeckglHandle* deck, const char* path);

/* Arrow C Data Interface structs, see https://arrow.apache.org/docs/format/CDataInterface.html */
struct ArrowSchema;
struct ArrowArray;

/** Register an Arrow table under a name. `array` must be a struct array whose fields are the
 *  columns. Ownership of `array` moves to the deck (its release callback runs when the table
 *  is replaced or the deck destroyed) and the caller's struct is marked released; `schema` is
 *  only read. Buffers are not copied. JSON layers use it as `"data": "@@table:<name>"` with
 *  column accessors such as `"getPosition": "@@column:geometry"` (FixedSizeList<f64, 2 or 3>)
 *  and `"getFillColor": "@@column:color"` (FixedSizeList<u8, 3 or 4>). Returns 0 on success. */
int32_t deckgl_set_arrow_table(DeckglHandle* deck,
                               const char* name,
                               const struct ArrowSchema* schema,
                               struct ArrowArray* array);

/** Forget a registered table. Layers already built keep their data. */
int32_t deckgl_remove_arrow_table(DeckglHandle* deck, const char* name);

/** Create a deck on a headless GPU device, on any platform, for tooling, tests and
 *  snapshots. Returns NULL without a GPU adapter. */
DeckglHandle* deckgl_headless_create(void);

/** Render the layers into `rgba`, which must hold width * height * 4 bytes (rows top to
 *  bottom, transparent background). Uses the camera from deckgl_set_camera, whose size must
 *  then match, or else the last JSON description's initialViewState at the given size.
 *  Works on any deck, headless or host bound. Returns 0 on success. */
int32_t deckgl_snapshot(DeckglHandle* deck, uint32_t width, uint32_t height, uint8_t* rgba);

/** Like deckgl_snapshot, written as a PNG file. Returns 0 on success. */
int32_t deckgl_snapshot_png(DeckglHandle* deck, uint32_t width, uint32_t height, const char* path);

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

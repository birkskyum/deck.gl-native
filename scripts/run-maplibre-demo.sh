#!/usr/bin/env bash
# Build and run the maplibre-native GLFW app with the deck.gl-native overlay (macOS, Metal).
#
# Usage: scripts/run-maplibre-demo.sh [path/to/maplibre-native] [extra mbgl-glfw args...]
#
# The maplibre-native checkout must contain the overlay hook (platform/glfw/deckgl_overlay.*
# and the MLN_DECKGL_OVERLAY option in platform/glfw/CMakeLists.txt).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MLN="${1:-$HOME/repos/maplibre-monorepo/maplibre-native}"
shift || true

echo "== building deck.gl-native C library"
(cd "$HERE" && cargo build --release -p deck-gl-ffi)

# The Metal backend needs the metal-cpp headers.
(cd "$MLN" && git submodule update --init vendor/metal-cpp)

# The macos preset uses ccache; skip it when the installed ccache cannot run.
LAUNCHER_ARGS=()
if ! ccache --version >/dev/null 2>&1; then
  echo "== ccache is not usable, building without it"
  LAUNCHER_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER=)
fi
# maplibre-tile-spec's FastPFOR needs simde on the include path.
SIMDE="$MLN/vendor/maplibre-tile-spec/cpp/vendor/simde"

echo "== configuring maplibre-native (macos-metal preset) with the overlay"
(cd "$MLN" && cmake --preset macos-metal \
  -DMLN_DECKGL_OVERLAY=ON \
  -DDECKGL_FFI_LIB="$HERE/target/release/libdeckgl.a" \
  -DDECKGL_FFI_INCLUDE="$HERE/crates/deck-gl-ffi/include" \
  "-DCMAKE_C_FLAGS=-I$SIMDE -DSIMDE_ENABLE_NATIVE_ALIASES" \
  "-DCMAKE_CXX_FLAGS=-I$SIMDE -DSIMDE_ENABLE_NATIVE_ALIASES" \
  "${LAUNCHER_ARGS[@]}")

echo "== building mbgl-glfw"
(cd "$MLN" && ninja -C build-macos-metal mbgl-glfw)

echo "== running"
exec "$MLN/build-macos-metal/platform/glfw/mbgl-glfw" \
  --style https://tiles.openfreemap.org/styles/liberty \
  --lon -122.42 --lat 37.775 --zoom 12.6 --pitch 50 --bearing -25 \
  "$@"

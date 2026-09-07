#!/usr/bin/env sh
# Builds the wasm module and its JavaScript bindings into www/pkg.
#
#   ./build.sh            WebGPU, the module the demos use
#   ./build.sh webgl      WebGL2, what the maplibre-gl-js custom layer needs
set -eu
cd "$(dirname "$0")"

case "${1:-webgpu}" in
  webgpu) features=""; name="deck_gl_web" ;;
  webgl)  features="--features webgl"; name="deck_gl_web_gl" ;;
  *) echo "usage: $0 [webgpu|webgl]" >&2; exit 2 ;;
esac

# The scenes are the repository's own, so the browser draws exactly what json_render does
mkdir -p www/scenes
for scene in osm-tiles san-francisco minimap data-filter; do
  cp "../json/$scene.json" www/scenes/
done

# shellcheck disable=SC2086
cargo build --release --target wasm32-unknown-unknown $features
"${WASM_BINDGEN:-wasm-bindgen}" target/wasm32-unknown-unknown/release/deck_gl_web.wasm \
  --out-dir www/pkg --out-name "$name" --target web --no-typescript

ls -lh "www/pkg/${name}_bg.wasm" | awk '{print $9, $5}'

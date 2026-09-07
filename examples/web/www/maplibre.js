// deck.gl-native as a maplibre-gl-js custom layer: the map hands its WebGL2 context to wgpu,
// which draws the layers into the framebuffer and depth buffer the map is already using.
// MapLibre GL JS 6 is ESM only and WebGL2 only, which is what this layer wants anyway.
// MapLibre's own ESM build, not a rebundled one: its worker only starts from this file.
// v6 has named exports rather than a default one.
import * as maplibregl from 'https://unpkg.com/maplibre-gl@6.7.0/dist/maplibre-gl.mjs'
import init, { create_overlay } from './pkg/deck_gl_web_gl.js'
const status = document.querySelector('#status')
const say = (message, isError = false) => {
  status.textContent = message
  status.classList.toggle('error', isError)
  if (isError) console.error(message)
}

await init()
const spec = await (await fetch('scenes/maplibre-arcs.json')).text()
const camera = JSON.parse(spec).initialViewState

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://tiles.openfreemap.org/styles/liberty',
  center: [camera.longitude, camera.latitude],
  zoom: camera.zoom,
  pitch: camera.pitch,
  bearing: camera.bearing,
  antialias: false, // a multisampled default framebuffer cannot be shared
})

const deckLayer = {
  id: 'deck-gl-native',
  type: 'custom',
  renderingMode: '3d',
  onAdd(map, gl) {
    this.map = map
    say('starting deck.gl-native in the map’s WebGL2 context')
    create_overlay(gl)
      .then((overlay) => {
        overlay.set_spec(spec)
        this.overlay = overlay
        const warnings = overlay.warnings
        say(warnings.length ? warnings.join('; ') : 'arcs drawn by deck.gl-native')
        map.triggerRepaint()
      })
      .catch((error) => say(String(error), true))
  },
  // `parameters` are the custom layer render parameters, which carry the map's own depth
  // planes. They are the public way to reach them: MapLibre 6 removed the private
  // `map.transform` that older integrations read. This is what @deck.gl/maplibre does too.
  render(gl, parameters, legacyParameters) {
    if (!this.overlay) return
    const camera = [parameters, legacyParameters].find(
      (p) => p && Number.isFinite(p.nearZ) && Number.isFinite(p.farZ),
    )
    if (!camera) {
      say('this MapLibre is too old to share its depth planes; needs 4.5.1 or later', true)
      this.overlay = null
      return
    }
    const center = this.map.getCenter()
    try {
      this.overlay.render(
        center.lng,
        center.lat,
        this.map.getZoom(),
        this.map.getPitch(),
        this.map.getBearing(),
        camera.nearZ,
        camera.farZ,
      )
    } catch (error) {
      say(String(error), true)
      this.overlay = null
      return
    }
    // wgpu left the context set up for its own drawing. maplibre keeps a cache of the GL
    // state and would otherwise go on believing its own, which shows up as symbols losing
    // their blending a frame later.
    gl.bindVertexArray(null)
    this.map.painter?.context?.setDirty?.()
  },
}

window.deckMap = map
map.on('load', () => {
  // Above the buildings in the layer order, but the depth buffer decides what is seen
  map.addLayer(deckLayer)
})
map.on('error', (event) => say(String(event.error ?? event), true))

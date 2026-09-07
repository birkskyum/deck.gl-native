// deck.gl-native as a maplibre-gl-js custom layer: the map hands its WebGL2 context to wgpu,
// which draws the layers into the framebuffer and depth buffer the map is already using.
import init, { create_overlay } from './pkg/deck_gl_web_gl.js'

const maplibregl = window.maplibregl
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
  render() {
    if (!this.overlay) return
    const center = this.map.getCenter()
    try {
      this.overlay.render(
        center.lng,
        center.lat,
        this.map.getZoom(),
        this.map.getPitch(),
        this.map.getBearing(),
      )
    } catch (error) {
      say(String(error), true)
      this.overlay = null
    }
  },
}

map.on('load', () => {
  // Above the buildings in the layer order, but the depth buffer decides what is seen
  map.addLayer(deckLayer)
})
map.on('error', (event) => say(String(event.error ?? event), true))

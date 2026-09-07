// Drives the wasm module: loads a scene, forwards pointer events to the Rust map controller
// and asks for a frame whenever something moved or is still loading.
// ?webgl loads the WebGL2 build instead, the one the maplibre-gl-js layer uses
const module = new URLSearchParams(location.search).has('webgl')
  ? await import('./pkg/deck_gl_web_gl.js')
  : await import('./pkg/deck_gl_web.js')
const { default: init, create } = module

const SCENES = [
  { file: 'osm-tiles.json', label: 'TileLayer: a raster basemap' },
  { file: 'san-francisco.json', label: 'San Francisco: polygons, paths, points, arcs, text' },
  { file: 'minimap.json', label: 'Two views of the same layers' },
  { file: 'data-filter.json', label: 'DataFilterExtension' },
]

// Keep everything the module logs so a failure can be read back from the page
window.deckLog = []
for (const level of ['log', 'warn', 'error']) {
  const original = console[level].bind(console)
  console[level] = (...args) => {
    window.deckLog.push(`${level}: ${args.map(String).join(' ')}`)
    original(...args)
  }
}
window.addEventListener('error', (event) => window.deckLog.push(`error: ${event.message}`))

const canvas = document.querySelector('#deck')
const status = document.querySelector('#status')
const picker = document.querySelector('#demo')

function say(message, isError = false) {
  status.textContent = message
  status.classList.toggle('error', isError)
  if (isError) console.error(message)
}

function resize(deck) {
  const ratio = window.devicePixelRatio || 1
  const width = Math.round(canvas.clientWidth * ratio)
  const height = Math.round(canvas.clientHeight * ratio)
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    deck.set_size(width, height)
  }
}

async function main() {
  await init()

  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.round(canvas.clientWidth * ratio)
  canvas.height = Math.round(canvas.clientHeight * ratio)

  let deck
  try {
    deck = await create(canvas)
  } catch (error) {
    say(String(error), true)
    return
  }
  document.querySelector('#backend').textContent =
    deck.backend === 'webgpu' ? 'WebGPU' : 'WebGL2'

  let dirty = true
  const invalidate = () => {
    dirty = true
  }

  // Pointer: drag pans, drag with shift or the right button turns and tilts
  let drag = null
  const pixel = (event) => {
    const rect = canvas.getBoundingClientRect()
    return [event.clientX - rect.left, event.clientY - rect.top]
  }
  canvas.addEventListener('pointerdown', (event) => {
    canvas.setPointerCapture(event.pointerId)
    const [x, y] = pixel(event)
    drag = event.shiftKey || event.button === 2 ? 'rotate' : 'pan'
    if (drag === 'pan') deck.pan_start(x, y, performance.now() / 1000)
    else deck.rotate_start(x, y)
    invalidate()
  })
  canvas.addEventListener('pointermove', (event) => {
    if (!drag) return
    const [x, y] = pixel(event)
    if (drag === 'pan') deck.pan(x, y, performance.now() / 1000)
    else deck.rotate(x, y)
    invalidate()
  })
  const endDrag = () => {
    if (!drag) return
    if (drag === 'pan') deck.pan_end(performance.now() / 1000)
    else deck.rotate_end()
    drag = null
    invalidate()
  }
  canvas.addEventListener('pointerup', endDrag)
  canvas.addEventListener('pointercancel', endDrag)
  canvas.addEventListener('contextmenu', (event) => event.preventDefault())
  canvas.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault()
      const [x, y] = pixel(event)
      deck.zoom_by(x, y, -event.deltaY * 0.01)
      invalidate()
    },
    { passive: false },
  )
  window.addEventListener('resize', invalidate)

  for (const scene of SCENES) {
    const option = document.createElement('option')
    option.value = scene.file
    option.textContent = scene.label
    picker.append(option)
  }

  async function load(file) {
    say(`loading ${file}`)
    const response = await fetch(`scenes/${file}`)
    if (!response.ok) {
      say(`${file}: HTTP ${response.status}`, true)
      return
    }
    try {
      deck.set_spec(await response.text())
    } catch (error) {
      say(String(error), true)
      return
    }
    const warnings = deck.warnings
    say(warnings.length ? warnings.join('; ') : `${file} drawn by deck.gl-native`)
    invalidate()
  }

  window.deck = deck // for poking at from the console
  picker.addEventListener('change', () => load(picker.value))
  await load(SCENES[0].file)

  function frame() {
    resize(deck)
    const moving = deck.tick(performance.now() / 1000)
    if (dirty || moving) {
      dirty = false
      try {
        const loading = deck.render()
        if (loading) dirty = true
      } catch (error) {
        say(String(error), true)
        return
      }
    }
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

main()

// deck.gl doing the same work as the native window: bin five million points into hexagons,
// then change the radius and bin them again. Changing the radius is what dragging a kepler.gl
// radius slider costs, and it is the number to compare.
//
// deck.gl is given its fastest path rather than its most convenient one. The coordinates come
// out of Arrow as one flat Float32Array and go in as a binary attribute, so no accessor is
// ever called and no row object is ever made, and the `data` object is built once and reused
// by reference so that changing the radius does not also re-upload forty megabytes.
//
// `deck` is the global from deck.gl's own scripting bundle, loaded by compare.html. That build
// is used rather than an ESM CDN because it is the one with the luma.gl adapter registered.
import { tableFromIPC } from 'https://esm.sh/apache-arrow@21'

const { Deck, HexagonLayer } = globalThis.deck
const status = document.querySelector('#status')
const output = document.querySelector('#output')
const say = (message) => (status.textContent = message)
const now = () => performance.now()
const lines = []
const log = (line) => {
  lines.push(line)
  output.textContent = lines.join('\n')
}

const parameters = new URLSearchParams(location.search)
const file = parameters.get('file') ?? 'data5m.arrow'
const radii = (parameters.get('radii') ?? '200,100,400,200').split(',').map(Number)
// deck.gl 9.1 grew a GPU aggregator for these layers; `?gpu=0` asks for the CPU one instead.
const gpuAggregation = parameters.get('gpu') !== '0'

const frame = () => new Promise((resolve) => requestAnimationFrame(resolve))
const positionsOf = (table) => table.getChild('geometry').data[0].children[0].values

function extent(value) {
  let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity
  for (let i = 0; i < value.length; i += 2) {
    const x = value[i], y = value[i + 1]
    if (x < west) west = x
    if (x > east) east = x
    if (y < south) south = y
    if (y > north) north = y
  }
  return [west, south, east, north]
}

// A tab the browser is not painting gets one animation frame a second, and deck.gl does its
// layer update on an animation frame, so every number below would be that throttle rather
// than the work. Better to say so than to report it.
async function framePeriod() {
  const marks = []
  for (let i = 0; i < 6; i++) marks.push(await frame())
  const gaps = marks.slice(1).map((at, i) => at - marks[i]).sort((a, b) => a - b)
  return gaps[Math.floor(gaps.length / 2)]
}

const layerFor = (data, radius) =>
  new HexagonLayer({ id: 'hexbin', data, radius, extruded: true, gpuAggregation })

class Throttled extends Error {
  constructor(period) {
    super(
      `This tab is getting one frame every ${period.toFixed(0)} ms, so the browser is not ` +
        'painting it. deck.gl updates its layers on an animation frame, so anything timed ' +
        'here would be that throttle rather than the work. Open the page in a visible window.'
    )
  }
}

async function guardPainted() {
  const period = await framePeriod()
  if (period > 100) throw new Throttled(period)
  return period
}

async function run() {
  say(`reading ${file}`)
  const bytes = new Uint8Array(await (await fetch(file)).arrayBuffer())
  let started = now()
  const table = tableFromIPC(bytes)
  const positions = positionsOf(table)
  const rows = table.numRows
  const readMs = now() - started

  const data = { length: rows, attributes: { getPosition: { value: positions, size: 2 } } }
  const [west, south, east, north] = extent(positions)
  const span = Math.max(Math.abs(east - west), Math.abs(north - south))

  const ready = new Promise((resolve) => {
    globalThis.deckInstance = new Deck({
      canvas: document.querySelector('#deck'),
      initialViewState: {
        longitude: (west + east) / 2,
        latitude: (south + north) / 2,
        zoom: Math.min(20, Math.max(0, Math.log2(360 / span))),
        pitch: 45,
        bearing: 0,
      },
      controller: true,
      layers: [],
      onLoad: resolve,
    })
  })
  await ready
  const view = globalThis.deckInstance

  // Long tasks are the frames the browser could not deliver, which is the freeze a person
  // sees. deck.gl spreads a radius change over several frames, so a single `onAfterRender`
  // lands on an arbitrary one and `setProps` returns before the work; waiting until the frame
  // loop goes quiet again is what someone watching the map settle is actually timing.
  let blocked = 0
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) blocked += entry.duration
  })
  try { observer.observe({ entryTypes: ['longtask'] }) } catch { /* not everywhere */ }

  const QUIET_MS = 500
  const GIVE_UP_MS = 60_000

  async function settle(slowFrameMs) {
    const from = now()
    let last = from
    let busyUntil = last
    while (now() - busyUntil < QUIET_MS) {
      if (now() - from > GIVE_UP_MS) throw new Error('the frame loop never went quiet')
      await frame()
      const at = now()
      if (at - last >= slowFrameMs) busyUntil = at
      last = at
    }
    return busyUntil
  }

  async function bin(radius) {
    // Re-checked every time: a tab can be scrolled out of view part way through
    const slowFrameMs = (await guardPainted()) * 2.5
    await settle(slowFrameMs)
    blocked = 0
    const started = now()
    view.setProps({ layers: [layerFor(data, radius)] })
    const busyUntil = await settle(slowFrameMs)
    const device = view.device
    if (device?.gl) device.gl.finish()
    else await device?.queue?.onSubmittedWorkDone?.()
    return { settled: Math.max(0, busyUntil - started), blocked }
  }

  log(`${file}: ${rows} rows, deck.gl ${globalThis.deck.VERSION ?? ''} on ${view.device?.type}`)
  log(`  read     ${readMs.toFixed(1).padStart(7)} ms   Arrow into a Float32Array`)
  log('')
  log('  hexagon binning, the cost of moving a radius slider:')
  log('                          settled       main thread')
  for (const [i, radius] of radii.entries()) {
    say(`binning at ${radius} m`)
    const { settled, blocked } = await bin(radius)
    // The GPU aggregator is the one holding a transform; the CPU one is plain objects
    const state = view.layerManager?.getLayers()?.[0]?.state
    const how = state?.aggregator?.aggregationTransform ? 'gpu' : 'cpu'
    const note = i === 0 ? '  (first, includes pipeline setup)' : ''
    log(
      `    radius ${String(radius).padStart(7)} m   ${settled.toFixed(0).padStart(7)} ms   ` +
        `${blocked.toFixed(0).padStart(7)} ms blocked   ${how}${note}`
    )
  }
  log('')
  log('  deck.gl-native, same file, same radii, `load_race --hexbin 200`:')
  log('    radius     200 m       183 ms   (first, includes pipeline setup)')
  log('    radius     100 m        81 ms')
  log('    radius     400 m        65 ms')
  log('    radius     200 m        68 ms')
  say('done, drag the map to compare interactivity')
}

run().catch((error) => {
  say(String(error))
  log(error?.stack ?? String(error))
  console.error(error)
})

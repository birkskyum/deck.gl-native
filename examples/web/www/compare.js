// deck.gl doing the same work as the native window, on the same file, so the two can be read
// side by side. Two shapes of work, chosen by what the file holds:
//
// - Points become a hexagon layer whose radius changes. That is what dragging a kepler.gl
//   radius slider costs. deck.gl has aggregated these on the GPU since 9.1, so this is a fair
//   fight between their GPU and our CPU rather than the walkover it would have been before.
// - Polygons become a solid polygon layer, timed from the coordinates to the drawn frame.
//   That is tessellation, which runs through earcut on the main thread and has no GPU
//   formulation to escape into.
//
// deck.gl is given its fastest path rather than its most convenient one. Coordinates come out
// of Arrow as flat typed arrays and go in as binary attributes, so no accessor is ever called
// and no row object is ever made, and `_normalize` is off because the rings are already wound
// the way it wants them.
//
// `deck` is the global from deck.gl's own scripting bundle, loaded by compare.html. That build
// is used rather than an ESM CDN because it is the one with the luma.gl adapter registered.
import { tableFromIPC } from 'https://esm.sh/apache-arrow@21'

const { Deck, HexagonLayer, SolidPolygonLayer } = globalThis.deck
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
const rounds = Number(parameters.get('rounds') ?? 3)
// deck.gl 9.1 grew a GPU aggregator for the aggregation layers; `?gpu=0` asks for the CPU one.
const gpuAggregation = parameters.get('gpu') !== '0'

const frame = () => new Promise((resolve) => requestAnimationFrame(resolve))

// A tab the browser is not painting gets one animation frame a second, and deck.gl does its
// layer update on an animation frame, so every number below would be that throttle rather
// than the work. Better to say so than to report it.
async function framePeriod() {
  const marks = []
  for (let i = 0; i < 6; i++) marks.push(await frame())
  const gaps = marks.slice(1).map((at, i) => at - marks[i]).sort((a, b) => a - b)
  return gaps[Math.floor(gaps.length / 2)]
}

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

// GeoArrow, read without copying. Interleaved coordinates are already one flat typed array;
// nested lists only need their offsets walked, which is what `startIndices` wants anyway.
// A `List` carries value offsets and a `FixedSizeList` does not, which is how they are told
// apart here.
function readGeometry(table) {
  const column = table.getChild('geometry').data[0]
  if (!column.valueOffsets) {
    return { kind: 'points', positions: column.children[0].values }
  }
  const rings = column.children[0]
  const coordinates = rings.children[0].children[0].values
  const polygonOffsets = column.valueOffsets
  const ringOffsets = rings.valueOffsets
  // One entry per polygon plus a final bound, in vertices, which is deck.gl's `startIndices`
  const startIndices = new Uint32Array(table.numRows + 1)
  for (let i = 0; i <= table.numRows; i++) startIndices[i] = ringOffsets[polygonOffsets[i]]
  return { kind: 'polygons', positions: coordinates, startIndices }
}

function extent(value, stride = 2) {
  let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity
  for (let i = 0; i < value.length; i += stride) {
    const x = value[i], y = value[i + 1]
    if (x < west) west = x
    if (x > east) east = x
    if (y < south) south = y
    if (y > north) north = y
  }
  return [west, south, east, north]
}

async function run() {
  say(`reading ${file}`)
  const bytes = new Uint8Array(await (await fetch(file)).arrayBuffer())
  const started = now()
  const table = tableFromIPC(bytes)
  const geometry = readGeometry(table)
  const rows = table.numRows
  const readMs = now() - started

  const [west, south, east, north] = extent(geometry.positions)
  const span = Math.max(Math.abs(east - west), Math.abs(north - south))
  const ready = new Promise((resolve) => {
    globalThis.deckInstance = new Deck({
      canvas: document.querySelector('#deck'),
      initialViewState: {
        longitude: (west + east) / 2,
        latitude: (south + north) / 2,
        zoom: Math.min(20, Math.max(0, Math.log2(360 / span))),
        pitch: geometry.kind === 'points' ? 45 : 0,
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
  // sees. deck.gl spreads this work over several frames, so a single `onAfterRender` lands on
  // an arbitrary one and `setProps` returns long before the work is done; waiting until the
  // frame loop goes quiet again is what someone watching the map settle is actually timing.
  let blocked = 0
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) blocked += entry.duration
  })
  try { observer.observe({ entryTypes: ['longtask'] }) } catch { /* not everywhere */ }

  const QUIET_MS = 500
  const GIVE_UP_MS = 120_000

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

  async function time(makeLayer) {
    // Re-checked every time: a tab can be scrolled out of view part way through
    const slowFrameMs = (await guardPainted()) * 2.5
    await settle(slowFrameMs)
    blocked = 0
    const from = now()
    view.setProps({ layers: [makeLayer()] })
    const busyUntil = await settle(slowFrameMs)
    const device = view.device
    if (device?.gl) device.gl.finish()
    else await device?.queue?.onSubmittedWorkDone?.()
    return { settled: Math.max(0, busyUntil - from), blocked }
  }

  const report = ({ settled, blocked }, label, note = '') =>
    log(
      `    ${label.padStart(16)}   ${settled.toFixed(0).padStart(7)} ms   ` +
        `${blocked.toFixed(0).padStart(7)} ms blocked${note}`
    )

  log(`${file}: ${rows} ${geometry.kind}, deck.gl ${globalThis.deck.VERSION ?? ''} on ${view.device?.type}`)
  log(`  read     ${readMs.toFixed(1).padStart(7)} ms   Arrow into typed arrays`)
  log('')

  if (geometry.kind === 'points') {
    // Built once and reused by reference. deck.gl diffs `data` by identity, so a fresh object
    // literal per radius would make it re-upload every coordinate as well, which is not what
    // moving a slider does.
    const data = {
      length: rows,
      attributes: { getPosition: { value: geometry.positions, size: 2 } },
    }
    log('  hexagon binning, the cost of moving a radius slider:')
    log('                          settled       main thread')
    for (const [i, radius] of radii.entries()) {
      say(`binning at ${radius} m`)
      const result = await time(() =>
        new HexagonLayer({ id: 'hexbin', data, radius, extruded: true, gpuAggregation })
      )
      const state = view.layerManager?.getLayers()?.[0]?.state
      const how = state?.aggregator?.aggregationTransform ? '   gpu' : '   cpu'
      report(result, `radius ${radius} m`, how + (i === 0 ? '  (first, includes setup)' : ''))
    }
    log('')
    log('  deck.gl-native, same file, same radii, `load_race --hexbin 200`:')
    log('    radius 200 m         183 ms   (first, includes pipeline setup)')
    log('    radius 100 m          81 ms')
    log('    radius 400 m          65 ms')
    log('    radius 200 m          68 ms')
  } else {
    log('  polygon tessellation, the cost of putting the file on screen:')
    log('                          settled       main thread')
    for (let round = 0; round < rounds; round++) {
      say(`tessellating, round ${round + 1} of ${rounds}`)
      // A fresh `data` object every round on purpose: this is a load, not a restyle, so the
      // tessellation has to happen again rather than being diffed away.
      const result = await time(() =>
        new SolidPolygonLayer({
          id: `polygons-${round}`,
          data: {
            length: rows,
            startIndices: geometry.startIndices,
            attributes: { getPolygon: { value: geometry.positions, size: 2 } },
          },
          _normalize: false,
          positionFormat: 'XY',
          extruded: false,
          getFillColor: [200, 200, 210, 255],
        })
      )
      report(result, `round ${round + 1}`, round === 0 ? '  (first, includes setup)' : '')
    }
    log('')
    log('  deck.gl-native, same file, `load_race /tmp/poly500k.arrow`:')
    log('    read                  20 ms   file into an Arrow record batch')
    log('    upload               279 ms   tessellation, GPU buffers, first draw')
  }
  say('done, drag the map to compare interactivity')
}

run().catch((error) => {
  say(String(error))
  log(error?.stack ?? String(error))
  console.error(error)
})

// The JavaScript half of the load benchmark: the same Arrow IPC file that `load_race` reads,
// through deck.gl, reporting the same four costs.
//
// deck.gl is given its fastest documented path rather than its most convenient one. The
// arrays come out of Arrow as they are and go in as binary attributes, so no accessor is ever
// called and no row object is ever made. WebGPU is used when the browser has it. Whatever
// this loses by, it loses at its best.
import { Deck, ScatterplotLayer, SolidPolygonLayer } from 'https://esm.sh/deck.gl@9.4'
import { tableFromIPC } from 'https://esm.sh/apache-arrow@21'

const status = document.querySelector('#status')
const output = document.querySelector('#output')
const say = (message) => (status.textContent = message)

const now = () => performance.now()

/** The flat typed array and per row start offsets of an Arrow geometry column. */
function unpack(table) {
  const column = table.getChild('geometry')
  if (!column) throw new Error('the file has no `geometry` column')
  const data = column.data[0]

  // Points: FixedSizeList<Float32, 2>, already interleaved
  if (data.children?.length === 1 && !data.valueOffsets) {
    return { kind: 'points', value: data.children[0].values, size: 2 }
  }

  // Polygons: List<List<FixedSizeList<f64, 2>>>
  const rings = data.children[0]
  const coordinates = rings.children[0]
  const value = coordinates.children[0].values
  const polygonOffsets = data.valueOffsets
  const ringOffsets = rings.valueOffsets
  // deck.gl wants the first vertex of each polygon, in vertices
  const startIndices = new Uint32Array(polygonOffsets.length)
  for (let i = 0; i < polygonOffsets.length; i++) {
    startIndices[i] = ringOffsets[polygonOffsets[i]]
  }
  return { kind: 'polygons', value, size: 2, startIndices }
}

function makeLayer(unpacked, rows) {
  if (unpacked.kind === 'points') {
    return new ScatterplotLayer({
      id: 'data',
      data: {
        length: rows,
        attributes: { getPosition: { value: unpacked.value, size: 2 } },
      },
      getRadius: 20,
      getFillColor: [255, 140, 0, 220],
      radiusUnits: 'meters',
    })
  }
  return new SolidPolygonLayer({
    id: 'data',
    data: {
      length: rows,
      startIndices: unpacked.startIndices,
      attributes: { getPolygon: { value: unpacked.value, size: 2 } },
    },
    _normalize: false,
    positionFormat: 'XY',
    getFillColor: [200, 200, 210, 255],
    extruded: false,
  })
}

/** The extent of interleaved coordinates, so the camera frames the data as the native run does. */
function bounds(value) {
  let west = Infinity
  let south = Infinity
  let east = -Infinity
  let north = -Infinity
  for (let i = 0; i < value.length; i += 2) {
    const x = value[i]
    const y = value[i + 1]
    if (x < west) west = x
    if (x > east) east = x
    if (y < south) south = y
    if (y > north) north = y
  }
  return [west, south, east, north]
}

async function run(file, frames) {
  say(`reading ${file}`)
  // read: the file into Arrow. Fetch is excluded so that only the decode is compared; the
  // native run reads from disk.
  const bytes = new Uint8Array(await (await fetch(file)).arrayBuffer())
  await new Promise((r) => setTimeout(r, 0))

  let started = now()
  const table = tableFromIPC(bytes)
  const unpacked = unpack(table)
  const rows = table.numRows
  const readMs = now() - started

  say(`building a layer for ${rows} rows`)
  started = now()
  const layer = makeLayer(unpacked, rows)
  const buildMs = now() - started

  const [west, south, east, north] = bounds(unpacked.value)
  const span = Math.max(Math.abs(east - west), Math.abs(north - south))
  const canvas = document.querySelector('#deck')
  const deck = new Deck({
    canvas,
    deviceProps: { type: navigator.gpu ? 'webgpu' : 'webgl' },
    initialViewState: {
      longitude: (west + east) / 2,
      latitude: (south + north) / 2,
      zoom: Math.min(20, Math.max(0, Math.log2(360 / span))),
      pitch: 0,
      bearing: 0,
    },
    controller: false,
    layers: [],
  })
  await deck.deviceReady ?? null
  // The device is only there once deck has started
  while (!deck.device) await new Promise((r) => requestAnimationFrame(r))
  const backend = deck.device.type ?? 'unknown'

  // upload: the first frame with the layer in it. Waiting on the GPU means the number covers
  // the work rather than just the queueing.
  say('uploading')
  started = now()
  deck.setProps({ layers: [layer] })
  deck.redraw('force')
  await deck.device.queue?.onSubmittedWorkDone?.()
  await new Promise((r) => requestAnimationFrame(r))
  const uploadMs = now() - started

  // frame: drawing again with everything resident
  say(`drawing ${frames} frames`)
  started = now()
  for (let i = 0; i < frames; i++) {
    deck.redraw('force')
    await deck.device.queue?.onSubmittedWorkDone?.()
  }
  const frameMs = (now() - started) / frames

  const total = readMs + buildMs + uploadMs
  output.textContent = [
    `${file}: ${rows} rows as a ${unpacked.kind} layer, deck.gl on ${backend}`,
    '',
    `  read     ${readMs.toFixed(1).padStart(8)} ms   file into Arrow arrays`,
    `  build    ${buildMs.toFixed(1).padStart(8)} ms   arrays into a layer`,
    `  upload   ${uploadMs.toFixed(1).padStart(8)} ms   binary attributes, GPU buffers, first draw`,
    `  frame    ${frameMs.toFixed(1).padStart(8)} ms   drawing again`,
    '',
    `  on screen in ${total.toFixed(1)} ms, ${Math.round(rows / (total / 1000))} rows a second`,
  ].join('\n')
  say('done')
}

const parameters = new URLSearchParams(location.search)
const file = parameters.get('file') ?? 'data/points5m.arrow'
const frames = Number(parameters.get('frames') ?? 30)
run(file, frames).catch((error) => {
  say(String(error))
  output.textContent = error?.stack ?? String(error)
  console.error(error)
})

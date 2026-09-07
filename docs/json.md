# JSON descriptions

`deck-gl-json` reads the JSON format of [`@deck.gl/json`](https://deck.gl/docs/api-reference/json/overview)
and [pydeck](https://pydeck.gl): a description with `initialViewState` and `layers`, where each
layer names its class with `@@type` and uses deck.gl's camelCase props. Descriptions written for
deck.gl's JSON playground or exported from pydeck work unchanged as long as the layers exist here.

```json
{
  "initialViewState": {"longitude": -122.42, "latitude": 37.775, "zoom": 12, "pitch": 45},
  "layers": [
    {
      "@@type": "ScatterplotLayer",
      "id": "stops",
      "data": [{"coordinates": [-122.47, 37.77], "passengers": 1500}],
      "pickable": true,
      "getPosition": "@@=coordinates",
      "getRadius": "@@=passengers / 10",
      "getFillColor": "@@=passengers > 2000 ? [255, 80, 80] : [255, 200, 80]"
    }
  ]
}
```

## Using it

```rust
let json = deck_gl_json::JsonConverter::parse_file("scene.json")?;
for warning in &json.warnings {
    eprintln!("{warning}");            // unknown layer types and props, like deck.gl's console warnings
}
let mut deck = Deck::new(&device, &queue, target, DeckProps {
    view_state: json.view_state.unwrap_or_default(),
    layers: json.layers,
    ..Default::default()
})?;
```

`JsonConverter::parse` takes text, `convert` takes a `serde_json::Value`, and both accept a bare
array of layers as well as a full description. `parse_file` resolves relative paths against the
file's directory; use `JsonConverter::with_base_dir` for text from elsewhere.

Every example binary reads `DECKGL_JSON=path/to/description.json` and shows that instead of the
built-in scene, and `json_render` renders a description straight to a PNG:

```sh
cargo run --release --bin json_render -- examples/json/san-francisco.json out.png 1600x1200
DECKGL_JSON=examples/json/vancouver-blocks.json cargo run --release --bin window
DECKGL_JSON=examples/json/heathrow-flights.json cargo run --release --manifest-path examples/maplibre-ffi/Cargo.toml
```

![heathrow-flights.json rendered by json_render](images/json-heathrow-flights.png)

*deck.gl's LineLayer website example, from `examples/json/heathrow-flights.json`.*

Over the C API, `deckgl_set_layers_json(deck, json, base_dir)` and
`deckgl_load_json_file(deck, path)` do the same for a host renderer; maplibre-native's GLFW
overlay reads `DECKGL_JSON` too.

## Data

`data` is an inline array of rows, a GeoJSON object (its features become the rows), or a
string naming a local file or an `http(s)` URL of either. URLs need the crate's default `fetch`
feature. `image` (BitmapLayer) and `iconAtlas` (IconLayer) are PNG or JPEG files or URLs, and
`iconMapping` is an inline object or a JSON file.

## Accessors and expressions

An accessor prop is either a constant (`"getRadius": 5`, `"getFillColor": [255, 0, 0]`) or a
string starting with `@@=`, which is an expression evaluated for every row. The expression
language is the one `@deck.gl/json` uses (jsep): field access with `.` and `[]`, array literals,
arithmetic, comparisons, `&&`, `||`, `!`, the `?:` conditional, and JavaScript coercion rules.
Function calls are rejected. `@@=-` and `this` refer to the row itself.

```
"getPosition": "@@=coordinates"
"getPosition": "@@=[lng, lat, altitude || 0]"
"getElevation": "@@=properties.valuePerSqm"
"getFillColor": "@@=properties.growth > 0.2 ? [255, 120, 60] : [199, 233, 180]"
"getIcon": "@@='marker-' + kind"
```

Expressions are evaluated once when the description is converted, and every row must produce
a value of the prop's type (a position is 2 or 3 numbers, a color 3 or 4 channels in 0..255, a
polygon a ring or an array of rings). Errors name the layer, the prop and the row.

Enumerations use `@@#`: `"coordinateSystem": "@@#COORDINATE_SYSTEM.METER_OFFSETS"` (the plain
number deck.gl uses also works). Units are strings: `"radiusUnits": "pixels"`.

## Supported layers and props

| Layer | Props |
| --- | --- |
| all layers | `id`, `visible`, `opacity`, `pickable`, `coordinateSystem`, `coordinateOrigin`, `modelMatrix`, `wrapLongitude`, `highlightColor`, `highlightedObjectIndex` |
| `ScatterplotLayer` | `radiusUnits`, `radiusScale`, `radiusMinPixels`, `radiusMaxPixels`, `lineWidthUnits`, `lineWidthScale`, `lineWidthMinPixels`, `lineWidthMaxPixels`, `stroked`, `filled`, `billboard`, `antialiasing`, `getPosition`, `getRadius`, `getFillColor`, `getLineColor`, `getLineWidth`, `getPixelOffset` |
| `LineLayer` | `widthUnits`, `widthScale`, `widthMinPixels`, `widthMaxPixels`, `getSourcePosition`, `getTargetPosition`, `getColor`, `getWidth` |
| `ArcLayer` | `greatCircle`, `numSegments`, `widthUnits`, `widthScale`, `widthMinPixels`, `widthMaxPixels`, `getSourcePosition`, `getTargetPosition`, `getSourceColor`, `getTargetColor`, `getWidth`, `getHeight`, `getTilt` |
| `PathLayer` | `widthUnits`, `widthScale`, `widthMinPixels`, `widthMaxPixels`, `jointRounded`, `capRounded`, `miterLimit`, `billboard`, `getPath`, `getColor`, `getWidth` |
| `SolidPolygonLayer` | `filled`, `extruded`, `wireframe`, `elevationScale`, `getPolygon`, `getElevation`, `getFillColor`, `getLineColor` |
| `PolygonLayer` | `stroked`, `filled`, `extruded`, `wireframe`, `elevationScale`, `lineWidthUnits`, `lineWidthScale`, `lineWidthMinPixels`, `lineWidthMaxPixels`, `lineJointRounded`, `lineMiterLimit`, `getPolygon`, `getFillColor`, `getLineColor`, `getLineWidth`, `getElevation` |
| `GeoJsonLayer` | `filled`, `stroked`, `extruded`, `wireframe`, `elevationScale`, `lineWidthUnits`, `lineWidthScale`, `lineWidthMinPixels`, `lineWidthMaxPixels`, `lineJointRounded`, `lineCapRounded`, `lineMiterLimit`, `pointRadiusUnits`, `pointRadiusScale`, `pointRadiusMinPixels`, `pointRadiusMaxPixels`, `getFillColor`, `getLineColor`, `getLineWidth`, `getPointRadius`, `getElevation` |
| `ColumnLayer` | `diskResolution`, `radius`, `angle`, `offset`, `coverage`, `elevationScale`, `radiusUnits`, `lineWidthUnits`, `lineWidthScale`, `lineWidthMinPixels`, `lineWidthMaxPixels`, `extruded`, `wireframe`, `filled`, `stroked`, `getPosition`, `getFillColor`, `getLineColor`, `getLineWidth`, `getElevation` |
| `HexagonLayer`, `GridLayer` | `radius` / `cellSize`, `getPosition`, `getColorWeight`, `getElevationWeight`, `colorAggregation`, `elevationAggregation` (`SUM`, `MEAN`, `MIN`, `MAX`, `COUNT`), `colorRange`, `colorDomain`, `colorScaleType`, `elevationDomain`, `elevationRange`, `elevationScale`, `elevationScaleType`, `lowerPercentile`, `upperPercentile`, `elevationLowerPercentile`, `elevationUpperPercentile`, `extruded`, `coverage` (CPU aggregation; `gpuAggregation` and `material` are accepted and ignored) |
| `GridCellLayer` | `cellSize`, `coverage`, `elevationScale`, `extruded`, `getPosition`, `getFillColor`, `getElevation` |
| `PointCloudLayer` | `sizeUnits`, `pointSize`, `getPosition`, `getNormal`, `getColor` |
| `TextLayer` | `getText`, `getPosition`, `getColor`, `getSize`, `getAngle`, `getTextAnchor`, `getAlignmentBaseline`, `getPixelOffset`, `sizeScale`, `sizeUnits`, `sizeMinPixels`, `sizeMaxPixels`, `billboard`, `background`, `getBackgroundColor`, `getBorderColor`, `getBorderWidth`, `backgroundPadding`, `backgroundBorderRadius`, `characterSet` (`"auto"`, a string or an array), `fontFamily` (a `.ttf`/`.otf` path or URL; CSS families fall back to the bundled Roboto Mono), `fontSettings` (`sdf`, `fontSize`, `buffer`, `radius`, `cutoff`, `smoothing`), `lineHeight`, `outlineWidth`, `outlineColor`, `wordBreak`, `maxWidth` |
| `IconLayer` | `iconAtlas`, `iconMapping`, `sizeUnits`, `sizeScale`, `sizeMinPixels`, `sizeMaxPixels`, `billboard`, `alphaCutoff`, `getPosition`, `getIcon`, `getColor`, `getSize`, `getAngle`, `getPixelOffset` |
| `BitmapLayer` | `image`, `bounds` (`[left, bottom, right, top]` or four corners), `desaturate`, `transparentColor`, `tintColor` |

Props deck.gl accepts but this port does not have yet (`extensions`, `parameters`, `transitions`,
`material`, `pointType`, ...) produce a warning and are skipped, as are layer types that do not
exist here yet. Callbacks such as `onHover` and `updateTriggers` are ignored silently since they
have no meaning in a static description.

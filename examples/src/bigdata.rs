//! Opening a large columnar file as a layer.
//!
//! Arrow IPC and Parquet hold the coordinates in the layout the GPU wants, so a file becomes a
//! layer by moving the record batch in rather than converting it. `load_race` measures that
//! and `window` flies through it.

use std::error::Error;
use std::path::Path;

use arrow_array::RecordBatch;
use deck_gl::{Accessor, Layer, LayerData, LayerProps, Unit};
use deck_gl_layers::{ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer, SolidPolygonLayerProps};

/// Whether [`open`] would read this path, rather than it being a JSON description.
pub fn is_data_file(path: &str) -> bool {
    matches!(extension(path).as_str(), "arrow" | "ipc" | "feather" | "parquet")
}

fn extension(path: &str) -> String {
    Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// Read a columnar file into one record batch. Parquet is decoded; Arrow IPC is already the
/// buffers, so this is close to the read itself.
pub fn read_batch(path: &str) -> Result<RecordBatch, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    match extension(path).as_str() {
        "parquet" => Ok(deck_gl_json::geo::read_parquet(bytes)?),
        "arrow" | "ipc" | "feather" => {
            let reader = arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None)?;
            let schema = reader.schema();
            let batches = reader.collect::<Result<Vec<_>, _>>()?;
            Ok(arrow_select::concat::concat_batches(&schema, &batches)?)
        }
        other => Err(format!("`{other}` files are not read here, use arrow or parquet").into()),
    }
}

/// The geometry column of a batch: what GeoParquet names, or the first that looks like one.
pub fn geometry_column(batch: &RecordBatch) -> Option<String> {
    if let Some((column, _)) = deck_gl_json::geo::geoparquet_column(batch.schema_ref()) {
        if batch.schema_ref().column_with_name(&column).is_some() {
            return Some(column);
        }
    }
    ["geometry", "geom", "position", "coordinates"]
        .into_iter()
        .find(|name| batch.schema_ref().column_with_name(name).is_some())
        .map(str::to_string)
}

/// Interleaved or separated coordinates draw as points; everything else as polygons.
pub fn guess_layer(batch: &RecordBatch, column: &str) -> String {
    use arrow_schema::DataType;
    let Some((_, field)) = batch.schema_ref().column_with_name(column) else {
        return "polygon".to_string();
    };
    match field.data_type() {
        DataType::FixedSizeList(_, _) | DataType::Struct(_) => "scatterplot".to_string(),
        _ => "polygon".to_string(),
    }
}

/// Build a layer around a record batch. The batch is moved in as it is, which is the whole
/// point: there is no row by row conversion between a columnar file and the GPU.
pub fn layer_from_batch(
    batch: RecordBatch,
    column: &str,
    kind: &str,
) -> Result<Box<dyn Layer>, Box<dyn Error>> {
    Ok(match kind {
        "scatterplot" => Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new("data"),
            data: LayerData::from_batch(batch),
            get_position: Accessor::column(column),
            get_radius: Accessor::Constant(20.0),
            get_fill_color: Accessor::Constant([255, 140, 0, 220]),
            radius_units: Unit::Meters,
            radius_min_pixels: 1.0,
            antialiasing: false,
            ..Default::default()
        })),
        "polygon" => Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
            base: LayerProps::new("data"),
            data: LayerData::from_batch(batch),
            get_polygon: Accessor::column(column),
            get_fill_color: Accessor::Constant([200, 200, 210, 255]),
            extruded: false,
            ..Default::default()
        })),
        other => return Err(format!("unknown layer `{other}`, use polygon or scatterplot").into()),
    })
}

/// A file opened as a layer: the layer, how many rows it holds and which kind it is.
pub struct Opened {
    pub layer: Box<dyn Layer>,
    pub rows: usize,
    pub kind: String,
}

/// A file as a layer.
pub fn open(path: &str) -> Result<Opened, Box<dyn Error>> {
    let batch = read_batch(path)?;
    let rows = batch.num_rows();
    let column = geometry_column(&batch).ok_or("the file has no geometry column")?;
    let kind = guess_layer(&batch, &column);
    let layer = layer_from_batch(batch, &column, &kind)?;
    Ok(Opened { layer, rows, kind })
}

/// A zoom that fits `[west, south, east, north]` into a square viewport.
pub fn fit_zoom(bounds: [f64; 4]) -> f64 {
    let span = (bounds[2] - bounds[0]).abs().max((bounds[3] - bounds[1]).abs());
    if span <= 0.0 {
        return 12.0;
    }
    (360.0 / span).log2().clamp(0.0, 20.0)
}

pub fn union(a: Option<[f64; 4]>, b: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (some, None) | (None, some) => some,
    }
}

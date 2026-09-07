//! How long it takes to get a large file on screen.
//!
//! ```sh
//! cargo run --release --bin load_race -- buildings.parquet
//! cargo run --release --bin load_race -- points.parquet --layer scatterplot --frames 120
//! ```
//!
//! Reports the four costs between a file on disk and a frame on screen, so they can be
//! compared with the same file in a browser:
//!
//! - **read**: the file into an Arrow record batch, which for Parquet is the decode
//! - **build**: the record batch into a layer, which is free when the columns are already
//!   what the GPU wants
//! - **upload**: the first frame, which resolves the accessors, tessellates, creates the GPU
//!   buffers and draws
//! - **frame**: drawing again with everything resident
//!
//! Rendering is headless and every frame is read back to the CPU, so the frame times are an
//! upper bound: a window that only presents is faster.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use deck_gl::luma_gl::device::create_headless_context;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Accessor, Deck, DeckProps, LayerData, LayerProps, ViewState};
use deck_gl_layers::{ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer, SolidPolygonLayerProps};

const SIZE: u32 = 1024;
const CLEAR: deck_gl::wgpu::Color = deck_gl::wgpu::Color {
    r: 0.02,
    g: 0.03,
    b: 0.06,
    a: 1.0,
};

fn main() -> Result<(), Box<dyn Error>> {
    deck_gl_examples::init_logging();
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!(
            "usage: load_race <file.parquet|file.arrow> [--layer polygon|scatterplot] \
             [--frames N] [--column NAME]"
        );
        std::process::exit(2);
    };
    let mut layer_kind = String::new();
    let mut frames = 60usize;
    let mut column = String::new();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--layer" => layer_kind = args.next().unwrap_or_default(),
            "--frames" => frames = args.next().unwrap_or_default().parse()?,
            "--column" => column = args.next().unwrap_or_default(),
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    // read: the file into Arrow. Nothing is converted into rows of objects on the way.
    let started = Instant::now();
    let bytes = std::fs::read(&path)?;
    let file_bytes = bytes.len();
    let batch = read_batch(&path, bytes)?;
    let read_ms = ms(started);
    let rows = batch.num_rows();
    let arrow_bytes = batch.get_array_memory_size();

    let column = if column.is_empty() {
        geometry_column(&batch).ok_or("no geometry column; name one with --column")?
    } else {
        column
    };
    let kind = if layer_kind.is_empty() {
        guess_layer(&batch, &column)
    } else {
        layer_kind.clone()
    };
    // build: the batch into a layer. The batch is moved in as it is.
    let started = Instant::now();
    let layer: Box<dyn deck_gl::Layer> = match kind.as_str() {
        "scatterplot" => Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
            base: LayerProps::new("data"),
            data: LayerData::from_batch(batch),
            get_position: Accessor::column(&column),
            get_radius: Accessor::Constant(20.0),
            get_fill_color: Accessor::Constant([255, 140, 0, 220]),
            radius_units: deck_gl::Unit::Meters,
            ..Default::default()
        })),
        "polygon" => Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
            base: LayerProps::new("data"),
            data: LayerData::from_batch(batch),
            get_polygon: Accessor::column(&column),
            get_fill_color: Accessor::Constant([200, 200, 210, 255]),
            extruded: false,
            ..Default::default()
        })),
        other => return Err(format!("unknown layer `{other}`, use polygon or scatterplot").into()),
    };
    let build_ms = ms(started);

    let ctx = create_headless_context()?;
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            layers: vec![layer],
            ..Default::default()
        },
    )?;

    // upload: the first frame. Accessors, tessellation, GPU buffers and a draw.
    let started = Instant::now();
    deck.snapshot(Some(CLEAR))?;
    let upload_ms = ms(started);
    let uploaded = deck.stats().uploaded_bytes;

    // The layer knows its extent once it has updated, so the camera can frame the data for
    // the frames that are timed and for the picture at the end.
    let bounds = deck.layers().fold(None, |all, layer| union(all, layer.bounds()));
    if let Some(b) = bounds {
        deck.set_view_state(ViewState {
            longitude: (b[0] + b[2]) / 2.0,
            latitude: (b[1] + b[3]) / 2.0,
            zoom: fit_zoom(b),
            pitch: 0.0,
            bearing: 0.0,
        });
    }

    // frame: everything is resident, so this is the drawing alone (plus the readback).
    let started = Instant::now();
    for _ in 0..frames.max(1) {
        deck.snapshot(Some(CLEAR))?;
    }
    let frame_ms = ms(started) / frames.max(1) as f64;

    let name = Path::new(&path).file_name().unwrap_or_default().to_string_lossy();
    println!();
    println!("{name}: {rows} rows as a {kind} layer on `{column}`");
    println!("  file            {:>10}", bytes_human(file_bytes));
    println!("  arrow in memory {:>10}", bytes_human(arrow_bytes));
    println!("  uploaded to GPU {:>10}", bytes_human(uploaded as usize));
    println!();
    println!("  read     {read_ms:>8.1} ms   file into an Arrow record batch");
    println!("  build    {build_ms:>8.1} ms   record batch into a layer");
    println!("  upload   {upload_ms:>8.1} ms   accessors, tessellation, GPU buffers, first draw");
    println!("  frame    {frame_ms:>8.1} ms   drawing again, including a full frame readback");
    println!();
    println!(
        "  on screen in {:.1} ms, {:.0} rows a second",
        read_ms + build_ms + upload_ms,
        rows as f64 / ((read_ms + build_ms + upload_ms) / 1000.0)
    );
    Ok(())
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

fn bytes_human(bytes: usize) -> String {
    const UNITS: [&str; 4] = ["B", "kB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn read_batch(path: &str, bytes: Vec<u8>) -> Result<arrow_array::RecordBatch, Box<dyn Error>> {
    let extension = Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "parquet" => Ok(deck_gl_json::geo::read_parquet(bytes)?),
        "arrow" | "ipc" | "feather" => {
            let reader = arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None)?;
            let schema = reader.schema();
            let batches = reader.collect::<Result<Vec<_>, _>>()?;
            Ok(arrow_select::concat::concat_batches(&schema, &batches)?)
        }
        other => Err(format!("`{other}` files are not read here, use parquet or arrow").into()),
    }
}

/// The geometry column of a batch: what GeoParquet names, or the first column that looks like
/// geometry.
fn geometry_column(batch: &arrow_array::RecordBatch) -> Option<String> {
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

/// Points draw as a scatterplot and anything else as polygons. A GeoArrow point column is a
/// fixed size list or a struct of coordinates; everything else here is treated as polygons.
fn guess_layer(batch: &arrow_array::RecordBatch, column: &str) -> String {
    use arrow_schema::DataType;
    let Some(field) = batch.schema_ref().column_with_name(column).map(|(_, f)| f) else {
        return "polygon".to_string();
    };
    match field.data_type() {
        DataType::FixedSizeList(_, _) | DataType::Struct(_) => "scatterplot".to_string(),
        _ => "polygon".to_string(),
    }
}

fn union(a: Option<[f64; 4]>, b: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (some, None) | (None, some) => some,
    }
}

/// A zoom that fits `[west, south, east, north]` into the square viewport.
fn fit_zoom(bounds: [f64; 4]) -> f64 {
    let span = (bounds[2] - bounds[0]).abs().max((bounds[3] - bounds[1]).abs());
    if span <= 0.0 {
        return 12.0;
    }
    (360.0 / span).log2().clamp(0.0, 20.0)
}

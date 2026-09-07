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
use deck_gl_examples::bigdata;
use deck_gl_layers::aggregation::AggregationProps;
use deck_gl_layers::{HexagonLayer, HexagonLayerProps};

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
    let mut hexbin: Option<f64> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--layer" => layer_kind = args.next().unwrap_or_default(),
            "--frames" => frames = args.next().unwrap_or_default().parse()?,
            "--column" => column = args.next().unwrap_or_default(),
            "--hexbin" => hexbin = args.next().unwrap_or_default().parse().ok(),
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    // read: the file into Arrow. Nothing is converted into rows of objects on the way.
    let file_bytes = std::fs::metadata(&path)?.len() as usize;
    let started = Instant::now();
    let batch = bigdata::read_batch(&path)?;
    let read_ms = ms(started);
    let rows = batch.num_rows();
    let arrow_bytes = batch.get_array_memory_size();

    let column = if column.is_empty() {
        bigdata::geometry_column(&batch).ok_or("no geometry column; name one with --column")?
    } else {
        column
    };
    let kind = if layer_kind.is_empty() {
        bigdata::guess_layer(&batch, &column)
    } else {
        layer_kind.clone()
    };
    // Arrow arrays are reference counted, so keeping a handle for the restyle below copies
    // nothing; it is the same buffers.
    let again = batch.clone();
    let for_hexbin = batch.clone();
    // build: the batch into a layer. The batch is moved in as it is.
    let started = Instant::now();
    let layer = bigdata::layer_from_batch(batch, &column, &kind)?;
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
    let bounds = deck
        .layers()
        .fold(None, |all, layer| bigdata::union(all, layer.bounds()));
    if let Some(b) = bounds {
        deck.set_view_state(ViewState {
            longitude: (b[0] + b[2]) / 2.0,
            latitude: (b[1] + b[3]) / 2.0,
            zoom: bigdata::fit_zoom(b),
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

    // restyle: the data stays where it is and the styling changes. This is what working with
    // a resident dataset costs, as against loading it again.
    let started = Instant::now();
    let restyled = bigdata::layer_from_batch(again, &column, &kind)?;
    deck.set_layers(vec![restyled]);
    deck.snapshot(Some(CLEAR))?;
    let restyle_ms = ms(started);

    // Binning is the work a kepler.gl style hexagon layer does every time its radius moves:
    // a projection and a bin lookup for every single row.
    if let Some(radius) = hexbin {
        println!();
        println!("  hexagon binning, the cost of moving a radius slider:");
        for scale in [1.0, 0.5, 2.0] {
            let props = HexagonLayerProps {
                base: LayerProps::new("hexbin"),
                data: LayerData::from_batch(for_hexbin.clone()),
                radius: radius * scale,
                aggregation: AggregationProps {
                    get_position: Accessor::column(&column),
                    ..Default::default()
                },
            };
            let started = Instant::now();
            let aggregation = HexagonLayer::aggregate(&props)?;
            println!(
                "    radius {:>7.0} m   {:>8.1} ms   {} bins",
                radius * scale,
                ms(started),
                aggregation.bins.len()
            );
        }
    }

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
    println!("  restyle  {restyle_ms:>8.1} ms   new styling on the same resident data");
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

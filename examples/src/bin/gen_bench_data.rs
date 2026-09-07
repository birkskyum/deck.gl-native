//! Writes the Arrow IPC files the load benchmark and its JavaScript counterpart both read.
//!
//! ```sh
//! cargo run --release --bin gen_bench_data -- points 5000000 /tmp/points5m.arrow
//! cargo run --release --bin gen_bench_data -- polygons 500000 /tmp/poly500k.arrow
//! ```
//!
//! Arrow IPC rather than Parquet so that both sides read the same bytes with no decode and no
//! conversion: the file holds exactly the buffers the GPU wants. deck.gl JS reads it with
//! `apache-arrow` and passes the arrays straight through as binary attributes, which is the
//! fastest path it documents, so the comparison is against its best rather than its default.
//!
//! - `points` writes `geometry` as GeoArrow interleaved coordinates, `FixedSizeList<Float32,
//!   2>`, which is one flat `Float32Array` on either side.
//! - `polygons` writes GeoArrow polygons, `List<List<FixedSizeList<Float64, 2>>>`, as squares
//!   of eight vertices. Neither side can avoid tessellating them, which is the point.

use std::error::Error;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Float64Builder, ListBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

/// Around San Francisco, so the two demos look at the same place.
const WEST: f64 = -122.6;
const SOUTH: f64 = 37.65;
const SPAN_X: f64 = 0.4;
const SPAN_Y: f64 = 0.3;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let kind = args.next().unwrap_or_default();
    let count: usize = args.next().unwrap_or_default().parse().unwrap_or(0);
    let path = args.next().unwrap_or_default();
    if count == 0 || path.is_empty() {
        eprintln!("usage: gen_bench_data <points|polygons> <count> <out.arrow>");
        std::process::exit(2);
    }

    let batch = match kind.as_str() {
        "points" => points(count),
        "polygons" => polygons(count),
        other => return Err(format!("unknown kind `{other}`, use points or polygons").into()),
    };

    let file = std::fs::File::create(&path)?;
    let mut writer = arrow_ipc::writer::FileWriter::try_new(file, batch.schema_ref())?;
    writer.write(&batch)?;
    writer.finish()?;
    let bytes = std::fs::metadata(&path)?.len();
    println!("{path}: {count} {kind}, {:.1} MB", bytes as f64 / 1_000_000.0);
    Ok(())
}

/// A deterministic value in `[0, 1)`. A fixed sequence so both runs draw the same picture.
fn noise(i: usize, salt: u64) -> f64 {
    let mut x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn points(count: usize) -> RecordBatch {
    let mut builder = FixedSizeListBuilder::with_capacity(Float32Builder::new(), 2, count);
    for i in 0..count {
        builder
            .values()
            .append_value((WEST + noise(i, 1) * SPAN_X) as f32);
        builder
            .values()
            .append_value((SOUTH + noise(i, 2) * SPAN_Y) as f32);
        builder.append(true);
    }
    let array = Arc::new(builder.finish()) as ArrayRef;
    let field = Field::new("geometry", array.data_type().clone(), false);
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).expect("points batch")
}

fn polygons(count: usize) -> RecordBatch {
    // A polygon is a list of rings; a ring is a list of interleaved coordinates.
    let coordinates = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    let ring = ListBuilder::new(coordinates);
    let mut builder = ListBuilder::with_capacity(ring, count);
    // Eight vertices, so the tessellation is real work rather than a single triangle
    let size = 0.0004;
    let corners: [(f64, f64); 8] = [
        (0.0, 1.0),
        (0.7, 0.7),
        (1.0, 0.0),
        (0.7, -0.7),
        (0.0, -1.0),
        (-0.7, -0.7),
        (-1.0, 0.0),
        (-0.7, 0.7),
    ];
    for i in 0..count {
        let x = WEST + noise(i, 3) * SPAN_X;
        let y = SOUTH + noise(i, 4) * SPAN_Y;
        let ring = builder.values();
        for (dx, dy) in corners {
            ring.values().values().append_value(x + dx * size);
            ring.values().values().append_value(y + dy * size);
            ring.values().append(true);
        }
        // Rings close on themselves
        ring.values().values().append_value(x + corners[0].0 * size);
        ring.values().values().append_value(y + corners[0].1 * size);
        ring.values().append(true);
        ring.append(true);
        builder.append(true);
    }
    let array = Arc::new(builder.finish()) as ArrayRef;
    let field = Field::new("geometry", array.data_type().clone(), false);
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).expect("polygons batch")
}

/// Keeps the unused import warning away when the polygon builder types change.
#[allow(dead_code)]
fn types() -> DataType {
    DataType::Float64
}

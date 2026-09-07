//! Upload and frame times of large layers: a million points, a hundred thousand polygons and
//! ten thousand paths. Run with `cargo bench -p deck-gl-layers --bench layers`; skipped without
//! a GPU adapter.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use deck_gl::luma_gl::device::{create_headless_context, HeadlessContext};
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Accessor, Deck, DeckProps, Layer, LayerData, LayerProps, Unit, ViewState};
use deck_gl_layers::{
    PathLayer, PathLayerProps, ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer,
    SolidPolygonLayerProps,
};

const CENTER: [f64; 2] = [-122.4, 37.8];
const SIZE: u32 = 1024;

fn deck(ctx: &HeadlessContext, layers: Vec<Box<dyn Layer>>) -> Deck {
    Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: SIZE,
            height: SIZE,
            view_state: ViewState {
                longitude: CENTER[0],
                latitude: CENTER[1],
                zoom: 11.0,
                pitch: 30.0,
                bearing: 0.0,
            },
            layers,
            ..Default::default()
        },
    )
    .expect("deck")
}

/// A deterministic pseudo random position around the centre.
fn position(i: usize, spread: f64) -> [f64; 3] {
    let a = (i as f64 * 0.618_033_988_749_895).fract();
    let b = (i as f64 * 0.381_966_011_250_105).fract();
    [
        CENTER[0] + (a - 0.5) * spread,
        CENTER[1] + (b - 0.5) * spread,
        0.0,
    ]
}

fn points(count: usize) -> Box<dyn Layer> {
    Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps::new("points"),
        data: LayerData::with_length(count),
        get_position: Accessor::Func(Arc::new(|i| position(i, 0.5))),
        get_fill_color: Accessor::Func(Arc::new(|i| [(i % 255) as u8, 80, 200, 255])),
        get_radius: Accessor::Constant(20.0),
        radius_units: Unit::Meters,
        ..Default::default()
    }))
}

fn polygons(count: usize) -> Box<dyn Layer> {
    Box::new(SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps::new("polygons"),
        data: LayerData::with_length(count),
        get_polygon: Accessor::Func(Arc::new(|i| {
            let [x, y, _] = position(i, 0.5);
            let s = 0.0005;
            vec![vec![
                [x, y, 0.0],
                [x + s, y, 0.0],
                [x + s, y + s, 0.0],
                [x, y + s, 0.0],
                [x, y, 0.0],
            ]]
        })),
        get_fill_color: Accessor::Func(Arc::new(|i| [200, (i % 255) as u8, 80, 255])),
        get_elevation: Accessor::Constant(50.0),
        extruded: true,
        ..Default::default()
    }))
}

fn paths(count: usize) -> Box<dyn Layer> {
    Box::new(PathLayer::new(PathLayerProps {
        base: LayerProps::new("paths"),
        data: LayerData::with_length(count),
        get_path: Accessor::Func(Arc::new(|i| {
            let [x, y, _] = position(i, 0.5);
            (0..20)
                .map(|k| {
                    let t = k as f64 * 0.0004;
                    [x + t, y + (t * 40.0).sin() * 0.0004, 0.0]
                })
                .collect()
        })),
        get_color: Accessor::Constant([40, 120, 220, 255]),
        get_width: Accessor::Constant(3.0),
        width_units: Unit::Pixels,
        ..Default::default()
    }))
}

/// Name, object count and layer constructor of one benchmark case.
type Case = (&'static str, usize, fn(usize) -> Box<dyn Layer>);

fn bench_layers(c: &mut Criterion) {
    let Ok(ctx) = create_headless_context() else {
        eprintln!("no GPU adapter, skipping");
        return;
    };
    let cases: Vec<Case> = vec![
        ("scatterplot", 1_000_000, points),
        ("solid_polygon", 100_000, polygons),
        ("path", 10_000, paths),
    ];
    for (name, count, make) in cases {
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Elements(count as u64));
        group.sample_size(10);
        // Upload: resolving the accessors, tessellating and creating the buffers, plus the
        // first frame
        group.bench_with_input(BenchmarkId::new("upload", count), &count, |b, &count| {
            b.iter(|| {
                let mut deck = deck(&ctx, vec![make(count)]);
                deck.snapshot(None).expect("frame");
            });
        });
        // Frame: drawing again with everything resident, read back included
        let mut deck = deck(&ctx, vec![make(count)]);
        deck.snapshot(None).expect("frame");
        group.bench_with_input(BenchmarkId::new("frame", count), &count, |b, _| {
            b.iter(|| deck.snapshot(None).expect("frame"));
        });
        group.finish();
    }

    // Two hundred small layers: shader compilation and pipelines are shared between them
    let mut group = c.benchmark_group("many_layers");
    group.sample_size(10);
    group.bench_function(BenchmarkId::new("init", 200), |b| {
        b.iter(|| {
            let layers: Vec<Box<dyn Layer>> = (0..200)
                .map(|i| {
                    Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
                        base: LayerProps::new(format!("points-{i}")),
                        data: LayerData::with_length(10),
                        get_position: Accessor::func(move |j| position(i * 10 + j, 0.5)),
                        get_radius: Accessor::Constant(3.0),
                        radius_units: Unit::Pixels,
                        ..Default::default()
                    })) as Box<dyn Layer>
                })
                .collect();
            let mut deck = deck(&ctx, layers);
            deck.snapshot(None).expect("frame");
        });
    });
    group.finish();

    // A million points from Arrow columns in the GPU layout: positions and colours upload as
    // they are
    let mut group = c.benchmark_group("scatterplot_arrow");
    group.throughput(Throughput::Elements(1_000_000));
    group.sample_size(10);
    let batch = {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, UInt8Builder};
        use arrow_array::{Array, RecordBatch};
        use arrow_schema::{Field, Schema};
        let mut positions = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        let mut colors = FixedSizeListBuilder::new(UInt8Builder::new(), 4);
        for i in 0..1_000_000usize {
            let p = position(i, 0.5);
            positions.values().append_value(p[0] as f32);
            positions.values().append_value(p[1] as f32);
            positions.values().append_value(0.0);
            positions.append(true);
            colors.values().append_slice(&[(i % 255) as u8, 80, 200, 255]);
            colors.append(true);
        }
        let positions = positions.finish();
        let colors = colors.finish();
        let schema = Schema::new(vec![
            Field::new("position", positions.data_type().clone(), false),
            Field::new("color", colors.data_type().clone(), false),
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(positions), Arc::new(colors)]).expect("batch")
    };
    group.bench_function(BenchmarkId::new("upload", 1_000_000), |b| {
        b.iter(|| {
            let layer = Box::new(ScatterplotLayer::new(ScatterplotLayerProps {
                base: LayerProps::new("arrow"),
                data: LayerData::from_batch(batch.clone()),
                get_position: Accessor::column("position"),
                get_fill_color: Accessor::column("color"),
                get_radius: Accessor::Constant(20.0),
                radius_units: Unit::Meters,
                ..Default::default()
            })) as Box<dyn Layer>;
            let mut deck = deck(&ctx, vec![layer]);
            deck.snapshot(None).expect("frame");
        });
    });
    group.finish();

    // Prop updates on the million points: a uniform only change and a colour accessor change
    let mut group = c.benchmark_group("scatterplot_update");
    group.sample_size(10);
    let count = 1_000_000;
    let mut deck = deck(&ctx, vec![points(count)]);
    deck.snapshot(None).expect("frame");
    let mut scale = 1.0f32;
    group.bench_function(BenchmarkId::new("radius_scale", count), |b| {
        b.iter(|| {
            scale += 0.01;
            let layer = deck.layer_mut("points").expect("layer");
            let layer = layer
                .as_any_mut()
                .downcast_mut::<ScatterplotLayer>()
                .expect("scatterplot");
            let mut props = layer.props().clone();
            props.radius_scale = scale;
            layer.set_props(props);
            deck.snapshot(None).expect("frame")
        });
    });
    let mut shade = 0u8;
    group.bench_function(BenchmarkId::new("fill_color", count), |b| {
        b.iter(|| {
            shade = shade.wrapping_add(1);
            let layer = deck.layer_mut("points").expect("layer");
            let layer = layer
                .as_any_mut()
                .downcast_mut::<ScatterplotLayer>()
                .expect("scatterplot");
            let mut props = layer.props().clone();
            props.get_fill_color = Accessor::Func(Arc::new(move |i| [(i % 255) as u8, shade, 200, 255]));
            layer.set_props(props);
            deck.snapshot(None).expect("frame")
        });
    });
    group.finish();
}

criterion_group!(benches, bench_layers);
criterion_main!(benches);

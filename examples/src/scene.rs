//! The example scene used by the `texture_render` and `window` binaries: a scatterplot fed
//! from Arrow columns, lines built by accessor functions, and extruded polygons with a hole.

use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, UInt8Builder};
use arrow_array::{Array, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use deck_gl::{Accessor, Layer, LayerData, LayerProps, Path, Polygon, Unit, ViewState};
use deck_gl_layers::{AggregationOperation, AggregationProps, HexagonLayer, HexagonLayerProps};
use deck_gl_layers::{
    ArcLayer, ArcLayerProps, BitmapImage, BitmapLayer, BitmapLayerProps, ColumnLayer, ColumnLayerProps,
    IconAtlas, IconLayer, IconLayerProps, IconMapping, LineLayer, LineLayerProps, PathLayer, PathLayerProps,
    PolygonLayer, PolygonLayerProps, ScatterplotLayer, ScatterplotLayerProps, SolidPolygonLayer,
    SolidPolygonLayerProps, TextLayer, TextLayerProps,
};

pub const CENTER: [f64; 2] = [-122.42, 37.775];

pub fn view_state(bearing: f64) -> ViewState {
    ViewState {
        longitude: CENTER[0],
        latitude: CENTER[1],
        zoom: 12.6,
        pitch: 50.0,
        bearing,
    }
}

/// Scatterplot data as an Arrow record batch: position, color and radius columns.
pub fn scatterplot_batch() -> RecordBatch {
    let mut positions = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    let mut colors = FixedSizeListBuilder::new(UInt8Builder::new(), 4);
    let mut radius = Vec::new();
    let n = 40;
    for i in 0..n {
        for j in 0..n {
            let fx = i as f64 / (n - 1) as f64;
            let fy = j as f64 / (n - 1) as f64;
            positions.values().append_value(CENTER[0] - 0.06 + fx * 0.12);
            positions.values().append_value(CENTER[1] - 0.045 + fy * 0.09);
            positions.append(true);
            colors
                .values()
                .append_slice(&[(255.0 * fx) as u8, 80, (255.0 * fy) as u8, 220]);
            colors.append(true);
            radius.push(20.0 + 60.0 * ((fx * 6.0).sin() * (fy * 6.0).cos()).abs() as f32);
        }
    }
    let positions = positions.finish();
    let colors = colors.finish();
    let schema = Schema::new(vec![
        Field::new("position", positions.data_type().clone(), false),
        Field::new("color", colors.data_type().clone(), false),
        Field::new("radius", DataType::Float32, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(positions),
            Arc::new(colors),
            Arc::new(Float32Array::from(radius)),
        ],
    )
    .expect("valid batch")
}

fn block(x: f64, y: f64, w: f64, h: f64) -> Polygon {
    vec![vec![
        [x, y, 0.0],
        [x + w, y, 0.0],
        [x + w, y + h, 0.0],
        [x, y + h, 0.0],
    ]]
}

/// Build the scene's layers, bottom to top.
pub fn layers() -> Vec<Box<dyn Layer>> {
    let scatterplot = ScatterplotLayer::new(ScatterplotLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("points")
        },
        data: LayerData::from_batch(scatterplot_batch()),
        get_position: Accessor::column("position"),
        get_fill_color: Accessor::column("color"),
        get_radius: Accessor::column("radius"),
        get_line_color: Accessor::Constant([255, 255, 255, 255]),
        radius_units: Unit::Meters,
        stroked: true,
        line_width_min_pixels: 1.0,
        ..Default::default()
    });

    let line_count = 24usize;
    let lines = LineLayer::new(LineLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("lines")
        },
        data: LayerData::with_length(line_count),
        get_source_position: Accessor::Constant([CENTER[0], CENTER[1], 0.0]),
        get_target_position: Accessor::func(move |i| {
            let a = i as f64 / line_count as f64 * std::f64::consts::TAU;
            [CENTER[0] + 0.07 * a.cos(), CENTER[1] + 0.055 * a.sin(), 0.0]
        }),
        get_color: Accessor::func(|i| [255, (i * 10 % 255) as u8, 60, 255]),
        get_width: Accessor::func(|i| 1.0 + (i % 4) as f32),
        width_units: Unit::Pixels,
        ..Default::default()
    });

    let mut polygons: Vec<Polygon> = Vec::new();
    let mut elevations = Vec::new();
    for i in 0..5 {
        for j in 0..4 {
            let x = CENTER[0] - 0.02 + i as f64 * 0.009;
            let y = CENTER[1] - 0.012 + j as f64 * 0.007;
            polygons.push(block(x, y, 0.006, 0.0045));
            elevations.push(150.0 + 250.0 * ((i * 7 + j * 3) % 5) as f32);
        }
    }
    polygons.push(vec![
        vec![
            [CENTER[0] + 0.03, CENTER[1] + 0.015, 0.0],
            [CENTER[0] + 0.055, CENTER[1] + 0.015, 0.0],
            [CENTER[0] + 0.055, CENTER[1] + 0.035, 0.0],
            [CENTER[0] + 0.03, CENTER[1] + 0.035, 0.0],
        ],
        vec![
            [CENTER[0] + 0.036, CENTER[1] + 0.02, 0.0],
            [CENTER[0] + 0.048, CENTER[1] + 0.02, 0.0],
            [CENTER[0] + 0.048, CENTER[1] + 0.03, 0.0],
            [CENTER[0] + 0.036, CENTER[1] + 0.03, 0.0],
        ],
    ]);
    elevations.push(0.0);
    let polygon_count = polygons.len();
    let polygons = Arc::new(polygons);
    let elevations = Arc::new(elevations);
    let solid_polygons = SolidPolygonLayer::new(SolidPolygonLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("blocks")
        },
        data: LayerData::with_length(polygon_count),
        extruded: true,
        wireframe: true,
        get_polygon: Accessor::func(move |i| polygons[i].clone()),
        get_elevation: Accessor::func(move |i| elevations[i]),
        get_fill_color: Accessor::func(move |i| {
            if i == polygon_count - 1 {
                [60, 180, 120, 200]
            } else {
                [230, 200, 80, 255]
            }
        }),
        get_line_color: Accessor::Constant([40, 40, 40, 255]),
        ..Default::default()
    });

    // A winding route drawn with joints and round caps
    let route: Path = (0..40)
        .map(|i| {
            let t = i as f64 / 39.0;
            let a = t * std::f64::consts::TAU * 1.5;
            [
                CENTER[0] - 0.05 + t * 0.1 + 0.006 * a.sin(),
                CENTER[1] - 0.03 + 0.012 * (a * 0.7).cos() + t * 0.02,
                0.0,
            ]
        })
        .collect();
    let route = Arc::new(route);
    let paths = PathLayer::new(PathLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("route")
        },
        data: LayerData::with_length(1),
        get_path: Accessor::func(move |_| (*route).clone()),
        get_color: Accessor::Constant([20, 120, 255, 255]),
        get_width: Accessor::Constant(6.0),
        width_units: Unit::Pixels,
        joint_rounded: true,
        cap_rounded: true,
        ..Default::default()
    });

    // Arcs from the center to points around the bay
    let arc_count = 8usize;
    let arcs = ArcLayer::new(ArcLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("arcs")
        },
        data: LayerData::with_length(arc_count),
        get_source_position: Accessor::Constant([CENTER[0], CENTER[1], 0.0]),
        get_target_position: Accessor::func(move |i| {
            let a = (i as f64 + 0.5) / arc_count as f64 * std::f64::consts::TAU;
            [CENTER[0] + 0.05 * a.cos(), CENTER[1] + 0.04 * a.sin(), 0.0]
        }),
        get_source_color: Accessor::Constant([255, 80, 0, 255]),
        get_target_color: Accessor::Constant([0, 200, 255, 255]),
        get_width: Accessor::Constant(3.0),
        get_height: Accessor::Constant(0.6),
        ..Default::default()
    });

    // A stroked polygon with a hole, drawn by the composite PolygonLayer
    let park: Polygon = vec![
        vec![
            [CENTER[0] - 0.058, CENTER[1] + 0.01, 0.0],
            [CENTER[0] - 0.03, CENTER[1] + 0.012, 0.0],
            [CENTER[0] - 0.028, CENTER[1] + 0.03, 0.0],
            [CENTER[0] - 0.056, CENTER[1] + 0.034, 0.0],
        ],
        vec![
            [CENTER[0] - 0.05, CENTER[1] + 0.017, 0.0],
            [CENTER[0] - 0.04, CENTER[1] + 0.017, 0.0],
            [CENTER[0] - 0.04, CENTER[1] + 0.026, 0.0],
            [CENTER[0] - 0.05, CENTER[1] + 0.026, 0.0],
        ],
    ];
    let park = Arc::new(park);
    let polygons = PolygonLayer::new(PolygonLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("park")
        },
        data: LayerData::with_length(1),
        get_polygon: Accessor::func(move |_| (*park).clone()),
        get_fill_color: Accessor::Constant([80, 200, 120, 160]),
        get_line_color: Accessor::Constant([20, 90, 50, 255]),
        get_line_width: Accessor::Constant(4.0),
        line_width_units: Unit::Pixels,
        line_joint_rounded: true,
        ..Default::default()
    });

    // A procedural image (color wheel over a checkerboard) draped south west of the center
    let size = 128u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f64 / size as f64 - 0.5, y as f64 / size as f64 - 0.5);
            let angle = fy.atan2(fx);
            let r = (fx * fx + fy * fy).sqrt();
            let check = ((x / 16 + y / 16) % 2) as f64;
            let hue = |offset: f64| (((angle + offset).sin() * 0.5 + 0.5) * 255.0) as u8;
            let inside = r < 0.45;
            rgba.extend_from_slice(&[
                if inside {
                    hue(0.0)
                } else {
                    (160.0 + 60.0 * check) as u8
                },
                if inside {
                    hue(2.1)
                } else {
                    (160.0 + 60.0 * check) as u8
                },
                if inside {
                    hue(4.2)
                } else {
                    (160.0 + 60.0 * check) as u8
                },
                if inside { 255 } else { 180 },
            ]);
        }
    }
    let bitmap = BitmapLayer::new(BitmapLayerProps {
        base: LayerProps::new("bitmap"),
        image: Some(BitmapImage::new(size, size, rgba)),
        bounds: [
            CENTER[0] - 0.062,
            CENTER[1] - 0.05,
            CENTER[0] - 0.032,
            CENTER[1] - 0.026,
        ],
        ..Default::default()
    });

    // Hexagonal columns in a small grid east of the blocks
    let column_count = 5 * 4;
    let columns = ColumnLayer::new(ColumnLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("columns")
        },
        data: LayerData::with_length(column_count),
        disk_resolution: 6,
        radius: 150.0,
        get_position: Accessor::func(|i| {
            [
                CENTER[0] + 0.04 + (i % 5) as f64 * 0.0045,
                CENTER[1] - 0.03 + (i / 5) as f64 * 0.0045,
                0.0,
            ]
        }),
        get_elevation: Accessor::func(|i| 100.0 + 80.0 * ((i * 5) % 7) as f32),
        get_fill_color: Accessor::func(|i| [90, 120 + ((i * 17) % 100) as u8, 220, 255]),
        get_line_color: Accessor::Constant([30, 30, 60, 255]),
        wireframe: true,
        ..Default::default()
    });

    // Icons from a procedural atlas: a pin and a ring, both masks colored per instance
    let icons = IconLayer::new(IconLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("icons")
        },
        data: LayerData::with_length(12),
        atlas: Some(Arc::new(icon_atlas())),
        get_position: Accessor::func(|i| {
            let a = i as f64 / 12.0 * std::f64::consts::TAU;
            [CENTER[0] + 0.075 * a.cos(), CENTER[1] + 0.058 * a.sin(), 0.0]
        }),
        get_icon: Accessor::func(|i| {
            if i.is_multiple_of(2) {
                "pin".to_string()
            } else {
                "ring".to_string()
            }
        }),
        get_color: Accessor::func(|i| [255, 40 + (i * 18) as u8, 80, 255]),
        get_size: Accessor::Constant(32.0),
        ..Default::default()
    });

    vec![
        Box::new(bitmap),
        Box::new(solid_polygons),
        Box::new(columns),
        Box::new(polygons),
        Box::new(scatterplot),
        Box::new(paths),
        Box::new(lines),
        Box::new(arcs),
        Box::new(icons),
        Box::new(hexagons()),
        Box::new(labels()),
    ]
}

/// A 64x32 atlas with a map pin on the left and a ring on the right, as alpha masks.
/// Pseudo random points south west of the centre, aggregated into extruded hexagons.
pub fn hexagons() -> HexagonLayer {
    // a small deterministic generator so the scene is stable across runs
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let origin = [CENTER[0] - 0.03, CENTER[1] - 0.036];
    let mut points = Vec::with_capacity(2500);
    for _ in 0..2500 {
        // two gaussian-ish blobs by summing uniforms
        let blob = if next() < 0.6 { [0.0, 0.0] } else { [0.011, 0.006] };
        let gx: f64 = (0..4).map(|_| next()).sum::<f64>() / 4.0 - 0.5;
        let gy: f64 = (0..4).map(|_| next()).sum::<f64>() / 4.0 - 0.5;
        points.push([
            origin[0] + blob[0] + gx * 0.024,
            origin[1] + blob[1] + gy * 0.016,
            0.0,
        ]);
    }
    let positions = Arc::new(points);
    let n = positions.len();
    HexagonLayer::new(HexagonLayerProps {
        base: LayerProps {
            pickable: true,
            opacity: 0.9,
            ..LayerProps::new("hexagons")
        },
        data: LayerData::with_length(n),
        radius: 120.0,
        aggregation: AggregationProps {
            get_position: Accessor::func(move |i| positions[i]),
            color_aggregation: AggregationOperation::Count,
            extruded: true,
            elevation_range: [0.0, 600.0],
            coverage: 0.9,
            ..Default::default()
        },
    })
}

/// District labels: SDF text with an outline, plus one boxed label.
pub fn labels() -> TextLayer {
    let places: Vec<(&str, [f64; 3])> = vec![
        ("Downtown", [CENTER[0] + 0.034, CENTER[1] - 0.004, 0.0]),
        ("Mission", [CENTER[0] + 0.005, CENTER[1] - 0.02, 0.0]),
        ("Golden Gate Park", [CENTER[0] - 0.046, CENTER[1] - 0.014, 0.0]),
        ("deck.gl-native", [CENTER[0] - 0.008, CENTER[1] + 0.036, 0.0]),
    ];
    let names: Vec<String> = places.iter().map(|p| p.0.to_string()).collect();
    let positions: Vec<[f64; 3]> = places.iter().map(|p| p.1).collect();
    let n = places.len();
    TextLayer::new(TextLayerProps {
        base: LayerProps {
            pickable: true,
            ..LayerProps::new("labels")
        },
        data: LayerData::with_length(n),
        get_text: Accessor::func(move |i| names[i].clone()),
        get_position: Accessor::func(move |i| positions[i]),
        get_size: Accessor::func(|i| if i == 3 { 28.0 } else { 20.0 }),
        get_color: Accessor::Constant([255, 255, 255, 255]),
        get_background_color: Accessor::Constant([20, 24, 40, 220]),
        get_border_color: Accessor::Constant([255, 200, 80, 255]),
        get_border_width: Accessor::func(|i| if i == 3 { 2.0 } else { 0.0 }),
        background: true,
        background_padding: [8.0, 4.0, 8.0, 4.0],
        background_border_radius: [6.0; 4],
        font: deck_gl_layers::FontSettings {
            sdf: true,
            character_set: deck_gl_layers::CharacterSet::Auto,
            ..Default::default()
        },
        outline_width: 4.0,
        outline_color: [0, 0, 0, 255],
        ..Default::default()
    })
}

pub fn icon_atlas() -> IconAtlas {
    let (w, h) = (64u32, 32u32);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
            let visible = if x < 32 {
                // pin: a disk with a hole on top of a triangle pointing down to the anchor
                let d = ((fx - 16.0).powi(2) + (fy - 11.0).powi(2)).sqrt();
                let tail = fy >= 11.0 && (fx - 16.0).abs() < (32.0 - fy) * 0.45;
                (d < 9.0 || tail) && d >= 3.5
            } else {
                let r = ((fx - 48.0).powi(2) + (fy - 16.0).powi(2)).sqrt();
                (9.0..14.0).contains(&r)
            };
            if visible {
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
    }
    let mut mapping = std::collections::HashMap::new();
    mapping.insert(
        "pin".to_string(),
        IconMapping::new(0, 0, 32, 32).mask().anchor(16.0, 32.0),
    );
    mapping.insert("ring".to_string(), IconMapping::new(32, 0, 32, 32).mask());
    IconAtlas {
        image: BitmapImage::new(w, h, rgba),
        mapping,
    }
}

pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.07,
    g: 0.08,
    b: 0.11,
    a: 1.0,
};

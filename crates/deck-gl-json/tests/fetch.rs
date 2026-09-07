//! URL data through a fetcher: layers appear once their data arrived.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use deck_gl_json::{ConvertOptions, Fetcher, JsonConverter, JsonError};
use serde_json::json;

fn serve(body: &'static str, delay: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = [0u8; 2048];
            // The request is read only to let the client finish sending it
            let _read = stream.read(&mut request).unwrap_or(0);
            std::thread::sleep(delay);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).ok();
            stream.write_all(body.as_bytes()).ok();
        }
    });
    base
}

/// The number of rows a converted scatterplot layer holds.
fn rows(layer: &mut Box<dyn deck_gl::Layer>) -> usize {
    layer
        .as_any_mut()
        .downcast_mut::<deck_gl_layers::ScatterplotLayer>()
        .expect("a scatterplot layer")
        .props()
        .data
        .len()
}

#[test]
fn url_data_loads_in_the_background() {
    let base = serve(
        r#"[{"position": [1.0, 2.0]}, {"position": [3.0, 4.0]}]"#,
        Duration::from_millis(150),
    );
    let url = format!("{base}/points.json");
    let fetcher = Fetcher::new(2);
    let converter = JsonConverter {
        options: ConvertOptions {
            fetcher: Some(fetcher.clone()),
            ..Default::default()
        },
    };
    let spec = json!({
        "layers": [
            {"@@type": "ScatterplotLayer", "id": "remote", "data": url, "getPosition": "@@=position"},
            {"@@type": "ScatterplotLayer", "id": "inline", "data": [{"position": [0, 0]}], "getPosition": "@@=position"}
        ]
    });
    let generation = fetcher.generation();
    let first = converter.convert(&spec).unwrap();
    assert_eq!(first.pending, vec![url.clone()]);
    assert_eq!(first.layers.len(), 1, "the inline layer is there at once");
    assert_eq!(first.layers[0].id(), "inline");
    // The bare layer array API reports the wait as an error instead
    let mut warnings = Vec::new();
    match converter.convert_layers(&spec["layers"], &mut warnings) {
        Err(JsonError::Pending { url: pending }) => assert_eq!(pending, url),
        other => panic!("expected a pending error, got {:?}", other.map(|l| l.len())),
    }
    let start = Instant::now();
    while fetcher.generation() == generation {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the fetch never finished"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut second = converter.convert(&spec).unwrap();
    assert!(second.pending.is_empty());
    assert_eq!(second.layers.len(), 2);
    assert_eq!(second.layers[0].id(), "remote");
    assert_eq!(rows(&mut second.layers[0]), 2);
    assert_eq!(fetcher.stats().completed, 1);
}

#[test]
fn without_a_fetcher_urls_block_and_use_the_shared_cache() {
    let base = serve(r#"[{"position": [5.0, 6.0]}]"#, Duration::ZERO);
    let url = format!("{base}/one.json");
    let converter = JsonConverter::new();
    let mut deck = converter
        .convert(&json!([{"@@type": "ScatterplotLayer", "data": url, "getPosition": "@@=position"}]))
        .unwrap();
    assert!(deck.pending.is_empty());
    assert_eq!(rows(&mut deck.layers[0]), 1);
    assert!(Fetcher::global().cached(&url).is_some());
}

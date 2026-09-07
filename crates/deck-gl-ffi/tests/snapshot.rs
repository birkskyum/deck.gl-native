//! `deckgl_snapshot` renders JSON layers headlessly through the C API.

use std::ffi::{CStr, CString};

use deckgl::{
    deckgl_destroy, deckgl_headless_create, deckgl_is_loading, deckgl_last_error, deckgl_set_layers_json,
    deckgl_snapshot,
};

#[test]
fn snapshot_renders_json_layers_at_the_initial_view_state() {
    let deck = unsafe { deckgl_headless_create() };
    if deck.is_null() {
        eprintln!("skipping GPU test");
        return;
    }
    let json = CString::new(
        r#"{
            "initialViewState": {"longitude": -122.4, "latitude": 37.8, "zoom": 14},
            "layers": [{
                "@@type": "ScatterplotLayer",
                "data": [{"position": [-122.4, 37.8]}],
                "getPosition": "@@=position",
                "getFillColor": [255, 0, 0],
                "getRadius": 8,
                "radiusUnits": "pixels"
            }]
        }"#,
    )
    .unwrap();
    assert_eq!(
        unsafe { deckgl_set_layers_json(deck, json.as_ptr(), std::ptr::null()) },
        0
    );
    let (width, height) = (32u32, 32u32);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    let status = unsafe { deckgl_snapshot(deck, width, height, rgba.as_mut_ptr()) };
    let error = unsafe { CStr::from_ptr(deckgl_last_error(deck)) }
        .to_string_lossy()
        .into_owned();
    assert_eq!(status, 0, "{error}");
    let pixel = |x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    };
    assert_eq!(pixel(16, 16), [255, 0, 0, 255], "red disk at the view center");
    assert_eq!(pixel(1, 1), [0, 0, 0, 0], "transparent background");
    // A size of zero is an error, not a crash
    assert_eq!(unsafe { deckgl_snapshot(deck, 0, 0, rgba.as_mut_ptr()) }, 1);
    unsafe { deckgl_destroy(deck) };
}

/// A server answering every request with `body` after a short delay.
fn serve(body: &'static str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = [0u8; 2048];
            // The request is read only to let the client finish sending it
            let _read = stream.read(&mut request).unwrap_or(0);
            std::thread::sleep(std::time::Duration::from_millis(100));
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

#[test]
fn url_data_arrives_on_a_later_frame() {
    let deck = unsafe { deckgl_headless_create() };
    if deck.is_null() {
        eprintln!("skipping GPU test");
        return;
    }
    let base = serve(r#"[{"position": [-122.4, 37.8]}]"#);
    let json = CString::new(format!(
        r#"{{
            "initialViewState": {{"longitude": -122.4, "latitude": 37.8, "zoom": 14}},
            "layers": [{{
                "@@type": "ScatterplotLayer",
                "data": "{base}/points.json",
                "getPosition": "@@=position",
                "getFillColor": [0, 0, 255],
                "getRadius": 8,
                "radiusUnits": "pixels"
            }}]
        }}"#
    ))
    .unwrap();
    assert_eq!(
        unsafe { deckgl_set_layers_json(deck, json.as_ptr(), std::ptr::null()) },
        0
    );
    assert_eq!(unsafe { deckgl_is_loading(deck) }, 1, "the URL is on its way");
    let (width, height) = (32u32, 32u32);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    let pixel = |rgba: &[u8], x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    };
    assert_eq!(
        unsafe { deckgl_snapshot(deck, width, height, rgba.as_mut_ptr()) },
        0
    );
    assert_eq!(
        pixel(&rgba, 16, 16),
        [0, 0, 0, 0],
        "nothing to draw before the data arrived"
    );
    let start = std::time::Instant::now();
    while unsafe { deckgl_is_loading(deck) } == 1 {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "the load never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        unsafe { deckgl_snapshot(deck, width, height, rgba.as_mut_ptr()) },
        0
    );
    assert_eq!(
        pixel(&rgba, 16, 16),
        [0, 0, 255, 255],
        "the layer shows once loaded"
    );
    unsafe { deckgl_destroy(deck) };
}

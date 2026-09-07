//! `deckgl_snapshot` renders JSON layers headlessly through the C API.

use std::ffi::{CStr, CString};

use deckgl::{
    deckgl_destroy, deckgl_headless_create, deckgl_last_error, deckgl_set_layers_json, deckgl_snapshot,
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

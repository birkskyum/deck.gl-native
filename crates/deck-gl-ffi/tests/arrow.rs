//! The C API accepts Arrow tables through the C Data Interface and JSON layers can use them.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float64Builder};
use arrow_array::ffi::to_ffi;
use arrow_array::{Array, StructArray};
use arrow_schema::{DataType, Field};
use deckgl::{
    deckgl_destroy, deckgl_headless_create, deckgl_last_error, deckgl_remove_arrow_table,
    deckgl_set_arrow_table, deckgl_set_layers_json,
};

fn table() -> StructArray {
    let mut positions = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    for (lng, lat) in [(-122.4, 37.8), (-122.41, 37.79)] {
        positions.values().append_value(lng);
        positions.values().append_value(lat);
        positions.append(true);
    }
    let positions = positions.finish();
    let weights = arrow_array::Float32Array::from(vec![1.0f32, 2.0]);
    StructArray::from(vec![
        (
            Arc::new(Field::new("position", positions.data_type().clone(), false)),
            Arc::new(positions) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new("weight", DataType::Float32, false)),
            Arc::new(weights) as Arc<dyn Array>,
        ),
    ])
}

#[test]
fn arrow_tables_cross_the_c_api() {
    let deck = unsafe { deckgl_headless_create() };
    if deck.is_null() {
        eprintln!("skipping GPU test");
        return;
    }
    let last_error = |deck| unsafe {
        CStr::from_ptr(deckgl_last_error(deck))
            .to_string_lossy()
            .into_owned()
    };

    let (mut array, schema) = to_ffi(&table().to_data()).unwrap();
    let name = CString::new("points").unwrap();
    let status = unsafe { deckgl_set_arrow_table(deck, name.as_ptr(), &schema, &mut array) };
    assert_eq!(status, 0, "{}", last_error(deck));
    assert!(array.is_released(), "ownership moved to the deck");

    let json = CString::new(
        r#"[{
          "@@type": "HexagonLayer",
          "id": "bins",
          "data": "@@table:points",
          "getPosition": "@@column:position",
          "getColorWeight": "@@column:weight",
          "radius": 200
        }, {
          "@@type": "ScatterplotLayer",
          "id": "dots",
          "data": "@@table:points",
          "getPosition": "@@=position",
          "getRadius": 5
        }]"#,
    )
    .unwrap();
    let status = unsafe { deckgl_set_layers_json(deck, json.as_ptr(), std::ptr::null()) };
    assert_eq!(status, 0, "{}", last_error(deck));

    // Unknown tables are reported
    let bad = CString::new(r#"[{"@@type": "ScatterplotLayer", "data": "@@table:nope"}]"#).unwrap();
    let status = unsafe { deckgl_set_layers_json(deck, bad.as_ptr(), std::ptr::null()) };
    assert_eq!(status, 1);
    assert!(last_error(deck).contains("nope"), "{}", last_error(deck));

    assert_eq!(unsafe { deckgl_remove_arrow_table(deck, name.as_ptr()) }, 0);
    let status = unsafe { deckgl_set_layers_json(deck, json.as_ptr(), std::ptr::null()) };
    assert_eq!(status, 1, "the table is gone");
    unsafe { deckgl_destroy(deck) };
}

//! Per frame counters: draw calls, instances and bytes uploaded to vertex and index buffers.
//! [`Model::draw`](crate::Model::draw) and the buffer helpers add to them; a deck resets them
//! at the start of a frame and reads them at the end. The counters are per thread, matching
//! a deck that updates and draws on one thread.

use std::cell::Cell;

thread_local! {
    static DRAW_CALLS: Cell<u64> = const { Cell::new(0) };
    static INSTANCES: Cell<u64> = const { Cell::new(0) };
    static UPLOADED_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Counters since the last [`reset`] on this thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub draw_calls: u64,
    pub instances: u64,
    pub uploaded_bytes: u64,
}

pub fn reset() {
    DRAW_CALLS.with(|c| c.set(0));
    INSTANCES.with(|c| c.set(0));
    UPLOADED_BYTES.with(|c| c.set(0));
}

pub fn snapshot() -> Counters {
    Counters {
        draw_calls: DRAW_CALLS.with(Cell::get),
        instances: INSTANCES.with(Cell::get),
        uploaded_bytes: UPLOADED_BYTES.with(Cell::get),
    }
}

pub(crate) fn count_draw(instances: u32) {
    DRAW_CALLS.with(|c| c.set(c.get() + 1));
    INSTANCES.with(|c| c.set(c.get() + u64::from(instances)));
}

pub(crate) fn count_upload(bytes: usize) {
    UPLOADED_BYTES.with(|c| c.set(c.get() + bytes as u64));
}

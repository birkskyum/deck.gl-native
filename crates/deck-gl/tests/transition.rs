//! `Deck::fly_to` moves the deck's own camera frame by frame.

use deck_gl::luma_gl::device::create_headless_context;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps, TransitionProps, ViewState};

#[test]
fn deck_transitions_move_the_viewport() {
    let Ok(ctx) = create_headless_context() else {
        eprintln!("skipping GPU test");
        return;
    };
    let start = ViewState {
        longitude: -122.4,
        latitude: 37.8,
        zoom: 12.0,
        pitch: 0.0,
        bearing: 0.0,
    };
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: 64,
            height: 64,
            view_state: start,
            ..Default::default()
        },
    )
    .unwrap();
    let end = ViewState {
        longitude: -74.0,
        latitude: 40.7,
        zoom: 10.0,
        pitch: 30.0,
        bearing: 0.0,
    };
    deck.fly_to(end, 0.0);
    let duration = deck.transition().expect("in flight").duration_ms();
    assert!(deck.tick(duration / 2.0));
    let mid = deck.view_state();
    assert!(
        mid.zoom < 10.0 && mid.longitude > -122.4 && mid.longitude < -74.0,
        "{mid:?}"
    );
    assert!(
        (deck.viewport().longitude - mid.longitude).abs() < 1e-9,
        "viewport follows"
    );
    assert!(deck.tick(duration + 1.0));
    assert!(!deck.tick(duration + 2.0), "done");
    let view = deck.view_state();
    assert!((view.longitude - end.longitude).abs() < 1e-6 && (view.pitch - 30.0).abs() < 1e-9);
    // A linear transition and a jump
    deck.transition_to(start, TransitionProps::linear(100.0), 5000.0);
    deck.tick(5050.0);
    assert!((deck.view_state().pitch - 15.0).abs() < 1e-9);
    deck.set_view_state(end);
    assert!(deck.transition().is_none());
}

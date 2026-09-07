//! `Deck::set_layers` keeps layers whose id and type match, so re-sending a layer list does not
//! rebuild GPU resources.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use deck_gl::luma_gl::device::create_headless_context;
use deck_gl::luma_gl::RenderTarget;
use deck_gl::{Deck, DeckProps, Layer, LayerContext, LayerProps, Result, Viewport};

/// A layer that counts how often it is initialized and which props it holds.
struct Counter {
    props: LayerProps,
    value: u32,
    initialized: Arc<AtomicUsize>,
    updates: Arc<AtomicUsize>,
}

impl Layer for Counter {
    fn props(&self) -> &LayerProps {
        &self.props
    }

    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        self.initialized.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn update(&mut self, _ctx: &LayerContext, _viewport: &Viewport) -> Result<()> {
        Ok(())
    }

    fn draw(&mut self, _ctx: &LayerContext, _pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn update_from(&mut self, incoming: &mut dyn Layer) -> bool {
        match incoming.as_any_mut().downcast_mut::<Self>() {
            Some(other) => {
                self.value = other.value;
                self.updates.fetch_add(1, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }
}

/// A different layer type.
struct Other {
    props: LayerProps,
}

impl Layer for Other {
    fn props(&self) -> &LayerProps {
        &self.props
    }
    fn initialize(&mut self, _ctx: &LayerContext) -> Result<()> {
        Ok(())
    }
    fn update(&mut self, _ctx: &LayerContext, _viewport: &Viewport) -> Result<()> {
        Ok(())
    }
    fn draw(&mut self, _ctx: &LayerContext, _pass: &mut wgpu::RenderPass<'_>) -> Result<()> {
        Ok(())
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn set_layers_reuses_layers_with_the_same_id_and_type() {
    let Ok(ctx) = create_headless_context() else {
        eprintln!("skipping GPU test");
        return;
    };
    let initialized = Arc::new(AtomicUsize::new(0));
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = |value: u32| -> Box<dyn Layer> {
        Box::new(Counter {
            props: LayerProps::new("counter"),
            value,
            initialized: initialized.clone(),
            updates: updates.clone(),
        })
    };
    let mut deck = Deck::new(
        &ctx.device,
        &ctx.queue,
        RenderTarget::default(),
        DeckProps {
            width: 16,
            height: 16,
            layers: vec![counter(1)],
            ..Default::default()
        },
    )
    .unwrap();
    deck.update().unwrap();
    assert_eq!(initialized.load(Ordering::SeqCst), 1);

    // Same id and type: the existing layer takes the new props, no new initialization
    deck.set_layers(vec![counter(2)]);
    deck.update().unwrap();
    assert_eq!(initialized.load(Ordering::SeqCst), 1);
    assert_eq!(updates.load(Ordering::SeqCst), 1);

    // A second entry with a new id is initialized; the first still matches
    deck.set_layers(vec![
        counter(3),
        Box::new(Counter {
            props: LayerProps::new("another"),
            value: 4,
            initialized: initialized.clone(),
            updates: updates.clone(),
        }),
    ]);
    deck.update().unwrap();
    assert_eq!(updates.load(Ordering::SeqCst), 2, "the first entry matched by id");
    assert_eq!(initialized.load(Ordering::SeqCst), 2, "the second entry is new");

    // Same id, different type: replaced
    deck.set_layers(vec![Box::new(Other {
        props: LayerProps::new("counter"),
    })]);
    deck.update().unwrap();
    assert_eq!(deck.layers().count(), 1);
    deck.set_layers(vec![counter(5)]);
    deck.update().unwrap();
    assert_eq!(
        initialized.load(Ordering::SeqCst),
        3,
        "a Counter replaced an Other"
    );
}

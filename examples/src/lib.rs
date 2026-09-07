//! Shared example scene and JSON description loading.

pub mod bigdata;
pub mod scene;
pub mod spec;

/// Log to stderr at the level of `DECKGL_LOG` or `RUST_LOG` (`warn` by default).
pub fn init_logging() {
    let filter = std::env::var("DECKGL_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| "warn".to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();
}

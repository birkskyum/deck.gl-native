//! The shared fetcher against a local HTTP server: caching, one request per URL,
//! cancellation and progress.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use deck_gl_layers::{FetchStatus, Fetcher};

/// A server answering every request with `body`, after `delay`, counting the requests.
fn serve(body: &'static [u8], delay: Duration) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    let requests = Arc::new(AtomicUsize::new(0));
    let counter = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            counter.fetch_add(1, Ordering::SeqCst);
            let mut request = [0u8; 2048];
            let read = stream.read(&mut request).unwrap_or(0);
            let line = String::from_utf8_lossy(&request[..read]);
            let missing = line.starts_with("GET /missing");
            std::thread::sleep(delay);
            let status = if missing { "404 Not Found" } else { "200 OK" };
            let body: &[u8] = if missing { b"" } else { body };
            let header = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).ok();
            stream.write_all(body).ok();
        }
    });
    (base, requests)
}

#[test]
fn fetches_once_and_serves_the_cache_afterwards() {
    let (base, requests) = serve(b"hello", Duration::ZERO);
    let fetcher = Fetcher::new(2);
    let url = format!("{base}/data.json");
    let handle = fetcher.fetch(&url);
    assert_eq!(handle.wait().unwrap().as_slice(), b"hello");
    assert_eq!(handle.status(), FetchStatus::Done);
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    let again = fetcher.fetch(&url);
    assert_eq!(
        again.status(),
        FetchStatus::Done,
        "cached answers are finished at once"
    );
    assert_eq!(again.result().unwrap().unwrap().as_slice(), b"hello");
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    assert_eq!(fetcher.cached(&url).unwrap().len(), 5);
    let stats = fetcher.stats();
    assert_eq!(
        (stats.completed, stats.cache_hits, stats.bytes, stats.cached_bytes),
        (1, 1, 5, 5)
    );
    assert!(fetcher.is_idle());
    fetcher.clear_cache();
    assert!(fetcher.cached(&url).is_none());
}

#[test]
fn one_request_serves_every_handle_for_a_url() {
    let (base, requests) = serve(b"shared", Duration::from_millis(150));
    let fetcher = Fetcher::new(4);
    let url = format!("{base}/shared");
    let a = fetcher.fetch(&url);
    let b = fetcher.fetch(&url);
    assert!(!fetcher.is_idle());
    assert_eq!(a.wait().unwrap().as_slice(), b"shared");
    assert_eq!(b.wait().unwrap().as_slice(), b"shared");
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    fetcher.wait_idle();
    assert_eq!(fetcher.stats().completed, 1);
}

#[test]
fn cancelled_requests_never_reach_the_server() {
    let (base, requests) = serve(b"slow", Duration::from_millis(200));
    let fetcher = Fetcher::new(1);
    let generation = fetcher.generation();
    let first = fetcher.fetch(&format!("{base}/first"));
    let second = fetcher.fetch(&format!("{base}/second"));
    let third = fetcher.fetch(&format!("{base}/second"));
    assert_eq!(second.status(), FetchStatus::Queued);
    second.cancel();
    assert_eq!(
        third.status(),
        FetchStatus::Queued,
        "another handle still wants it"
    );
    third.cancel();
    assert_eq!(second.status(), FetchStatus::Cancelled);
    assert!(third.wait().is_err());
    assert!(first.wait().is_ok());
    fetcher.wait_idle();
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "the cancelled URL was never requested"
    );
    let stats = fetcher.stats();
    assert_eq!((stats.completed, stats.cancelled, stats.pending()), (1, 1, 0));
    assert!(fetcher.generation() > generation);
}

#[test]
fn failures_are_reported_and_not_cached() {
    let (base, requests) = serve(b"", Duration::ZERO);
    let fetcher = Fetcher::new(1);
    let url = format!("{base}/missing");
    let error = fetcher.fetch_blocking(&url).unwrap_err();
    assert!(error.contains("404") || error.contains("status"), "{error}");
    assert!(fetcher.cached(&url).is_none());
    assert!(fetcher.fetch_blocking(&url).is_err());
    assert_eq!(requests.load(Ordering::SeqCst), 2, "failures are retried");
    assert_eq!(fetcher.stats().failed, 2);
    let unreachable = fetcher.fetch_blocking("http://127.0.0.1:1/nothing");
    assert!(unreachable.is_err());
}

#[test]
fn the_cache_budget_evicts_old_responses() {
    let (base, _) = serve(b"0123456789", Duration::ZERO);
    let fetcher = Fetcher::new(1).with_cache_limit(25);
    for name in ["a", "b", "c"] {
        fetcher.fetch_blocking(&format!("{base}/{name}")).unwrap();
    }
    assert!(
        fetcher.cached(&format!("{base}/a")).is_none(),
        "the oldest went first"
    );
    assert!(fetcher.cached(&format!("{base}/b")).is_some());
    assert!(fetcher.cached(&format!("{base}/c")).is_some());
    assert_eq!(fetcher.stats().cached_bytes, 20);
    fetcher.insert("manual", vec![1, 2, 3]);
    assert_eq!(fetcher.cached("manual").unwrap().as_slice(), &[1, 2, 3]);
}

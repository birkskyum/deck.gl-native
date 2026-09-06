//! Debug aid: dump the composited frame to a PNG.
//!
//! Set `DECKGL_SCREENSHOT=/path/to/frame.png` to capture the host's color attachment right
//! after deck has drawn into it. The capture happens on the first frame rendered at least
//! `DECKGL_SCREENSHOT_AFTER_MS` milliseconds (default 8000, to give the basemap time to load
//! its tiles) after the first deck frame. It waits for the GPU, so it stalls that one frame.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use deck_gl::luma_gl::device::read_texture_rgba8;

use crate::DeckglHandle;

static DONE: AtomicBool = AtomicBool::new(false);
static FIRST_FRAME: OnceLock<Instant> = OnceLock::new();

pub(crate) fn maybe_capture(handle: &DeckglHandle, texture: &wgpu::Texture, format: wgpu::TextureFormat) {
    if DONE.load(Ordering::Relaxed) {
        return;
    }
    let Ok(path) = std::env::var("DECKGL_SCREENSHOT") else {
        return;
    };
    let after_ms: u64 = std::env::var("DECKGL_SCREENSHOT_AFTER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000);
    let first = FIRST_FRAME.get_or_init(Instant::now);
    if handle.frame < 2 || first.elapsed() < Duration::from_millis(after_ms) {
        return;
    }
    DONE.store(true, Ordering::Relaxed);

    let mut pixels = match read_texture_rgba8(&handle.device, &handle.queue, texture) {
        Ok(pixels) => pixels,
        Err(e) => {
            eprintln!("deck.gl-native: screenshot readback failed: {e}");
            return;
        }
    };
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    let size = texture.size();
    match write_png(&path, size.width, size.height, &pixels) {
        Ok(()) => eprintln!("deck.gl-native: wrote {path}"),
        Err(e) => eprintln!("deck.gl-native: screenshot failed: {e}"),
    }
}

/// Minimal PNG writer (RGBA8, stored deflate blocks) so the FFI crate has no image dependency.
fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    0xEDB8_8320 ^ (crc >> 1)
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
    fn adler32(data: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &d in data {
            a = (a + d as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = Vec::with_capacity(4 + data.len());
        body.extend_from_slice(kind);
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }

    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in rgba.chunks_exact(stride) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    // zlib stream with stored (uncompressed) deflate blocks
    let mut z = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    while let Some(block) = blocks.next() {
        let last = blocks.peek().is_none();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(block.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out)
}

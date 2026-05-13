"""Tests for the ffmpeg PCM tapper buffer."""

import asyncio

import numpy as np

from whospeaks.addon import tapper


def _make_reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


def _collect_windows(payload: bytes) -> list[np.ndarray]:
    async def _go():
        reader = _make_reader(payload)
        return [w async for w in tapper.stream_windows(reader)]

    return asyncio.run(_go())


def test_stream_windows_emits_after_buffer_primes():
    """4s of audio → 3 windows: at 2s, 3s, 4s."""
    samples = np.arange(4 * tapper.SAMPLE_RATE, dtype=np.int16)
    windows = _collect_windows(samples.tobytes())

    assert len(windows) == 3
    for w in windows:
        assert w.shape == (tapper.WINDOW_SAMPLES,)
        assert w.dtype == np.float32
        assert np.all(w >= -1.0) and np.all(w <= 1.0)


def test_stream_windows_sliding_buffer_content():
    """Adjacent windows overlap by WINDOW_SAMPLES - STRIDE_SAMPLES samples."""
    samples = np.arange(4 * tapper.SAMPLE_RATE, dtype=np.int16)
    windows = _collect_windows(samples.tobytes())

    w0 = (windows[0] * 32768.0).astype(np.int16)
    w1 = (windows[1] * 32768.0).astype(np.int16)
    overlap_w0 = w0[tapper.STRIDE_SAMPLES:]
    overlap_w1 = w1[: tapper.WINDOW_SAMPLES - tapper.STRIDE_SAMPLES]
    assert np.array_equal(overlap_w0, overlap_w1)


def test_stream_windows_eof_below_window():
    """If fewer than WINDOW_SAMPLES bytes are produced, yield nothing."""
    samples = np.zeros(tapper.STRIDE_SAMPLES, dtype=np.int16)  # only 1s
    windows = _collect_windows(samples.tobytes())
    assert windows == []


def test_stream_windows_partial_trailing_bytes_dropped():
    """2s + a few extra samples → only the 2s window is emitted before EOF."""
    samples = np.arange(2 * tapper.SAMPLE_RATE + 100, dtype=np.int16)
    windows = _collect_windows(samples.tobytes())
    assert len(windows) == 1

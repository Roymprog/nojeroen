"""Tap a live HTTPS audio stream via ffmpeg into a 2s rolling PCM buffer.

`open_stream(url)` spawns ffmpeg to decode `url` to raw 16 kHz mono s16le PCM
on stdout. `stream_windows(proc)` reads that PCM and yields a fresh 2-second
float32 window (in [-1, 1]) every STRIDE_S seconds, after the buffer is primed.
"""

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE_S = 2.0
STRIDE_S = 1.0
BYTES_PER_SAMPLE = 2  # s16le
WINDOW_SAMPLES = int(WINDOW_SIZE_S * SAMPLE_RATE)
STRIDE_SAMPLES = int(STRIDE_S * SAMPLE_RATE)
STRIDE_BYTES = STRIDE_SAMPLES * BYTES_PER_SAMPLE

FFMPEG_BIN = "ffmpeg"


async def open_stream(url: str) -> asyncio.subprocess.Process:
    """Spawn ffmpeg decoding `url` to raw 16 kHz mono s16le PCM on stdout."""
    return await asyncio.create_subprocess_exec(
        FFMPEG_BIN,
        "-loglevel", "error",
        "-nostdin",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "10",
        "-i", url,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def stream_windows(stdout: asyncio.StreamReader):
    """Yield 2-second float32 windows every STRIDE_S seconds.

    Reads STRIDE_BYTES at a time from `stdout`, maintains a rolling buffer
    of WINDOW_SAMPLES, and yields a fresh copy once the buffer is full.
    Stops when `stdout` reaches EOF.
    """
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    filled = 0

    while True:
        try:
            raw = await stdout.readexactly(STRIDE_BYTES)
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                logger.debug("ffmpeg EOF with %d partial bytes", len(exc.partial))
            return

        chunk = np.frombuffer(raw, dtype=np.int16)
        buffer = np.concatenate([buffer[len(chunk):], chunk])
        filled = min(filled + len(chunk), WINDOW_SAMPLES)

        if filled >= WINDOW_SAMPLES:
            yield buffer.astype(np.float32) / 32768.0


async def drain_stderr(stderr: asyncio.StreamReader, log: logging.Logger) -> None:
    """Forward ffmpeg's stderr lines into the addon's logger."""
    while True:
        line = await stderr.readline()
        if not line:
            return
        log.warning("ffmpeg: %s", line.decode("utf-8", errors="replace").rstrip())

"""Shared fixtures for WhoSpeaks tests."""

import os
import struct
import sys
import tempfile

import numpy as np
import pytest

# Add src to path so whospeaks package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SAMPLE_RATE = 16000


@pytest.fixture
def audio_2s():
    """2-second audio at 16kHz (random noise)."""
    return np.random.randn(int(2.0 * SAMPLE_RATE)).astype(np.float32)


@pytest.fixture
def audio_6s():
    """6-second audio at 16kHz (random noise)."""
    return np.random.randn(int(6.0 * SAMPLE_RATE)).astype(np.float32)


@pytest.fixture
def audio_1s():
    """1-second audio at 16kHz (shorter than window)."""
    return np.random.randn(int(1.0 * SAMPLE_RATE)).astype(np.float32)


@pytest.fixture
def tmp_model_dir():
    """Temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory(prefix="whospeaks_test_") as d:
        yield d


def create_wav_bytes(duration_s=2.0, sample_rate=SAMPLE_RATE):
    """Create a minimal valid WAV file in memory (16-bit PCM, mono)."""
    n_samples = int(duration_s * sample_rate)
    samples = np.random.randint(-32768, 32767, size=n_samples, dtype=np.int16)
    data = samples.tobytes()

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


def create_wav_file(path, duration_s=2.0, sample_rate=SAMPLE_RATE):
    """Write a WAV file to disk."""
    data = create_wav_bytes(duration_s, sample_rate)
    with open(path, "wb") as f:
        f.write(data)
    return path

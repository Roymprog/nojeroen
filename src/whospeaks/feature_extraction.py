import re

import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from whospeaks.config import (
    EMBEDDING_DIM,
    MIN_TAIL_S,
    POSITIVE_LABEL,
    SAMPLE_RATE,
    STRIDE_S,
    WINDOW_SIZE_S,
)

# Module-level encoder, loaded lazily
_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def parse_label(filename):
    """Extract label from segment filename like segment_1.0_3.5_JEROEN_VAN_INKEL.wav."""
    match = re.match(r"segment_[\d.]+_[\d.]+_(.+)\.wav$", filename)
    if match:
        return match.group(1)
    return None


def extract_windows_from_audio(
    audio,
    sr=SAMPLE_RATE,
    window_size_s=WINDOW_SIZE_S,
    stride_s=STRIDE_S,
    min_tail_s=MIN_TAIL_S,
):
    """Extract embeddings from fixed-size sliding windows over audio samples.

    Returns list of 256-dim embedding vectors.
    """
    encoder = get_encoder()
    duration = len(audio) / sr
    window_samples = int(window_size_s * sr)
    stride_samples = int(stride_s * sr)
    min_processed_samples = int(0.5 * sr)

    embeddings = []
    start_sample = 0

    while start_sample + window_samples <= len(audio):
        chunk = audio[start_sample : start_sample + window_samples]
        processed = preprocess_wav(chunk, source_sr=sr)
        if len(processed) >= min_processed_samples:
            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)
        start_sample += stride_samples

    # Handle remaining tail
    remaining_samples = len(audio) - start_sample
    remaining_s = remaining_samples / sr
    if remaining_s >= min_tail_s and start_sample < len(audio):
        chunk = audio[start_sample:]
        processed = preprocess_wav(chunk, source_sr=sr)
        if len(processed) >= min_processed_samples:
            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)

    return embeddings


def extract_windows_from_file(
    wav_path,
    sr=SAMPLE_RATE,
    window_size_s=WINDOW_SIZE_S,
    stride_s=STRIDE_S,
    min_tail_s=MIN_TAIL_S,
):
    """Load a WAV file and extract embeddings from sliding windows."""
    audio, _ = librosa.load(wav_path, sr=sr)
    return extract_windows_from_audio(
        audio,
        sr=sr,
        window_size_s=window_size_s,
        stride_s=stride_s,
        min_tail_s=min_tail_s,
    )


def embed_chunk(audio_chunk, sr=SAMPLE_RATE):
    """Embed a single audio chunk (for real-time inference)."""
    encoder = get_encoder()
    processed = preprocess_wav(audio_chunk, source_sr=sr)
    if len(processed) < int(0.5 * sr):
        return None
    return encoder.embed_utterance(processed)

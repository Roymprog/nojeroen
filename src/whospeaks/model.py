"""Model wrapper: load trained model from disk and provide predictions."""

import json
import os
import time

import joblib
import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

_REQUIRED_CONFIG_KEYS = {"threshold", "sample_rate", "embedding_dim", "feature_type"}


class SpeakerPredictor:
    """Loads a trained model and provides predictions."""

    def __init__(self, model, encoder, threshold: float, config: dict, sample_rate: int = 16000):
        self._model = model
        self._encoder = encoder
        self._threshold = threshold
        self._config = config
        self._sr = sample_rate

    @classmethod
    def load(cls, model_dir: str = "models/") -> "SpeakerPredictor":
        """Load model.joblib and config.json from model_dir.

        Raises:
            FileNotFoundError: If model.joblib or config.json is missing.
            ValueError: If config is missing required keys or has invalid values.
        """
        model_path = os.path.join(model_dir, "model.joblib")
        config_path = os.path.join(model_dir, "config.json")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        model_data = joblib.load(model_path)

        # train.py saves {"model": lgbm, "threshold": float}
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
        else:
            model = model_data

        with open(config_path) as f:
            config = json.load(f)

        missing = _REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing:
            raise ValueError(f"Config missing required keys: {missing}")

        if config["feature_type"] != "resemblyzer_ge2e":
            raise ValueError(
                f"Unsupported feature_type: {config['feature_type']}. "
                "Expected 'resemblyzer_ge2e'."
            )

        threshold = config["threshold"]
        sample_rate = config["sample_rate"]

        encoder = VoiceEncoder()

        return cls(model=model, encoder=encoder, threshold=threshold, config=config, sample_rate=sample_rate)

    def predict_from_embedding(self, embedding: np.ndarray) -> dict:
        """Predict from a pre-computed 256-dim embedding vector.

        Args:
            embedding: 1D numpy array of shape (256,).

        Returns:
            {"label": "JEROEN_VAN_INKEL" | "OTHER", "confidence": float}
        """
        prob = self._model.predict_proba(embedding.reshape(1, -1))[0, 1]
        label = "JEROEN_VAN_INKEL" if prob >= self._threshold else "OTHER"
        confidence = float(prob) if label == "JEROEN_VAN_INKEL" else float(1.0 - prob)
        return {"label": label, "confidence": round(confidence, 4)}

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        """Predict speaker for an audio segment.

        Args:
            audio: 1D numpy array of audio samples (mono, any length >= 2s).
            sr: Sample rate in Hz.

        Returns:
            {"label": "JEROEN_VAN_INKEL" | "OTHER", "confidence": float}

        Raises:
            ValueError: If audio is empty or not 1D.
        """
        if audio.ndim != 1 or len(audio) == 0:
            raise ValueError(
                f"Audio must be a non-empty 1D array, got shape {audio.shape}"
            )

        if sr != self._sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sr)

        processed = preprocess_wav(audio, source_sr=self._sr)
        embedding = self._encoder.embed_utterance(processed)
        prob = self._model.predict_proba(embedding.reshape(1, -1))[0, 1]

        label = "JEROEN_VAN_INKEL" if prob >= self._threshold else "OTHER"
        confidence = float(prob) if label == "JEROEN_VAN_INKEL" else float(1.0 - prob)

        return {"label": label, "confidence": round(confidence, 4)}

    def predict_window(self, file_path: str, position: float) -> dict:
        """Predict speaker for a 2s window ending at position in the given file.

        Per RFC-007, windows shorter than 2s are padded with silence on the left to reach 2s.

        Args:
            file_path: path to WAV file.
            position: end timestamp in seconds (must be >= 0.0 and <= file_duration).

        Returns:
            Same dict as predict().

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If position < 0.0 or exceeds file duration.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check position doesn't exceed file duration
        file_duration = librosa.get_duration(path=file_path)
        if position < 0.0 or position > file_duration:
            raise ValueError(
                f"Position must be in [0.0, {file_duration:.2f}s], got {position}"
            )

        window_sec = 2.0
        offset = max(0.0, position - window_sec)
        duration = position - offset

        audio, sr = librosa.load(file_path, sr=self._sr, offset=offset, duration=duration)

        # RFC-007: pad short windows on the left with silence to reach 2s
        if len(audio) < window_sec * sr:
            pad_samples = int((window_sec * sr) - len(audio))
            audio = np.concatenate([np.zeros(pad_samples, dtype=audio.dtype), audio])

        return self.predict(audio, sr)

    def compute_inference_latency(self, duration_s: float = 2.0) -> float:
        """Benchmark feature extraction + prediction on a synthetic audio window.

        Args:
            duration_s: Length of synthetic audio in seconds (default 2.0).

        Returns:
            Latency in milliseconds.
        """
        audio = np.random.randn(int(duration_s * self._sr)).astype(np.float32)
        start = time.perf_counter()
        self.predict(audio, self._sr)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return round(elapsed_ms, 2)

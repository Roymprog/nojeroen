"""Unit tests for SpeakerPredictor (model.py) — task #6."""

import json
import os
import tempfile

import joblib
import numpy as np
import pytest

from whospeaks.model import SpeakerPredictor

SAMPLE_RATE = 16000


class _FakeLGBM:
    """Picklable fake LightGBM model for testing."""

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


def _make_mock_lgbm():
    return _FakeLGBM()


def _make_config(**overrides):
    config = {
        "feature_type": "resemblyzer_ge2e",
        "embedding_dim": 256,
        "window_size_s": 2.0,
        "stride_s": 1.0,
        "sample_rate": 16000,
        "threshold": 0.65,
    }
    config.update(overrides)
    return config


def _write_model_artifacts(model_dir, model=None, config=None):
    """Write model.joblib and config.json to model_dir."""
    if model is None:
        model = _make_mock_lgbm()
    if config is None:
        config = _make_config()

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)


class TestLoad:
    def test_load_missing_model_file(self, tmp_path):
        """SpeakerPredictor.load() raises FileNotFoundError when model.joblib is missing."""
        config = _make_config()
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(FileNotFoundError, match="model.joblib"):
            SpeakerPredictor.load(str(tmp_path))

    def test_load_missing_config_file(self, tmp_path):
        """SpeakerPredictor.load() raises FileNotFoundError when config.json is missing."""
        joblib.dump(_make_mock_lgbm(), tmp_path / "model.joblib")

        with pytest.raises(FileNotFoundError, match="config.json"):
            SpeakerPredictor.load(str(tmp_path))

    def test_load_config_missing_keys(self, tmp_path):
        """SpeakerPredictor.load() raises ValueError when config lacks required keys."""
        joblib.dump(_make_mock_lgbm(), tmp_path / "model.joblib")
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"threshold": 0.5}, f)

        with pytest.raises(ValueError, match="missing required keys"):
            SpeakerPredictor.load(str(tmp_path))

    def test_load_invalid_feature_type(self, tmp_path):
        """SpeakerPredictor.load() raises ValueError for unsupported feature_type."""
        config = _make_config(feature_type="mfcc_summary")
        _write_model_artifacts(str(tmp_path), config=config)

        with pytest.raises(ValueError, match="Unsupported feature_type"):
            SpeakerPredictor.load(str(tmp_path))

    def test_load_success(self, tmp_path):
        """SpeakerPredictor.load() succeeds with valid artifacts."""
        _write_model_artifacts(str(tmp_path))
        predictor = SpeakerPredictor.load(str(tmp_path))
        assert predictor._threshold == 0.65
        assert predictor._sr == 16000


class TestPredict:
    @pytest.fixture
    def predictor(self, tmp_path):
        _write_model_artifacts(str(tmp_path))
        return SpeakerPredictor.load(str(tmp_path))

    def test_predict_output_shape_and_range(self, predictor):
        """predict() returns dict with valid label and confidence in [0.0, 1.0]."""
        audio = np.random.randn(int(2.0 * SAMPLE_RATE)).astype(np.float32)
        result = predictor.predict(audio, SAMPLE_RATE)

        assert "label" in result
        assert result["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty_audio_raises(self, predictor):
        """predict() raises ValueError on empty audio."""
        with pytest.raises(ValueError, match="non-empty 1D"):
            predictor.predict(np.array([]), SAMPLE_RATE)

    def test_predict_2d_audio_raises(self, predictor):
        """predict() raises ValueError on 2D audio."""
        audio_2d = np.random.randn(2, SAMPLE_RATE).astype(np.float32)
        with pytest.raises(ValueError, match="non-empty 1D"):
            predictor.predict(audio_2d, SAMPLE_RATE)

    def test_predict_resamples_non_16khz(self, predictor):
        """predict() handles audio at a different sample rate."""
        audio_22k = np.random.randn(int(2.0 * 22050)).astype(np.float32)
        result = predictor.predict(audio_22k, 22050)
        assert result["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_short_audio(self, predictor):
        """predict() handles short audio (< 2s) gracefully — returns valid prediction."""
        audio_short = np.random.randn(int(0.5 * SAMPLE_RATE)).astype(np.float32)
        result = predictor.predict(audio_short, SAMPLE_RATE)
        assert result["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert 0.0 <= result["confidence"] <= 1.0


class TestPredictWindow:
    @pytest.fixture
    def predictor(self, tmp_path):
        _write_model_artifacts(str(tmp_path))
        return SpeakerPredictor.load(str(tmp_path))

    def _make_wav(self, tmp_path, duration_s=6.0):
        import wave
        import io

        path = os.path.join(str(tmp_path), "test.wav")
        n_samples = int(duration_s * SAMPLE_RATE)
        samples = np.random.randint(-32768, 32767, size=n_samples, dtype=np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
        return path

    def test_predict_window_various_positions(self, predictor, tmp_path):
        """predict_window() returns valid predictions at multiple positions."""
        wav_path = self._make_wav(tmp_path, 6.0)
        for pos in [2.0, 3.5, 5.0]:
            result = predictor.predict_window(wav_path, pos)
            assert result["label"] in ("JEROEN_VAN_INKEL", "OTHER")
            assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_window_missing_file(self, predictor):
        """predict_window() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            predictor.predict_window("/nonexistent/test.wav", 3.0)

    def test_predict_window_position_too_low(self, predictor, tmp_path):
        """predict_window() pads short windows on the left per RFC-007."""
        wav_path = self._make_wav(tmp_path, 3.0)
        # Position < 2.0 should succeed with left-padding to reach 2s
        result = predictor.predict_window(wav_path, 1.5)
        assert result["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_window_position_beyond_duration(self, predictor, tmp_path):
        """predict_window() raises ValueError when position exceeds file duration."""
        wav_path = self._make_wav(tmp_path, 6.0)
        with pytest.raises(ValueError, match="Position must be in"):
            predictor.predict_window(wav_path, 100.0)


class TestComputeInferenceLatency:
    @pytest.fixture
    def predictor(self, tmp_path):
        _write_model_artifacts(str(tmp_path))
        return SpeakerPredictor.load(str(tmp_path))

    def test_compute_inference_latency_returns_positive(self, predictor):
        """compute_inference_latency() returns a positive float in ms."""
        latency_ms = predictor.compute_inference_latency()
        assert isinstance(latency_ms, float)
        assert latency_ms > 0

    def test_compute_inference_latency_under_1s(self, predictor):
        """AC-006: compute_inference_latency() completes within 1 second."""
        latency_ms = predictor.compute_inference_latency()
        assert latency_ms < 1000, f"Latency {latency_ms}ms exceeds 1s (AC-006 requirement)"

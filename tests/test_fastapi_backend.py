"""
Integration tests for the FastAPI backend.

Covers acceptance criteria:
- AC-002: Model loads from disk on startup without retraining
- AC-003: App starts via single command, HTTP 200 on root URL
- AC-004: Accepts any WAV file (not restricted to data/ directory)
- AC-006: Prediction latency < 1 second behind playback
- AC-007: Prediction response shows binary label + confidence score

Reference: RFC-007 specifies prediction endpoint returns:
    {"label": "JEROEN_VAN_INKEL" | "OTHER", "confidence": 0.0 to 1.0}
"""

import io
import os
import struct
import time

import numpy as np
import pytest

# --- Test helpers ---

SAMPLE_RATE = 16000
WINDOW_SIZE_S = 2.0


def create_wav_bytes(duration_s=2.0, sample_rate=SAMPLE_RATE):
    """Create a minimal valid WAV file in memory (16-bit PCM, mono)."""
    n_samples = int(duration_s * sample_rate)
    samples = np.random.randint(-32768, 32767, size=n_samples, dtype=np.int16)
    data = samples.tobytes()

    # WAV header (44 bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,       # chunk size
        1,        # PCM format
        1,        # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,        # block align
        16,       # bits per sample
        b"data",
        len(data),
    )
    return header + data


# ============================================================
# AC-003: App starts via single command, HTTP 200 on root
# ============================================================
class TestAppStartup:

    def test_root_returns_200(self):
        """AC-003: The web application is accessible and returns HTTP 200 on root."""
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # client = TestClient(app)
        # response = client.get("/")
        # assert response.status_code == 200
        pytest.skip("Awaiting production code -- module not yet available")

    def test_model_loads_on_startup(self):
        """
        AC-002: The app loads a pre-trained model on startup without retraining.
        Verify model file is read, not training pipeline.
        """
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # Verify model is loaded by checking the app state
        # client = TestClient(app)
        # assert app.state.predictor is not None
        # assert app.state.predictor.model is not None
        pytest.skip("Awaiting production code -- module not yet available")


# ============================================================
# AC-007: Prediction endpoint returns label + confidence
# ============================================================
class TestPredictEndpoint:

    def test_predict_returns_label_and_confidence(self):
        """
        AC-007: Response contains binary label (JEROEN_VAN_INKEL / OTHER)
        and a 0-100% confidence score.
        """
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # client = TestClient(app)
        # wav_bytes = create_wav_bytes(duration_s=2.0)
        # response = client.post(
        #     "/predict",
        #     files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        # )
        # assert response.status_code == 200
        # data = response.json()
        # assert "label" in data
        # assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        # assert "confidence" in data
        # assert 0.0 <= data["confidence"] <= 1.0
        pytest.skip("Awaiting production code -- module not yet available")

    def test_predict_confidence_range(self):
        """Confidence is always in [0.0, 1.0] regardless of input."""
        # TODO: Import from production module
        # Test with several different audio samples
        # for _ in range(5):
        #     wav_bytes = create_wav_bytes(duration_s=2.0)
        #     response = client.post("/predict", ...)
        #     data = response.json()
        #     assert 0.0 <= data["confidence"] <= 1.0
        pytest.skip("Awaiting production code -- module not yet available")


# ============================================================
# AC-004: Accepts any WAV file
# ============================================================
class TestAudioAcceptance:

    def test_accepts_wav_not_in_data_dir(self):
        """AC-004: The app accepts any WAV file, not restricted to data/ directory."""
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # wav_bytes = create_wav_bytes(duration_s=3.0)
        # client = TestClient(app)
        # response = client.post(
        #     "/predict",
        #     files={"file": ("outside_data.wav", io.BytesIO(wav_bytes), "audio/wav")},
        # )
        # assert response.status_code == 200
        pytest.skip("Awaiting production code -- module not yet available")

    def test_rejects_non_wav(self):
        """Error handling: non-WAV files should return an appropriate error."""
        # TODO: Import from production module
        # client = TestClient(app)
        # response = client.post(
        #     "/predict",
        #     files={"file": ("test.mp3", io.BytesIO(b"not a wav"), "audio/mpeg")},
        # )
        # assert response.status_code in (400, 422)
        pytest.skip("Awaiting production code -- module not yet available")


# ============================================================
# AC-006: Prediction latency < 1 second
# ============================================================
class TestLatency:

    def test_predict_latency_under_1_second(self):
        """AC-006: Prediction latency is under 1 second behind live playback position."""
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # client = TestClient(app)
        # wav_bytes = create_wav_bytes(duration_s=2.0)
        #
        # # Warm up
        # client.post("/predict", files={"file": ("w.wav", io.BytesIO(wav_bytes), "audio/wav")})
        #
        # # Timed run
        # start = time.perf_counter()
        # response = client.post(
        #     "/predict",
        #     files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        # )
        # elapsed_ms = (time.perf_counter() - start) * 1000
        # assert response.status_code == 200
        # assert elapsed_ms < 1000, f"Prediction took {elapsed_ms:.0f}ms, exceeds 1s"
        pytest.skip("Awaiting production code -- module not yet available")


# ============================================================
# Error handling
# ============================================================
class TestErrorHandling:

    def test_missing_model_file(self):
        """App should fail gracefully if model file is missing."""
        # TODO: Import from production module
        # Temporarily rename model file, try to start app, verify clear error
        pytest.skip("Awaiting production code -- module not yet available")

    def test_invalid_audio_data(self):
        """Endpoint handles corrupted audio gracefully."""
        # TODO: Import from production module
        # client = TestClient(app)
        # response = client.post(
        #     "/predict",
        #     files={"file": ("bad.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
        # )
        # assert response.status_code in (400, 422, 500)
        pytest.skip("Awaiting production code -- module not yet available")


# ============================================================
# WebSocket streaming predictions (if implemented per RFC-007)
# ============================================================
class TestWebSocket:

    def test_websocket_prediction_stream(self):
        """
        If WebSocket is used for streaming: connect, send audio chunk,
        receive prediction with label + confidence.
        """
        # TODO: Import from production module
        # from fastapi.testclient import TestClient
        # from whospeaks.app import app
        #
        # client = TestClient(app)
        # with client.websocket_connect("/ws/predict") as ws:
        #     wav_bytes = create_wav_bytes(duration_s=2.0)
        #     ws.send_bytes(wav_bytes)
        #     data = ws.receive_json()
        #     assert "label" in data
        #     assert "confidence" in data
        pytest.skip("Awaiting production code -- module not yet available")

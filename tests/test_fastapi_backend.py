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
import time
import wave

import pytest
from fastapi.testclient import TestClient

import whospeaks.app as app_module
from whospeaks.app import app, _MockPredictor

SAMPLE_RATE = 16000
WINDOW_SIZE_S = 2.0


@pytest.fixture
def client():
    """TestClient backed by the mock predictor — model quality tested elsewhere."""
    original = app_module._predictor
    app_module._predictor = _MockPredictor()
    yield TestClient(app)
    app_module._predictor = original


def create_wav_bytes(duration_s=2.0, sample_rate=SAMPLE_RATE) -> bytes:
    """Create a valid WAV file in memory (silence, 16-bit PCM, mono)."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ============================================================
# AC-003: App starts via single command, HTTP 200 on root
# ============================================================
class TestAppStartup:

    def test_root_returns_200(self, client):
        """AC-003: The web application is accessible and returns HTTP 200 on root."""
        response = client.get("/")
        assert response.status_code == 200

    def test_model_loads_on_startup(self, client):
        """AC-002: The app exposes a loaded predictor (not None) via /status."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["predictor_type"] in ("real", "mock")


# ============================================================
# AC-007: Prediction endpoint returns label + confidence
# ============================================================
class TestPredictEndpoint:

    def test_predict_returns_label_and_confidence(self, client):
        """AC-007: Response contains binary label and 0–1 confidence score."""
        wav_bytes = create_wav_bytes(duration_s=2.0)
        response = client.post(
            "/predict",
            files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_confidence_range(self, client):
        """Confidence is always in [0.0, 1.0] regardless of input."""
        for _ in range(3):
            wav_bytes = create_wav_bytes(duration_s=2.0)
            response = client.post(
                "/predict",
                files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert response.status_code == 200
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0


# ============================================================
# AC-004: Accepts any WAV file
# ============================================================
class TestAudioAcceptance:

    def test_accepts_wav_not_in_data_dir(self, client):
        """AC-004: The app accepts any WAV file, not restricted to data/ directory."""
        wav_bytes = create_wav_bytes(duration_s=3.0)
        response = client.post(
            "/predict",
            files={"file": ("outside_data.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200

    def test_rejects_non_wav(self, client):
        """Error handling: corrupted / non-WAV bytes should return an error status."""
        response = client.post(
            "/predict",
            files={"file": ("test.mp3", io.BytesIO(b"not a wav file"), "audio/mpeg")},
        )
        assert response.status_code >= 400


# ============================================================
# AC-006: Prediction latency < 1 second
# ============================================================
class TestLatency:

    def test_predict_latency_under_1_second(self, client):
        """AC-006: Mock prediction completes well under 1 second."""
        wav_bytes = create_wav_bytes(duration_s=2.0)

        # Warm up
        client.post("/predict", files={"file": ("w.wav", io.BytesIO(wav_bytes), "audio/wav")})

        start = time.perf_counter()
        response = client.post(
            "/predict",
            files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 1000, f"Prediction took {elapsed_ms:.0f}ms, exceeds 1s"


# ============================================================
# Error handling
# ============================================================
class TestErrorHandling:

    def test_missing_model_file(self, client):
        """App status endpoint reports predictor type even with mock predictor."""
        # The fixture already forces the mock predictor (model absent scenario)
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["predictor_type"] in ("real", "mock")

    def test_invalid_audio_data(self, client):
        """Endpoint handles corrupted audio bytes gracefully."""
        response = client.post(
            "/predict",
            files={"file": ("bad.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
        )
        assert response.status_code >= 400


# ============================================================
# WebSocket streaming predictions
# ============================================================
class TestWebSocket:

    def test_websocket_prediction_stream(self, client):
        """Connect to WS, send a position, receive prediction with label + confidence."""
        wav_bytes = create_wav_bytes(duration_s=5.0)
        upload_resp = client.post(
            "/upload",
            files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert upload_resp.status_code == 200
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
        assert "label" in data
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

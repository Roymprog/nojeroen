"""
End-to-end tests with the real SpeakerPredictor.

Covers the full flow we validated manually:
  1. Server starts with the real model loaded (predictor_type == "real")
  2. Upload a WAV file  → file_id, duration, sample_rate
  3. GET /audio/{file_id} → serves the WAV bytes back
  4. WebSocket /ws/predict/{file_id} → streams predictions for multiple positions

These tests require models/model.joblib and models/config.json to exist.
They are skipped automatically when the model files are absent.
"""

import io
import os
import wave

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Skip guard — skip the whole module if the model artefacts aren't present.
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_model_missing = not (
    os.path.isfile(os.path.join(MODEL_DIR, "model.joblib"))
    and os.path.isfile(os.path.join(MODEL_DIR, "config.json"))
)
pytestmark = pytest.mark.skipif(
    _model_missing, reason="models/model.joblib or models/config.json not found"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with the real lifespan (loads SpeakerPredictor once)."""
    # Import order matters: lightgbm native code must load before torch (resemblyzer)
    # to avoid the OpenMP double-initialisation crash on macOS. app.py already
    # enforces this via its top-level `import lightgbm` statement.
    from whospeaks.app import app
    with TestClient(app) as c:
        yield c


def _make_wav(duration_s: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Return a valid mono WAV file (white-noise, 16-bit PCM)."""
    import numpy as np
    n = int(duration_s * sample_rate)
    samples = (np.random.randn(n) * 0.5 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def uploaded_file(client):
    """Upload a 5-second WAV once and return (file_id, duration, sample_rate)."""
    wav = _make_wav(duration_s=5.0, sample_rate=16000)
    resp = client.post("/upload", files={"file": ("e2e.wav", wav, "audio/wav")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    return data["file_id"], data["duration"], data["sample_rate"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServerStartup:
    def test_status_ok(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_real_model_loaded(self, client):
        data = client.get("/status").json()
        assert data["predictor_type"] == "real", (
            f"Expected real predictor, got {data['predictor_type']!r}"
        )

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestUpload:
    def test_upload_returns_file_id(self, client, uploaded_file):
        file_id, duration, sample_rate = uploaded_file
        assert file_id  # non-empty UUID string

    def test_upload_duration_correct(self, client, uploaded_file):
        _, duration, _ = uploaded_file
        assert abs(duration - 5.0) < 0.1, f"Expected ~5s, got {duration}"

    def test_upload_sample_rate_correct(self, client, uploaded_file):
        _, _, sample_rate = uploaded_file
        assert sample_rate == 16000


class TestAudioRetrieval:
    def test_get_audio_200(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        resp = client.get(f"/audio/{file_id}")
        assert resp.status_code == 200

    def test_get_audio_content_type(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        resp = client.get(f"/audio/{file_id}")
        assert "audio" in resp.headers["content-type"]

    def test_get_audio_returns_wav_bytes(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        resp = client.get(f"/audio/{file_id}")
        # WAV files start with the RIFF header
        assert resp.content[:4] == b"RIFF"

    def test_get_audio_unknown_id_returns_404(self, client):
        resp = client.get("/audio/does-not-exist")
        assert resp.status_code == 404


class TestWebSocketPredictions:
    def test_connects_and_receives_prediction(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
        assert "label" in data
        assert "confidence" in data
        assert "position" in data

    def test_label_is_binary(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")

    def test_confidence_in_range(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_position_echoed(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 2.5})
            data = ws.receive_json()
        assert data["position"] == pytest.approx(2.5)

    def test_multiple_positions_stream(self, client, uploaded_file):
        file_id, _, _ = uploaded_file
        positions = [0.5, 1.0, 2.0, 3.0, 4.0]
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            for pos in positions:
                ws.send_json({"position": pos})
                data = ws.receive_json()
                assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER"), \
                    f"Unexpected label at pos={pos}: {data}"
                assert 0.0 <= data["confidence"] <= 1.0, \
                    f"Confidence out of range at pos={pos}: {data['confidence']}"

    def test_short_window_padded(self, client, uploaded_file):
        """Positions < 2s are padded per RFC-007 — should still return a valid prediction."""
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 0.5})
            data = ws.receive_json()
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_caching_same_quantized_position(self, client, uploaded_file):
        """Two positions that quantize to the same 0.5s bucket return identical results."""
        file_id, _, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            r1 = ws.receive_json()
            ws.send_json({"position": 3.1})  # quantizes to 3.0
            r2 = ws.receive_json()
        assert r1["label"] == r2["label"]
        assert r1["confidence"] == r2["confidence"]

    def test_out_of_range_position_returns_error(self, client, uploaded_file):
        file_id, duration, _ = uploaded_file
        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": duration + 10.0})
            data = ws.receive_json()
        assert "error" in data

    def test_invalid_file_id_returns_error(self, client):
        with client.websocket_connect("/ws/predict/no-such-file") as ws:
            data = ws.receive_json()
        assert "error" in data

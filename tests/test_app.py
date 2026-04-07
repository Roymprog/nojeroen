"""Integration tests for the WhoSpeaks FastAPI app."""

import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from whospeaks.app import app


@pytest.fixture
def client():
    return TestClient(app)


def make_wav_bytes(duration_s: float = 3.0, sample_rate: int = 22050) -> bytes:
    """Generate a valid WAV file in memory."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence
        wf.writeframes(b"\x00\x00" * n_samples)
    buf.seek(0)
    return buf.read()


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]
        assert "<audio" in resp.text

    def test_root_has_upload_form(self, client):
        resp = client.get("/")
        assert 'type="file"' in resp.text

    def test_root_has_prediction_display(self, client):
        resp = client.get("/")
        assert "prediction-label" in resp.text
        assert "confidence-bar" in resp.text
        assert "confidence-value" in resp.text


class TestUploadEndpoint:
    def test_upload_wav(self, client):
        wav = make_wav_bytes(3.0, 22050)
        resp = client.post("/upload", files={"file": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        data = resp.json()
        assert "file_id" in data
        assert "duration" in data
        assert "sample_rate" in data
        assert data["duration"] > 2.0

    def test_upload_returns_correct_metadata(self, client):
        wav = make_wav_bytes(5.0, 16000)
        resp = client.post("/upload", files={"file": ("test.wav", wav, "audio/wav")})
        data = resp.json()
        assert abs(data["duration"] - 5.0) < 0.1
        assert data["sample_rate"] == 16000


class TestAudioEndpoint:
    def test_get_audio(self, client):
        wav = make_wav_bytes(3.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]
        resp = client.get(f"/audio/{file_id}")
        assert resp.status_code == 200
        assert "audio" in resp.headers["content-type"]

    def test_get_audio_not_found(self, client):
        resp = client.get("/audio/nonexistent-id")
        assert resp.status_code != 200


class TestWebSocketPrediction:
    def test_websocket_prediction(self, client):
        # Upload a file first
        wav = make_wav_bytes(5.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            # Send position >= 2.0 for valid prediction
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
            assert "label" in data
            assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "position" in data

    def test_websocket_insufficient_audio(self, client):
        """Per RFC-007, short windows are padded with silence, so all positions return valid predictions."""
        wav = make_wav_bytes(5.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            # Position < 2.0 is now valid (padded on the left per RFC-007)
            ws.send_json({"position": 1.0})
            data = ws.receive_json()
            assert "label" in data
            assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["position"] == 1.0

    def test_websocket_multiple_positions(self, client):
        wav = make_wav_bytes(10.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            for pos in [2.5, 3.0, 4.5, 6.0]:
                ws.send_json({"position": pos})
                data = ws.receive_json()
                assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
                assert 0.0 <= data["confidence"] <= 1.0

    def test_websocket_caching(self, client):
        """Same quantized position should return cached result."""
        wav = make_wav_bytes(5.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data1 = ws.receive_json()
            ws.send_json({"position": 3.1})  # same quantized pos (3.0)
            data2 = ws.receive_json()
            assert data1["label"] == data2["label"]
            assert data1["confidence"] == data2["confidence"]

    def test_websocket_invalid_file_id(self, client):
        with client.websocket_connect("/ws/predict/nonexistent") as ws:
            data = ws.receive_json()
            assert "error" in data


class TestStatusEndpoint:
    def test_status_returns_ok(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_returns_label_and_confidence(self, client):
        wav = make_wav_bytes(3.0, 16000)
        resp = client.post("/predict", files={"file": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0


class TestBinaryLabelAndConfidence:
    """AC-007: Both binary label and confidence score must be present."""

    def test_prediction_has_label_and_confidence(self, client):
        wav = make_wav_bytes(5.0)
        upload_resp = client.post(
            "/upload", files={"file": ("test.wav", wav, "audio/wav")}
        )
        file_id = upload_resp.json()["file_id"]

        with client.websocket_connect(f"/ws/predict/{file_id}") as ws:
            ws.send_json({"position": 3.0})
            data = ws.receive_json()
            # Binary label
            assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
            # Confidence as float 0-1
            assert isinstance(data["confidence"], float)
            assert 0.0 <= data["confidence"] <= 1.0

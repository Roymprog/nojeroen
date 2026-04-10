"""
End-to-end tests for the web application (frontend + backend).

Covers acceptance criteria:
- AC-005: Predictions update continuously as sliding window during playback
- AC-007: Binary label + 0-100% confidence score visible in UI
- AC-010: Confidence bar or visual indicator alongside numeric score

Note: Full browser-based UI testing (AC-005, AC-010) requires manual verification
or a browser automation tool. The tests below verify the backend contracts that
the frontend depends on.
"""

import io
import time
import wave

import pytest
from fastapi.testclient import TestClient

import whospeaks.app as app_module
from whospeaks.app import app, _MockPredictor

SAMPLE_RATE = 16000


@pytest.fixture
def client():
    """TestClient backed by the mock predictor."""
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
# AC-005: Sliding window predictions during playback
# ============================================================
class TestSlidingWindowPredictions:

    def test_multiple_sequential_predictions(self, client):
        """AC-005: Backend handles rapid sequential predictions for a sliding window."""
        wav_bytes = create_wav_bytes(duration_s=2.0)
        predictions = []

        for _ in range(5):
            response = client.post(
                "/predict",
                files={"file": ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert response.status_code == 200
            data = response.json()
            predictions.append(data)
            assert "label" in data
            assert "confidence" in data

        assert len(predictions) == 5

    def test_prediction_throughput_supports_realtime(self, client):
        """AC-005 + AC-006: At least 1 prediction per second (1s stride window)."""
        wav_bytes = create_wav_bytes(duration_s=2.0)

        # Warm up
        client.post("/predict", files={"file": ("w.wav", io.BytesIO(wav_bytes), "audio/wav")})

        start = time.perf_counter()
        for _ in range(5):
            response = client.post(
                "/predict",
                files={"file": ("chunk.wav", io.BytesIO(create_wav_bytes(2.0)), "audio/wav")},
            )
            assert response.status_code == 200
        total_s = time.perf_counter() - start

        assert total_s < 5.0, f"5 predictions took {total_s:.1f}s, too slow for real-time"


# ============================================================
# AC-007 + AC-010: UI response contract
# ============================================================
class TestUIResponseContract:

    def test_response_includes_all_ui_fields(self, client):
        """AC-007 + AC-010: Response includes all fields needed by the UI."""
        wav_bytes = create_wav_bytes(duration_s=2.0)
        response = client.post(
            "/predict",
            files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        data = response.json()

        assert "label" in data
        assert "confidence" in data
        assert data["label"] in ("JEROEN_VAN_INKEL", "OTHER")
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0


# ============================================================
# AC-004: Upload flow for arbitrary WAV files
# ============================================================
class TestUploadFlow:

    def test_upload_arbitrary_wav(self, client):
        """AC-004: Upload a WAV file not from data/ directory and get predictions."""
        wav_bytes = create_wav_bytes(duration_s=5.0)
        response = client.post(
            "/upload",
            files={"file": ("my_recording.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "duration" in data

    def test_upload_long_audio(self, client):
        """Verify upload of longer audio files (30s) works correctly."""
        wav_bytes = create_wav_bytes(duration_s=30.0)
        response = client.post(
            "/upload",
            files={"file": ("long.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert abs(data["duration"] - 30.0) < 0.5


# ============================================================
# Frontend static assets (served by FastAPI)
# ============================================================
class TestStaticAssets:

    def test_frontend_html_served(self, client):
        """Verify the frontend HTML page is served by FastAPI."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "audio" in response.text.lower()

    def test_frontend_has_confidence_indicator(self, client):
        """AC-010: Frontend HTML includes a confidence bar or visual indicator element."""
        response = client.get("/")
        html = response.text
        has_indicator = any(tag in html for tag in [
            "progress", "meter", "confidence-bar", "confidence_bar",
            "bar", "indicator",
        ])
        assert has_indicator, "Frontend missing visual confidence indicator (AC-010)"

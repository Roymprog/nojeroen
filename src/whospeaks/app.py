"""FastAPI application for WhoSpeaks audio classifier."""

import math
import os
import tempfile
import uuid
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="WhoSpeaks")

# Temp storage for uploaded files
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="whospeaks_"))
# file_id -> {path, duration, sample_rate}
uploaded_files: dict[str, dict] = {}
# prediction cache: file_id -> {quantized_position -> prediction}
prediction_cache: dict[str, dict[float, dict]] = {}

STATIC_DIR = Path(__file__).parent / "static"


# --- Model loading ---
# Try to load real SpeakerPredictor; fall back to mock
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    try:
        from whospeaks.model import SpeakerPredictor

        _predictor = SpeakerPredictor.load()
        return _predictor
    except Exception:
        pass

    # Mock predictor for frontend development
    _predictor = _MockPredictor()
    return _predictor


class _MockPredictor:
    """Returns mock predictions for frontend development."""

    def predict_window(self, file_path: str, position: float) -> dict:
        # Deterministic mock based on position for consistency
        sin_val = math.sin(position * 0.5)
        confidence = 0.3 + 0.4 * abs(sin_val)
        if sin_val > 0.3:
            return {"label": "JEROEN_VAN_INKEL", "confidence": round(confidence, 3)}
        return {"label": "OTHER", "confidence": round(1.0 - confidence, 3)}

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        return {"label": "OTHER", "confidence": 0.5}


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.get("/status")
async def status():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/upload")
async def upload_file(file: UploadFile):
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}.wav"

    content = await file.read()
    dest.write_bytes(content)

    y, sr = librosa.load(str(dest), sr=None)
    duration = float(len(y)) / sr

    uploaded_files[file_id] = {
        "path": str(dest),
        "duration": duration,
        "sample_rate": sr,
    }
    prediction_cache[file_id] = {}

    return {"file_id": file_id, "duration": duration, "sample_rate": sr}


@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        uploaded_files[file_id]["path"], media_type="audio/wav"
    )


@app.post("/predict")
async def predict(file: UploadFile):
    """Accept a WAV audio chunk and return a prediction. AC-007."""
    content = await file.read()

    # Save to temp file for librosa to load
    tmp_path = UPLOAD_DIR / f"predict_{uuid.uuid4()}.wav"
    tmp_path.write_bytes(content)

    try:
        predictor = _get_predictor()
        audio, sr = librosa.load(str(tmp_path), sr=None)
        result = predictor.predict(audio, sr)
        return {"label": result["label"], "confidence": result["confidence"]}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.websocket("/ws/predict/{file_id}")
async def websocket_predict(websocket: WebSocket, file_id: str):
    await websocket.accept()

    if file_id not in uploaded_files:
        await websocket.send_json({"error": "File not found"})
        await websocket.close()
        return

    file_info = uploaded_files[file_id]
    cache = prediction_cache.setdefault(file_id, {})
    predictor = _get_predictor()

    try:
        while True:
            data = await websocket.receive_json()
            position = data.get("position", 0.0)

            # Quantize to 0.5s boundaries.
            # Per RFC-007, segments shorter than the 2s window are padded
            # with silence by feature_extraction, so all positions return
            # a real prediction (no insufficient_audio short-circuit).
            quantized = round(position * 2) / 2

            # Check cache
            if quantized in cache:
                result = cache[quantized]
            else:
                result = predictor.predict_window(
                    file_info["path"], quantized
                )
                cache[quantized] = result

            await websocket.send_json(
                {
                    "position": position,
                    "label": result["label"],
                    "confidence": result["confidence"],
                }
            )
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""FastAPI application for WhoSpeaks audio classifier."""

import logging
import math
import os

# Prevent OpenMP/MKL thread pool deadlock when PyTorch (resemblyzer) and
# LightGBM are both loaded in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import librosa
import lightgbm  # noqa: F401 — must load LightGBM native code before torch to avoid OpenMP runtime conflict on macOS
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from whospeaks.model import SpeakerPredictor

# Temp storage for uploaded files
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="whospeaks_"))
# file_id -> {path, duration, sample_rate}
uploaded_files: dict[str, dict] = {}
# prediction cache: file_id -> {quantized_position -> prediction}
prediction_cache: dict[str, dict[float, dict]] = {}

STATIC_DIR = Path(__file__).parent / "static"


class _MockPredictor:
    """Returns mock predictions for frontend development."""

    def predict_window(self, _file_path: str, position: float) -> dict:
        # Deterministic mock based on position for consistency
        sin_val = math.sin(position * 0.5)
        confidence = 0.3 + 0.4 * abs(sin_val)
        if sin_val > 0.3:
            return {"label": "JEROEN_VAN_INKEL", "confidence": round(confidence, 3)}
        return {"label": "OTHER", "confidence": round(1.0 - confidence, 3)}

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        return {"label": "OTHER", "confidence": 0.5}


_predictor: "_MockPredictor | SpeakerPredictor | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SpeakerPredictor once on startup.

    Falls back to the mock predictor if the persisted model file is
    unavailable, so the app still boots in dev environments without a
    trained model on disk.
    """
    global _predictor
    try:
        _predictor = SpeakerPredictor.load()
    except Exception as e:
        logging.warning("Failed to load SpeakerPredictor, falling back to mock: %s", e)
        _predictor = _MockPredictor()
    yield


app = FastAPI(title="WhoSpeaks", lifespan=lifespan)


def _get_predictor():
    return _predictor


PredictorDep = Annotated[_MockPredictor, Depends(_get_predictor)]


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.get("/status")
async def status(predictor: PredictorDep):
    predictor_type = "mock" if isinstance(predictor, _MockPredictor) else "real"
    return {"status": "ok", "model_loaded": True, "predictor_type": predictor_type}


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
async def predict(file: UploadFile, predictor: PredictorDep):
    """Accept a WAV audio chunk and return a prediction. AC-007."""
    content = await file.read()

    # Save to temp file for librosa to load
    tmp_path = UPLOAD_DIR / f"predict_{uuid.uuid4()}.wav"
    tmp_path.write_bytes(content)

    try:
        audio, sr = librosa.load(str(tmp_path), sr=None)
        result = predictor.predict(audio, sr)
        return {"label": result["label"], "confidence": result["confidence"]}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.websocket("/ws/predict/{file_id}")
async def websocket_predict(websocket: WebSocket, file_id: str, predictor: PredictorDep):
    await websocket.accept()

    if file_id not in uploaded_files:
        await websocket.send_json({"error": "File not found"})
        await websocket.close()
        return

    file_info = uploaded_files[file_id]
    cache = prediction_cache.setdefault(file_id, {})

    try:
        while True:
            data = await websocket.receive_json()
            position = float(data.get("position", 0.0))

            # Reject positions outside the file. Negative or beyond-EOF
            # positions cannot be padded into a meaningful 2s window.
            if position < 0.0 or position > file_info["duration"]:
                await websocket.send_json(
                    {
                        "position": position,
                        "error": "position out of range",
                    }
                )
                continue

            # Quantize to 0.5s boundaries.
            # Per RFC-007, segments shorter than the 2s window are padded
            # with silence by feature_extraction, so all positions return
            # a real prediction (no insufficient_audio short-circuit).
            quantized = round(position * 2) / 2

            # Check cache
            if quantized in cache:
                result = cache[quantized]
            else:
                try:
                    result = predictor.predict_window(
                        file_info["path"], quantized
                    )
                except Exception as exc:
                    logging.warning("predict_window error at %.2f: %s", quantized, exc)
                    await websocket.send_json({"position": position, "error": str(exc)})
                    continue
                cache[quantized] = result

            confidence = result["confidence"]
            if not math.isfinite(confidence):
                confidence = 0.0
            await websocket.send_json(
                {
                    "position": position,
                    "label": result["label"],
                    "confidence": confidence,
                }
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logging.warning("WebSocket handler error: %s", exc, exc_info=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


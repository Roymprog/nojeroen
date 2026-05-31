# WhoSpeaks: Real-Time Speaker Identification for Audio Playback

A local web application that identifies when a specific speaker (JEROEN_VAN_INKEL) is speaking in audio files. Features live, real-time predictions during playback with confidence scoring.

## Overview

**WhoSpeaks** is a binary audio classifier built with:
- **Feature Extraction**: Resemblyzer GE2E speaker embeddings (256-dim vectors)
- **Model**: LightGBM binary classifier (JEROEN_VAN_INKEL vs OTHER)
- **Interface**: FastAPI backend + HTML/JS frontend with WebSocket streaming
- **Training**: Leave-One-Session-Out cross-validation (LOSO-CV) with session-based train/test split

### Key Features
- ✅ Train once, persist to disk, reload on app startup (no retraining)
- ✅ Upload any WAV file and play with live speaker predictions
- ✅ Sliding-window analysis (2s windows, 1s stride) with silence-padding for short segments
- ✅ Sub-1-second prediction latency for responsive playback
- ✅ Conservative threshold tuning (precision >= 95%)
- ✅ Cross-show generalization (trained on S1+S2, validated on S3, tested on S4)

---

## Quick Start

### Prerequisites
- Python 3.10+
- `uv` package manager (https://docs.astral.sh/uv/)
- WAV audio files (any sample rate, app resamples to 16kHz internally)

### Installation

```bash
# Clone or navigate to the project directory
cd /workspaces/whospeaks_agent

# Install dependencies
uv sync

# Create necessary directories
mkdir -p models data/labeled
```

### Train the Model

Before running the app, you must train the model on labeled data:

```bash
# Prepare labeled data in data/labeled/
# Expected format: segment_{start}_{end}_{LABEL}.wav
# where LABEL is JEROEN_VAN_INKEL or OTHER

# Run the full training pipeline
uv run python -c "from whospeaks.train import run_training_pipeline; run_training_pipeline()"
```

This will:
1. Load labeled segments from `data/labeled/`
2. Extract 256-dim speaker embeddings (resemblyzer)
3. Train LightGBM classifier on S1+S2 (LOSO-CV)
4. Tune threshold on S3 validation set for >= 95% precision
5. Evaluate on S4 test set (held-out cross-show data)
6. Save model to `models/model.joblib` and config to `models/config.json`

**Expected output**: Prints LOSO-CV metrics for all 4 folds and final S4 test metrics.

### Run the Web Application

```bash
# Start the app
uv run python src/whospeaks/app.py

# App is now running at http://localhost:8000
```

Open in a web browser:
- **Upload**: Click "Choose File" to select a WAV file
- **Play**: Audio player appears; press Play
- **Predictions**: Real-time speaker predictions update every ~500ms as audio plays
- **Confidence**: Binary label (JEROEN_VAN_INKEL / OTHER) + confidence score (0-100%)

### Run Tests

```bash
# Run all tests (unit + integration + E2E)
uv run pytest tests/ -v

# Run just integration tests (model pipeline validation)
uv run pytest tests/test_model.py::TestIntegrationPipelineEndToEnd -v

# Run app/WebSocket tests
uv run pytest tests/test_app.py -v
```

---

## Home Assistant deployment

The repo also ships a Home Assistant add-on that taps a Sonos radio stream live and publishes the classifier's verdict to MQTT, where HA picks it up via auto-discovery. Full design notes are in [`docs/home-assistant-addon.md`](docs/home-assistant-addon.md); the steps below are the operator path.

### Prerequisites on the HA host

- Home Assistant OS or Supervised (so the **Add-on Store** is available — Container/Core installs cannot install add-ons).
- The **Mosquitto broker** add-on installed and running, plus the **MQTT** integration added in *Settings → Devices & Services*.
- The **Sonos** integration added, with the target `media_player.*` entity you want to tap.
- The **Samba share** add-on (or any way to drop files into `/share/`) — used to deliver the trained model.

### 1. Add this repo as an add-on repository

1. In HA, go to **Settings → Add-ons → Add-on Store**.
2. Top-right menu (⋮) → **Repositories**.
3. Paste the HTTPS URL of this Git repo and click **Add**, then **Close**.
4. The store reloads; scroll to the new **WhoSpeaks** section and open the **WhoSpeaks** add-on tile.

### 2. Install and configure

1. Click **Install** on the add-on page. First build takes ~5–10 minutes on Yellow (it compiles the Python deps for ARM).
2. Open the **Configuration** tab and set:

   ```yaml
   sonos_entity_id: media_player.sonos_woonkamer   # your Sonos entity
   stations:
     "NPO Radio 2": https://icecast.omroep.nl/radio2-bb-mp3
     "BNR Nieuwsradio": https://stream.bnr.nl/bnr_mp3_128_03
   log_level: INFO
   ```

   `stations` keys must match the `media_title` Sonos reports for that station exactly. Check *Developer Tools → States → media_player.sonos_…* while the station is playing if you're unsure.

3. **Save**.

### 3. Drop in the trained model

The add-on does **not** ship a model — it loads one from `/share/whospeaks/` on the HA host. Until the model is present, the entities exist but stay `unavailable`.

1. Train locally with the workflow above (`run_training_pipeline()`); this produces `models/model.joblib` and `models/config.json`.
2. Mount the host's `share` folder via the Samba add-on (or `scp` for HA Supervised).
3. Create `share/whospeaks/` and copy both files into it. The final layout on the host:

   ```
   /share/whospeaks/
   ├── model.joblib
   └── config.json
   ```

### 4. Start the add-on

1. On the add-on page, **Start**.
2. Open the **Log** tab. A healthy boot logs `model loaded from /share/whospeaks` and `connecting to MQTT …`.
3. In HA, two new entities appear (MQTT discovery):
   - `sensor.whospeaks_current_speaker` — state is one of `JEROEN_VAN_INKEL` / `OTHER` / `idle` / `unavailable`, with attributes `confidence`, `station`, `station_url`, `last_classified_at`, `raw_label`.
   - `binary_sensor.whospeaks_jeroen_present` — `on` while the sensor state is `JEROEN_VAN_INKEL`.

Enable **Watchdog** and **Start on boot** on the add-on page once you've confirmed it works.

### 5. Shadow-rollout first

Per the spec's accepted risks, **do not write automations against `binary_sensor.whospeaks_jeroen_present` for the first week of running**. Watch the History graph of `confidence` and `raw_label` while the configured stations play, look for systematic false positives, and only then wire up automations. If precision looks bad, retrain locally (see [`docs/home-assistant-addon.md`](docs/home-assistant-addon.md) §"Accepted risks" for the MP3 round-trip retrain path) and copy the new artifacts back to `/share/whospeaks/`.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Entities never appear in HA | MQTT integration not added, or Mosquitto add-on not running. |
| Entities show `unavailable` permanently | `/share/whospeaks/model.joblib` or `config.json` missing/corrupt — check add-on log for `failed to load model`. |
| Sensor stuck on `idle` while Sonos is playing | `media_title` in HA doesn't match any key under `stations` — copy the exact title from *Developer Tools → States*. |
| Frequent reconnects, log shows `ffmpeg: …` warnings | The station's HTTPS endpoint is throttling or 404-ing — verify the URL in a browser. |

---

## Project Structure

```
/workspaces/whospeaks_agent/
├── src/whospeaks/
│   ├── app.py                    # FastAPI server, WebSocket endpoints
│   ├── model.py                  # SpeakerPredictor class (load, predict, predict_window)
│   ├── train.py                  # Training pipeline (embeddings, LightGBM, threshold tuning)
│   ├── evaluate.py               # Evaluation: metrics, confusion matrix, visualizations
│   ├── feature_extraction.py     # Resemblyzer embedding extraction (2s windows, 1s stride)
│   ├── data_loader.py            # Session-based train/test split
│   ├── config.py                 # Constants (window size, stride, sample rate, etc.)
│   ├── predict.py                # Batch prediction script (if needed)
│   └── static/
│       ├── index.html            # Web UI (audio player, predictions, confidence bar)
│       ├── style.css             # UI styling
│       └── script.js             # Frontend logic (WebSocket, quantization, UI updates)
├── tests/
│   ├── test_model.py             # Unit + integration tests for SpeakerPredictor
│   ├── test_app.py               # Integration tests for FastAPI routes + WebSocket
│   ├── test_training_pipeline.py # Training pipeline tests
│   ├── test_evaluate.py          # Evaluation tests
│   └── conftest.py               # Pytest fixtures
├── models/                       # Persisted model artifacts (generated during training)
│   ├── model.joblib              # Trained LightGBM + threshold
│   └── config.json               # Config (feature type, window size, threshold, etc.)
├── data/
│   └── labeled/                  # Labeled training data (user-provided)
│       ├── segment_1.0_3.5_JEROEN_VAN_INKEL.wav
│       ├── segment_4.2_6.8_OTHER.wav
│       └── ...
├── pyproject.toml                # Project config, dependencies
├── uv.lock                       # Locked dependency versions
└── README.md                     # This file
```

---

## How It Works

### Training Pipeline (Phase 2)

1. **Feature Extraction**:
   - Load each labeled WAV segment
   - Resample to 16kHz
   - Extract 2-second windows with 1-second stride
   - For each window: compute 256-dim resemblyzer embedding
   - Inherit parent segment's label (JEROEN_VAN_INKEL or OTHER)

2. **Model Training** (Leave-One-Session-Out CV):
   - Fold 1: Train on S2+S3+S4, validate on S1
   - Fold 2: Train on S1+S3+S4, validate on S2
   - Fold 3: Train on S1+S2+S4, validate on S3
   - Fold 4: Train on S1+S2+S3, validate on S4 (cross-show, primary focus)
   - Use LightGBM with `scale_pos_weight` for class imbalance

3. **Threshold Tuning**:
   - On S3 validation set, sweep thresholds from 0.3 to 0.95
   - Select threshold achieving >= 95% precision
   - Report precision, recall, F1 on S4 test set

4. **Model Persistence**:
   - Save LightGBM classifier to `models/model.joblib`
   - Save config to `models/config.json` (feature type, window size, threshold, etc.)

### Inference Pipeline (App)

1. **On Startup**:
   - Load model from disk via `SpeakerPredictor.load()`
   - Fall back to mock predictor if models/ unavailable (dev-only)

2. **File Upload** (`POST /upload`):
   - Accept WAV file
   - Store in temp directory
   - Return file_id, duration, sample_rate to frontend

3. **WebSocket Streaming** (`/ws/predict/{file_id}`):
   - Frontend sends current playback position (every ~500ms)
   - Backend quantizes position to 0.5s boundary
   - Load 2-second window ending at position (or pad if position < 2s)
   - Compute embedding via resemblyzer
   - Predict via LightGBM
   - Return `{label, confidence}` to frontend
   - Cache results to avoid re-computation

4. **UI Update**:
   - Display binary label (JEROEN_VAN_INKEL / OTHER)
   - Show confidence as numeric (0-100%) and visual bar


### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve index.html (UI) |
| `/status` | GET | Health check, model loaded status |
| `/upload` | POST | Accept WAV file, return file_id + metadata |
| `/audio/{file_id}` | GET | Retrieve uploaded audio file |
| `/ws/predict/{file_id}` | WebSocket | Stream predictions during playback |
| `/predict` | POST | Single prediction (legacy, not used by UI) |

### WebSocket Message Format

**Frontend sends**:
```json
{"position": 3.5}
```

**Backend returns**:
```json
{
  "position": 3.5,
  "label": "JEROEN_VAN_INKEL",
  "confidence": 0.87
}
```

## Development

### Adding Tests
Tests use pytest. See `tests/test_model.py` and `tests/test_app.py` for patterns.

```bash
# Run a single test
uv run pytest tests/test_model.py::TestIntegrationPipelineEndToEnd::test_predict_window_boundary_positions -v
```

### Running Linting (if configured)
```bash
uv run ruff check src/ tests/
uv run mypy src/ --no-error-summary  # Type checking (optional)
```

---

## License

This project is provided as-is for research and development purposes.

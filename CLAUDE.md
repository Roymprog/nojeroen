# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WhoSpeaks: a local FastAPI web app that does real-time binary speaker identification (`JEROEN_VAN_INKEL` vs `OTHER`) during WAV playback. Pipeline: resemblyzer GE2E 256-dim embeddings → LightGBM classifier → threshold tuned for precision ≥ 0.95.

## Commands

Dependency management is via `uv`. Tests skip the `slow` marker by default (see `pyproject.toml`).

```bash
uv sync                                                  # install
uv run python src/whospeaks/app.py                       # run server on :8000
uv run python -c "from whospeaks.train import run_training_pipeline; run_training_pipeline()"
uv run pytest                                            # fast tests only
uv run pytest -m slow                                    # full training pipeline on real data
uv run pytest tests/test_model.py::TestIntegrationPipelineEndToEnd::test_predict_window_boundary_positions -v
uv run ruff check src/ tests/
```

`data/` is gitignored. Training expects WAVs under `data/labeled/<session_dir>/segment_{start}_{end}_{LABEL}.wav` where `<session_dir>` matches the four session strings in `config.SESSIONS` and `LABEL` is `JEROEN_VAN_INKEL` or `OTHER`. Override locations with `WHOSPEAKS_DATA_DIR` / `WHOSPEAKS_MODEL_DIR`.

## Architecture

**Session-based split, not random.** `config.py` hard-codes four sessions; `TRAIN_SESSIONS=[S1,S2]`, val=S3, test=S4 (a different show — S4 measures cross-show generalization). LOSO-CV in `train.run_loso_cv` rotates each session as the held-out fold; the final model is trained on TRAIN_SESSIONS only and the threshold is tuned on S3 to achieve `TARGET_PRECISION` (lowest threshold meeting the bar wins). `data_loader.load_split_data` returns both the split dict and the per-session dict so the same loaded embeddings feed both LOSO-CV and the final fit.

**Window contract is shared across train and inference.** Training extracts 2s/1s-stride sliding windows via `feature_extraction.extract_windows_from_file`; each window becomes one training row inheriting its parent segment's label. Inference (`SpeakerPredictor.predict_window`) loads a 2s window *ending* at `position` and left-pads with silence if `position < 2s` (per RFC-007 — there is no "insufficient audio" short-circuit, every in-range position returns a real prediction). The WebSocket handler quantizes incoming positions to 0.5s boundaries and caches per `(file_id, quantized_position)`.

**Model persistence format.** `train.save_model` writes `{"model": lgbm, "threshold": float}` to `models/model.joblib` plus `models/config.json` with `feature_type`, `embedding_dim`, `sample_rate`, `threshold`, etc. `SpeakerPredictor.load` validates `_REQUIRED_CONFIG_KEYS` and rejects any `feature_type` other than `resemblyzer_ge2e`. Confidence in the predictor output is `prob` when the label is positive and `1 - prob` when negative — not raw probability.

**App lifespan and the mock predictor.** `app.py` loads `SpeakerPredictor` once in the FastAPI `lifespan` context. If loading raises, it falls back to a deterministic `_MockPredictor` so the frontend still boots without a trained model. `/status` reports `predictor_type: "mock" | "real"` — check this when debugging "why are predictions always nonsense".

## Gotchas

- **Import order in `app.py`.** `os.environ.setdefault` for `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` runs *before* any heavy imports, and `lightgbm` is imported before anything that pulls in torch (resemblyzer). Both prevent OpenMP thread-pool deadlocks. Don't reorder these.
- **Resemblyzer encoder is a module-level singleton** in `feature_extraction._encoder` (loaded lazily via `get_encoder()`); `SpeakerPredictor` keeps its own `VoiceEncoder` instance. Tests that monkeypatch one won't affect the other.
- **`position` validation in the WebSocket.** Positions outside `[0, file_duration]` return `{"error": "position out of range"}` and the loop continues — they do not close the socket.
- **Test markers.** `pyproject.toml` sets `addopts = "-m 'not slow'"` so slow tests are skipped by default; opt in explicitly with `-m slow` (or `-m ""` for everything).

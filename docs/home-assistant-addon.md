# WhoSpeaks Home Assistant add-on

Live Jeroen-van-Inkel detection on a Sonos audio stream, surfaced as Home Assistant entities.

## Summary

A Home Assistant add-on (Docker container, deployed via the HA Supervisor) taps the live HTTPS audio stream of whichever radio station the user's Sonos is currently playing, runs the existing resemblyzer + LightGBM classifier (`SpeakerPredictor`) once per second on a rolling 2-second window, and publishes a debounced classification to MQTT. Home Assistant auto-discovers the resulting entities via MQTT discovery. No HA custom component is required.

Target host: **Home Assistant Yellow** (CM4-class hardware, 4 cores, 4–8 GB RAM).

## Goals

- Expose `sensor.whospeaks_current_speaker` and `binary_sensor.whospeaks_jeroen_present` to Home Assistant.
- Update automatically based on whatever Sonos is playing, with no per-session user action.
- Support multiple radio stations via a user-configured `media_title` → HTTPS URL table.
- Be debugable from inside HA (logs + history graphs) without ssh-ing into anything.

## Non-goals

- Classifying audio from Spotify, Apple Music, Tidal, AirPlay, TV, or line-in (non-tappable sources).
- Voice activity detection upstream of the classifier (music classifies as `OTHER`; that is acceptable).
- Threshold tuning UI (the trained `threshold` in `config.json` is the source of truth).
- On-device training or retraining inside the add-on.
- A shadow-mode toggle as a code feature (shadow rollout is a discipline: "deploy, don't write automations yet").

## Architecture

```
+-------------------+        +--------------------+        +-----------------+
| Home Assistant    | <----- | Mosquitto (MQTT)   | <----- | WhoSpeaks       |
| (Sonos media_     | WS API |                    |        | add-on (Docker) |
|  player entity,   |------->|                    |------->|                 |
|  MQTT discovery)  |        |                    |        | ffmpeg + model  |
+-------------------+        +--------------------+        +-------+---------+
                                                                    |
                                                                    | HTTPS GET
                                                                    v
                                                          +-------------------+
                                                          | Icecast / radio   |
                                                          | stream URL        |
                                                          +-------------------+
```

The add-on is the only moving part this project ships. HA, Mosquitto, and the Sonos integration are pre-existing.

## Entities (MQTT discovery)

### `sensor.whospeaks_current_speaker`

| Field | Value |
|---|---|
| State | `JEROEN_VAN_INKEL` \| `OTHER` \| `idle` \| `unavailable` |
| `confidence` | float in [0, 1] — model's confidence in the **current** label (per `SpeakerPredictor.predict`: `prob` if positive, `1 - prob` if negative). Updates every cycle. |
| `station` | string — `media_title` reported by Sonos for the currently tapped station, or `null` when `idle`. |
| `station_url` | string — resolved HTTPS URL being tapped, or `null` when `idle`. |
| `last_classified_at` | ISO-8601 timestamp of the most recent classification. |
| `raw_label` | string — most recent **pre-hysteresis** classification. Useful for debugging "why is the sensor still `OTHER`". Updates every cycle. |

### `binary_sensor.whospeaks_jeroen_present`

| Field | Value |
|---|---|
| State | `on` iff `sensor.whospeaks_current_speaker` state == `JEROEN_VAN_INKEL`, else `off`. `unavailable` when the sensor is `unavailable`. |

Automations triggered by "Jeroen is now on the radio" should subscribe to `state_changed` on this binary sensor.

## State machine

States: `JEROEN_VAN_INKEL`, `OTHER`, `idle`, `unavailable`.

| Trigger | New state |
|---|---|
| Sonos state → `playing` and `media_title` ∈ station table | enter classifier; start in `OTHER`; hysteresis counters = 0 |
| Sonos state → `paused`, `idle`, `off` | `idle` |
| Sonos plays a station **not** in the station table | `idle` (same code path as paused; deliberately not a distinct state) |
| Station change (e.g. NPO R2 → BNR) | **Hard reset**: close stream, drop buffer, counters = 0, state → `idle` for one cycle, then start fresh from `OTHER` |
| Stream disconnect / decoder failure | Hard reset + reconnect with exponential backoff (1s → 2s → 4s → 8s → cap 30s); sensor stays `idle` while retrying |
| Add-on cannot reach MQTT, model fails to load, supervisor token invalid | `unavailable` (driven by MQTT LWT — happens automatically without explicit code on most failures) |

### Hysteresis

After each per-cycle raw classification, the state machine commits to a new state only when the raw label has been consistent for K cycles:

- `K_enter = 3`: require 3 consecutive `JEROEN_VAN_INKEL` raw predictions to transition into `JEROEN_VAN_INKEL`.
- `K_leave = 2`: require 2 consecutive `OTHER` raw predictions to transition out of `JEROEN_VAN_INKEL`.

Asymmetric — harder to enter, easier to leave — compounding with the model's already-precision-tuned threshold of 0.77. Defaults baked into the add-on; not user-configurable in v1.

## Signal flow

1. Add-on boots, loads `model.joblib` + `config.json` from `/share/whospeaks/`. If either is missing or invalid, publish `unavailable` on the LWT topic and exit/retry — sensor shows `unavailable` in HA.
2. Add-on opens a WebSocket to `http://supervisor/core/websocket` using the `SUPERVISOR_TOKEN` env var. Subscribes to `state_changed` events filtered to the configured `media_player.*` entity.
3. On Sonos state event:
   - If `playing` and `media_title` ∈ `stations` config: resolve to HTTPS URL, spawn ffmpeg subprocess, start tapper loop.
   - Otherwise: tear down any running tapper, publish `idle`.
4. Tapper loop (one per active stream):
   - Read raw 16 kHz mono PCM from ffmpeg's stdout into a 2-second rolling buffer.
   - Once per second, copy the last 2 seconds out of the buffer, run `SpeakerPredictor.predict(audio, sr=16000)`.
   - Feed the raw label into the hysteresis state machine.
   - Publish to MQTT:
     - **Every cycle**: full JSON attribute payload (`confidence`, `station`, `station_url`, `last_classified_at`, `raw_label`).
     - **On committed transitions**: new state value on the state topic.
   - HA's recorder captures attribute updates → free history graph of `confidence` and `raw_label` over time.
5. On any error in (3) or (4): hard reset.

## Configuration (HA add-on UI)

Minimal. Three keys.

```yaml
sonos_entity_id: media_player.sonos_woonkamer  # required
stations:                                       # required, dict of media_title -> URL
  "NPO Radio 2": "https://icecast.omroep.nl/radio2-bb-mp3"
  "BNR Nieuwsradio": "https://stream.bnr.nl/bnr_mp3_128_03"
log_level: INFO                                 # optional; default INFO
```

Deliberately **not** in the config UI:
- `threshold` — locked to model's `config.json`.
- `K_enter`, `K_leave` — locked to 3 / 2.
- `window_size_s`, `sample_rate`, `embedding_dim`, `feature_type` — locked to model's `config.json`.
- MQTT broker credentials — obtained from HA Supervisor via `services: ["mqtt:need"]`.
- Model path — locked to `/share/whospeaks/`.

## Model delivery

Model artifacts (`model.joblib`, `config.json`) live on the HA host filesystem at `/share/whospeaks/` and are mounted into the add-on container via the standard HA `map: ["share"]` mechanism. The user copies new model artifacts in via Samba (HA's `samba` add-on exposes `/share/` over SMB) and restarts the add-on to pick them up.

The add-on **does not** ship a model in its Docker image. On first boot with no model present, the sensor goes to `unavailable` and the add-on logs an error. This is the same failure mode as a corrupt or unreadable model file.

Retraining stays on the developer's workstation (the existing `run_training_pipeline()` flow); the result is a new `model.joblib` + `config.json` pair the user drops into `/share/whospeaks/`.

## Add-on packaging

### `config.yaml` (HA Supervisor schema)

```yaml
name: WhoSpeaks
version: 0.1.0
slug: whospeaks
description: Real-time Jeroen-van-Inkel detection on Sonos radio
arch: [aarch64, amd64]
init: false
hassio_api: false
homeassistant_api: true
services:
  - mqtt:need
map:
  - share
schema:
  sonos_entity_id: str
  stations:
    "[str]": url
  log_level: list(DEBUG|INFO|WARNING|ERROR)?
```

### Container

- Base image: a slim Python 3.11+ image with `ffmpeg` apt-installed.
- Python deps: existing `pyproject.toml` minus the FastAPI/uvicorn stack (the add-on is a long-running script, not a web service). Specifically: `resemblyzer`, `lightgbm`, `librosa`, `numpy`, `joblib`, plus `aiohttp` (for HA WS), `paho-mqtt` (or `aiomqtt`) for MQTT.

### Runtime tweaks

- `torch.set_num_threads(2)` at startup. Yellow has 4 cores; leaving 2 for HA itself prevents the supervisor from going laggy when the classifier is busy.
- The env-var ordering gotcha from `app.py` (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` set **before** any heavy imports; `lightgbm` imported before `resemblyzer`) carries over to the add-on entrypoint.

## MQTT topic layout

| Purpose | Topic |
|---|---|
| Availability (LWT) | `whospeaks/availability` (`online` / `offline`) |
| Sensor state | `whospeaks/current_speaker/state` |
| Sensor attributes | `whospeaks/current_speaker/attributes` (JSON) |
| Binary sensor state | `whospeaks/jeroen_present/state` (`ON` / `OFF`) |
| Discovery: sensor | `homeassistant/sensor/whospeaks_current_speaker/config` |
| Discovery: binary_sensor | `homeassistant/binary_sensor/whospeaks_jeroen_present/config` |

Both discovery payloads reference `availability_topic: whospeaks/availability` so the entities auto-go-`unavailable` when the add-on disconnects or crashes.

## Performance budget (Yellow)

| Operation | Approximate cost |
|---|---|
| Resemblyzer encoder on a 2 s window | 300–500 ms (one core) |
| LightGBM `predict_proba` | < 1 ms |
| ffmpeg MP3 → PCM at 128 kbps | < 5% of one core continuously |
| MQTT publish | negligible |
| HA WebSocket subscribe | idle except on Sonos events |

Per-cycle inference ≈ 35–50% of one core sustained while a tappable station is playing. Idle when nothing is playing. No GPU. Memory footprint expected ≈ 400–600 MB RSS for the container.

## Accepted risks

1. **WAV-vs-MP3 domain shift.** The trained model has not been exposed to MP3 compression artifacts. If the live Icecast stream's 128 kbps MP3 → PCM round-trip degrades resemblyzer embeddings enough to break the precision tuning, false-positive `JEROEN_VAN_INKEL` events will appear.
   **Mitigation path if/when it happens**: round-trip every training WAV through `ffmpeg -b:a 128k -codec:a libmp3lame -ar 16000 -ac 1` before feature extraction, retrain, re-run threshold tuning, redeploy the model artifact via `/share/whospeaks/`. **Do not** add a threshold UI knob as a workaround.

2. **Cross-station / cross-show generalization.** The model was trained on the specific sessions in `whospeaks.config.SESSIONS`. Other shows on NPO Radio 2 (different co-hosts, jingles, mixing) and other stations entirely (different stream conditioning) are out-of-distribution. Precision may degrade on new stations.
   **Mitigation path**: collect labeled segments from the new station, append to `data/labeled/`, retrain, redeploy via `/share/whospeaks/`. The mount-based delivery story makes this iteration cheap.

3. **No automated end-to-end validation pre-deploy.** Rollout discipline is "deploy the add-on, write no automations against `binary_sensor.whospeaks_jeroen_present` for the first week, watch the History graph of `confidence` and `raw_label`, retrain if it looks bad." There is no shadow-mode flag in the code.

## Out of scope

- Sources other than tappable HTTPS radio streams (Spotify, Apple Music, AirPlay, TV audio, line-in, room mic).
- Synchronization with Sonos `media_position` for on-demand/podcast playback (live radio has no sync problem; drift between tap and Sonos doesn't matter).
- HLS support beyond what ffmpeg handles natively.
- Multiple Sonos groups / multi-room awareness (single configured `media_player.*` entity).
- VAD, music classification, ad detection.
- On-device retraining; training workflow stays on the developer workstation.
- Threshold or hysteresis UI exposure.
- A REST/WebSocket interface for external consumers (everything goes through MQTT + HA).

## Open follow-ups (deferred until evidence demands them)

- If shadow rollout reveals systematic false positives → execute the MP3 round-trip retrain path (Accepted risks #1).
- If users add many stations and the per-station performance varies → consider per-station thresholds shipped in `config.json`.
- If `K_enter` / `K_leave` defaults feel wrong after a week of use → promote them to add-on config.
- If the add-on is ever distributed to others → revisit model delivery: either bake into the image, or download from a release URL.

"""Tests for whospeaks.addon.options."""

import json

import pytest

from whospeaks.addon.options import Options


def _write(tmp_path, payload):
    p = tmp_path / "options.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_load_minimal_valid(tmp_path, monkeypatch):
    monkeypatch.delenv("WHOSPEAKS_MODEL_DIR", raising=False)
    path = _write(
        tmp_path,
        {
            "sonos_entity_id": "media_player.sonos",
            "stations": {"NPO Radio 2": "https://icecast.omroep.nl/radio2-bb-mp3"},
        },
    )
    opts = Options.load(path)
    assert opts.sonos_entity_id == "media_player.sonos"
    assert opts.stations == {"NPO Radio 2": "https://icecast.omroep.nl/radio2-bb-mp3"}
    assert opts.log_level == "INFO"
    assert opts.model_dir == "/share/whospeaks"


def test_load_respects_model_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WHOSPEAKS_MODEL_DIR", "/tmp/custom-models")
    path = _write(
        tmp_path,
        {
            "sonos_entity_id": "media_player.sonos",
            "stations": {"BNR": "https://stream.bnr.nl/bnr_mp3_128_03"},
        },
    )
    opts = Options.load(path)
    assert opts.model_dir == "/tmp/custom-models"


def test_rejects_missing_entity(tmp_path):
    path = _write(tmp_path, {"stations": {"X": "https://x"}})
    with pytest.raises(ValueError, match="sonos_entity_id"):
        Options.load(path)


def test_rejects_empty_stations(tmp_path):
    path = _write(tmp_path, {"sonos_entity_id": "media_player.x", "stations": {}})
    with pytest.raises(ValueError, match="stations"):
        Options.load(path)


def test_rejects_non_http_url(tmp_path):
    path = _write(
        tmp_path,
        {"sonos_entity_id": "media_player.x", "stations": {"X": "ftp://no"}},
    )
    with pytest.raises(ValueError, match="http"):
        Options.load(path)


def test_rejects_unknown_log_level(tmp_path):
    path = _write(
        tmp_path,
        {
            "sonos_entity_id": "media_player.x",
            "stations": {"X": "https://x"},
            "log_level": "TRACE",
        },
    )
    with pytest.raises(ValueError, match="log_level"):
        Options.load(path)

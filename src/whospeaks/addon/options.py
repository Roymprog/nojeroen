"""Add-on options loaded from /data/options.json (Supervisor convention)."""

import json
import os
from dataclasses import dataclass


DEFAULT_OPTIONS_PATH = "/data/options.json"
DEFAULT_MODEL_DIR = "/share/whospeaks"


@dataclass(frozen=True)
class Options:
    sonos_entity_id: str
    stations: dict[str, str]
    log_level: str
    model_dir: str
    options_path: str

    @classmethod
    def load(cls, path: str | None = None) -> "Options":
        path = path or os.environ.get("WHOSPEAKS_OPTIONS_PATH", DEFAULT_OPTIONS_PATH)
        with open(path) as f:
            raw = json.load(f)

        sonos_entity_id = raw.get("sonos_entity_id")
        if not isinstance(sonos_entity_id, str) or not sonos_entity_id:
            raise ValueError("sonos_entity_id missing or empty")

        stations_raw = raw.get("stations")
        if not isinstance(stations_raw, list) or not stations_raw:
            raise ValueError("stations missing or empty")
        stations: dict[str, str] = {}
        for entry in stations_raw:
            if not isinstance(entry, str) or "|" not in entry:
                raise ValueError(f"stations entry must be 'Name|URL', got {entry!r}")
            title, url = entry.split("|", 1)
            title = title.strip()
            url = url.strip()
            if not title or not url:
                raise ValueError(f"stations entry has empty name or URL: {entry!r}")
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"stations[{title!r}] is not an http(s) URL: {url!r}")
            stations[title] = url

        log_level = raw.get("log_level", "INFO")
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"log_level must be one of DEBUG/INFO/WARNING/ERROR, got {log_level!r}")

        model_dir = os.environ.get("WHOSPEAKS_MODEL_DIR", DEFAULT_MODEL_DIR)

        return cls(
            sonos_entity_id=sonos_entity_id,
            stations=dict(stations),
            log_level=log_level,
            model_dir=model_dir,
            options_path=path,
        )

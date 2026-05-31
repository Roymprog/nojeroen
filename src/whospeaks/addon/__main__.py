"""Add-on entrypoint.

Wires options → MQTT (with HA discovery) → HA WebSocket subscription → ffmpeg
tapper → SpeakerPredictor → hysteresis FSM → MQTT publishes.

Per CLAUDE.md gotchas:
  * Thread-pool env vars are set before any heavy imports.
  * `lightgbm` is imported before `resemblyzer` (the latter pulls in torch).
"""

import os

# Must come before any import that pulls torch / lightgbm / numpy-with-MKL.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import asyncio
import contextlib
import logging
import signal
import sys
from datetime import datetime, timezone

import aiohttp
import lightgbm  # noqa: F401  — must load before resemblyzer (torch) on some platforms

import torch  # noqa: E402

torch.set_num_threads(2)

from whospeaks.model import SpeakerPredictor  # noqa: E402

from .backoff import Backoff  # noqa: E402
from .ha_client import HomeAssistantClient, SonosState  # noqa: E402
from .mqtt_client import MqttPublisher  # noqa: E402
from .options import Options  # noqa: E402
from .state_machine import (  # noqa: E402
    STATE_IDLE,
    STATE_OTHER,
    HysteresisFSM,
)
from .supervisor import fetch_mqtt_broker  # noqa: E402
from .tapper import SAMPLE_RATE, drain_stderr, open_stream, stream_windows  # noqa: E402

logger = logging.getLogger("whospeaks.addon")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _idle_attrs() -> dict:
    return {
        "confidence": None,
        "station": None,
        "station_url": None,
        "last_classified_at": _now_iso(),
        "raw_label": None,
    }


class Orchestrator:
    """Owns the FSM, the active tapper task, and the MQTT publishes."""

    def __init__(self, options: Options, predictor: SpeakerPredictor, mqtt: MqttPublisher):
        self._options = options
        self._predictor = predictor
        self._mqtt = mqtt
        self._fsm = HysteresisFSM()
        self._tapper_task: asyncio.Task | None = None
        self._current_station: str | None = None
        self._current_url: str | None = None

    async def start(self) -> None:
        """Publish initial idle state once on startup."""
        await self._mqtt.publish_state(STATE_IDLE)
        await self._mqtt.publish_attributes(_idle_attrs())

    async def shutdown(self) -> None:
        await self._stop_tapper()

    async def handle_sonos(self, sonos: SonosState) -> None:
        title = sonos.media_title
        is_tappable = (sonos.state == "playing" and title in self._options.stations)

        if not is_tappable:
            if self._tapper_task is not None or self._fsm.state != STATE_IDLE:
                logger.info(
                    "Sonos state=%s title=%r — stopping tapper",
                    sonos.state, title,
                )
                await self._go_idle()
            return

        url = self._options.stations[title]
        if (
            self._current_station == title
            and self._tapper_task is not None
            and not self._tapper_task.done()
        ):
            return  # already tapping this exact station

        logger.info("Starting tapper: station=%r url=%s", title, url)
        await self._stop_tapper()
        # Hard reset between stations: one publish of idle, then start fresh in OTHER.
        self._fsm.reset(STATE_IDLE)
        await self._mqtt.publish_state(STATE_IDLE)
        await self._mqtt.publish_attributes(_idle_attrs())
        self._fsm.start_streaming()
        await self._mqtt.publish_state(STATE_OTHER)
        self._current_station = title
        self._current_url = url
        self._tapper_task = asyncio.create_task(
            self._run_tapper(title, url),
            name=f"whospeaks-tapper:{title}",
        )

    async def _go_idle(self) -> None:
        await self._stop_tapper()
        self._current_station = None
        self._current_url = None
        transition = self._fsm.reset(STATE_IDLE)
        if transition.changed:
            await self._mqtt.publish_state(STATE_IDLE)
        await self._mqtt.publish_attributes(_idle_attrs())

    async def _stop_tapper(self) -> None:
        task = self._tapper_task
        self._tapper_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_tapper(self, station_title: str, station_url: str) -> None:
        """Drive ffmpeg → classifier → MQTT for as long as we're tapping this station.

        On stream/decoder errors: hard-reset the FSM, publish idle, sleep with
        exponential backoff, then re-enter classifier in OTHER and retry.
        """
        backoff = Backoff(initial=1.0, cap=30.0)
        while True:
            proc = None
            stderr_task: asyncio.Task | None = None
            try:
                proc = await open_stream(station_url)
                assert proc.stdout is not None and proc.stderr is not None
                stderr_task = asyncio.create_task(drain_stderr(proc.stderr, logger))
                backoff.reset()

                async for window in stream_windows(proc.stdout):
                    result = await asyncio.to_thread(
                        self._predictor.predict, window, SAMPLE_RATE,
                    )
                    raw_label = result["label"]
                    confidence = float(result["confidence"])
                    transition = self._fsm.feed(raw_label)

                    attrs = {
                        "confidence": confidence,
                        "station": station_title,
                        "station_url": station_url,
                        "last_classified_at": _now_iso(),
                        "raw_label": raw_label,
                    }
                    await self._mqtt.publish_attributes(attrs)
                    if transition.changed:
                        await self._mqtt.publish_state(transition.committed_state)

                logger.warning("ffmpeg stdout closed; will reconnect")
                raise ConnectionError("ffmpeg stdout EOF")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delay = backoff.next()
                logger.warning(
                    "tapper error on station=%r: %s; retrying in %.1fs",
                    station_title, exc, delay,
                )
                self._fsm.reset(STATE_IDLE)
                await self._mqtt.publish_state(STATE_IDLE)
                await self._mqtt.publish_attributes(_idle_attrs())
                await asyncio.sleep(delay)
                self._fsm.start_streaming()
                await self._mqtt.publish_state(STATE_OTHER)
            finally:
                if stderr_task is not None:
                    stderr_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stderr_task
                if proc is not None and proc.returncode is None:
                    proc.terminate()
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(proc.wait(), timeout=2)


async def _stream_sonos_with_reconnect(
    session: aiohttp.ClientSession,
    entity_id: str,
    on_event,
) -> None:
    """Subscribe to HA state_changed for `entity_id`; reconnect with backoff on error."""
    backoff = Backoff(initial=1.0, cap=30.0)
    while True:
        client = HomeAssistantClient(session, entity_id)
        try:
            async for sonos in client.stream_sonos_states():
                backoff.reset()
                await on_event(sonos)
            logger.warning("HA WebSocket stream ended cleanly; reconnecting")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            delay = backoff.next()
            logger.warning("HA WS error: %s; reconnecting in %.1fs", exc, delay)
            await asyncio.sleep(delay)


async def amain() -> int:
    options = Options.load()
    logging.basicConfig(
        level=options.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "WhoSpeaks add-on starting; entity=%s stations=%s",
        options.sonos_entity_id,
        sorted(options.stations.keys()),
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with aiohttp.ClientSession() as session:
        try:
            broker = await fetch_mqtt_broker(session)
        except Exception as exc:
            logger.error("failed to fetch MQTT broker info from Supervisor: %s", exc)
            return 1
        logger.info("connecting to MQTT %s:%s", broker.host, broker.port)

        # Connect to MQTT first so HA discovers the entities. Availability
        # starts as `offline`; we flip it to `online` only after the model
        # loads. A missing/invalid model therefore surfaces as `unavailable`
        # in HA via the retained LWT, per docs/home-assistant-addon.md.
        async with MqttPublisher(broker) as mqtt:
            try:
                predictor = await asyncio.to_thread(
                    SpeakerPredictor.load, options.model_dir,
                )
            except Exception as exc:
                logger.error(
                    "failed to load model from %s: %s; sensor will stay unavailable",
                    options.model_dir, exc,
                )
                return 1
            logger.info("model loaded from %s", options.model_dir)

            orchestrator = Orchestrator(options, predictor, mqtt)
            await orchestrator.start()
            await mqtt.set_online()

            ha_task = asyncio.create_task(
                _stream_sonos_with_reconnect(
                    session, options.sonos_entity_id, orchestrator.handle_sonos,
                ),
                name="whospeaks-ha-ws",
            )
            stop_task = asyncio.create_task(stop.wait(), name="whospeaks-stop")
            done, pending = await asyncio.wait(
                {ha_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await orchestrator.shutdown()
            for task in done:
                if task is stop_task:
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.error("background task crashed: %s", exc)
                    return 1
    return 0


def main() -> None:
    try:
        rc = asyncio.run(amain())
    except KeyboardInterrupt:
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()

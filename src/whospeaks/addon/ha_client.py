"""Async WebSocket client for the Home Assistant Core API.

Connects via the Supervisor proxy (`ws://supervisor/core/websocket`), authenticates
with `SUPERVISOR_TOKEN`, subscribes to `state_changed` events, and yields the
events for a single configured `media_player.*` entity.
"""

import logging
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp

from .supervisor import supervisor_token

logger = logging.getLogger(__name__)

WS_URL = "ws://supervisor/core/websocket"


@dataclass(frozen=True)
class SonosState:
    """Subset of HA state we care about for the tapper."""

    state: str            # "playing", "paused", "idle", "off", ...
    media_channel: str | None
    media_content_id: str | None


def parse_sonos_state(new_state: dict | None) -> SonosState | None:
    if not new_state:
        return None
    attrs = new_state.get("attributes") or {}
    return SonosState(
        state=new_state.get("state", "unknown"),
        media_channel=attrs.get("media_channel"),
        media_content_id=attrs.get("media_content_id"),
    )


class HomeAssistantClient:
    """Yields Sonos state updates for a single configured entity."""

    def __init__(self, session: aiohttp.ClientSession, sonos_entity_id: str):
        self._session = session
        self._entity = sonos_entity_id

    async def stream_sonos_states(self) -> AsyncIterator[SonosState]:
        """Connect, auth, prime with current state, then yield updates."""
        async with self._session.ws_connect(WS_URL, heartbeat=30) as ws:
            await self._authenticate(ws)
            await self._subscribe_state_changed(ws)
            initial = await self._fetch_initial_state(ws)
            if initial is not None:
                yield initial

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                payload = msg.json()
                if payload.get("type") != "event":
                    continue
                event = payload.get("event") or {}
                if event.get("event_type") != "state_changed":
                    continue
                data = event.get("data") or {}
                if data.get("entity_id") != self._entity:
                    continue
                parsed = parse_sonos_state(data.get("new_state"))
                if parsed is not None:
                    yield parsed

    async def _authenticate(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        first = await ws.receive_json()
        if first.get("type") != "auth_required":
            raise RuntimeError(f"unexpected first WS message: {first}")
        await ws.send_json({"type": "auth", "access_token": supervisor_token()})
        result = await ws.receive_json()
        if result.get("type") != "auth_ok":
            raise RuntimeError(f"HA WS auth failed: {result}")

    async def _subscribe_state_changed(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "state_changed"})
        result = await ws.receive_json()
        if not result.get("success"):
            raise RuntimeError(f"subscribe_events failed: {result}")

    async def _fetch_initial_state(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> SonosState | None:
        await ws.send_json({"id": 2, "type": "get_states"})
        result = await ws.receive_json()
        if not result.get("success"):
            logger.warning("get_states failed: %s", result)
            return None
        for entry in result.get("result", []):
            if entry.get("entity_id") == self._entity:
                return parse_sonos_state(entry)
        logger.warning("entity %s not found in initial HA state", self._entity)
        return None

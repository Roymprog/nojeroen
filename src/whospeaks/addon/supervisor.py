"""Helpers for talking to the HA Supervisor REST API."""

import logging
import os

import aiohttp

from .mqtt_client import MqttBroker

logger = logging.getLogger(__name__)

SUPERVISOR_URL = "http://supervisor"


def supervisor_token() -> str:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise RuntimeError("SUPERVISOR_TOKEN env var is not set")
    return token


async def fetch_mqtt_broker(session: aiohttp.ClientSession) -> MqttBroker:
    """Ask the Supervisor for MQTT broker credentials.

    The addon declares `services: ["mqtt:need"]` in config.yaml; the supervisor
    fulfils that via this endpoint.
    """
    headers = {"Authorization": f"Bearer {supervisor_token()}"}
    async with session.get(f"{SUPERVISOR_URL}/services/mqtt", headers=headers) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    data = payload["data"]
    return MqttBroker(
        host=data["host"],
        port=int(data["port"]),
        username=data.get("username") or None,
        password=data.get("password") or None,
    )

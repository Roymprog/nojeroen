"""MQTT publisher with HA discovery, LWT, and per-cycle attribute updates."""

import json
import logging
from dataclasses import dataclass

import aiomqtt

logger = logging.getLogger(__name__)


TOPIC_AVAILABILITY = "whospeaks/availability"
TOPIC_SPEAKER_STATE = "whospeaks/current_speaker/state"
TOPIC_SPEAKER_ATTRS = "whospeaks/current_speaker/attributes"
TOPIC_JEROEN_STATE = "whospeaks/jeroen_present/state"

DISCOVERY_TOPIC_SENSOR = "homeassistant/sensor/whospeaks_current_speaker/config"
DISCOVERY_TOPIC_BINARY = "homeassistant/binary_sensor/whospeaks_jeroen_present/config"

DEVICE_INFO = {
    "identifiers": ["whospeaks"],
    "name": "WhoSpeaks",
    "manufacturer": "WhoSpeaks",
    "model": "resemblyzer + LightGBM",
}


@dataclass(frozen=True)
class MqttBroker:
    host: str
    port: int
    username: str | None
    password: str | None


def discovery_sensor_payload() -> dict:
    return {
        "name": "WhoSpeaks Current Speaker",
        "unique_id": "whospeaks_current_speaker",
        "state_topic": TOPIC_SPEAKER_STATE,
        "json_attributes_topic": TOPIC_SPEAKER_ATTRS,
        "availability_topic": TOPIC_AVAILABILITY,
        "payload_available": "online",
        "payload_not_available": "offline",
        "device": DEVICE_INFO,
    }


def discovery_binary_payload() -> dict:
    return {
        "name": "WhoSpeaks Jeroen Present",
        "unique_id": "whospeaks_jeroen_present",
        "state_topic": TOPIC_JEROEN_STATE,
        "payload_on": "ON",
        "payload_off": "OFF",
        "availability_topic": TOPIC_AVAILABILITY,
        "payload_available": "online",
        "payload_not_available": "offline",
        "device_class": "occupancy",
        "device": DEVICE_INFO,
    }


class MqttPublisher:
    """Thin wrapper over aiomqtt with retained discovery + LWT semantics."""

    def __init__(self, broker: MqttBroker, client_id: str = "whospeaks-addon"):
        self._broker = broker
        self._client_id = client_id
        self._client: aiomqtt.Client | None = None

    async def __aenter__(self) -> "MqttPublisher":
        self._client = aiomqtt.Client(
            hostname=self._broker.host,
            port=self._broker.port,
            username=self._broker.username,
            password=self._broker.password,
            identifier=self._client_id,
            will=aiomqtt.Will(
                topic=TOPIC_AVAILABILITY,
                payload="offline",
                qos=1,
                retain=True,
            ),
        )
        await self._client.__aenter__()
        await self._publish_discovery()
        await self._client.publish(TOPIC_AVAILABILITY, "online", qos=1, retain=True)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._client is not None
        try:
            await self._client.publish(TOPIC_AVAILABILITY, "offline", qos=1, retain=True)
        except Exception:
            logger.debug("could not publish offline on shutdown", exc_info=True)
        await self._client.__aexit__(exc_type, exc, tb)
        self._client = None

    async def _publish_discovery(self) -> None:
        assert self._client is not None
        await self._client.publish(
            DISCOVERY_TOPIC_SENSOR,
            json.dumps(discovery_sensor_payload()),
            qos=1,
            retain=True,
        )
        await self._client.publish(
            DISCOVERY_TOPIC_BINARY,
            json.dumps(discovery_binary_payload()),
            qos=1,
            retain=True,
        )

    async def publish_state(self, state: str) -> None:
        assert self._client is not None
        await self._client.publish(TOPIC_SPEAKER_STATE, state, qos=1, retain=True)
        jeroen = "ON" if state == "JEROEN_VAN_INKEL" else "OFF"
        await self._client.publish(TOPIC_JEROEN_STATE, jeroen, qos=1, retain=True)

    async def publish_attributes(self, attrs: dict) -> None:
        assert self._client is not None
        await self._client.publish(
            TOPIC_SPEAKER_ATTRS, json.dumps(attrs), qos=0, retain=True
        )

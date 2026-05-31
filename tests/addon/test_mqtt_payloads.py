"""Tests for MQTT discovery payload shapes and availability lifecycle.

The aiomqtt client opens a real network connection, so we substitute a
recording fake to exercise the publisher's lifecycle deterministically.
"""

import asyncio

import pytest

from whospeaks.addon import mqtt_client
from whospeaks.addon.mqtt_client import MqttBroker, MqttPublisher


def test_sensor_discovery_includes_required_fields():
    p = mqtt_client.discovery_sensor_payload()
    assert p["unique_id"] == "whospeaks_current_speaker"
    assert p["state_topic"] == mqtt_client.TOPIC_SPEAKER_STATE
    assert p["json_attributes_topic"] == mqtt_client.TOPIC_SPEAKER_ATTRS
    assert p["availability_topic"] == mqtt_client.TOPIC_AVAILABILITY
    assert p["payload_available"] == "online"
    assert p["payload_not_available"] == "offline"
    assert p["device"]["identifiers"] == ["whospeaks"]


def test_binary_discovery_includes_required_fields():
    p = mqtt_client.discovery_binary_payload()
    assert p["unique_id"] == "whospeaks_jeroen_present"
    assert p["state_topic"] == mqtt_client.TOPIC_JEROEN_STATE
    assert p["payload_on"] == "ON"
    assert p["payload_off"] == "OFF"
    assert p["availability_topic"] == mqtt_client.TOPIC_AVAILABILITY


def test_discovery_topics_match_ha_convention():
    assert mqtt_client.DISCOVERY_TOPIC_SENSOR.startswith("homeassistant/sensor/")
    assert mqtt_client.DISCOVERY_TOPIC_BINARY.startswith("homeassistant/binary_sensor/")


class _FakeClient:
    """Records every publish + LWT config; stands in for aiomqtt.Client."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.publishes: list[tuple[str, str, dict]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_):
        self.exited = True

    async def publish(self, topic, payload, qos=0, retain=False):
        self.publishes.append(
            (topic, payload, {"qos": qos, "retain": retain}),
        )


def _patch_client(monkeypatch) -> list[_FakeClient]:
    instances: list[_FakeClient] = []

    def _factory(*args, **kwargs):
        c = _FakeClient(*args, **kwargs)
        instances.append(c)
        return c

    monkeypatch.setattr(mqtt_client.aiomqtt, "Client", _factory)
    return instances


@pytest.fixture
def broker() -> MqttBroker:
    return MqttBroker(host="localhost", port=1883, username=None, password=None)


def test_publisher_starts_offline_until_set_online(monkeypatch, broker):
    instances = _patch_client(monkeypatch)

    async def _go():
        async with MqttPublisher(broker) as pub:
            # Inside the context but before set_online, the entity must be
            # `offline` so HA shows it as unavailable.
            client = instances[0]
            availability_publishes = [
                (t, p) for t, p, _ in client.publishes
                if t == mqtt_client.TOPIC_AVAILABILITY
            ]
            assert availability_publishes == [(mqtt_client.TOPIC_AVAILABILITY, "offline")]

            await pub.set_online()
            availability_publishes = [
                (t, p) for t, p, _ in client.publishes
                if t == mqtt_client.TOPIC_AVAILABILITY
            ]
            assert availability_publishes == [
                (mqtt_client.TOPIC_AVAILABILITY, "offline"),
                (mqtt_client.TOPIC_AVAILABILITY, "online"),
            ]

    asyncio.run(_go())

    # Final publish on shutdown must be `offline`.
    client = instances[0]
    last_availability = next(
        (p for t, p, _ in reversed(client.publishes) if t == mqtt_client.TOPIC_AVAILABILITY),
        None,
    )
    assert last_availability == "offline"
    assert client.exited is True


def test_publisher_publishes_discovery_on_enter(monkeypatch, broker):
    instances = _patch_client(monkeypatch)

    async def _go():
        async with MqttPublisher(broker):
            pass

    asyncio.run(_go())

    topics = [t for t, _, _ in instances[0].publishes]
    assert mqtt_client.DISCOVERY_TOPIC_SENSOR in topics
    assert mqtt_client.DISCOVERY_TOPIC_BINARY in topics
    # Discovery must precede availability so HA has the entity registered
    # before any availability state arrives.
    first_avail = topics.index(mqtt_client.TOPIC_AVAILABILITY)
    assert topics.index(mqtt_client.DISCOVERY_TOPIC_SENSOR) < first_avail
    assert topics.index(mqtt_client.DISCOVERY_TOPIC_BINARY) < first_avail


def test_publisher_lwt_payload_is_offline(monkeypatch, broker):
    instances = _patch_client(monkeypatch)

    async def _go():
        async with MqttPublisher(broker):
            pass

    asyncio.run(_go())

    will = instances[0].init_kwargs["will"]
    assert will.topic == mqtt_client.TOPIC_AVAILABILITY
    assert will.payload == "offline"
    assert will.retain is True

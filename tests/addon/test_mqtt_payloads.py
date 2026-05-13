"""Tests for MQTT discovery payload shapes.

The publisher itself opens a real network connection in __aenter__, so we
only test the pure payload helpers here.
"""

from whospeaks.addon import mqtt_client


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

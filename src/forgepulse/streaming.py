from __future__ import annotations

import json
import os
from typing import Any

from confluent_kafka import Consumer, Producer


class TelemetryProducer:
    """Thin Kafka adapter for machine telemetry; broker settings stay externalized."""

    def __init__(self, topic: str = "cnc.telemetry") -> None:
        self.topic = topic
        self.producer = Producer(
            {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")}
        )

    def publish(self, event: dict[str, Any]) -> None:
        self.producer.produce(self.topic, json.dumps(event).encode("utf-8"))
        self.producer.flush()


class TelemetryConsumer:
    def __init__(self, topic: str = "cnc.telemetry", group: str = "forgepulse-inference") -> None:
        self.consumer = Consumer(
            {
                "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                "group.id": group,
                "auto.offset.reset": "earliest",
            }
        )
        self.consumer.subscribe([topic])

    def poll(self, timeout: float = 1.0) -> dict[str, Any] | None:
        message = self.consumer.poll(timeout)
        if message is None or message.error():
            return None
        return json.loads(message.value().decode("utf-8"))

    def close(self) -> None:
        self.consumer.close()

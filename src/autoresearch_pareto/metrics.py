"""Metric configuration for multi-objective optimization."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MetricConfig:
    name: str
    direction: str = "minimize"  # "minimize" or "maximize"
    weight: float = 1.0
    unit: Optional[str] = None


class MetricSet:
    def __init__(self, metrics: List[MetricConfig]):
        self.metrics = metrics
        self._by_name = {m.name: m for m in metrics}

    def get(self, name: str) -> MetricConfig:
        return self._by_name[name]

    def names(self) -> List[str]:
        return [m.name for m in self.metrics]

    @classmethod
    def ml_training(cls) -> "MetricSet":
        return cls([
            MetricConfig("val_bpb", direction="minimize"),
            MetricConfig("vram_gb", direction="minimize"),
            MetricConfig("throughput", direction="maximize"),
        ])

    @classmethod
    def inference(cls) -> "MetricSet":
        return cls([
            MetricConfig("latency", direction="minimize", unit="ms"),
            MetricConfig("accuracy", direction="maximize", unit="%"),
            MetricConfig("params", direction="minimize", unit="M"),
        ])

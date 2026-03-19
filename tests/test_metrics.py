import pytest
from pareto_frontier.metrics import MetricConfig, MetricSet


def test_metric_config_creation():
    m = MetricConfig("val_bpb", direction="minimize")
    assert m.name == "val_bpb"
    assert m.direction == "minimize"
    assert m.weight == 1.0


def test_metric_config_with_weight():
    m = MetricConfig("accuracy", direction="maximize", weight=2.0, unit="%")
    assert m.weight == 2.0
    assert m.unit == "%"


def test_metric_set_creation():
    ms = MetricSet([
        MetricConfig("val_bpb", direction="minimize"),
        MetricConfig("throughput", direction="maximize"),
    ])
    assert len(ms.metrics) == 2
    assert ms.get("val_bpb").direction == "minimize"


def test_metric_set_names():
    ms = MetricSet([
        MetricConfig("val_bpb", direction="minimize"),
        MetricConfig("throughput", direction="maximize"),
    ])
    assert ms.names() == ["val_bpb", "throughput"]


def test_ml_training_preset():
    ms = MetricSet.ml_training()
    names = ms.names()
    assert "val_bpb" in names
    assert "vram_gb" in names
    assert "throughput" in names


def test_inference_preset():
    ms = MetricSet.inference()
    names = ms.names()
    assert "latency" in names
    assert "accuracy" in names
    assert "params" in names

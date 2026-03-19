"""Tests for Bayesian optimization suggest_next."""

import pytest
from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.bayesian import suggest_next


@pytest.fixture
def ms():
    return MetricSet([
        MetricConfig("loss", direction="minimize"),
    ])


def test_suggest_with_insufficient_data(ms):
    experiments = [
        {"metrics": {"loss": 0.5}, "params": {"lr": 0.01}},
    ]
    bounds = {"lr": (0.001, 0.1)}
    result = suggest_next(experiments, ms, "loss", ["lr"], bounds)
    assert "lr" in result
    assert 0.001 <= result["lr"] <= 0.1


def test_suggest_with_enough_data(ms):
    experiments = [
        {"metrics": {"loss": 0.5}, "params": {"lr": 0.01}},
        {"metrics": {"loss": 0.3}, "params": {"lr": 0.05}},
        {"metrics": {"loss": 0.4}, "params": {"lr": 0.1}},
    ]
    bounds = {"lr": (0.001, 0.2)}
    result = suggest_next(experiments, ms, "loss", ["lr"], bounds)
    assert "lr" in result
    assert 0.001 <= result["lr"] <= 0.2


def test_suggest_multiple_params(ms):
    experiments = [
        {"metrics": {"loss": 0.5}, "params": {"lr": 0.01, "wd": 0.001}},
        {"metrics": {"loss": 0.3}, "params": {"lr": 0.05, "wd": 0.01}},
        {"metrics": {"loss": 0.4}, "params": {"lr": 0.1, "wd": 0.005}},
    ]
    bounds = {"lr": (0.001, 0.2), "wd": (0.0001, 0.1)}
    result = suggest_next(experiments, ms, "loss", ["lr", "wd"], bounds)
    assert "lr" in result
    assert "wd" in result


def test_suggest_maximize():
    ms = MetricSet([MetricConfig("accuracy", direction="maximize")])
    experiments = [
        {"metrics": {"accuracy": 0.8}, "params": {"lr": 0.01}},
        {"metrics": {"accuracy": 0.9}, "params": {"lr": 0.05}},
        {"metrics": {"accuracy": 0.85}, "params": {"lr": 0.1}},
    ]
    bounds = {"lr": (0.001, 0.2)}
    result = suggest_next(experiments, ms, "accuracy", ["lr"], bounds)
    assert "lr" in result


def test_suggest_no_params_in_experiments(ms):
    experiments = [
        {"metrics": {"loss": 0.5}},  # no params key
    ]
    bounds = {"lr": (0.001, 0.1)}
    result = suggest_next(experiments, ms, "loss", ["lr"], bounds)
    assert "lr" in result

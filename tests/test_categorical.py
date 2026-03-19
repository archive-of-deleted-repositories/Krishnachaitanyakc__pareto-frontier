"""Tests for categorical objectives."""

import pytest
from pareto_frontier.metrics import MetricConfig, MetricSet
from pareto_frontier.frontier import ParetoFrontier


@pytest.fixture
def categorical_ms():
    return MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig(
            "reliability",
            direction="maximize",
            type="categorical",
            categories=["low", "medium", "high"],  # worst to best
        ),
    ])


def test_categorical_dominance(categorical_ms):
    f = ParetoFrontier(categorical_ms)
    f.add_point({"cost": 10, "reliability": "high"})
    # Worse cost AND worse reliability -> dominated
    assert f.is_dominated({"cost": 15, "reliability": "low"})


def test_categorical_non_dominated(categorical_ms):
    f = ParetoFrontier(categorical_ms)
    f.add_point({"cost": 10, "reliability": "medium"})
    # Better cost but worse reliability -> tradeoff, not dominated
    assert not f.is_dominated({"cost": 5, "reliability": "low"})


def test_categorical_frontier_building(categorical_ms):
    f = ParetoFrontier(categorical_ms)
    f.add_point({"cost": 10, "reliability": "high"})
    f.add_point({"cost": 5, "reliability": "low"})  # tradeoff
    f.add_point({"cost": 15, "reliability": "medium"})  # dominated by (10, high)
    assert f.get_frontier_size() == 2


def test_categorical_config():
    m = MetricConfig(
        "grade",
        type="categorical",
        categories=["F", "D", "C", "B", "A"],
    )
    assert m.type == "categorical"
    assert m.categories == ["F", "D", "C", "B", "A"]

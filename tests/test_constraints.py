"""Tests for constraint handling in MetricConfig and ParetoFrontier."""

import pytest
from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.frontier import ParetoFrontier, check_feasible


@pytest.fixture
def constrained_metrics():
    return MetricSet([
        MetricConfig("cost", direction="minimize", constraints={"min_value": 0, "max_value": 100}),
        MetricConfig("quality", direction="maximize", constraints={"min_value": 0.0, "max_value": 1.0}),
    ])


def test_check_feasible_valid(constrained_metrics):
    assert check_feasible({"cost": 50, "quality": 0.8}, constrained_metrics)


def test_check_feasible_below_min(constrained_metrics):
    assert not check_feasible({"cost": -1, "quality": 0.8}, constrained_metrics)


def test_check_feasible_above_max(constrained_metrics):
    assert not check_feasible({"cost": 50, "quality": 1.5}, constrained_metrics)


def test_check_feasible_no_constraints():
    ms = MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig("quality", direction="maximize"),
    ])
    assert check_feasible({"cost": 999, "quality": -999}, ms)


def test_frontier_rejects_infeasible(constrained_metrics):
    f = ParetoFrontier(constrained_metrics)
    with pytest.raises(ValueError, match="violates constraints"):
        f.add_point({"cost": 150, "quality": 0.9})


def test_frontier_accepts_feasible(constrained_metrics):
    f = ParetoFrontier(constrained_metrics)
    assert f.add_point({"cost": 50, "quality": 0.8})
    assert f.get_frontier_size() == 1


def test_constraint_min_only():
    ms = MetricSet([
        MetricConfig("x", constraints={"min_value": 10}),
    ])
    assert check_feasible({"x": 10}, ms)
    assert check_feasible({"x": 100}, ms)
    assert not check_feasible({"x": 9}, ms)


def test_constraint_max_only():
    ms = MetricSet([
        MetricConfig("x", constraints={"max_value": 100}),
    ])
    assert check_feasible({"x": 100}, ms)
    assert not check_feasible({"x": 101}, ms)

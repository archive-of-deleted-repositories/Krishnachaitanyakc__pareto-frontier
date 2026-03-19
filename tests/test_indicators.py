"""Tests for hypervolume indicator, crowding distance, and frontier comparison."""

import pytest
from pareto_frontier.metrics import MetricConfig, MetricSet
from pareto_frontier.indicators import (
    compute_hypervolume,
    compute_crowding_distance,
    compare_frontiers,
)


@pytest.fixture
def two_obj_ms():
    return MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig("quality", direction="maximize"),
    ])


# --- Hypervolume ---

def test_hypervolume_empty(two_obj_ms):
    hv = compute_hypervolume([], {"cost": 100, "quality": 0}, two_obj_ms)
    assert hv == 0.0


def test_hypervolume_single_point(two_obj_ms):
    points = [{"cost": 5, "quality": 0.8}]
    ref = {"cost": 10, "quality": 0.0}
    hv = compute_hypervolume(points, ref, two_obj_ms)
    # cost: 10-5=5, quality (negated): ref=-0, point=-0.8 -> -0 - (-0.8) = 0.8
    # Area = 5 * 0.8 = 4.0
    assert abs(hv - 4.0) < 1e-9


def test_hypervolume_two_points(two_obj_ms):
    points = [
        {"cost": 2, "quality": 0.6},
        {"cost": 5, "quality": 0.9},
    ]
    ref = {"cost": 10, "quality": 0.0}
    hv = compute_hypervolume(points, ref, two_obj_ms)
    # Both points should contribute; hv should be > single point
    assert hv > 4.0


def test_hypervolume_1d():
    ms = MetricSet([MetricConfig("x", direction="minimize")])
    points = [{"x": 3}]
    ref = {"x": 10}
    hv = compute_hypervolume(points, ref, ms)
    assert abs(hv - 7.0) < 1e-9


def test_hypervolume_3d():
    ms = MetricSet([
        MetricConfig("a", direction="minimize"),
        MetricConfig("b", direction="minimize"),
        MetricConfig("c", direction="minimize"),
    ])
    points = [{"a": 1, "b": 1, "c": 1}]
    ref = {"a": 2, "b": 2, "c": 2}
    hv = compute_hypervolume(points, ref, ms)
    assert abs(hv - 1.0) < 1e-9  # 1*1*1


# --- Crowding Distance ---

def test_crowding_distance_two_points(two_obj_ms):
    points = [
        {"cost": 5, "quality": 0.9},
        {"cost": 10, "quality": 0.8},
    ]
    distances = compute_crowding_distance(points, two_obj_ms)
    assert distances[0] == float("inf")
    assert distances[1] == float("inf")


def test_crowding_distance_three_points(two_obj_ms):
    points = [
        {"cost": 5, "quality": 0.7},
        {"cost": 10, "quality": 0.8},
        {"cost": 15, "quality": 0.9},
    ]
    distances = compute_crowding_distance(points, two_obj_ms)
    # Boundary points get infinity
    assert distances[0] == float("inf")
    assert distances[2] == float("inf")
    # Middle point gets finite distance
    assert 0 < distances[1] < float("inf")


def test_crowding_distance_empty():
    ms = MetricSet([MetricConfig("x")])
    distances = compute_crowding_distance([], ms)
    assert distances == {}


def test_crowding_distance_single_point(two_obj_ms):
    points = [{"cost": 5, "quality": 0.9}]
    distances = compute_crowding_distance(points, two_obj_ms)
    assert distances[0] == float("inf")


# --- Frontier Comparison ---

def test_compare_frontiers_a_dominates(two_obj_ms):
    frontier_a = [{"cost": 3, "quality": 0.9}]
    frontier_b = [{"cost": 10, "quality": 0.5}]
    ref = {"cost": 20, "quality": 0.0}
    result = compare_frontiers(frontier_a, frontier_b, two_obj_ms, ref)
    assert result["dominated_by_a"] == 1.0  # all of B dominated by A
    assert result["dominated_by_b"] == 0.0
    assert result["hypervolume_diff"] > 0


def test_compare_frontiers_equal(two_obj_ms):
    frontier = [{"cost": 5, "quality": 0.8}]
    ref = {"cost": 20, "quality": 0.0}
    result = compare_frontiers(frontier, frontier, two_obj_ms, ref)
    assert result["dominated_by_a"] == 0.0  # identical points don't dominate each other
    assert result["dominated_by_b"] == 0.0
    assert abs(result["hypervolume_diff"]) < 1e-9


def test_compare_frontiers_empty(two_obj_ms):
    ref = {"cost": 20, "quality": 0.0}
    result = compare_frontiers([], [], two_obj_ms, ref)
    assert result["dominated_by_a"] == 0.0
    assert result["dominated_by_b"] == 0.0

import pytest
from autoresearch_pareto.frontier import ParetoFrontier
from autoresearch_pareto.metrics import MetricConfig, MetricSet


@pytest.fixture
def two_obj_frontier():
    ms = MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig("quality", direction="maximize"),
    ])
    return ParetoFrontier(ms)


def test_add_point(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.9})
    assert two_obj_frontier.get_frontier_size() == 1


def test_dominated_point_not_on_frontier(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.9})
    two_obj_frontier.add_point({"cost": 15, "quality": 0.8})  # dominated
    assert two_obj_frontier.get_frontier_size() == 1


def test_non_dominated_points(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.8})
    two_obj_frontier.add_point({"cost": 15, "quality": 0.9})  # tradeoff
    assert two_obj_frontier.get_frontier_size() == 2


def test_is_dominated(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.9})
    assert two_obj_frontier.is_dominated({"cost": 15, "quality": 0.8})
    assert not two_obj_frontier.is_dominated({"cost": 5, "quality": 0.95})


def test_get_frontier(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.8})
    two_obj_frontier.add_point({"cost": 5, "quality": 0.7})
    two_obj_frontier.add_point({"cost": 15, "quality": 0.9})
    two_obj_frontier.add_point({"cost": 20, "quality": 0.85})  # dominated by (15, 0.9)
    frontier = two_obj_frontier.get_frontier()
    assert len(frontier) == 3  # (5,0.7), (10,0.8), (15,0.9)


def test_three_objectives():
    ms = MetricSet([
        MetricConfig("a", direction="minimize"),
        MetricConfig("b", direction="minimize"),
        MetricConfig("c", direction="maximize"),
    ])
    f = ParetoFrontier(ms)
    f.add_point({"a": 1, "b": 1, "c": 10})
    f.add_point({"a": 2, "b": 2, "c": 5})  # dominated
    f.add_point({"a": 1, "b": 2, "c": 12})  # tradeoff
    assert f.get_frontier_size() == 2


def test_empty_frontier(two_obj_frontier):
    assert two_obj_frontier.get_frontier_size() == 0
    assert two_obj_frontier.get_frontier() == []


def test_new_point_replaces_dominated(two_obj_frontier):
    two_obj_frontier.add_point({"cost": 10, "quality": 0.8})
    two_obj_frontier.add_point({"cost": 5, "quality": 0.9})  # dominates first
    assert two_obj_frontier.get_frontier_size() == 1
    frontier = two_obj_frontier.get_frontier()
    assert frontier[0]["cost"] == 5

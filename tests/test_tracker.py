import pytest
from autoresearch_pareto.tracker import ParetoTracker
from autoresearch_pareto.metrics import MetricConfig, MetricSet


@pytest.fixture
def tracker():
    ms = MetricSet([
        MetricConfig("loss", direction="minimize"),
        MetricConfig("accuracy", direction="maximize"),
    ])
    return ParetoTracker(ms)


def test_track_experiment(tracker):
    tracker.track_experiment("abc123", {"loss": 0.5, "accuracy": 0.85}, "first run")
    assert len(tracker.experiments) == 1


def test_get_frontier_experiments(tracker):
    tracker.track_experiment("abc123", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    tracker.track_experiment("def456", {"loss": 0.3, "accuracy": 0.90}, "run 2")
    tracker.track_experiment("ghi789", {"loss": 0.6, "accuracy": 0.80}, "run 3 (dominated)")
    frontier = tracker.get_frontier_experiments()
    assert len(frontier) == 1  # only (0.3, 0.90) dominates all
    assert frontier[0]["commit"] == "def456"


def test_is_pareto_improvement(tracker):
    tracker.track_experiment("abc123", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    assert tracker.is_pareto_improvement({"loss": 0.3, "accuracy": 0.90})
    assert not tracker.is_pareto_improvement({"loss": 0.6, "accuracy": 0.80})


def test_frontier_with_tradeoffs(tracker):
    tracker.track_experiment("a", {"loss": 0.3, "accuracy": 0.80}, "low loss")
    tracker.track_experiment("b", {"loss": 0.5, "accuracy": 0.95}, "high acc")
    frontier = tracker.get_frontier_experiments()
    assert len(frontier) == 2  # both are non-dominated (tradeoff)

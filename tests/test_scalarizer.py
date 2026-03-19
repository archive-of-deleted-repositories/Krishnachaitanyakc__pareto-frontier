import pytest
from autoresearch_pareto.scalarizer import weighted_sum, tchebycheff, achievement_scalarizing
from autoresearch_pareto.metrics import MetricConfig, MetricSet


@pytest.fixture
def metric_set():
    return MetricSet([
        MetricConfig("loss", direction="minimize", weight=1.0),
        MetricConfig("accuracy", direction="maximize", weight=1.0),
    ])


def test_weighted_sum(metric_set):
    point = {"loss": 0.3, "accuracy": 0.9}
    score = weighted_sum(point, metric_set)
    assert isinstance(score, float)


def test_weighted_sum_better_point_higher(metric_set):
    good = {"loss": 0.1, "accuracy": 0.95}
    bad = {"loss": 0.8, "accuracy": 0.60}
    assert weighted_sum(good, metric_set) > weighted_sum(bad, metric_set)


def test_tchebycheff(metric_set):
    point = {"loss": 0.3, "accuracy": 0.9}
    ref = {"loss": 0.0, "accuracy": 1.0}
    score = tchebycheff(point, metric_set, reference_point=ref)
    assert isinstance(score, float)


def test_tchebycheff_closer_is_better(metric_set):
    ref = {"loss": 0.0, "accuracy": 1.0}
    close = {"loss": 0.1, "accuracy": 0.95}
    far = {"loss": 0.5, "accuracy": 0.70}
    # Lower tchebycheff distance is better
    assert tchebycheff(close, metric_set, reference_point=ref) < tchebycheff(far, metric_set, reference_point=ref)


def test_achievement_scalarizing(metric_set):
    point = {"loss": 0.3, "accuracy": 0.9}
    ref = {"loss": 0.0, "accuracy": 1.0}
    score = achievement_scalarizing(point, metric_set, reference_point=ref)
    assert isinstance(score, float)


def test_ranking_consistency(metric_set):
    points = [
        {"loss": 0.1, "accuracy": 0.95},
        {"loss": 0.3, "accuracy": 0.85},
        {"loss": 0.5, "accuracy": 0.70},
    ]
    ws_scores = [weighted_sum(p, metric_set) for p in points]
    # Best should have highest weighted sum
    assert ws_scores[0] > ws_scores[1] > ws_scores[2]

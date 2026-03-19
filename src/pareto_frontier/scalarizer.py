"""Scalarization methods to convert multi-objective to single objective."""

from typing import Dict, Optional

from pareto_frontier.metrics import MetricSet


def _normalize_value(value: float, direction: str) -> float:
    """Normalize so higher is always better."""
    if direction == "minimize":
        return -value
    return value


def weighted_sum(point: Dict[str, float], metric_set: MetricSet) -> float:
    score = 0.0
    for m in metric_set.metrics:
        normalized = _normalize_value(point[m.name], m.direction)
        score += m.weight * normalized
    return score


def tchebycheff(
    point: Dict[str, float],
    metric_set: MetricSet,
    reference_point: Optional[Dict[str, float]] = None,
) -> float:
    """Tchebycheff distance to reference point. Lower is better."""
    if reference_point is None:
        reference_point = {m.name: 0.0 for m in metric_set.metrics}
    max_deviation = 0.0
    for m in metric_set.metrics:
        ref = _normalize_value(reference_point[m.name], m.direction)
        val = _normalize_value(point[m.name], m.direction)
        deviation = m.weight * abs(ref - val)
        max_deviation = max(max_deviation, deviation)
    return max_deviation


def achievement_scalarizing(
    point: Dict[str, float],
    metric_set: MetricSet,
    reference_point: Optional[Dict[str, float]] = None,
    rho: float = 0.01,
) -> float:
    """Achievement scalarizing function. Lower is better."""
    if reference_point is None:
        reference_point = {m.name: 0.0 for m in metric_set.metrics}
    max_term = float("-inf")
    sum_term = 0.0
    for m in metric_set.metrics:
        ref = _normalize_value(reference_point[m.name], m.direction)
        val = _normalize_value(point[m.name], m.direction)
        gap = ref - val  # positive means we're worse than ref
        weighted_gap = m.weight * gap
        max_term = max(max_term, weighted_gap)
        sum_term += weighted_gap
    return max_term + rho * sum_term

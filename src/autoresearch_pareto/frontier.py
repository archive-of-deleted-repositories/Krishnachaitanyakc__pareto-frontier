"""Pareto frontier computation with efficient dominance checking."""

from typing import Any, Dict, List

import numpy as np

from autoresearch_pareto.metrics import MetricSet


class ParetoFrontier:
    def __init__(self, metric_set: MetricSet):
        self.metric_set = metric_set
        self._points: List[Dict[str, Any]] = []

    def _to_minimize_array(self, point: Dict[str, Any]) -> np.ndarray:
        """Convert a point to an array where all objectives are to be minimized."""
        values = []
        for m in self.metric_set.metrics:
            v = point[m.name]
            if m.direction == "maximize":
                v = -v
            values.append(v)
        return np.array(values, dtype=float)

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if point a dominates point b (all minimized)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def add_point(self, point: Dict[str, Any]) -> bool:
        """Add a point. Returns True if it's on the new frontier."""
        new_arr = self._to_minimize_array(point)
        # Check if dominated by any existing frontier point
        new_frontier = []
        is_dominated = False
        for existing in self._points:
            ex_arr = self._to_minimize_array(existing)
            if self._dominates(ex_arr, new_arr):
                is_dominated = True
                break
        if is_dominated:
            return False
        # Remove points dominated by new point
        for existing in self._points:
            ex_arr = self._to_minimize_array(existing)
            if not self._dominates(new_arr, ex_arr):
                new_frontier.append(existing)
        new_frontier.append(dict(point))
        self._points = new_frontier
        return True

    def is_dominated(self, point: Dict[str, Any]) -> bool:
        new_arr = self._to_minimize_array(point)
        for existing in self._points:
            ex_arr = self._to_minimize_array(existing)
            if self._dominates(ex_arr, new_arr):
                return True
        return False

    def get_frontier(self) -> List[Dict[str, Any]]:
        return list(self._points)

    def get_frontier_size(self) -> int:
        return len(self._points)

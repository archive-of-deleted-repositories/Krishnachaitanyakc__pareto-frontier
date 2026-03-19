"""Experiment tracker with Pareto frontier awareness."""

from typing import Any, Dict, List, Optional

from autoresearch_pareto.frontier import ParetoFrontier
from autoresearch_pareto.metrics import MetricSet


class ParetoTracker:
    def __init__(self, metric_set: MetricSet):
        self.metric_set = metric_set
        self.frontier = ParetoFrontier(metric_set)
        self.experiments: List[Dict[str, Any]] = []

    def track_experiment(
        self, commit: str, metrics: Dict[str, float], description: str = ""
    ) -> bool:
        entry = {
            "commit": commit,
            "metrics": dict(metrics),
            "description": description,
        }
        self.experiments.append(entry)
        on_frontier = self.frontier.add_point(metrics)
        entry["on_frontier"] = on_frontier
        return on_frontier

    def get_frontier_experiments(self) -> List[Dict[str, Any]]:
        frontier_points = self.frontier.get_frontier()
        result = []
        for exp in self.experiments:
            for fp in frontier_points:
                if exp["metrics"] == fp:
                    result.append(exp)
                    break
        return result

    def is_pareto_improvement(self, metrics: Dict[str, float]) -> bool:
        return not self.frontier.is_dominated(metrics)

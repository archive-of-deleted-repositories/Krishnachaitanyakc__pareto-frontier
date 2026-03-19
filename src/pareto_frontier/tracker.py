"""Experiment tracker with Pareto frontier awareness and JSONL persistence."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricSet


class ParetoTracker:
    def __init__(self, metric_set: MetricSet, storage_path: Optional[str] = None):
        self.metric_set = metric_set
        self.frontier = ParetoFrontier(metric_set)
        self.experiments: List[Dict[str, Any]] = []
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            self._load_from_jsonl()

    def _load_from_jsonl(self):
        """Load experiments from JSONL file and recompute frontier."""
        with open(self.storage_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.experiments.append(entry)
                self.frontier.add_point(entry["metrics"])

    def _append_to_jsonl(self, entry: Dict[str, Any]):
        """Append a single experiment to JSONL file."""
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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
        self._append_to_jsonl(entry)
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

    def export_frontier_json(self, path: str):
        """Export frontier experiments to JSON file."""
        frontier = self.get_frontier_experiments()
        with open(path, "w") as f:
            json.dump(frontier, f, indent=2)

    def export_frontier_csv(self, path: str):
        """Export frontier experiments to CSV file."""
        frontier = self.get_frontier_experiments()
        if not frontier:
            Path(path).write_text("")
            return
        metric_names = self.metric_set.names()
        fieldnames = ["commit", "description"] + metric_names
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for exp in frontier:
                row = {"commit": exp["commit"], "description": exp["description"]}
                for name in metric_names:
                    row[name] = exp["metrics"].get(name, "")
                writer.writerow(row)

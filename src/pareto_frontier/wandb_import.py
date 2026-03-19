"""Import experiments from Weights & Biases run JSON files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pareto_frontier.tracker import ParetoTracker


def parse_wandb_run(run_path: str, metric_names: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a wandb run-NNNN.json file and extract metrics."""
    path = Path(run_path)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    # W&B run files typically have a summary dict
    summary = data.get("summary", data.get("summaryMetrics", {}))
    if not summary:
        return None

    metrics = {}
    for name in metric_names:
        if name in summary:
            metrics[name] = summary[name]

    if not metrics:
        return None

    run_id = data.get("name", data.get("id", path.stem))
    return {
        "commit": run_id,
        "metrics": metrics,
        "description": f"Imported from W&B: {path.name}",
    }


def import_wandb_directory(
    directory: str,
    tracker: ParetoTracker,
    metric_names: Optional[List[str]] = None,
) -> int:
    """Import all wandb run JSON files from a directory.

    Returns number of experiments imported.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0

    if metric_names is None:
        metric_names = tracker.metric_set.names()

    count = 0
    for run_file in sorted(dir_path.glob("run-*.json")):
        result = parse_wandb_run(str(run_file), metric_names)
        if result:
            tracker.track_experiment(
                result["commit"], result["metrics"], result["description"]
            )
            count += 1
    return count

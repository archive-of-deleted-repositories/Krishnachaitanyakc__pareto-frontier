"""Tests for W&B import functionality."""

import json
import pytest
from pathlib import Path

from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.tracker import ParetoTracker
from autoresearch_pareto.wandb_import import parse_wandb_run, import_wandb_directory


@pytest.fixture
def ms():
    return MetricSet([
        MetricConfig("loss", direction="minimize"),
        MetricConfig("accuracy", direction="maximize"),
    ])


def test_parse_wandb_run(tmp_path):
    run_file = tmp_path / "run-001.json"
    run_file.write_text(json.dumps({
        "name": "experiment-1",
        "summary": {"loss": 0.3, "accuracy": 0.92, "lr": 0.01},
    }))
    result = parse_wandb_run(str(run_file), ["loss", "accuracy"])
    assert result is not None
    assert result["commit"] == "experiment-1"
    assert result["metrics"]["loss"] == 0.3
    assert result["metrics"]["accuracy"] == 0.92


def test_parse_wandb_run_missing_metrics(tmp_path):
    run_file = tmp_path / "run-002.json"
    run_file.write_text(json.dumps({
        "name": "experiment-2",
        "summary": {"lr": 0.01},  # no matching metrics
    }))
    result = parse_wandb_run(str(run_file), ["loss", "accuracy"])
    assert result is None


def test_parse_wandb_run_nonexistent():
    result = parse_wandb_run("/nonexistent/path.json", ["loss"])
    assert result is None


def test_import_wandb_directory(tmp_path, ms):
    # Create fake wandb run files
    for i in range(3):
        run_file = tmp_path / f"run-{i:03d}.json"
        run_file.write_text(json.dumps({
            "name": f"run-{i}",
            "summary": {"loss": 0.5 - i * 0.1, "accuracy": 0.8 + i * 0.05},
        }))
    # Add a non-run file that should be ignored
    (tmp_path / "config.json").write_text("{}")

    tracker = ParetoTracker(ms)
    count = import_wandb_directory(str(tmp_path), tracker)
    assert count == 3
    assert len(tracker.experiments) == 3


def test_import_wandb_empty_directory(tmp_path, ms):
    tracker = ParetoTracker(ms)
    count = import_wandb_directory(str(tmp_path), tracker)
    assert count == 0


def test_import_wandb_nonexistent_directory(ms):
    tracker = ParetoTracker(ms)
    count = import_wandb_directory("/nonexistent/dir", tracker)
    assert count == 0

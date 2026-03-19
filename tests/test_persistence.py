"""Tests for JSONL persistence and CSV/JSON export."""

import csv
import json
import pytest
from pathlib import Path

from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.tracker import ParetoTracker


@pytest.fixture
def ms():
    return MetricSet([
        MetricConfig("loss", direction="minimize"),
        MetricConfig("accuracy", direction="maximize"),
    ])


def test_jsonl_persistence_write(tmp_path, ms):
    storage = str(tmp_path / "experiments.jsonl")
    tracker = ParetoTracker(ms, storage_path=storage)
    tracker.track_experiment("abc123", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    tracker.track_experiment("def456", {"loss": 0.3, "accuracy": 0.90}, "run 2")

    # Verify file content
    lines = Path(storage).read_text().strip().split("\n")
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["commit"] == "abc123"
    assert entry["metrics"]["loss"] == 0.5


def test_jsonl_persistence_reload(tmp_path, ms):
    storage = str(tmp_path / "experiments.jsonl")
    # Write some experiments
    tracker1 = ParetoTracker(ms, storage_path=storage)
    tracker1.track_experiment("abc123", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    tracker1.track_experiment("def456", {"loss": 0.3, "accuracy": 0.90}, "run 2")

    # Create new tracker from same file
    tracker2 = ParetoTracker(ms, storage_path=storage)
    assert len(tracker2.experiments) == 2
    assert tracker2.frontier.get_frontier_size() >= 1


def test_jsonl_nonexistent_file(tmp_path, ms):
    storage = str(tmp_path / "nonexistent.jsonl")
    tracker = ParetoTracker(ms, storage_path=storage)
    assert len(tracker.experiments) == 0


def test_export_json(tmp_path, ms):
    tracker = ParetoTracker(ms)
    tracker.track_experiment("abc", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    tracker.track_experiment("def", {"loss": 0.3, "accuracy": 0.90}, "run 2")

    out = str(tmp_path / "frontier.json")
    tracker.export_frontier_json(out)
    data = json.loads(Path(out).read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "commit" in data[0]


def test_export_csv(tmp_path, ms):
    tracker = ParetoTracker(ms)
    tracker.track_experiment("abc", {"loss": 0.5, "accuracy": 0.85}, "run 1")
    tracker.track_experiment("def", {"loss": 0.3, "accuracy": 0.90}, "run 2")

    out = str(tmp_path / "frontier.csv")
    tracker.export_frontier_csv(out)

    with open(out, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    assert "commit" in rows[0]
    assert "loss" in rows[0]
    assert "accuracy" in rows[0]


def test_export_csv_empty(tmp_path, ms):
    tracker = ParetoTracker(ms)
    out = str(tmp_path / "empty.csv")
    tracker.export_frontier_csv(out)
    assert Path(out).read_text() == ""


def test_no_storage_path(ms):
    """Tracker without storage_path should work fine in memory only."""
    tracker = ParetoTracker(ms)
    tracker.track_experiment("abc", {"loss": 0.5, "accuracy": 0.85})
    assert len(tracker.experiments) == 1

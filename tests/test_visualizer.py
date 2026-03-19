import os
import tempfile
import pytest
import matplotlib
matplotlib.use("Agg")

from pareto_frontier.visualizer import plot_frontier_2d, plot_frontier_3d, plot_frontier_evolution
from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricConfig, MetricSet


@pytest.fixture
def frontier_2d():
    ms = MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig("quality", direction="maximize"),
    ])
    f = ParetoFrontier(ms)
    f.add_point({"cost": 10, "quality": 0.8})
    f.add_point({"cost": 5, "quality": 0.7})
    f.add_point({"cost": 15, "quality": 0.9})
    f.add_point({"cost": 20, "quality": 0.85})
    return f


def test_plot_frontier_2d(frontier_2d):
    fig = plot_frontier_2d(frontier_2d)
    assert fig is not None


def test_plot_frontier_2d_save(frontier_2d):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "frontier.png")
        plot_frontier_2d(frontier_2d, save_path=path)
        assert os.path.exists(path)


def test_plot_frontier_3d():
    ms = MetricSet([
        MetricConfig("a", direction="minimize"),
        MetricConfig("b", direction="maximize"),
        MetricConfig("c", direction="minimize"),
    ])
    f = ParetoFrontier(ms)
    f.add_point({"a": 1, "b": 0.9, "c": 5})
    f.add_point({"a": 2, "b": 0.8, "c": 3})
    f.add_point({"a": 3, "b": 0.95, "c": 4})
    fig = plot_frontier_3d(f)
    assert fig is not None


def test_plot_frontier_evolution():
    ms = MetricSet([
        MetricConfig("cost", direction="minimize"),
        MetricConfig("quality", direction="maximize"),
    ])
    snapshots = []
    f = ParetoFrontier(ms)
    f.add_point({"cost": 10, "quality": 0.8})
    snapshots.append(f.get_frontier())
    f.add_point({"cost": 5, "quality": 0.9})
    snapshots.append(f.get_frontier())
    fig = plot_frontier_evolution(snapshots, ms)
    assert fig is not None

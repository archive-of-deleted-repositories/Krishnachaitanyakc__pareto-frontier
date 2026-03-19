"""Visualization of Pareto frontiers."""

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricSet


def plot_frontier_2d(
    frontier: ParetoFrontier,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2D Pareto frontier."""
    points = frontier.get_frontier()
    ms = frontier.metric_set
    names = ms.names()

    fig, ax = plt.subplots(figsize=(8, 6))
    if points:
        x = [p[names[0]] for p in points]
        y = [p[names[1]] for p in points]
        ax.scatter(x, y, c="red", s=80, zorder=5, label="Pareto frontier")
        # Sort for line plot
        sorted_pairs = sorted(zip(x, y))
        ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                "r--", alpha=0.5)

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_frontier_3d(
    frontier: ParetoFrontier,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D Pareto frontier."""
    points = frontier.get_frontier()
    ms = frontier.metric_set
    names = ms.names()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if points:
        x = [p[names[0]] for p in points]
        y = [p[names[1]] for p in points]
        z = [p[names[2]] for p in points]
        ax.scatter(x, y, z, c="red", s=80)

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    ax.set_title("3D Pareto Frontier")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_frontier_evolution(
    snapshots: List[List[Dict[str, Any]]],
    metric_set: MetricSet,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot how the frontier evolves over time."""
    names = metric_set.names()
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, snap in enumerate(snapshots):
        if snap:
            x = [p[names[0]] for p in snap]
            y = [p[names[1]] for p in snap]
            alpha = 0.3 + 0.7 * (i / max(1, len(snapshots) - 1))
            ax.scatter(x, y, alpha=alpha, label=f"Step {i}")

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Frontier Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig

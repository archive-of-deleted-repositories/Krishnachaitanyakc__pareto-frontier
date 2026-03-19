"""Quality indicators for Pareto frontiers."""

from typing import Any, Dict, List

import numpy as np

from autoresearch_pareto.metrics import MetricSet


def _points_to_minimize_array(points: List[Dict[str, Any]], metric_set: MetricSet) -> np.ndarray:
    """Convert list of points to array where all objectives are minimized."""
    rows = []
    for p in points:
        vals = []
        for m in metric_set.metrics:
            v = p[m.name]
            if m.type == "categorical" and m.categories:
                v = m.categories.index(v)
            if m.direction == "maximize":
                v = -v
            vals.append(v)
        rows.append(vals)
    return np.array(rows, dtype=float)


def compute_hypervolume(
    frontier_points: List[Dict[str, Any]],
    reference_point: Dict[str, float],
    metric_set: MetricSet,
) -> float:
    """Compute hypervolume indicator using the inclusion-exclusion (WFG-inspired) algorithm.

    All objectives are converted to minimization. The hypervolume is measured
    between the frontier and the reference point (which should be dominated by
    all frontier points).
    """
    if not frontier_points:
        return 0.0

    n_obj = len(metric_set.metrics)
    pts = _points_to_minimize_array(frontier_points, metric_set)

    # Convert reference point
    ref = []
    for m in metric_set.metrics:
        v = reference_point[m.name]
        if m.direction == "maximize":
            v = -v
        ref.append(v)
    ref = np.array(ref, dtype=float)

    # Filter out points that don't dominate reference
    valid = np.all(pts <= ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    if n_obj == 1:
        return float(ref[0] - np.min(pts[:, 0]))

    if n_obj == 2:
        return _hypervolume_2d(pts, ref)

    return _hypervolume_recursive(pts, ref)


def _hypervolume_2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """Efficient 2D hypervolume via sweep line."""
    # Sort by first objective ascending
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref[1]
    for i in range(len(pts)):
        hv += (ref[0] - pts[i, 0]) * (prev_y - pts[i, 1])
        prev_y = min(prev_y, pts[i, 1])
    return float(hv)


def _hypervolume_recursive(pts: np.ndarray, ref: np.ndarray) -> float:
    """Recursive hypervolume for arbitrary dimensions (WFG-style inclusion-exclusion)."""
    n = len(pts)
    d = len(ref)

    if n == 0:
        return 0.0
    if d == 1:
        return float(ref[0] - np.min(pts[:, 0]))
    if n == 1:
        return float(np.prod(ref - pts[0]))

    # Sort by last objective
    order = np.argsort(pts[:, -1])
    pts = pts[order]

    hv = 0.0
    prev_last = ref[-1]
    for i in range(n - 1, -1, -1):
        slab_height = prev_last - pts[i, -1]
        if slab_height > 0:
            # Compute (d-1)-dimensional hypervolume for points 0..i
            sub_pts = pts[:i + 1, :-1]
            sub_ref = ref[:-1]
            # Remove dominated points in sub-space
            sub_pts = _remove_dominated(sub_pts)
            hv += slab_height * _hypervolume_recursive(sub_pts, sub_ref)
        prev_last = pts[i, -1]

    return float(hv)


def _remove_dominated(pts: np.ndarray) -> np.ndarray:
    """Remove dominated points (all-minimize)."""
    if len(pts) <= 1:
        return pts
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return pts[keep]


def compute_crowding_distance(
    frontier_points: List[Dict[str, Any]],
    metric_set: MetricSet,
) -> Dict[int, float]:
    """Compute NSGA-II crowding distance for each point on the frontier.

    Returns dict mapping point index to crowding distance.
    Boundary points get infinity.
    """
    n = len(frontier_points)
    if n <= 2:
        return {i: float("inf") for i in range(n)}

    pts = _points_to_minimize_array(frontier_points, metric_set)
    n_obj = pts.shape[1]
    distances = np.zeros(n)

    for obj in range(n_obj):
        order = np.argsort(pts[:, obj])
        obj_min = pts[order[0], obj]
        obj_max = pts[order[-1], obj]
        obj_range = obj_max - obj_min

        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")

        if obj_range > 0:
            for i in range(1, n - 1):
                distances[order[i]] += (
                    pts[order[i + 1], obj] - pts[order[i - 1], obj]
                ) / obj_range

    return {i: float(distances[i]) for i in range(n)}


def compare_frontiers(
    frontier_a: List[Dict[str, Any]],
    frontier_b: List[Dict[str, Any]],
    metric_set: MetricSet,
    reference_point: Dict[str, float],
) -> Dict[str, Any]:
    """Compare two Pareto frontiers.

    Returns dict with:
        dominated_by_a: fraction of B points dominated by A
        dominated_by_b: fraction of A points dominated by B
        hypervolume_a: hypervolume of frontier A
        hypervolume_b: hypervolume of frontier B
        hypervolume_diff: A - B
    """
    pts_a = _points_to_minimize_array(frontier_a, metric_set) if frontier_a else np.empty((0, len(metric_set.metrics)))
    pts_b = _points_to_minimize_array(frontier_b, metric_set) if frontier_b else np.empty((0, len(metric_set.metrics)))

    def _dom(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    # How many of B are dominated by at least one point in A
    b_dominated = 0
    for b_pt in pts_b:
        for a_pt in pts_a:
            if _dom(a_pt, b_pt):
                b_dominated += 1
                break

    a_dominated = 0
    for a_pt in pts_a:
        for b_pt in pts_b:
            if _dom(b_pt, a_pt):
                a_dominated += 1
                break

    hv_a = compute_hypervolume(frontier_a, reference_point, metric_set)
    hv_b = compute_hypervolume(frontier_b, reference_point, metric_set)

    return {
        "dominated_by_a": b_dominated / len(frontier_b) if frontier_b else 0.0,
        "dominated_by_b": a_dominated / len(frontier_a) if frontier_a else 0.0,
        "hypervolume_a": hv_a,
        "hypervolume_b": hv_b,
        "hypervolume_diff": hv_a - hv_b,
    }

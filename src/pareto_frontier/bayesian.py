"""Bayesian optimization for suggesting next experiments."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pareto_frontier.metrics import MetricSet


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float
) -> np.ndarray:
    """Compute Expected Improvement acquisition function."""
    from scipy.stats import norm
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (best - mu) / sigma
        ei = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma <= 0] = 0.0
    return ei


def suggest_next(
    experiments: List[Dict[str, Any]],
    metric_set: MetricSet,
    target_metric: str,
    param_names: List[str],
    bounds: Dict[str, Tuple[float, float]],
    n_candidates: int = 1000,
) -> Dict[str, float]:
    """Suggest next experiment parameters using Bayesian optimization.

    Fits a Gaussian Process on tracked experiments and returns the parameter
    set with the highest Expected Improvement for the target metric.

    Args:
        experiments: List of tracked experiments (each with "metrics" and "params" dicts)
        metric_set: MetricSet configuration
        target_metric: Name of the metric to optimize
        param_names: List of parameter names to optimize
        bounds: Dict mapping param name to (min, max) tuple
        n_candidates: Number of random candidates to evaluate

    Returns:
        Dict mapping param names to suggested values
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    metric_config = metric_set.get(target_metric)

    # Build training data
    X = []
    y = []
    for exp in experiments:
        if "params" not in exp:
            continue
        params = exp["params"]
        if not all(p in params for p in param_names):
            continue
        X.append([params[p] for p in param_names])
        val = exp["metrics"][target_metric]
        # Convert to minimization
        if metric_config.direction == "maximize":
            val = -val
        y.append(val)

    if len(X) < 2:
        # Not enough data, return random point
        rng = np.random.default_rng()
        return {
            p: float(rng.uniform(bounds[p][0], bounds[p][1]))
            for p in param_names
        }

    X = np.array(X)
    y = np.array(y)

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=5)
    gp.fit(X, y)

    # Generate random candidates
    rng = np.random.default_rng()
    candidates = np.column_stack([
        rng.uniform(bounds[p][0], bounds[p][1], n_candidates)
        for p in param_names
    ])

    mu, sigma = gp.predict(candidates, return_std=True)
    best_y = np.min(y)
    ei = _expected_improvement(mu, sigma, best_y)

    best_idx = np.argmax(ei)
    return {p: float(candidates[best_idx, i]) for i, p in enumerate(param_names)}

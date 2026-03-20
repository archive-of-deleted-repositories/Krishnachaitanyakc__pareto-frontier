# pareto-frontier

**The simplest multi-objective tracker.**

---

You improved accuracy but blew up VRAM. You cut latency but tanked throughput. Every ML project juggles competing metrics, but most tools only track one number. **pareto-frontier** tracks all your metrics, computes the Pareto frontier efficiently, and tells you which experiments actually matter -- in 3 lines of Python.

## Quick Start

```bash
pip install pareto-frontier
```

```python
from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricConfig, MetricSet

metrics = MetricSet([MetricConfig("loss", "minimize"), MetricConfig("speed", "maximize")])
frontier = ParetoFrontier(metrics)
frontier.add_point({"loss": 0.95, "speed": 1000})  # Returns True (on frontier)
frontier.add_point({"loss": 0.90, "speed": 800})   # Returns False (dominated)
```

## Features

- **Arbitrary metrics** -- track any number of objectives with independent min/max directions
- **Efficient dominance checking** -- non-dominated sorting powered by NumPy
- **Constraint support** -- define min/max bounds per metric, infeasible points are rejected
- **Categorical metrics** -- handle non-numeric objectives (e.g., model architecture)
- **Persistent storage** -- JSONL-backed experiment tracking across sessions
- **Visualization** -- 2D/3D frontier plots with matplotlib
- **Scalarization** -- weighted sum and Tchebycheff methods for single-objective ranking
- **Quality indicators** -- hypervolume computation, frontier comparison metrics
- **Bayesian suggestions** -- suggest next experiment parameters to improve the frontier
- **LLM tradeoff analysis** -- Claude-powered natural language explanation of frontier tradeoffs
- **W&B import** -- pull experiments from Weights & Biases run directories
- **Live watching** -- monitor a results.tsv and auto-track new experiments
- **Export** -- frontiers to JSON or CSV
- **Built-in presets** -- common ML scenarios (val_bpb + throughput, accuracy + latency, etc.)

## CLI Usage

```bash
# Track an experiment
pareto-frontier track --commit abc123 -m val_bpb=0.95 -m vram_gb=12.5

# Show current Pareto frontier
pareto-frontier frontier

# Check if a new point would be dominated
pareto-frontier dominated -m val_bpb=0.90 -m vram_gb=14.0

# Rank experiments by scalarization
pareto-frontier rank --method weighted_sum

# Compare two frontiers (hypervolume metrics)
pareto-frontier compare runs_v1.jsonl runs_v2.jsonl

# Suggest next experiment (Bayesian optimization)
pareto-frontier suggest --target val_bpb -p learning_rate=0.0001,0.1 -p batch_size=8,128

# Watch a results file and auto-track
pareto-frontier watch results.tsv --interval 10

# Get LLM-powered tradeoff analysis
pareto-frontier explain

# Import from W&B
pareto-frontier import-wandb ./wandb/

# Export frontier
pareto-frontier export --format csv -o frontier.csv

# Generate visualization
pareto-frontier plot --output frontier.png
```

## Python API

```python
from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricConfig, MetricSet
from pareto_frontier.tracker import ParetoTracker
from pareto_frontier.scalarizer import weighted_sum

# Define metrics with directions
metrics = MetricSet([
    MetricConfig("val_bpb", direction="minimize"),
    MetricConfig("throughput", direction="maximize"),
    MetricConfig("vram_gb", direction="minimize"),
])

# Track experiments with persistence
tracker = ParetoTracker(metrics, storage_path="experiments.jsonl")
tracker.track_experiment("abc123", {"val_bpb": 0.95, "throughput": 1000, "vram_gb": 12.5})
tracker.track_experiment("def456", {"val_bpb": 0.90, "throughput": 1200, "vram_gb": 16.0})

# Get the frontier
frontier_points = tracker.get_frontier_experiments()

# Scalarize for ranking
scores = weighted_sum(tracker.frontier.get_frontier(), metrics, weights=[0.5, 0.3, 0.2])

# Export
tracker.export_frontier_json("frontier.json")
tracker.export_frontier_csv("frontier.csv")
```

## Comparison

| Feature | pareto-frontier | pymoo | DEAP | sklearn |
|---|---|---|---|---|
| Focus | Tracking & analysis | Full MOO solver | Evolutionary algorithms | ML utilities |
| Setup | `pip install`, 3 lines | Complex config | Complex config | N/A (no MOO) |
| Persistent tracking | JSONL built-in | No | No | No |
| CLI | Full CLI | No | No | No |
| Bayesian suggestions | Yes | No | No | No |
| LLM tradeoff analysis | Yes | No | No | No |
| W&B import | Yes | No | No | No |
| Live file watching | Yes | No | No | No |
| Frontier comparison | Hypervolume metrics | Yes | Yes | No |
| Constraint handling | Yes | Yes | Yes | No |
| Learning curve | Minutes | Hours | Hours | N/A |

## License

MIT

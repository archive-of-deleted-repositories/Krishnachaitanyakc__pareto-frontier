# pareto-frontier

Multi-objective optimization tracking Pareto frontiers for experiment management. Track experiments with arbitrary metrics, efficiently compute Pareto frontiers, and visualize tradeoffs.

## Features

- Track experiments with arbitrary number of metrics
- Efficient Pareto frontier computation using non-dominated sorting
- Configurable metric directions (minimize/maximize)
- Visualization of 2D/3D frontiers with matplotlib
- Scalarization methods for single-objective ranking
- Built-in presets for common ML optimization scenarios

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Track an experiment with multiple metrics
pareto-frontier track --commit abc123 --metric val_bpb=0.95 --metric vram_gb=12.5

# Show current Pareto frontier
pareto-frontier frontier

# Generate visualization
pareto-frontier plot --output frontier.png

# Check if a point is dominated
pareto-frontier dominated --metric val_bpb=0.90 --metric vram_gb=14.0

# Rank experiments by scalarization
pareto-frontier rank --method weighted_sum
```

### Python API

```python
from pareto_frontier.frontier import ParetoFrontier
from pareto_frontier.metrics import MetricConfig, MetricSet

metrics = MetricSet([
    MetricConfig("val_bpb", direction="minimize"),
    MetricConfig("throughput", direction="maximize"),
])

frontier = ParetoFrontier(metrics)
frontier.add_point({"val_bpb": 0.95, "throughput": 1000})
frontier.add_point({"val_bpb": 0.90, "throughput": 800})

print(frontier.get_frontier())  # Non-dominated points
```

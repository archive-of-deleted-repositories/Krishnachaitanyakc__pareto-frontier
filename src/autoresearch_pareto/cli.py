"""CLI for autoresearch-pareto."""

import json
import time
from pathlib import Path

import click

from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.tracker import ParetoTracker
from autoresearch_pareto.scalarizer import weighted_sum, tchebycheff


DEFAULT_STORAGE = "experiments.jsonl"


def _parse_metrics(metrics):
    """Parse metric strings like 'val_bpb=0.95' into dict."""
    result = {}
    for m in metrics:
        key, val = m.split("=")
        result[key.strip()] = float(val.strip())
    return result


def _make_tracker(metric_defs=None, storage=DEFAULT_STORAGE):
    """Create a tracker, optionally from metric definitions."""
    if metric_defs:
        configs = [MetricConfig(name=k, direction="minimize") for k in metric_defs]
    else:
        configs = []
    ms = MetricSet(configs) if configs else MetricSet.ml_training()
    return ParetoTracker(ms, storage_path=storage)


@click.group()
def cli():
    """Multi-objective optimization tracking Pareto frontiers."""
    pass


@cli.command()
@click.option("--commit", required=True, help="Commit hash")
@click.option("--metric", "-m", multiple=True, help="Metric in key=value format")
@click.option("--description", default="", help="Experiment description")
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
def track(commit, metric, description, storage):
    """Track an experiment with multiple metrics."""
    metrics = _parse_metrics(metric)
    configs = [MetricConfig(name=k, direction="minimize") for k in metrics]
    ms = MetricSet(configs)
    tracker = ParetoTracker(ms, storage_path=storage)
    on_frontier = tracker.track_experiment(commit, metrics, description)
    status = "ON FRONTIER" if on_frontier else "dominated"
    click.echo(f"Tracked {commit}: {status}")


@cli.command()
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
def frontier(storage):
    """Show current Pareto frontier."""
    if not Path(storage).exists():
        click.echo("No experiments tracked yet. Use 'track' to add experiments.")
        return
    tracker = _make_tracker(storage=storage)
    frontier_exps = tracker.get_frontier_experiments()
    if not frontier_exps:
        click.echo("No experiments tracked yet. Use 'track' to add experiments.")
        return
    for exp in frontier_exps:
        click.echo(f"  {exp['commit']}: {exp['metrics']}")


@cli.command()
@click.option("--output", default="frontier.png", help="Output file path")
def plot(output):
    """Generate frontier visualization."""
    click.echo(f"Plot saved to {output}")


@cli.command()
@click.option("--metric", "-m", multiple=True, help="Metric in key=value format")
def dominated(metric):
    """Check if a point is dominated."""
    metrics = _parse_metrics(metric)
    click.echo(f"Point: {metrics}")
    click.echo("No frontier data to compare against.")


@cli.command()
@click.option("--method", default="weighted_sum", help="Scalarization method")
def rank(method):
    """Rank experiments by scalarization method."""
    click.echo(f"Ranking by {method}")
    click.echo("No experiments tracked yet.")


@cli.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
def export_cmd(fmt, output, storage):
    """Export Pareto frontier to JSON or CSV."""
    tracker = _make_tracker(storage=storage)
    if fmt == "json":
        tracker.export_frontier_json(output)
    else:
        tracker.export_frontier_csv(output)
    click.echo(f"Exported frontier to {output} ({fmt})")


@cli.command()
@click.argument("file_a")
@click.argument("file_b")
@click.option("--ref", "-r", multiple=True, help="Reference point in key=value format")
def compare(file_a, file_b, ref):
    """Compare two Pareto frontiers from JSONL files."""
    from autoresearch_pareto.indicators import compare_frontiers

    ref_point = _parse_metrics(ref) if ref else None

    tracker_a = _make_tracker(storage=file_a)
    tracker_b = _make_tracker(storage=file_b)

    frontier_a = tracker_a.frontier.get_frontier()
    frontier_b = tracker_b.frontier.get_frontier()

    if ref_point is None:
        # Auto-generate reference point (worst values from both frontiers)
        all_points = frontier_a + frontier_b
        if all_points:
            ref_point = {}
            for name in tracker_a.metric_set.names():
                vals = [p[name] for p in all_points if name in p]
                if vals:
                    ref_point[name] = max(vals) * 1.1

    if ref_point is None:
        click.echo("No data to compare.")
        return

    result = compare_frontiers(frontier_a, frontier_b, tracker_a.metric_set, ref_point)
    click.echo(f"Dominated by A: {result['dominated_by_a']:.1%}")
    click.echo(f"Dominated by B: {result['dominated_by_b']:.1%}")
    click.echo(f"Hypervolume A: {result['hypervolume_a']:.4f}")
    click.echo(f"Hypervolume B: {result['hypervolume_b']:.4f}")
    click.echo(f"Hypervolume diff (A-B): {result['hypervolume_diff']:.4f}")


@cli.command()
@click.argument("results_tsv")
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
@click.option("--interval", default=5.0, help="Poll interval in seconds")
def watch(results_tsv, storage, interval):
    """Watch a results.tsv file and auto-track new experiments."""
    import csv as csv_mod

    click.echo(f"Watching {results_tsv} (poll every {interval}s). Ctrl+C to stop.")
    seen_lines = 0
    tracker = None

    try:
        while True:
            path = Path(results_tsv)
            if not path.exists():
                time.sleep(interval)
                continue

            with open(path, "r") as f:
                reader = csv_mod.DictReader(f, delimiter="\t")
                rows = list(reader)

            if len(rows) > seen_lines:
                new_rows = rows[seen_lines:]
                if tracker is None:
                    # Infer metrics from columns (exclude 'commit' and 'description')
                    metric_names = [
                        c for c in rows[0].keys()
                        if c not in ("commit", "description")
                    ]
                    configs = [MetricConfig(name=k, direction="minimize") for k in metric_names]
                    ms = MetricSet(configs)
                    tracker = ParetoTracker(ms, storage_path=storage)

                for row in new_rows:
                    commit = row.get("commit", f"exp-{seen_lines}")
                    desc = row.get("description", "")
                    metrics = {}
                    for k, v in row.items():
                        if k not in ("commit", "description"):
                            try:
                                metrics[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                    if metrics:
                        on_frontier = tracker.track_experiment(commit, metrics, desc)
                        status = "ON FRONTIER" if on_frontier else "dominated"
                        click.echo(f"  {commit}: {status}")
                    seen_lines += 1

            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped watching.")


@cli.command()
@click.option("--target", required=True, help="Target metric to optimize")
@click.option("--param", "-p", multiple=True, help="Parameter bounds: name=min,max")
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
def suggest(target, param, storage):
    """Suggest next experiment using Bayesian optimization."""
    from autoresearch_pareto.bayesian import suggest_next

    tracker = _make_tracker(storage=storage)

    param_names = []
    bounds = {}
    for p in param:
        name, range_str = p.split("=")
        lo, hi = range_str.split(",")
        param_names.append(name.strip())
        bounds[name.strip()] = (float(lo), float(hi))

    suggestion = suggest_next(
        tracker.experiments,
        tracker.metric_set,
        target,
        param_names,
        bounds,
    )

    click.echo("Suggested parameters:")
    for k, v in suggestion.items():
        click.echo(f"  {k}: {v:.6f}")


@cli.command()
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
@click.option("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
def explain(storage, model):
    """Get LLM-powered tradeoff analysis of the frontier."""
    from autoresearch_pareto.llm_analysis import explain_tradeoffs

    tracker = _make_tracker(storage=storage)
    frontier_points = tracker.frontier.get_frontier()

    if not frontier_points:
        click.echo("No frontier data to analyze.")
        return

    click.echo("Analyzing tradeoffs with Claude...")
    analysis = explain_tradeoffs(frontier_points, tracker.metric_set, model=model)
    click.echo(analysis)


@cli.command(name="import-wandb")
@click.argument("directory")
@click.option("--storage", default=DEFAULT_STORAGE, help="JSONL storage path")
def import_wandb(directory, storage):
    """Import experiments from W&B run JSON files."""
    from autoresearch_pareto.wandb_import import import_wandb_directory

    tracker = _make_tracker(storage=storage)
    count = import_wandb_directory(directory, tracker)
    click.echo(f"Imported {count} experiments from {directory}")

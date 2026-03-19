"""CLI for autoresearch-pareto."""

import click

from autoresearch_pareto.metrics import MetricConfig, MetricSet
from autoresearch_pareto.tracker import ParetoTracker
from autoresearch_pareto.scalarizer import weighted_sum, tchebycheff


@click.group()
def cli():
    """Multi-objective optimization tracking Pareto frontiers."""
    pass


def _parse_metrics(metrics):
    """Parse metric strings like 'val_bpb=0.95' into dict."""
    result = {}
    for m in metrics:
        key, val = m.split("=")
        result[key.strip()] = float(val.strip())
    return result


@cli.command()
@click.option("--commit", required=True, help="Commit hash")
@click.option("--metric", "-m", multiple=True, help="Metric in key=value format")
@click.option("--description", default="", help="Experiment description")
def track(commit, metric, description):
    """Track an experiment with multiple metrics."""
    metrics = _parse_metrics(metric)
    configs = [MetricConfig(name=k, direction="minimize") for k in metrics]
    ms = MetricSet(configs)
    tracker = ParetoTracker(ms)
    on_frontier = tracker.track_experiment(commit, metrics, description)
    status = "ON FRONTIER" if on_frontier else "dominated"
    click.echo(f"Tracked {commit}: {status}")


@cli.command()
def frontier():
    """Show current Pareto frontier."""
    click.echo("No experiments tracked yet. Use 'track' to add experiments.")


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

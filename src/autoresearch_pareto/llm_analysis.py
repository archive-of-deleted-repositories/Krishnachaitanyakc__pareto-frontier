"""LLM-powered tradeoff analysis using Anthropic Claude."""

from typing import Any, Dict, List

from autoresearch_pareto.metrics import MetricSet


def explain_tradeoffs(
    frontier_points: List[Dict[str, Any]],
    metric_set: MetricSet,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send frontier points to Claude for natural language tradeoff explanation."""
    import anthropic

    client = anthropic.Anthropic()

    # Build the prompt
    metric_descriptions = []
    for m in metric_set.metrics:
        desc = f"- {m.name}: {m.direction}"
        if m.unit:
            desc += f" ({m.unit})"
        metric_descriptions.append(desc)

    points_text = ""
    for i, p in enumerate(frontier_points):
        vals = ", ".join(f"{k}={v}" for k, v in p.items())
        points_text += f"  Point {i + 1}: {vals}\n"

    prompt = f"""Analyze the following Pareto frontier from a multi-objective optimization.

Metrics:
{chr(10).join(metric_descriptions)}

Pareto-optimal points:
{points_text}

Provide a concise analysis of:
1. The key tradeoffs between these solutions
2. Which solutions might be preferred in different scenarios
3. Any notable patterns or clusters in the frontier"""

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text

"""Console-only grouping for the upstream RSL-RL logger.

TensorBoard/W&B keys are deliberately untouched; this only makes the large
terminal summary easier to scan.
"""

from __future__ import annotations

import builtins
import io

from rsl_rl.utils.logger import Logger


class GroupedLogger(Logger):
    """Use the upstream logger and reorder its printed metric block."""

    def log(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        capture = io.StringIO()
        original_print = builtins.print

        def capture_print(*values, **options):  # type: ignore[no-untyped-def]
            options = dict(options)
            options["file"] = capture
            original_print(*values, **options)

        builtins.print = capture_print
        try:
            result = super().log(*args, **kwargs)
        finally:
            builtins.print = original_print

        block = capture.getvalue()
        if block:
            original_print(_group_metric_lines(block), end="")
        return result


def _group_metric_lines(block: str) -> str:
    lines = block.splitlines(keepends=True)
    groups: dict[str, list[str]] = {
        "training": [],
        "curriculum": [],
        "rewards": [],
        "terminations": [],
        "metrics": [],
        "other": [],
    }
    order = ("training", "curriculum", "rewards", "terminations", "metrics", "other")
    for line in lines:
        key = line.strip()
        if not key or key.startswith("#") or key.startswith("-"):
            continue
        if "Learning iteration" in key or key.startswith(("Iteration time:", "Time elapsed:", "ETA:")):
            continue
        if line.lstrip().startswith(("Run name:", "Total steps:", "Steps per second:", "Collection time:", "Learning time:")):
            groups["training"].append(line)
        elif "Curriculum/" in line:
            groups["curriculum"].append(line)
        elif "Episode_Reward/" in line or line.lstrip().startswith(("Mean reward:", "Mean episode length:")):
            groups["rewards"].append(line)
        elif "Episode_Termination/" in line:
            groups["terminations"].append(line)
        elif "Metrics/" in line or "Episode_Metrics/" in line:
            groups["metrics"].append(line)
        elif line.lstrip().startswith(("Mean value loss:", "Mean surrogate loss:", "Mean entropy loss:", "Mean action std:")):
            groups["training"].append(line)
        else:
            groups["other"].append(line)

    # The original formatter emits a fixed-width header and footer. Preserve
    # them, but place the metric groups between the header and footer.
    header = [line for line in lines if line.startswith("#") or "Learning iteration" in line]
    footer = [line for line in lines if line.startswith("-") or line.lstrip().startswith(("Iteration time:", "Time elapsed:", "ETA:"))]
    body = []
    labels = {
        "training": "Training",
        "curriculum": "Curriculum",
        "rewards": "Episode rewards",
        "terminations": "Terminations",
        "metrics": "Metrics",
        "other": "Other",
    }
    for group in order:
        entries = groups[group]
        if entries:
            body.append(f"\n[{labels[group]}]\n")
            body.extend(entries)
    return "".join(header + body + footer)

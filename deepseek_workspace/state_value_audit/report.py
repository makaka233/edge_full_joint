"""Generate compact state-value audit summaries and plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def report(results_dir):
    from matplotlib import pyplot as plt

    root = Path(results_dir)
    groups = []
    for path in sorted((root / "groups").glob("*.json")):
        groups.append(json.loads(path.read_text(encoding="utf8"))["group"])
    from .evaluate import evaluate

    payload = evaluate(groups)
    payload["groups"] = [
        {
            "id": group["id"],
            "seed": group["seed"],
            "stage": group["stage"],
            "profile": group["profile"],
            "states": len(group["states"]),
        }
        for group in groups
    ]
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf8",
    )

    stages = ("GS0", "GS16", "GS64", "SC64")
    profiles = ("natural", "sustained070")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    x = range(len(stages))
    width = .35
    for profile_index, profile in enumerate(profiles):
        values = [payload["q_ranking"].get(f"{stage}.{profile}") for stage in stages]
        for arm_index, name in enumerate(("prior_top1", "q_top1")):
            axes[0].bar(
                [i + (profile_index * 2 + arm_index - 1.5) * width / 2 for i in x],
                [value[name] if value else 0 for value in values],
                width=width / 2,
                label=f"{profile}:{name}",
            )
        for arm_index, name in enumerate(("prior_regret", "q_regret")):
            axes[1].bar(
                [i + (profile_index * 2 + arm_index - 1.5) * width / 2 for i in x],
                [value[name] if value else 0 for value in values],
                width=width / 2,
                label=f"{profile}:{name}",
            )
    offline = payload["offline"]
    for arm_index, representation in enumerate(("original", "extended")):
        axes[2].bar(
            np.arange(2) + (arm_index - .5) * width,
            [
                offline[representation]["top1_accuracy"]["mean"],
                offline[representation]["mean_regret"]["mean"],
            ],
            width=width,
            label=representation,
        )
    axes[0].set_xticks(list(x), stages)
    axes[0].set_title("Enumerated Top-1")
    axes[1].set_xticks(list(x), stages)
    axes[1].set_title("Mean Reward Regret")
    axes[2].set_xticks([0, 1], ["Top-1", "Regret"])
    axes[2].set_title("Offline Test")
    axes[0].legend(fontsize=6)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    fig.savefig(root / "state_value_audit.png", dpi=150)
    plt.close(fig)
    return payload

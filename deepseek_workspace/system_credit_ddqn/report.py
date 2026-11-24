"""Summaries and plots for the DeepSeek system-credit experiment."""

from __future__ import annotations

from pathlib import Path
import json

from .evaluate import compare


def records(path):
    with Path(path).open(encoding="utf8") as file:
        return [json.loads(line) for line in file if line.strip()]


def report(results_dir, output=None):
    from matplotlib import pyplot as plt

    root = Path(results_dir)
    files = sorted(root.glob("run.*.json"))
    if not files:
        raise FileNotFoundError("No run.*.json evaluation files")
    payload = {"runs": {}}
    for path in files:
        run = json.loads(path.read_text(encoding="utf8"))
        payload["runs"][path.stem] = {**run, "comparison": compare(run)}
    destination = Path(output) if output is not None else root / "summary.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for run, data in payload["runs"].items():
        for case, cell in data["cases"].items():
            rows = cell["system"]["rows"]
            axes[0].plot([r["period"] for r in rows], [r["reward_raw"] for r in rows], label=f"{run}:{case}")
            axes[1].plot([r["period"] for r in rows], [r["e2e_completed"] for r in rows], label=f"{run}:{case}")
    axes[0].set_title("Raw Reward")
    axes[1].set_title("Completed E2E (s)")
    for axis in axes:
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    fig.savefig(root / "system_credit.raw.png", dpi=150)
    plt.close(fig)

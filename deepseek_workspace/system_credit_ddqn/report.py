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
    files = sorted(root.glob("run.seed*.json"))
    if not files:
        raise FileNotFoundError("No run.*.json evaluation files")
    registration = root / "protocol.registered.json"
    payload = {
        "registration": (
            json.loads(registration.read_text(encoding="utf8"))
            if registration.is_file()
            else None
        ),
        "runs": {},
    }
    plotted = {}
    for path in files:
        run = json.loads(path.read_text(encoding="utf8"))
        comparison = compare(run)
        by_profile = {}
        for profile in ("natural", "burst", "sustained055", "sustained070"):
            selected = [value for key, value in comparison.items() if key.startswith(profile + ".")]
            if selected:
                by_profile[profile] = {
                    arm: {
                        name: sum(row[arm][name] for row in selected) / len(selected)
                        for name in ("reward", "e2e", "e2e_penalized", "p95", "p99", "spikes",
                                     "censored", "post_cutoff", "terminal", "dropped",
                                     "backlog", "cpu", "wire")
                    }
                    for arm in ("system", "local_period", "prior_only")
                }
        payload["runs"][path.stem] = {
            "seed": run["seed"],
            "cases": comparison,
            "by_profile": by_profile,
        }
        plotted[path.stem] = run
    destination = Path(output) if output is not None else root / "summary.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for run, data in plotted.items():
        for case, cell in data["cases"].items():
            rows = cell["cells"]["system"]["rows"]
            axes[0].plot([r["period"] for r in rows], [r["reward_raw"] for r in rows], label=f"{run}:{case}")
            axes[1].plot([r["period"] for r in rows], [r["e2e_completed"] for r in rows], label=f"{run}:{case}")
    axes[0].set_title("Raw Reward")
    axes[1].set_title("Completed E2E (s)")
    for axis in axes:
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    fig.savefig(root / "system_credit.raw.png", dpi=150)
    plt.close(fig)

    colors = {
        "system": "#b2182b",
        "local_period": "#2166ac",
        "local_episode": "#4d9221",
    }
    for seed in (20261113, 20261115, 20261117):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
        for arm in ("system", "local_period", "local_episode"):
            path = root / f"{arm}.seed{seed}" / "train.jsonl"
            if not path.is_file():
                continue
            rows = records(path)
            axes[0].plot(
                [row["gs"] for row in rows],
                [row["reward_raw"] for row in rows],
                label=arm,
                color=colors[arm],
                lw=0.9,
            )
            axes[1].plot(
                [row["gs"] for row in rows],
                [row["e2e_completed"] for row in rows],
                label=arm,
                color=colors[arm],
                lw=0.9,
            )
        axes[0].set_ylabel("Raw Reward")
        axes[1].set_ylabel("Completed E2E (s)")
        axes[1].set_xlabel("Global business periods (GS)")
        axes[0].legend()
        for axis in axes:
            axis.grid(alpha=.25)
        fig.suptitle(f"DeepSeek system credit · Seed {seed}")
        fig.savefig(root / f"training.seed{seed}.png", dpi=150)
        plt.close(fig)

    profiles = ("natural", "burst", "sustained055", "sustained070")
    arms = ("system", "local_period", "prior_only")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    width = .24
    xs = range(len(profiles))
    for arm_index, arm in enumerate(arms):
        selected = []
        for profile in profiles:
            values = []
            for data in payload["runs"].values():
                if profile in data["by_profile"]:
                    values.append(data["by_profile"][profile][arm])
            selected.append({
                name: sum(value[name] for value in values) / len(values)
                for name in ("reward", "e2e", "p99", "censored", "terminal")
            })
        for axis, metric in zip(axes, ("reward", "e2e", "p99"), strict=True):
            axis.bar(
                [x + (arm_index - 1) * width for x in xs],
                [value[metric] for value in selected],
                width=width,
                label=arm,
            )
    for axis, title in zip(axes, ("Mean raw Reward", "Mean E2E (s)", "Mean P99 (s)"), strict=True):
        axis.set_xticks(list(xs), profiles, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(root / "holdout.by_profile.png", dpi=150)
    plt.close(fig)

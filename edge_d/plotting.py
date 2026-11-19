"""Unsmoothed raw per-physical-trajectory plots. No deleted bad samples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _read(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]

def plots(run_dir:Path):
    run_dir=Path(run_dir);figs=run_dir/"figures";figs.mkdir(exist_ok=True,parents=True)
    train=_read(run_dir/"train.jsonl")
    evals=_read(run_dir/"eval_summary.jsonl")
    if not evals:evals=_read(run_dir/"eval_periodic_summary.jsonl")
    paths=[]
    if train:
        for metric,name,label in (("reward","raw_reward","Physical Reward"),
                                  ("penalized_e2e","raw_e2e","Penalty-adjusted E2E (s)"),
                                  ("decisions","raw_actions","Construction actions"),
                                  ("censored","raw_censored","Window-censored requests")):
            fig,ax=plt.subplots(figsize=(10,4.2))
            ax.plot(range(1,len(train)+1),[r[metric] for r in train],marker=".",ms=3,lw=.9)
            ax.set(xlabel="Real training physical trajectory (unfiltered)",ylabel=label,
                   title=f"Raw {label}: no smoothing/clipping/deletion")
            ax.grid(alpha=.2)
            fig.tight_layout();path=figs/f"{name}.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    fixed=[x for x in evals if x["split"]=="fixed"]
    if fixed:
        fig,ax=plt.subplots(figsize=(8,4.))
        ax.plot([v.get("update",v.get("step")) for v in fixed],[v["reward"] for v in fixed],marker="o")
        ax.set(xlabel="Training update index (not necessarily macro GS)",ylabel="Mean fixed Reward",
               title="Fixed-split evaluation (not holdout)")
        ax.grid(alpha=.2)
        fig.tight_layout();path=figs/"fixed_reward.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    updates=_read(run_dir/"updates.jsonl")
    if updates:
        for field,name,ylabel in (("s_loss_mean","s_loss","S DDQN mean Huber loss"),
                                   ("s_replay_size","s_replay","S replay size"),
                                   ("s_transitions_collected","s_samples","S transitions per update")):
            if any(u.get(field) is not None for u in updates):
                fig,ax=plt.subplots(figsize=(9,4))
                ax.plot([u.get("step",u.get("update")) for u in updates],
                        [u.get(field,float("nan")) for u in updates],marker=".",lw=.9)
                ax.set(xlabel="Training update",ylabel=ylabel,title=ylabel)
                ax.grid(alpha=.2);fig.tight_layout()
                path=figs/f"{name}.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    return paths

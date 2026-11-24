"""Single entry for the DeepSeek system-credit DDQN prototype."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
from itertools import product
import json
from pathlib import Path

import torch

from comparison.monolithic_service.environment import world
from comparison.profile_audit import profiles_from_catalog
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source, train as train_local
from edge_joint.adaptive_storage import RunFiles, atomic, line, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp037_run import cfg_for as exp037_cfg
from .evaluate import evaluate_seed
from .report import report
from .training import train


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS = (20261113, 20261115, 20261117)
PROFILES = {
    "natural": (),
    "burst": (None,) * 8 + (0.70,) * 4 + (None,) * 12,
    "sustained055": (0.55,) * 24,
    "sustained070": (0.70,) * 24,
}


def _parent_plan():
    path = Path("runs/results/EXP037/protocol.json")
    return json.loads(path.read_text(encoding="utf8"))["plan"]


def _cfg(seed, arm):
    parent = _parent_plan()
    cfg = exp037_cfg(parent, seed, "resource")
    return replace(cfg, s_update_timing="episode" if arm in ("system", "local_episode") else "period")


def _workload():
    parent = _parent_plan()
    catalog = _profile_catalog(parent)
    return profiles_from_catalog(catalog)[parent["historical"]["training_profile"]]


def _profile_catalog(parent):
    return parent["historical"]["workload_catalog"]


def _world_factory():
    profile = _workload()
    return partial(world, profile=profile)


def protocol():
    parent = _parent_plan()
    cases = [
        dict(
            profile=name,
            environment=stable_seed("DEEPSEEK_SYSTEM_CREDIT", "population", name, index),
            scene=stable_seed("DEEPSEEK_SYSTEM_CREDIT", "scene", name, index),
            loads=list(loads),
        )
        for name, loads in PROFILES.items()
        for index in range(2)
    ]
    return dict(
        experiment="DEEPSEEK_SYSTEM_CREDIT_DDQN",
        owner="DeepSeek",
        parent=fingerprint(parent),
        seeds=list(SEEDS),
        screening_episodes=2,
        maximum_episodes=8,
        credit="cohort residence + deadline penalty, global decision successor",
        gamma_per_second=0.99,
        cases=cases,
    )


def register():
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "protocol.registered.json"
    plan = protocol()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))
        if saved["protocol"] != plan:
            raise ValueError("DeepSeek system-credit protocol changed")
        return saved
    record = dict(protocol=plan, source=source())
    atomic(path, lambda file: file.write(line(record)))
    return record


def train_one(arm, seed, target, resume=False):
    torch.set_num_threads(1)
    cfg = _cfg(seed, arm)
    out = RESULTS / f"{arm}.seed{seed}"
    environment_id = fingerprint(protocol())
    if arm == "system":
        with run_lock(RunFiles(out, nested=True)):
            return train(
                cfg,
                out,
                target,
                world_factory=_world_factory(),
                environment_id=environment_id,
                resume=resume,
            )
    if arm not in ("local_period", "local_episode"):
        raise ValueError(f"Unknown arm: {arm}")
    with run_lock(RunFiles(out, nested=True)):
        train_local(
            cfg,
            RunFiles(out, nested=True),
            target,
            resume=resume,
            world_factory=_world_factory(),
            environment_id=environment_id,
        )
    return dict(out=str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train", "evaluate", "report", "all"), default="all")
    parser.add_argument("--target-episodes", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("system", "local_period", "local_episode"),
                        default=("system", "local_period", "local_episode"))
    args = parser.parse_args()
    if args.target_episodes not in (1, 2, 4, 8):
        parser.error("target episodes must be one of 1, 2, 4, or 8")
    registered = register()
    if args.stage in ("train", "all"):
        for arm, seed in product(args.arms, SEEDS):
            train_one(arm, seed, args.target_episodes, args.resume)
    if args.stage in ("evaluate", "all"):
        if args.target_episodes != 8:
            raise SystemExit("Independent evaluation requires the registered GS64 checkpoint")
        cases = registered["protocol"]["cases"]
        for seed in SEEDS:
            evaluate_seed(
                seed=seed,
                cfg_by_arm={arm: _cfg(seed, arm) for arm in args.arms},
                checkpoints={
                    arm: RESULTS / f"{arm}.seed{seed}" / "last.pt"
                    for arm in args.arms
                },
                cases=cases,
                out=RESULTS / f"run.seed{seed}.json",
                world_factory=_world_factory(),
            )
    if args.stage in ("report", "all"):
        report(RESULTS)
    return registered


if __name__ == "__main__":
    main()

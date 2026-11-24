"""Collect real states and frozen multi-action physical counterfactuals."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from comparison.run import MeasuredMultistageSimulator, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp027_run import verify_periods
from scripts.exp029_run import digest
from scripts.exp037_run import cfg_for, workload
from scripts.exp038_run import outputs as exp038_outputs
from scripts.exp037_run import run_files as exp037_files
from deepseek_workspace.state_value_audit.counterfactual import (
    DecisionSimulator,
    FullGreedyPolicy,
    action_set,
    fork as _fork,
    run_branch,
    selected_decisions,
)
from deepseek_workspace.state_value_audit.features import (
    action_features,
    extended_state_features,
    state_signature,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GROUPS = RESULTS / "groups"
SEEDS = (20261113, 20261115, 20261117)
CHECKPOINTS = ("GS0", "GS16", "GS64", "SC64")
PROFILES = ("natural", "sustained070")
PERIODS = 7
TARGET_PERIODS = (3, 4, 5, 6)


def parent_plan():
    return json.loads(Path("runs/results/EXP037/protocol.json").read_text(encoding="utf8"))["plan"]


def checkpoint(seed, stage):
    if stage == "GS0":
        return None
    if stage == "GS16":
        return exp037_files(seed, "resource") / "resume_latest.pt"
    if stage == "GS64":
        return exp038_outputs(seed) / "resume_latest.pt"
    if stage == "SC64":
        return (
            ROOT.parent
            / "system_credit_ddqn"
            / "results"
            / f"system.seed{seed}"
            / "resume_latest.pt"
        )
    raise ValueError(stage)


def load_actor(seed, stage):
    parent = parent_plan()
    cfg = cfg_for(parent, seed, "resource")
    d, s, _, _, gate, _ = components(cfg)
    index = 0
    saved = checkpoint(seed, stage)
    if saved is not None:
        state = torch.load(saved, weights_only=True, map_location="cpu", mmap=True)
        expected = 2 if stage == "GS16" else 8
        if state["episode"] != expected:
            raise ValueError(f"{stage} checkpoint has episode {state['episode']}")
        s.load_state_dict(state["s"])
        load_model(state, d, s, gate)
        index = expected
    return cfg, index, d, s, gate


def loads(profile):
    return () if profile == "natural" else (0.70,) * PERIODS


def copy_state(world_value, sim):
    saved = sim.agent, sim.policy, sim.decision_log
    sim.agent = sim.policy = sim.decision_log = None
    try:
        copied_env, copied_sim = _fork(world_value, sim)
    finally:
        sim.agent, sim.policy, sim.decision_log = saved
    copied_sim.agent = copied_sim.policy = copied_sim.decision_log = None
    return copied_env, copied_sim


def group_id(seed, stage, profile):
    return f"seed{seed}.{stage}.{profile}"


def protocol():
    parent = parent_plan()
    records = {}
    for seed in SEEDS:
        for stage in CHECKPOINTS:
            path = checkpoint(seed, stage)
            records[group_id(seed, stage, "checkpoint")] = (
                {"initial": "same-seed random initialization"} if path is None
                else {"path": str(path), "sha256": digest(path)}
            )
    module_hash = hashlib.sha256()
    for path in sorted(ROOT.rglob("*.py")):
        module_hash.update(path.name.encode("utf8"))
        module_hash.update(path.read_bytes())
    return {
        "experiment": "DEEPSEEK_STATE_VALUE_AUDIT",
        "owner": "DeepSeek",
        "source": source(),
        "module_sha256": module_hash.hexdigest(),
        "parent": digest(Path("runs/results/EXP037/protocol.json")),
        "seeds": list(SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "profiles": list(PROFILES),
        "periods": PERIODS,
        "target_periods": list(TARGET_PERIODS),
        "branch_periods": 2,
        "max_actions_per_state": 3,
        "checkpoint_sources": records,
        "decision_rule": (
            "first residual-over-prior flip at least 4 seconds into the target period; "
            "otherwise first multi-legal decision"
        ),
        "action_rule": (
            "prior argmax, full-Q argmax, residual argmax, then full-Q argmin; "
            "take the first three distinct legal actions"
        ),
        "split": {
            "train_seeds": [20261113, 20261115],
            "test_seed": 20261117,
            "unit": "complete source trajectory; all states and actions remain together",
        },
    }


def register():
    RESULTS.mkdir(parents=True, exist_ok=True)
    plan = protocol()
    path = RESULTS / "protocol.registered.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))
        if saved["protocol"] != plan:
            raise ValueError("DeepSeek state-value audit protocol changed")
        return saved
    record = {"protocol": plan, "source": source()}
    atomic(path, lambda file: file.write(line(record)))
    return record


def run_group(seed, stage, profile):
    started = time.perf_counter()
    torch.set_num_threads(1)
    path = GROUPS / f"{group_id(seed, stage, profile)}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))
        if saved["protocol"] == digest(RESULTS / "protocol.registered.json"):
            return saved["group"]

    parent = parent_plan()
    cfg, index, d, s, gate = load_actor(seed, stage)
    env_seed = stable_seed("DEEPSEEK_STATE_VALUE", "environment", seed, profile)
    scene = stable_seed("DEEPSEEK_STATE_VALUE", "scene", seed, profile)
    load_profile = loads(profile)
    decision_log = []
    captured = {}
    holder = {}

    def simulator_factory(*args, **kwargs):
        sim = DecisionSimulator(
            *args,
            agent=s,
            policy=FullGreedyPolicy(s),
            decision_log=decision_log,
            **kwargs,
        )
        holder["sim"] = sim
        return sim

    def boundary(world_value, sim, period, phase):
        if sim is not None and phase == "start" and period in TARGET_PERIODS:
            captured[period] = copy_state(world_value, sim)

    env = world(env_seed, scene, "holdout", profile=workload(parent))
    rows = episode(
        replace(cfg, periods=PERIODS, freeze_ds=True),
        env,
        d,
        s,
        gate,
        scene,
        "holdout",
        index,
        control=RolloutControl(
            environment_seed=env_seed,
            loads=load_profile,
            epsilon=0.0,
            sample_d=False,
            trace_requests=True,
        ),
        simulator_factory=simulator_factory,
        boundary=boundary,
    )[0]
    verify_periods(env, rows)
    if set(captured) != set(TARGET_PERIODS):
        raise AssertionError("Did not capture every registered state boundary")

    states = []
    for period in TARGET_PERIODS:
        decision = selected_decisions(decision_log, period, (period - 1) * cfg.period_seconds)
        if decision is None:
            continue
        actions = action_set(decision)
        branches = []
        for action in actions:
            branches.append(run_branch(
                state=captured[period],
                source_rows=rows,
                cfg=cfg,
                d=d,
                agent=s,
                gate=gate,
                scene=scene,
                env_seed=env_seed,
                loads=load_profile,
                index=index,
                period=period,
                target_action=action,
                decision=decision,
                tail_seconds=env.tail,
            ))
        full_branch = next(item for item in branches if item["action"] == decision["full_node"])
        full_other = full_branch["other_e2e"]
        for branch in branches:
            common = set(full_other) & set(branch["other_e2e"])
            branch["other_e2e_delta_sum"] = float(sum(
                branch["other_e2e"][key] - full_other[key] for key in common
            ))
            branch["focus_e2e_delta"] = float(
                branch["focus_e2e"] - full_branch["focus_e2e"]
            )
            branch["features_a"] = action_features(
                np.asarray(decision["obs"], dtype=np.float32),
                np.asarray(decision["mask"], dtype=bool),
                branch["action"],
                np.asarray(decision["extended"], dtype=np.float32),
                include_extension=False,
            ).tolist()
            branch["features_b"] = action_features(
                np.asarray(decision["obs"], dtype=np.float32),
                np.asarray(decision["mask"], dtype=bool),
                branch["action"],
                np.asarray(decision["extended"], dtype=np.float32),
                include_extension=True,
            ).tolist()
            score_index = decision["legal"].index(branch["action"])
            branch["prior"] = float(decision["prior"][branch["action"]])
            branch["residual"] = float(decision["residual"][branch["action"]])
            branch["q"] = float(decision["q"][branch["action"]])
        best = max(branches, key=lambda item: item["reward_sum"])
        prior_node = decision["prior_node"]
        full_node = decision["full_node"]
        states.append({
            "group": group_id(seed, stage, profile),
            "seed": seed,
            "stage": stage,
            "profile": profile,
            "period": period,
            "environment": env_seed,
            "scene": scene,
            "decision": decision,
            "state_signature": state_signature(
                np.asarray(decision["obs"], dtype=np.float32),
                np.asarray(decision["extended"], dtype=np.float32),
            ).tolist(),
            "actions": branches,
            "best_action": best["action"],
            "best_reward": best["reward_sum"],
            "prior_regret": float(best["reward_sum"] - next(
                item["reward_sum"] for item in branches if item["action"] == prior_node
            )),
            "full_regret": float(full_branch["reward_sum"] - best["reward_sum"]),
        })
    group = {
        "id": group_id(seed, stage, profile),
        "seed": seed,
        "stage": stage,
        "profile": profile,
        "environment": env_seed,
        "scene": scene,
        "states": states,
        "source_rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic(path, lambda file: file.write(line({
        "protocol": digest(RESULTS / "protocol.registered.json"),
        "source": source(),
        "group": group,
    })))
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS)
    parser.add_argument("--stage", action="append", choices=CHECKPOINTS)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    args = parser.parse_args()
    register()
    GROUPS.mkdir(parents=True, exist_ok=True)
    work = [
        (seed, stage, profile)
        for seed in (args.seed or SEEDS)
        for stage in (args.stage or CHECKPOINTS)
        for profile in (args.profile or PROFILES)
    ]
    if args.workers == 1:
        for item in work:
            print(run_group(*item)["id"], flush=True)
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_group, *item) for item in work]
                for future in as_completed(futures):
                    print(future.result()["id"], flush=True)
        except PermissionError:
            print("Windows process pipes unavailable; running the same audit serially", flush=True)
            for item in work:
                print(run_group(*item)["id"], flush=True)


if __name__ == "__main__":
    main()

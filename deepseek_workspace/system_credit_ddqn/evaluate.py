"""Frozen evaluation for local, system-credit and prior-only schedulers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator
from edge_joint.adaptive_storage import atomic, line
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts.exp029_report import summarize
from .scheduling_agent import SystemCreditScheduler


def _flow(rows):
    return [
        (row["start"], row["end"], row["requests"], row["trace_sha256"])
        for row in rows
    ]


def load_actor(checkpoint, cfg, arm):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if state["config"] != cfg.__dict__:
        raise ValueError(f"Checkpoint config differs: {checkpoint}")
    d, baseline_s, _, _, gate, _ = components(cfg)
    s = SystemCreditScheduler(cfg.seed) if arm == "system" else baseline_s
    load_model(state, d, s, gate)
    return d, s, gate, state


@torch.inference_mode()
def rollout(cfg, d, s, gate, case, world_factory, index):
    env = world_factory(case["environment"], case["scene"], "holdout")
    control = RolloutControl(
        environment_seed=case["environment"],
        loads=tuple(case["loads"]),
        epsilon=0.0,
        sample_d=False,
        trace_requests=True,
    )
    rows, _, _ = episode(
        replace(cfg, periods=24, freeze_ds=True),
        env,
        d,
        s,
        gate,
        case["scene"],
        "holdout",
        index,
        control=control,
        simulator_factory=MeasuredMultistageSimulator,
    )
    if any(row["epsilon"] != 0 for row in rows):
        raise AssertionError("Frozen evaluation used exploration")
    return rows


def evaluate_seed(
    *,
    seed,
    cfg_by_arm,
    checkpoints,
    cases,
    out,
    world_factory,
):
    """Evaluate all available arms with paired exogenous flows."""

    loaded = {
        arm: load_actor(path, cfg_by_arm[arm], arm)
        for arm, path in checkpoints.items()
        if Path(path).is_file()
    }
    if "system" not in loaded:
        raise FileNotFoundError("System-credit checkpoint is required")
    result = {"seed": seed, "cases": {}}
    for case in cases:
        cells = {}
        reference_flow = None
        for arm, (d, s, gate, state) in loaded.items():
            rows = rollout(
                cfg_by_arm[arm],
                d,
                s,
                gate,
                case,
                world_factory,
                int(state["episode"]),
            )
            flow = _flow(rows)
            if reference_flow is None:
                reference_flow = flow
            elif flow != reference_flow:
                raise AssertionError(f"Paired exogenous flow differs for {arm}")
            cells[arm] = dict(rows=rows, metric=summarize(rows), episode=int(state["episode"]))

        d, s, gate, state = loaded["system"]
        with torch.no_grad():
            saved = {key: value.detach().clone() for key, value in s.online.state_dict().items()}
            for parameter in s.online.parameters():
                parameter.zero_()
            prior_rows = rollout(
                cfg_by_arm["system"],
                d,
                s,
                gate,
                case,
                world_factory,
                int(state["episode"]),
            )
            s.online.load_state_dict(saved)
        if _flow(prior_rows) != reference_flow:
            raise AssertionError("Prior-only ablation received different exogenous flow")
        if any(a["active_matrix"] != b["active_matrix"] for a, b in zip(
                cells["system"]["rows"], prior_rows, strict=True)):
            raise AssertionError("Prior-only ablation changed D deployment")
        cells["prior_only"] = dict(
            rows=prior_rows,
            metric=summarize(prior_rows),
            episode=int(state["episode"]),
        )
        result["cases"][f"{case['profile']}.{case['scene']}"] = dict(case=case, cells=cells)
    atomic(out, lambda file: file.write(line(result)))
    return result


def compare(run):
    """Derive policy-level residuals without comparing internal TD losses."""

    output = {}
    for key, case in run["cases"].items():
        cells = case["cells"]
        system = cells["system"]["metric"]
        prior = cells["prior_only"]["metric"]
        value = {
            "system": system,
            "prior_only": prior,
            "residual": {
                name: system[name] - prior[name]
                for name in ("reward", "e2e", "e2e_penalized", "p95", "p99",
                             "censored", "terminal", "dropped")
            },
        }
        for arm in ("local_period", "local_episode"):
            if arm in cells:
                local = cells[arm]["metric"]
                value[arm] = local
                value[f"system_minus_{arm}"] = {
                    name: system[name] - local[name]
                    for name in ("reward", "e2e", "e2e_penalized", "p95", "p99",
                                 "censored", "terminal", "dropped")
                }
        output[key] = value
    return output

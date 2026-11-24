"""Bounded training for the system-credit DDQN candidate."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import FIXED, source
from edge_joint.adaptive_physics import PeriodSimulator
from edge_joint.adaptive_storage import append, atomic, line, publish, random_state, restore_random
from edge_joint.adaptive_training import components, episode, model, update
from edge_joint.inventory_joint import parameters, recover_ledgers
from scripts.exp027_run import fingerprint
from .credit_ledger import replay_snapshot, restore_replay
from .scheduling_agent import SystemCreditScheduler
from .simulator_adapter import SystemCreditSimulator, cohort_windows


FORMAT = "deepseek-system-credit-ddqn-v1"


def _config_dict(cfg):
    return asdict(cfg)


def initial_components(cfg):
    """Keep D/gate initialization exactly aligned with the resource-prior baseline."""

    d, _, opt, replay, gate, gate_opt = components(cfg)
    s = SystemCreditScheduler(cfg.seed)
    return d, s, opt, replay, gate, gate_opt


def fixed(cfg, d, s, gate, episode_index, *, world_factory):
    rows = []
    actions = s.actions
    with torch.inference_mode():
        for scene in FIXED:
            env = world_factory(cfg.seed, scene, "fixed")
            simulator = MeasuredMultistageSimulator if hasattr(env, "reference_work") else PeriodSimulator
            rows.extend(episode(
                replace(cfg, freeze_ds=True),
                env,
                d,
                s,
                gate,
                scene,
                "fixed",
                episode_index,
                simulator_factory=simulator,
            )[0])
    s.actions = actions
    return rows


def _collect_episode(cfg, d, s, gate, index, *, world_factory):
    scene = stable_seed("EXP019", cfg.seed, "train-scene", index)
    env = world_factory(cfg.seed, scene, "train")
    windows = cohort_windows(
        start=0.0,
        periods=cfg.periods,
        period_seconds=cfg.period_seconds,
        tail_seconds=env.tail,
    )
    holder = {}

    def simulator_factory(*args, **kwargs):
        sim = SystemCreditSimulator(
            *args,
            cohort_windows=windows,
            credit_enabled=True,
            **kwargs,
        )
        holder["sim"] = sim
        return sim

    d_audit = {} if cfg.d_diagnostics else None
    rows, credit, transitions = episode(
        cfg,
        env,
        d,
        s,
        gate,
        scene,
        "train",
        index,
        period_updates=None,
        simulator_factory=simulator_factory,
        d_audit=d_audit,
    )
    sim = holder.get("sim")
    if sim is None or sim.credit_audit is None:
        raise AssertionError("System-credit simulator did not publish its ledger")
    return rows, credit, transitions, sim.credit_audit, d_audit, sim


def _system_s_update(cfg, s, replay, transitions, index):
    if not transitions:
        raise ValueError("System-credit episode produced no global transitions")
    replay.add(transitions)
    replay.prune(index * cfg.periods)
    total_updates = cfg.s_updates_per_gs * cfg.periods
    if total_updates <= 0 or cfg.s_updates_per_gs <= 0:
        raise ValueError("System-credit DDQN requires a positive update budget")
    rng = np.random.default_rng(stable_seed("DEEPSEEK", cfg.seed, index, "system-replay"))
    metrics = []
    for update_index in range(total_updates):
        batch = replay.sample(cfg.s_batch, index * cfg.periods, rng)
        metrics.append(s.update_event_time(batch))
        if (update_index + 1) % cfg.s_updates_per_gs == 0:
            s.version += 1
    return dict(
        s_updates_this_episode=total_updates,
        s_loss=float(np.mean([row["s_loss"] for row in metrics])),
        s_grad_norm=float(np.mean([row["s_grad_norm"] for row in metrics])),
        mean_elapsed_seconds=float(np.mean([row["mean_elapsed_seconds"] for row in metrics])),
        min_discount=float(min(row["min_discount"] for row in metrics)),
        max_discount=float(max(row["max_discount"] for row in metrics)),
        mean_target=float(np.mean([row["mean_target"] for row in metrics])),
    )


def _model(d, s, gate):
    return dict(d=d.state_dict(), s_online=s.online.state_dict(), gate=gate.state_dict())


def _publish_state(path, state):
    publish(path, state)


def train(
    cfg,
    out,
    target,
    *,
    world_factory,
    environment_id=None,
    resume=False,
):
    """Train three-seed-compatible system-credit checkpoints."""

    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d, s, opt, replay, gate, gate_opt = initial_components(cfg)
    d_credit_offset = torch.zeros(cfg.periods) if cfg.d_advantage_calibration else None
    paths = ("train.jsonl", "updates.jsonl", "fixed.jsonl", "decisions.jsonl")
    resume_path = out / "resume_latest.pt"
    index = d_updates = 0
    offsets = dict.fromkeys(paths, 0)
    wall = 0.0
    best = None
    history = []
    if resume:
        saved = torch.load(resume_path, weights_only=True, map_location="cpu")
        if (
            saved.get("format") != FORMAT
            or saved["config"] != _config_dict(cfg)
            or saved.get("environment_id") != environment_id
        ):
            raise ValueError("System-credit resume configuration or environment differs")
        if target < saved["episode"]:
            raise ValueError("System-credit target precedes the current checkpoint")
        index, d_updates = saved["episode"], saved["d_updates"]
        offsets, wall = saved["offsets"], saved["wall_seconds"]
        best, history = saved["best"], saved["source_history"]
        recover_ledgers(out, dict(offsets=offsets, gs=index * cfg.periods))
        d.load_state_dict(saved["d"])
        s.load_state_dict(saved["s"])
        s.actions = saved["s_actions"]
        opt.load_state_dict(saved["d_optimizer"])
        gate.load_state_dict(saved["gate"])
        gate_opt.load_state_dict(saved["gate_optimizer"])
        restore_replay(replay, saved["replay"])
        restore_random(saved["rng"])
        if d_credit_offset is not None:
            d_credit_offset.copy_(torch.as_tensor(saved["d_credit_offset"]))
        if target == index and best is not None:
            return dict(episode=index, gs=index * cfg.periods, d_updates=d_updates, s_updates=s.updates)
    elif any((out / name).exists() for name in ("resume_latest.pt", *paths)):
        raise ValueError("Existing system-credit evidence requires --resume")

    history = [*history, dict(start_episode=index, target_episode=target, **source())]
    start = time.perf_counter()

    def snapshot(evaluate):
        nonlocal best
        current = dict(
            format=FORMAT,
            config=_config_dict(cfg),
            episode=index,
            origin={"kind": "random"},
            **_model(d, s, gate),
        )
        if environment_id is not None:
            current["environment_id"] = environment_id
        if evaluate:
            rows = fixed(cfg, d, s, gate, index, world_factory=world_factory)
            offsets["fixed.jsonl"] = append(out / "fixed.jsonl", rows)
            score = float(np.mean([row["reward_net"] for row in rows]))
            if best is None or score > best["fixed_reward_net"]:
                best = dict(current, fixed_reward_net=score)
        state = dict(
            current,
            s=s.state_dict(),
            s_actions=s.actions,
            d_optimizer=opt.state_dict(),
            gate_optimizer=gate_opt.state_dict(),
            replay=replay_snapshot(replay),
            rng=random_state(),
            d_updates=d_updates,
            best=best,
            offsets=offsets,
            source_history=history,
            wall_seconds=wall + time.perf_counter() - start,
        )
        if d_credit_offset is not None:
            state["d_credit_offset"] = d_credit_offset.tolist()
        _publish_state(resume_path, state)
        _publish_state(out / "last.pt", current)
        if best is not None:
            _publish_state(out / "best.pt", best)
        return state

    if resume:
        snapshot(False)
    if best is None:
        snapshot(True)
    metadata = dict(
        format=FORMAT,
        config=_config_dict(cfg),
        origin={"kind": "random"},
        source_history=history,
        fixed_scenes=FIXED,
        credit="system-cohort-residence",
        transition="global-decision",
    )
    if environment_id is not None:
        metadata["environment_id"] = environment_id
    atomic(out / "manifest.json", lambda file: file.write(line(metadata)))

    for index in range(index + 1, target + 1):
        episode_start = time.perf_counter()
        rows, credit, transitions, audit, d_audit, sim = _collect_episode(
            cfg, d, s, gate, index, world_factory=world_factory
        )
        s_before = parameters(s.online)
        target_before = parameters(s.target)
        change = update(
            cfg,
            d,
            s,
            opt,
            replay,
            gate,
            gate_opt,
            rows,
            credit,
            [],
            index,
            d_updates,
            period_updates=None,
            d_audit=d_audit,
            d_credit_offset=d_credit_offset,
        )
        d_updates = change["d_updates"]
        system_tick = time.perf_counter()
        system_metrics = _system_s_update(cfg, s, replay, transitions, index)
        system_seconds = time.perf_counter() - system_tick
        change.update(
            **system_metrics,
            s_updates=s.updates,
            s_policy_version=s.version,
            s_change=float((parameters(s.online) - s_before).norm()),
            s_target_change=float((parameters(s.target) - target_before).norm()),
            admitted_to_replay=len(transitions),
            replay_size=len(replay),
            credit_identity_error=audit.identity_error,
            credit_total_punished_delay=audit.total_punished_delay,
            credit_total_normalized_delay=audit.total_normalized_delay,
            credit_pre_decision_normalized_delay=audit.pre_decision_normalized_delay,
            credit_transition_normalized_delay=audit.transition_normalized_delay,
            episode_collect_seconds=time.perf_counter() - episode_start,
            ddqn_update_wall_seconds=system_seconds,
            credit_build_seconds=getattr(sim, "credit_build_seconds", None),
            events=sum(row["events"] + row.get("drain_decisions", 0) for row in rows),
            s_decisions=sim.decisions,
        )
        offsets["train.jsonl"] = append(out / "train.jsonl", rows)
        offsets["updates.jsonl"] = append(out / "updates.jsonl", [change])
        offsets["decisions.jsonl"] = append(
            out / "decisions.jsonl",
            [
                dict(
                    episode=index,
                    decision_index=item.decision_index,
                    time=item.decision_time,
                    elapsed_seconds=item.elapsed_seconds,
                    reward=item.reward,
                    action=item.action,
                    request_id=item.request_id,
                    stage_index=item.stage_index,
                    collection_step=item.collection_step,
                    version=item.version,
                    terminated=item.terminated,
                )
                for item in transitions
            ],
        )
        snapshot(index % cfg.eval_every == 0 or index == target)
        print(
            f"DeepSeek system-credit seed={cfg.seed} episode={index}/{target} "
            f"GS={index * cfg.periods} D={d_updates} S={s.updates} "
            f"reward={np.mean([row['reward_net'] for row in rows]):.6f}",
            flush=True,
        )
    return dict(
        episode=target,
        gs=target * cfg.periods,
        d_updates=d_updates,
        s_updates=s.updates,
        best_gs=None if best is None else best["episode"] * cfg.periods,
    )

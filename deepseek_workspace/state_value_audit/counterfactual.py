"""Real-state decision capture and frozen single-action counterfactuals."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import torch

from edge_d.world import stable_seed
from edge_joint.adaptive_training import RolloutControl, episode
from scripts.exp019_report import unpack
from scripts.exp027_cases import fork
from scripts.exp027_run import ObservedSimulator, WorkLedger, fingerprint, verify_periods
from .features import extended_state_features, q_scores


class FullGreedyPolicy:
    def __init__(self, agent, forced=None):
        self.agent = agent
        self.forced = forced
        self.forced_seen = False

    @torch.no_grad()
    def __call__(self, sim, request, obs, mask):
        legal = np.flatnonzero(mask).tolist()
        prior, residual, total = q_scores(self.agent, obs)
        full_node = legal[int(np.argmax(total[legal]))]
        prior_node = legal[int(np.argmax(prior[legal]))]
        residual_node = legal[int(np.argmax(residual[legal]))]
        selected = full_node
        if self.forced is not None:
            key = (int(request.id), int(request.index))
            forced = self.forced
            if key == tuple(forced["key"]):
                if (
                    not np.isclose(sim.now, forced["time"], atol=1e-9, rtol=0)
                    or not np.array_equal(obs, np.asarray(forced["obs"], dtype=np.float32))
                    or not mask[int(forced["action"])]
                ):
                    raise AssertionError("Counterfactual did not reach the exact decision state")
                selected = int(forced["action"])
                self.forced_seen = True
        return selected


class DecisionSimulator(ObservedSimulator):
    """Observe full event state while preserving the original physics."""

    def __init__(self, env, *args, agent, policy, decision_log, **kwargs):
        self.agent = agent
        self.policy = policy
        self.decision_log = decision_log
        super().__init__(env, *args, **kwargs)
        self.cpu_reference_work = env.reference_work

    def ready(self, request, previous=None):
        if self.policy is None:
            return super().ready(request, previous)
        original = self.scheduler

        def select(obs, mask):
            action = int(self.policy(self, request, obs, mask))
            if not bool(mask[action]):
                raise AssertionError("Audit policy selected an illegal node")
            prior, residual, total = q_scores(self.agent, obs)
            self.decision_log.append({
                "period": int(self.window_index + 1),
                "time": float(self.now),
                "key": [int(request.id), int(request.index)],
                "request_id": int(request.id),
                "stage_index": int(request.index),
                "service": int(request.arrival.service),
                "stage": int(self.c.services[request.arrival.service][request.index]),
                "source": int(request.source),
                "obs": np.asarray(obs, dtype=np.float32).tolist(),
                "mask": np.asarray(mask, dtype=bool).tolist(),
                "legal": np.flatnonzero(mask).tolist(),
                "prior": prior.tolist(),
                "residual": residual.tolist(),
                "q": total.tolist(),
                "prior_node": int(np.flatnonzero(mask)[np.argmax(prior[mask])]),
                "full_node": int(np.flatnonzero(mask)[np.argmax(total[mask])]),
                "residual_node": int(np.flatnonzero(mask)[np.argmax(residual[mask])]),
                "action": action,
                "extended": extended_state_features(self, request).tolist(),
            })
            return action

        self.scheduler = select
        try:
            return super().ready(request, previous)
        finally:
            self.scheduler = original

    def transmit(self, request, route, hop):
        super().transmit(request, route, hop)
        if hop == len(route):
            resource = self.cpu[request.destination]
            job = resource.jobs.get(self.job_counter)
            if job is not None:
                job.stage_id = self.c.services[request.arrival.service][request.index]


def selected_decisions(decisions, period, start_time):
    choices = [
        item for item in decisions
        if item["period"] == period and len(item["legal"]) >= 2
    ]
    if not choices:
        return None
    late = [item for item in choices if item["time"] >= start_time + 4.0]
    pool = late or choices
    flips = [item for item in pool if item["prior_node"] != item["full_node"]]
    return (flips or pool)[0]


def action_set(decision):
    legal = decision["legal"]
    preferred = [
        decision["prior_node"],
        decision["full_node"],
        decision["residual_node"],
    ]
    full = np.asarray(decision["q"], dtype=float)[legal]
    worst = legal[int(np.argmin(full))]
    preferred.append(worst)
    ordered = list(dict.fromkeys(node for node in preferred if node in legal))
    target = min(3, len(legal))
    for node in legal:
        if len(ordered) >= target:
            break
        if node not in ordered:
            ordered.append(node)
    return ordered[:target]


def _request_e2e(request, deadline):
    if request.finish is not None and request.finish <= deadline:
        return float(request.finish - request.arrival.time)
    return float(deadline - request.arrival.time + 10.0)


def run_branch(
    *,
    state,
    source_rows,
    cfg,
    d,
    agent,
    gate,
    scene,
    env_seed,
    loads,
    index,
    period,
    target_action,
    decision,
    tail_seconds,
):
    env, sim = fork(*state)
    sim.dispatch_tags = None
    sim.agent = agent
    policy = FullGreedyPolicy(agent, forced=dict(
        key=decision["key"],
        time=decision["time"],
        obs=decision["obs"],
        action=int(target_action),
    ))
    sim.policy = policy
    sim.decision_log = []
    adjusted = list(loads[:period + 1])
    if adjusted:
        adjusted[period - 1] = None
    control = RolloutControl(
        environment_seed=env_seed,
        loads=tuple(adjusted),
        epsilon=0.0,
        sample_d=False,
        trace_requests=True,
    )
    matrices = [None] * (period - 1) + [
        unpack(env, row["active_matrix"])
        for row in source_rows[period - 1:period + 1]
    ]
    ledger = WorkLedger(period)
    rows = episode(
        replace(cfg, periods=period + 1, freeze_ds=True),
        env,
        d,
        agent,
        gate,
        scene,
        "holdout",
        index,
        recorded_matrices=matrices,
        control=control,
        boundary=ledger,
        continuation=(period - 1, sim),
    )[0]
    verify_periods(env, rows)
    if [row["trace_sha256"] for row in rows] != [
        row["trace_sha256"] for row in source_rows[period - 1:period + 1]
    ]:
        raise AssertionError("Counterfactual changed exogenous requests")
    if [row["active_matrix"] for row in rows] != [
        row["active_matrix"] for row in source_rows[period - 1:period + 1]
    ]:
        raise AssertionError("Counterfactual changed the registered deployment sequence")
    if not policy.forced_seen:
        raise AssertionError("Counterfactual never executed the registered focal action")
    initial_state = state[1].snapshot()
    deadline = rows[0]["end"] + tail_seconds
    focus = next(item for item in sim.requests if item.id == decision["request_id"])
    cohort = [
        item for item in sim.requests
        if rows[0]["start"] <= item.arrival.time < rows[0]["end"]
    ]
    other = [item for item in cohort if item.id != focus.id]
    return {
        "action": int(target_action),
        "reward_sum": float(sum(row["reward_raw"] for row in rows)),
        "penalized_delay_sum": float(sum(
            row["e2e_penalized"] * row["requests"] for row in rows
        )),
        "focus_e2e": _request_e2e(focus, deadline),
        "other_mean_e2e": float(np.mean([
            _request_e2e(item, deadline) for item in other
        ])) if other else 0.0,
        "other_e2e": {
            int(item.id): _request_e2e(item, deadline) for item in other
        },
        "censored": int(sum(row["censored"] for row in rows)),
        "terminal": int(rows[-1].get("residual_requests", 0)),
        "p95": float(np.mean([row["p95"] for row in rows])),
        "p99": float(np.mean([row["p99"] for row in rows])),
        "rows": rows,
        "work": ledger.work,
        "selected_actions": [item["action"] for item in sim.decision_log],
        "initial_state_sha256": fingerprint(initial_state),
    }

"""Causal feature extraction from the pre-action physical state."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from edge_s.agent import resource_advantage


OBSERVATION_COLUMNS = (
    "cpu_capacity_div20",
    "cpu_job_count_div10",
    "cpu_remaining_div20",
    "source_hops_div10",
    "route_job_count_div20",
    "route_transfer_seconds",
    "is_source",
    "legal_mask",
    "stage_compute_div2",
    "stage_input_mbit_div4",
    "stage_output_mbit_div4",
    "stage_id_normalized",
    "next_stage_legal",
    "next_stage_min_transfer_seconds",
    "next_stage_min_transfer_plus_cpu",
    "next_stage_context_available",
    "log1p_cpu_remaining_time",
    "log1p_inflight_cpu_time",
    "log1p_route_backlog",
    "log1p_route_bottleneck",
    "log1p_node_pressure",
    "log1p_stage_pressure",
    "log1p_next_stage_estimate",
    "log1p_request_wait_age",
)


def _job_remaining(sim, resource, job):
    return max(0.0, float(job.remaining - job.rate * (sim.now - resource.last)))


@torch.no_grad()
def q_scores(agent, obs):
    x = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device)
    prior = resource_advantage(x, agent.config.latency_ref_seconds)[0].cpu().numpy()
    residual = agent.online(x)[0].cpu().numpy()
    return prior.astype(np.float32), residual.astype(np.float32), (prior + residual).astype(np.float32)


def extended_state_features(sim, request):
    """Small action-independent summary of concurrent and future system work."""

    service_count = len(sim.c.services)
    service_stage_count = np.zeros(service_count, dtype=np.float64)
    service_replicas = np.zeros(service_count, dtype=np.float64)
    for service, chain in enumerate(sim.c.services):
        service_stage_count[service] = len(chain)
        service_replicas[service] = sim.deployment[list(chain)].sum() / len(chain)

    cpu_by_service = np.zeros(service_count, dtype=np.float64)
    for resource in sim.cpu:
        for job in resource.jobs.values():
            if job.stage_id >= 0:
                service = sim.w.stage_service[job.stage_id]
                cpu_by_service[service] += _job_remaining(sim, resource, job)

    active = [item for item in sim.requests if item.finish is None]
    future_by_service = np.zeros(service_count, dtype=np.float64)
    ages = []
    for item in active:
        ages.append(max(0.0, sim.now - item.arrival.time))
        chain = sim.c.services[item.arrival.service]
        for stage in chain[item.index:]:
            future_by_service[item.arrival.service] += sim.w.stages[stage, 0]

    caps = sim.w.nodes[:, 0]
    cpu_queue = np.array([resource.outstanding() for resource in sim.cpu])
    link_work = np.array([resource.outstanding() / resource.capacity for resource in sim.wired])
    stage = sim.c.services[request.arrival.service][request.index]
    legal = np.flatnonzero(sim.deployment[stage])
    replica_count = sim.deployment.sum(1)
    single_support = replica_count == 1
    age_array = np.asarray(ages) if ages else np.zeros(1)

    global_features = np.array([
        len(active) / 20.0,
        float(age_array.mean()) / 20.0,
        float(age_array.max()) / 20.0,
        float(cpu_queue.max()) / 20.0,
        float(link_work.max(initial=0.0)),
        float(sim.stage_pressure[stage]),
        float(single_support.sum()) / max(1, len(single_support)),
        float(replica_count.mean()),
        float(len(legal)) / sim.c.n,
        float(sim.now - request.arrival.time) / 20.0,
        float(sim.now % sim.c.period_seconds) / sim.c.period_seconds,
        float(np.count_nonzero(cpu_by_service)) / service_count,
    ], dtype=np.float32)
    return np.concatenate([
        global_features,
        np.log1p(cpu_by_service).astype(np.float32),
        np.log1p(future_by_service).astype(np.float32),
        (service_replicas / max(1, sim.c.n)).astype(np.float32),
    ]).astype(np.float32)


def pooled_observation(obs, mask):
    legal = np.asarray(mask, dtype=bool)
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    maximum = obs.max(axis=0)
    return np.concatenate([mean, std, maximum, [legal.sum() / len(legal)]]).astype(np.float32)


def action_features(obs, mask, action, extended, *, include_extension):
    base = np.concatenate([
        np.asarray(obs[action], dtype=np.float32),
        pooled_observation(obs, mask),
    ])
    if include_extension:
        return np.concatenate([base, np.asarray(extended, dtype=np.float32)])
    return base.astype(np.float32)


def state_signature(obs, extended=None):
    """Current 24-D observation signature for nearest-neighbor diagnostics."""

    mask = np.asarray(obs[:, 7] > 0.5, dtype=bool)
    return pooled_observation(obs, mask)


def service_work(sim, request):
    stage = sim.c.services[request.arrival.service][request.index]
    by_service = defaultdict(float)
    for item in sim.requests:
        if item.finish is None:
            for downstream in sim.c.services[item.arrival.service][item.index:]:
                by_service[sim.w.stage_service[downstream]] += sim.w.stages[downstream, 0]
    return dict(stage=int(stage), by_service=dict(by_service))

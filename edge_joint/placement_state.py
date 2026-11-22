"""Read the live event queue at a deployment decision, without advancing it."""

from dataclasses import dataclass

import numpy as np
import torch


def feature_count(nodes, stages, links):
    return 4*nodes + 5*stages + links


@dataclass(frozen=True)
class PlacementResources:
    cpu_work: np.ndarray
    cpu_jobs: np.ndarray
    inflight_work: np.ndarray
    inflight_jobs: np.ndarray
    wired_work: np.ndarray

    def as_log(self):
        return {key: value.tolist() for key, value in vars(self).items()}


def capture_resources(env, sim):
    """Use stage IDs from running CPU jobs and dispatches still in transit."""
    cpu_work = np.zeros((env.K, env.N))
    cpu_jobs = np.zeros_like(cpu_work, dtype=int)
    inflight_work = np.zeros_like(cpu_work)
    inflight_jobs = np.zeros_like(cpu_jobs)
    wired_work = np.zeros(len(env.w.directed))
    if sim is not None:
        if sim.cpu_requests is None:
            raise ValueError('Resource capture requires dispatch tracking from bootstrap')
        for n, resource in enumerate(sim.cpu):
            for job in resource.jobs.values():
                if job.stage_id < 0:
                    raise AssertionError('A running CPU job lacks its deployment stage')
                cpu_work[job.stage_id, n] += max(0., job.remaining-job.rate*(sim.now-resource.last))
                cpu_jobs[job.stage_id, n] += 1
        for request in sim.requests:
            if request.pending is None or request.id in sim.cpu_requests:
                continue
            stage = env.c.services[request.arrival.service][request.index]
            inflight_work[stage, request.destination] += env.w.stages[stage, 0]
            inflight_jobs[stage, request.destination] += 1
        wired_work = np.asarray([edge.outstanding() for edge in sim.wired])
    return PlacementResources(cpu_work, cpu_jobs, inflight_work, inflight_jobs, wired_work)


def resource_features(env, state, demand):
    """Identity-ordered node/stage/link features, normalized without clipping."""
    cpu = env.w.nodes[:, 0]
    old = env.old
    offered = demand.sum(0)[env.w.stage_service]*env.w.stages[:, 0]
    counts = old.sum(1)
    node_seconds = state.cpu_work.sum(0)/cpu
    stage_cpu = (state.cpu_work/cpu).max(1)
    stage_wire = (state.inflight_work/cpu).max(1)
    share = (offered/np.maximum(counts, 1))@old/cpu
    legal_seconds = (old*node_seconds).max(1)
    single_pressure = np.divide(offered, old@cpu, out=np.zeros(env.K), where=counts==1)
    link_seconds = state.wired_work/np.asarray([edge[2] for edge in env.w.directed])

    def scaled(values, reference):
        return np.log1p(values)/np.log1p(reference)

    nodes = np.stack((scaled(node_seconds, 5.), scaled(state.cpu_jobs.sum(0), 16.),
                      scaled(state.inflight_work.sum(0)/cpu, 5.), scaled(share, 1.)), axis=1)
    stages = np.stack((scaled(legal_seconds, 5.), scaled(stage_cpu, 5.), scaled(stage_wire, 5.),
                       scaled(single_pressure, 1.), scaled((state.cpu_jobs+state.inflight_jobs).sum(1), 16.)), axis=1)
    return torch.tensor(np.concatenate((nodes.ravel(), stages.ravel(), scaled(link_seconds, 2.))), dtype=torch.float32)

"""Versioned continuous-time processor-sharing resources and request chains."""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import time
from collections import defaultdict
import numpy as np
from .model import World
from .cpu_allocation import cpu_rates, MODES


def kkt(work, capacity):
    work = np.asarray(work, dtype=float)
    if capacity <= 0 or np.any(work <= 0): raise ValueError("KKT requires positive work/capacity")
    roots = np.sqrt(work)
    return capacity*roots/roots.sum() if len(roots) else roots


@dataclass
class Job:
    remaining: float
    rate: float
    callback: object
    credit_owner: object = None
    stage_id: int = -1


class Resource:
    """Rates are constant between membership changes, as specified by the model.

    Every mutation settles OLD rates first. Generations invalidate heap entries.
    All jobs finishing together are removed atomically before callbacks.
    """
    def __init__(self, sim, capacity, kind):
        self.sim, self.capacity, self.kind = sim, float(capacity), kind
        self.jobs, self.last, self.generation = {}, sim.now, 0
        self.submitted = self.served = 0.
        self.max_capacity_error = 0.
        self.count_integral=0.; self.busy_seconds=0.; self.peak_active=0
        self.count_seconds=defaultdict(float)

    def settle(self):
        dt = self.sim.now-self.last
        if dt < -1e-8: raise RuntimeError("Physical time reversed")
        count=len(self.jobs)
        self.count_integral+=count*max(dt,0.)
        self.busy_seconds+=max(dt,0.) if count else 0.
        self.count_seconds[count]+=max(dt,0.)
        for job in self.jobs.values():
            if job.credit_owner is not None:
                # Conserving attribution of the resource's aggregate job-residence time.
                # Rates are precisely those in force over this interval.
                cost=count*max(dt,0.)*job.rate/self.capacity
                job.credit_owner.stage_credit+=cost
                self.sim.attributed_cost+=cost
            amount = min(job.remaining, max(0., dt)*job.rate)
            job.remaining -= amount
            self.served += amount
        self.last = self.sim.now

    def allocate(self):
        tick = time.perf_counter()
        self.generation += 1
        if self.jobs:
            jobs = list(self.jobs.values())
            if self.kind == "wireless":
                rates = np.full(len(jobs), self.capacity/len(jobs))
            elif self.kind == "cpu" and getattr(self.sim, "cpu_allocator", "kkt") != "kkt":
                rates = cpu_rates([max(j.remaining, 1e-14) for j in jobs], self.capacity,
                                  self.sim.cpu_allocator, self.sim.cpu_reference_work)
            else:
                rates = kkt([max(j.remaining, 1e-14) for j in jobs], self.capacity)
            self.max_capacity_error = max(self.max_capacity_error, abs(float(rates.sum())-self.capacity))
            delay = float("inf")
            for job, rate in zip(jobs, rates):
                job.rate = float(rate)
                if job.rate > 0:
                    delay = min(delay, job.remaining/job.rate)
            self.sim.push(self.sim.now+max(delay, 1e-10), "resource", (self, self.generation))
        self.sim.profile["kkt_seconds"] += time.perf_counter()-tick

    def add(self, amount, callback, credit_owner=None, stage_id=-1):
        if amount <= 0: raise ValueError("Work must be positive")
        self.settle()
        self.sim.job_counter += 1
        self.jobs[self.sim.job_counter] = Job(float(amount), 0., callback, credit_owner,stage_id)
        self.peak_active=max(self.peak_active,len(self.jobs))
        self.submitted += amount
        self.allocate()

    def complete(self, generation):
        if generation != self.generation:
            self.sim.stale_events += 1
            return
        self.settle()
        done = [key for key, j in self.jobs.items() if j.remaining <= max(1e-8, j.rate*1e-8)]
        if not done:  # absolute logical timestamps may round slightly below completion
            self.allocate()
            return
        callbacks = []
        for key in done:
            j = self.jobs.pop(key)
            self.served += j.remaining  # bounded roundoff correction, conserved exactly
            callbacks.append(j.callback)
        self.allocate()
        for callback in callbacks: callback()

    def outstanding(self):
        return sum(max(0., j.remaining-j.rate*(self.sim.now-self.last)) for j in self.jobs.values())


@dataclass
class Request:
    id: int
    arrival: object
    index: int = 0
    source: int = 0
    ready: float = 0.
    destination: int = -1
    pending: object = None
    finish: float | None = None
    trace: list = field(default_factory=list)
    stage_credit: float = 0.
    first_ready: float | None = None


@dataclass
class STransition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray
    terminated: bool
    truncated: bool
    version: int
    collection_step: int
    request_id: int
    stage_index: int
    window_index: int = 0


class Simulator:
    def __init__(self, world: World, deployment, scheduler, start=0., version=0,
                 collection_step=0, collect=True, trace=False, window_index=0, cohort=None):
        self.w, self.c = world, world.c
        self.cpu_allocator = getattr(self.c, "cpu_allocator", "kkt")
        if self.cpu_allocator not in MODES: raise ValueError("Unknown CPU allocator")
        self.cpu_reference_work = min(float(s[0]) for s in self.c.stages)
        self.now = float(start)
        self.start=float(start)
        self.activity_last=float(start); self.active_requests=0; self.active_integral=0.; self.peak_requests=0
        self.arrival_cpu_work=0.
        self.deployment = np.asarray(deployment, dtype=bool).copy()
        self.scheduler, self.version, self.collection_step = scheduler, version, collection_step
        self.collect, self.trace_enabled = collect, trace
        self.window_index=window_index
        self.system_credit=getattr(self.c,'s_system_credit',False)
        self.resource_credit=getattr(self.c,'s_resource_credit',False);self.attributed_cost=0.
        if self.system_credit and self.resource_credit:raise ValueError("Credit modes are exclusive")
        self.cohort=cohort
        if self.system_credit and cohort is None: raise ValueError('system credit requires explicit cohort bounds')
        self.cohort_active=0; self.cohort_integral=0.; self.cohort_last=float(start)
        self.credit_pending=None; self.credit_integral=0.; self.credit_decisions=0
        self.credit_scale=self.c.latency_ref_seconds*getattr(self.c,'s_system_credit_scale',20.)
        self.heap, self.serial, self.job_counter = [], 0, 0
        self.requests, self.transitions = [], []
        self.local_transitions=[]
        self.events = self.stale_events = self.decisions = 0
        self.profile = {"kkt_seconds": 0., "s_inference_seconds": 0., "physical_seconds": 0., "feature_seconds": 0.}
        self.cpu = [Resource(self, n[0], "cpu") for n in self.c.nodes]
        self.wired = [Resource(self, b, "wired") for _, _, b in world.directed]
        self.wireless = [Resource(self, n[3], "wireless") for n in self.c.nodes]
        self.validate_deployment()
        self.competition_context=getattr(self.c,'s_competition_context',False)
        self.reservation_context=getattr(self.c,'s_reservation_context',False)
        self.structured_context=getattr(self.c,'s_structured_context',False)
        if self.structured_context and (not self.competition_context or self.reservation_context or self.system_credit):raise ValueError('structured context requires local 24-feature competition context')
        if self.structured_context:self.stage_inflight=np.zeros((self.c.k,self.c.n))
        if self.reservation_context and (not self.competition_context or self.system_credit):raise ValueError('reservation context needs local competition context')
        if self.competition_context:
            self.inflight_cpu=np.zeros(self.c.n)
            if self.reservation_context:self.inflight_forced=np.zeros(self.c.n)
            self.link_capacity=np.array([r.capacity for r in self.wired])
            self.refresh_pressure()

    def refresh_pressure(self):
        # Current revealed intensity only, never future realized arrivals.
        stage_rate=self.w.node_demand().sum(0)[self.w.stage_service]
        work=stage_rate*self.w.stages[:,0]
        caps=self.w.nodes[:,0]
        totals=self.deployment@caps
        self.stage_pressure=work/np.maximum(totals,1e-12)
        self.node_pressure=self.deployment.T@self.stage_pressure
        if self.reservation_context:
            self.stage_work_rate=work;self.single_stage=self.deployment.sum(1)==1
            self.forced_work=self.deployment.T@(work*self.single_stage)
            self.forced_pressure=self.forced_work/caps

    @property
    def resources(self): return self.cpu+self.wired+self.wireless

    def validate_deployment(self):
        d = self.deployment
        if d.shape != (self.c.k, self.c.n) or not d.any(1).all(): raise ValueError("Uncovered deployment")
        if np.any(d.T @ self.w.stages[:, 3:5] > self.w.nodes[:, 1:3]+1e-8):
            raise ValueError("Deployment exceeds RAM/SSD")

    def change_deployment(self, matrix):
        self.deployment = np.asarray(matrix, dtype=bool).copy()
        self.validate_deployment()
        if self.competition_context:self.refresh_pressure()
        # Already dispatched stages (including in-flight data) retain their destination.

    def push(self, when, kind, payload):
        if when < self.now-1e-8: raise ValueError("Event in past")
        self.serial += 1
        heapq.heappush(self.heap, (float(when), self.serial, kind, payload))

    def add_arrivals(self, arrivals):
        for arrival in arrivals: self.push(arrival.time, "arrival", arrival)

    def structured_tokens(self, request):
        # Exact currently dispatched work by stage and destination; no future arrivals.
        assigned=self.stage_inflight.copy()
        for n,res in enumerate(self.cpu):
            for job in res.jobs.values():
                assert job.stage_id>=0
                assigned[job.stage_id,n]+=max(0.,job.remaining-job.rate*(self.now-res.last))
        # Remaining not-yet-dispatched service-chain workload of active requests.
        # Exclude current decision's work (already in base features), include wireless
        # arrivals and other active requests' downstream stages.
        future=np.zeros(self.c.k)
        for r in self.requests:
            if r.finish is not None:continue
            chain=self.c.services[r.arrival.service]
            start=r.index+1 if (r is request or r.pending is not None) else r.index
            for k in chain[start:]:future[k]+=self.w.stages[k,0]
        cap=self.w.nodes[:,0]
        # Four stage-node channels; stage identity retained, not pooled in simulator.
        tokens=np.stack([self.deployment.T.astype(float),
            np.broadcast_to(np.log1p(self.stage_pressure)[None,:],(self.c.n,self.c.k)),
            np.log1p(assigned.T/cap[:,None]),
            np.broadcast_to(np.log1p(future/20.)[None,:],(self.c.n,self.c.k))],axis=-1)
        return tokens.astype(np.float32)

    def observation(self, request):
        tick = time.perf_counter()
        stage = self.c.services[request.arrival.service][request.index]
        spec = self.w.stages[stage]
        mask = self.deployment[stage].copy()
        counts = np.array([len(r.jobs) for r in self.cpu])
        remaining = np.array([r.outstanding() for r in self.cpu])
        link_counts = np.array([len(r.jobs) for r in self.wired])
        path = self.w.path_incidence[request.source]
        path_load = path @ link_counts
        features = np.column_stack([
            self.w.nodes[:, 0]/20, counts/10, remaining/20,
            self.w.hops[request.source]/10, path_load/20,
            self.w.path_seconds_per_mbit[request.source]*spec[1],
            np.arange(self.c.n) == request.source, mask,
            np.full(self.c.n, spec[0]/2), np.full(self.c.n, spec[1]/4),
            np.full(self.c.n, spec[2]/4), np.full(self.c.n, stage/max(1,self.c.k-1)),
        ]).astype(np.float32)
        if getattr(self.c, "s_next_context", False):
            # State information only: no action choice or reward shaping.
            # Distances are from each possible current execution node to the
            # next stage's available replicas under the actual deployment.
            service = self.c.services[request.arrival.service]
            if request.index + 1 < len(service):
                ns = service[request.index + 1]
                replicas = np.flatnonzero(self.deployment[ns])
                next_spec = self.w.stages[ns]
                comm = self.w.path_seconds_per_mbit[:, replicas] * next_spec[1]
                work = next_spec[0] / self.w.nodes[replicas, 0]
                extra = np.column_stack([self.deployment[ns], comm.min(axis=1),
                                         (comm + work[None, :]).min(axis=1),
                                         np.ones(self.c.n)])
            else:
                extra = np.zeros((self.c.n, 4))
            features = np.concatenate([features, extra.astype(np.float32)], axis=1)
        if self.competition_context:
            caps=self.w.nodes[:,0]
            link_work=np.array([r.outstanding() for r in self.wired])/self.link_capacity
            route_work=path@link_work
            # Maximum existing link backlog along each candidate route.
            bottleneck=(path*link_work[None,:]).max(axis=1)
            cpu_time=remaining/caps
            inbound_time=self.inflight_cpu/caps
            service=self.c.services[request.arrival.service]
            if request.index+1<len(service):
                ns=service[request.index+1];replicas=np.flatnonzero(self.deployment[ns])
                nr=self.w.stages[ns]
                nxt=(self.w.path_seconds_per_mbit[:,replicas]*nr[1]+(self.w.path_incidence[:,replicas]@link_work)
                     +(remaining[replicas]+self.inflight_cpu[replicas]+nr[0])[None,:]/caps[replicas][None,:]).min(axis=1)
            else:nxt=np.zeros(self.c.n)
            extra=np.column_stack([cpu_time,inbound_time,route_work,bottleneck,
                self.node_pressure,np.full(self.c.n,self.stage_pressure[stage]),nxt,
                np.full(self.c.n,self.now-request.arrival.time)])
            # Smooth monotone scale, no clipping of congestion observations.
            features=np.concatenate([features,np.log1p(np.maximum(extra,0)).astype(np.float32)],axis=1)
        if self.reservation_context:
            caps=self.w.nodes[:,0]
            other=self.forced_pressure-(self.deployment[stage]*self.stage_work_rate[stage]/caps if self.single_stage[stage] else 0.)
            spare=np.maximum(caps-self.forced_work,0.)
            alternative=((mask*spare).sum()-mask*spare)/max(self.stage_work_rate[stage],1e-9)
            forced_remaining=np.array([sum(max(0.,j.remaining-j.rate*(self.now-res.last)) for j in res.jobs.values() if j.stage_id>=0 and self.single_stage[j.stage_id]) for res in self.cpu])
            extra=np.column_stack([self.forced_pressure,other,alternative,(forced_remaining+self.inflight_forced)/caps])
            features=np.concatenate([features,np.log1p(np.maximum(extra,0)).astype(np.float32)],axis=1)
        if self.structured_context:
            features=np.concatenate([features,self.structured_tokens(request).reshape(self.c.n,-1)],axis=1)
        if self.system_credit:
            b,e,limit=self.cohort
            extra=np.tile([self.cohort_active/20.,(e-self.now)/max(e-b,1e-9),(limit-self.now)/max(limit-b,1e-9)],(self.c.n,1))
            features=np.concatenate([features,extra.astype(np.float32)],axis=1)
        self.profile["feature_seconds"] += time.perf_counter()-tick
        return features, mask

    def cohort_member(self,r):
        return self.cohort is not None and self.cohort[0]<=r.arrival.time<self.cohort[1]

    def settle_cohort(self):
        self.cohort_integral+=self.cohort_active*(self.now-self.cohort_last)
        self.cohort_last=self.now

    def credit_transition(self,obs,mask,terminal=False,penalty=0.):
        self.settle_cohort()
        if self.credit_pending is None: return
        old_obs,old_mask,action,rid,stage=self.credit_pending
        # Integral of the number of unfinished cohort requests is their total elapsed delay.
        scale=self.credit_scale
        reward=-(self.cohort_integral-self.credit_integral+penalty)/scale
        self.transitions.append(STransition(old_obs,old_mask,action,reward,obs,mask,terminal,False,self.version,self.collection_step,rid,stage,self.window_index))
        self.credit_integral=self.cohort_integral

    def finish_credit(self,penalty_seconds):
        if self.system_credit and self.credit_pending is not None:
            self.credit_transition(np.zeros_like(self.credit_pending[0]),np.ones(self.c.n,bool),True,penalty_seconds*self.cohort_active)
            self.credit_pending=None

    def arrive(self, arrival):
        self.settle_activity()
        self.active_requests+=1; self.peak_requests=max(self.peak_requests,self.active_requests)
        self.arrival_cpu_work+=sum(self.c.stages[k][0] for k in self.c.services[arrival.service])
        r = Request(len(self.requests), arrival, source=arrival.access)
        self.requests.append(r)
        if self.system_credit and self.cohort_member(r):
            self.settle_cohort();self.cohort_active+=1
        initial = self.c.stages[self.c.services[arrival.service][0]][1]
        self.wireless[arrival.access].add(initial, lambda: self.ready(r))

    def ready(self, r, previous=None):
        r.ready = self.now
        if r.index==0:r.first_ready=self.now
        obs, mask = self.observation(r)
        if previous is not None: self.record(r, previous, obs, mask, False)
        if self.system_credit and self.collect: self.credit_transition(obs,mask)
        if self.resource_credit:r.stage_credit=0.
        tick = time.perf_counter()
        action = int(self.scheduler(obs, mask))
        self.profile["s_inference_seconds"] += time.perf_counter()-tick
        if not 0 <= action < self.c.n or not mask[action]: raise ValueError("Illegal scheduling action")
        self.decisions += 1
        if self.system_credit and self.collect:
            self.credit_pending=(obs,mask,action,r.id,r.index);self.credit_decisions+=1
        r.destination = action
        r.pending = (obs, mask, action, r.ready, r.index)
        if self.trace_enabled: r.trace.append((r.index, "ready", self.now, action))
        if self.competition_context:
            stage=self.c.services[r.arrival.service][r.index]
            self.inflight_cpu[action]+=self.c.stages[stage][0]
            if self.structured_context:self.stage_inflight[stage,action]+=self.c.stages[stage][0]
            if self.reservation_context and self.single_stage[stage]:self.inflight_forced[action]+=self.c.stages[stage][0]
        route = self.w.routes[r.source][action]
        self.transmit(r, route, 0)

    def transmit(self, r, route, hop):
        if hop == len(route):
            stage = self.c.services[r.arrival.service][r.index]
            if self.trace_enabled: r.trace.append((r.index, "compute", self.now, r.destination))
            if self.competition_context:
                self.inflight_cpu[r.destination]-=self.c.stages[stage][0]
                if self.structured_context:
                    self.stage_inflight[stage,r.destination]-=self.c.stages[stage][0]
                    if self.stage_inflight[stage,r.destination]<-1e-7:raise RuntimeError("stage inflight underflow")
                    self.stage_inflight[stage,r.destination]=max(0.,self.stage_inflight[stage,r.destination])
                if self.inflight_cpu[r.destination]<-1e-7:raise RuntimeError('inflight CPU work underflow')
                self.inflight_cpu[r.destination]=max(0.,self.inflight_cpu[r.destination])
                if self.reservation_context and self.single_stage[stage]:
                    self.inflight_forced[r.destination]-=self.c.stages[stage][0]
                    if self.inflight_forced[r.destination]<-1e-7:raise RuntimeError('inflight forced CPU underflow')
                    self.inflight_forced[r.destination]=max(0.,self.inflight_forced[r.destination])
            self.cpu[r.destination].add(self.c.stages[stage][0], lambda: self.finish_stage(r),r if self.resource_credit else None,stage if (self.reservation_context or self.structured_context) else -1)
        else:
            stage = self.c.services[r.arrival.service][r.index]
            self.wired[route[hop]].add(self.c.stages[stage][1], lambda: self.transmit(r, route, hop+1),r if self.resource_credit else None)

    def record(self, r, pending, next_obs, next_mask, terminated):
        obs, mask, action, ready, stage_index = pending
        if self.collect and (not self.system_credit or getattr(self.c,'s_local_auxiliary',False)):
            target=self.local_transitions if self.system_credit else self.transitions
            target.append(STransition(obs, mask, action,
                -(r.stage_credit if self.resource_credit else self.now-ready)/self.c.latency_ref_seconds, next_obs, next_mask,
                terminated, False, self.version, self.collection_step, r.id, stage_index, self.window_index))

    def finish_stage(self, r):
        if r.pending is None: raise RuntimeError("Duplicate stage completion")
        if self.trace_enabled: r.trace.append((r.index, "finish", self.now, r.destination))
        previous, r.pending = r.pending, None
        r.source = r.destination
        if r.index+1 == len(self.c.services[r.arrival.service]):
            self.settle_activity(); self.active_requests-=1
            if self.system_credit and self.cohort_member(r):
                self.settle_cohort();self.cohort_active-=1
            r.finish = self.now
            self.record(r, previous, np.zeros_like(previous[0]), np.ones(self.c.n, bool), True)
        else:
            r.index += 1
            self.ready(r, previous)

    def run_until(self, end):
        tick = time.perf_counter()
        while self.heap and self.heap[0][0] <= end:
            self.now, _, kind, payload = heapq.heappop(self.heap)
            self.events += 1
            if kind == "arrival": self.arrive(payload)
            elif kind == "resource": payload[0].complete(payload[1])
            elif kind == "callback": payload()
            else: raise RuntimeError(kind)
        self.now = float(end)
        self.profile["physical_seconds"] += time.perf_counter()-tick

    def result(self, arrival_start, arrival_end, details=False):
        targets = [r for r in self.requests if arrival_start <= r.arrival.time < arrival_end]
        completed = [r for r in targets if r.finish is not None]
        censored = [r for r in targets if r.finish is None]
        observed = [(r.finish if r.finish is not None else self.now)-r.arrival.time for r in targets]
        lower = sum(observed)
        penalized = lower + len(censored)*self.c.censor_penalty_seconds
        out = dict(arrivals=len(targets), completed=len(completed), censored=len(censored),
                   delay_lower_bound_sum=lower, delay_penalized_sum=penalized,
                   completed_delay_sum=sum(r.finish-r.arrival.time for r in completed),
                   mean_delay_lower_bound=lower/len(targets) if targets else None,
                   mean_completed_delay=(sum(r.finish-r.arrival.time for r in completed)/len(completed)) if completed else None,
                   censor_fraction=len(censored)/len(targets) if targets else 0.,
                   events=self.events, stale_events=self.stale_events, s_decisions=self.decisions,
                   s_transitions=len(self.transitions), profile=self.profile.copy(),
                   active_jobs=sum(len(r.jobs) for r in self.resources),
                   censored_pending=sum(r.pending is not None for r in censored))
        delays=[r.finish-r.arrival.time for r in completed]
        out["p95_completed_delay"]=float(np.quantile(delays,.95)) if delays else None
        out["p95_delay_lower_bound"]=float(np.quantile(observed,.95)) if observed else None
        if details:
            out["censored_records"] = [dict(request_id=r.id, arrival=r.arrival.time, observed_age=self.now-r.arrival.time,
                                             stage_index=r.index, pending=r.pending is not None) for r in censored]
            out["region_service"] = []
            for region in range(self.c.regions):
                for service in range(len(self.c.services)):
                    group = [r for r in targets if self.w.regions[r.arrival.access] == region and r.arrival.service == service]
                    out["region_service"].append(dict(region=region, service=service, arrivals=len(group),
                        censored=sum(r.finish is None for r in group), delay_lower_bound_sum=sum(
                            (r.finish if r.finish is not None else self.now)-r.arrival.time for r in group)))
        for resource in self.resources: resource.settle()
        out["max_capacity_error"]=max(r.max_capacity_error for r in self.resources)
        out["max_work_conservation_error"]=max(abs(r.submitted-r.served-sum(j.remaining for j in r.jobs.values())) for r in self.resources)
        return out

    def settle_activity(self):
        self.active_integral+=self.active_requests*(self.now-self.activity_last)
        self.activity_last=self.now

    def telemetry(self):
        """Exact time integrals; this settles old rates but never reallocates."""
        self.settle_activity()
        duration=max(self.now-self.start,1e-12)
        def summarize(resources):
            result=[]
            for resource in resources:
                resource.settle()
                result.append(dict(mean_active=resource.count_integral/duration,peak_active=resource.peak_active,
                    busy_fraction=resource.busy_seconds/duration,served=resource.served,
                    utilization=resource.served/(resource.capacity*duration),
                    active_count_time_fraction={str(k):v/duration for k,v in resource.count_seconds.items()}))
            return result
        return dict(duration=duration,mean_active_requests=self.active_integral/duration,
                    peak_active_requests=self.peak_requests,arrival_cpu_work=self.arrival_cpu_work,
                    arrival_cpu_work_per_second=self.arrival_cpu_work/duration,
                    cpu=summarize(self.cpu),wired=summarize(self.wired),wireless=summarize(self.wireless))


class HeuristicScheduler:
    def __init__(self, world): self.w = world

    def __call__(self, obs, mask):
        # Current workload + processing + loaded fixed-route transfer estimate.
        score = (obs[:, 2]*20+obs[:, 8]*2*(1+obs[:, 1]*10))/(obs[:, 0]*20)
        score += obs[:, 5]*(1+obs[:, 4]*20/np.maximum(1, obs[:, 3]*10))
        return int(np.where(mask, score, np.inf).argmin())

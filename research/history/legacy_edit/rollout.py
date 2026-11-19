"""Complete matrix generation + single real-physics submission.

No winner-picking, candidate full-layout pool, projection, forced FINISH, or
reward clipping. Nontermination is an explicit failure, not a fabricated reward.
"""
from __future__ import annotations
import hashlib, time
import numpy as np
import torch
from torch.nn import functional as F

from .actions import NativeLegalEdits
from edge_d.world import make_world, observe, ResourceFeatures, cpu_lower_bound
from .policy import lightweight_state, choose_action, json_action
from edge_d.physical import evaluate_layout


class NonterminationError(RuntimeError):
    """A policy failed to construct a valid terminal layout."""


def sample_layout(*, seed:int, scene:int, split:str, actor,
                  policy_seed:int, physical_seed:int,
                  cpu_finish_guard:bool=True,
                  warmup_seconds:float=20.,
                  max_actions:int=1100,
                  train:bool=False, scheduler=None, collect_s:bool=False,
                  return_transitions:bool=False, gs:int=1, existing_world=None):
    """Return (physical_record, log_probability_list).

    'train' keeps the action log-probability computation graph. The physical
    simulator does not participate in backprop; optimization uses REINFORCE.
    """
    e=existing_world if existing_world is not None else make_world(seed,scene,split)
    base=observe(e)
    e.draft[:]=False
    e.phase=0
    e._mask=None
    e.apply(e.BUILD)
    feature_maker=ResourceFeatures(e)
    rng=np.random.default_rng(policy_seed)
    logps=[]
    actions=[]
    counts={"ADD":0,"REMOVE":0,"RELOCATE":0,"FINISH":0}
    blocked_finish=0
    began=time.perf_counter()

    for _ in range(max_actions):
        state=lightweight_state(e,base,feature_maker)
        denied=False
        if cpu_finish_guard and e.phase==2:
            denied=cpu_lower_bound(e,e.draft) >= 1.-1e-8
            blocked_finish+=int(denied)
        if denied:
            try:
                action,logp,_=choose_action(actor,state,e,"EDIT",rng,force_continue=True)
            except ValueError as error:
                raise NonterminationError("Unsafe draft has no valid continuation") from error
            # Exact conditional log-probability after FINISH has been masked.
            _,stop=actor(state)
            logp=logp-F.logsigmoid(-stop)
        else:
            action,logp,_=choose_action(actor,state,e,"EDIT",rng)
        kind,stage,node,destination=action
        if train: logps.append(logp)
        actions.append(json_action(action))
        counts[kind]+=1
        if kind=="FINISH":
            if e.phase!=2 or not e.legal()[e.FINISH]:
                raise AssertionError("FINISH violated coverage/RAM/SSD legality")
            matrix=e.apply(e.FINISH)
            break
        if e.phase==1:
            if kind!="ADD": raise AssertionError("Coverage must use ADD")
            e.apply(stage*e.N+node)
        else:
            matrix_next=NativeLegalEdits(e,e.draft).apply(action)
            e.draft[:]=matrix_next
            e._mask=None
    else:
        raise NonterminationError(f"Policy did not FINISH in {max_actions} construction actions")

    if not (matrix.shape==(28,32) and matrix.any(1).all()
            and np.all(matrix.T@e.spec<=e.capacity+1e-7)):
        raise AssertionError("Invalid policy-generated matrix")
    if not np.array_equal(matrix,e.draft):
        raise AssertionError("Submitted matrix differs from policy draft")
    rho=cpu_lower_bound(e,matrix)
    if cpu_finish_guard and not rho<1.-1e-8:
        raise AssertionError("CPU FINISH gate was bypassed")
    physical=evaluate_layout(e,matrix,physical_seed,warmup_seconds,
            scheduler=scheduler,collect=collect_s,return_transitions=return_transitions,gs=gs)
    if return_transitions: result,transitions=physical
    else: result=physical;transitions=None
    record={
        "seed":seed,"scene":scene,"split":split,
        "policy_seed":policy_seed,"physical_seed":physical_seed,
        "reward":float(result["reward"]),
        "penalized_e2e":float(result["penalized_delay"]),
        "completed_e2e":(None if result["mean_completed"] is None else float(result["mean_completed"])),
        "censored":int(result["failed"]),
        "requests":int(result["n"]),
        "replicas":int(matrix.sum()),"decisions":len(actions),
        "cpu_rho_lower_bound":float(rho),
        "cpu_finish_guard":bool(cpu_finish_guard),
        "blocked_finish":blocked_finish,
        "kind_counts":counts,
        "matrix_sha256":hashlib.sha256(matrix.tobytes()).hexdigest(),
        "matrix_packed_hex":np.packbits(matrix.astype(np.uint8).reshape(-1)).tobytes().hex(),
        "actions":actions,
        "physical_trace_sha256":result["nested_trace_hash"],
        "evaluation_warmup_seconds":float(warmup_seconds),
        "physical_reward_unmodified":True,
        "physical_failed_semantics":"censored_at_window_end_plus_tail",
        "posthoc_changed_bits":0,
        "wall_seconds":time.perf_counter()-began,
    }
    return (record,logps,transitions) if return_transitions else (record,logps)

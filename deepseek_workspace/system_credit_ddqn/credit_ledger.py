"""Exact cohort-residence credit for ordered global Agent-S decisions.

The ledger is intentionally independent from the physical event engine. It
consumes the real request timestamps and global decision timestamps produced by
the simulator, then reconstructs the same residence integrals exactly.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, fields
import math
from typing import Iterable, Sequence

import numpy as np
import torch

from edge_d._engine.edge_rl.physics import STransition


@dataclass(frozen=True)
class CohortWindow:
    """Arrival, score deadline and denominator ownership of one business period."""

    period: int
    start: float
    end: float
    deadline: float

    def __post_init__(self):
        if self.period < 1 or not (self.start < self.end < self.deadline):
            raise ValueError("Invalid cohort window")


@dataclass(frozen=True)
class RequestSpan:
    """Physical request timestamps needed by the cohort-delay identity."""

    request_id: int
    arrival: float
    finish: float | None
    cohort: int

    def __post_init__(self):
        if self.arrival < 0 or (self.finish is not None and self.finish < self.arrival):
            raise ValueError("Invalid request span")


@dataclass(frozen=True)
class DecisionEvent:
    """One global Agent-S decision in actual event execution order."""

    sequence: int
    time: float
    obs: np.ndarray
    mask: np.ndarray
    action: int
    version: int
    collection_step: int
    request_id: int
    stage_index: int
    window_index: int

    def __post_init__(self):
        if self.sequence < 0 or self.time < 0 or not self.mask[self.action]:
            raise ValueError("Invalid global decision event")


@dataclass
class SystemTransition(STransition):
    """Transition over the next global decision, not the next request stage."""

    elapsed_seconds: float = 1.0
    decision_index: int = 0
    decision_time: float = 0.0


@dataclass(frozen=True)
class CohortCost:
    period: int
    requests: int
    residence_seconds: float
    deadline_requests: int
    punished_delay_seconds: float
    normalized_delay_seconds: float


@dataclass(frozen=True)
class CreditAudit:
    cohort_costs: tuple[CohortCost, ...]
    pre_decision_punished_delay: float
    pre_decision_normalized_delay: float
    transition_punished_delay: float
    transition_normalized_delay: float
    total_punished_delay: float
    total_normalized_delay: float
    identity_error: float
    decision_count: int
    first_decision_time: float | None
    final_time: float

    def as_dict(self):
        return {
            "cohort_costs": [
                {
                    "period": row.period,
                    "requests": row.requests,
                    "residence_seconds": row.residence_seconds,
                    "deadline_requests": row.deadline_requests,
                    "punished_delay_seconds": row.punished_delay_seconds,
                    "normalized_delay_seconds": row.normalized_delay_seconds,
                }
                for row in self.cohort_costs
            ],
            "pre_decision_punished_delay": self.pre_decision_punished_delay,
            "pre_decision_normalized_delay": self.pre_decision_normalized_delay,
            "transition_punished_delay": self.transition_punished_delay,
            "transition_normalized_delay": self.transition_normalized_delay,
            "total_punished_delay": self.total_punished_delay,
            "total_normalized_delay": self.total_normalized_delay,
            "identity_error": self.identity_error,
            "decision_count": self.decision_count,
            "first_decision_time": self.first_decision_time,
            "final_time": self.final_time,
        }


def _cohort_for(arrival: float, windows: Sequence[CohortWindow]) -> int:
    for window in windows:
        if window.start <= arrival < window.end:
            return window.period
    raise ValueError(f"Request arrival {arrival} is outside all cohort windows")


def request_spans(requests: Iterable[object], windows: Sequence[CohortWindow]) -> list[RequestSpan]:
    """Convert physical request objects to a stable read-only ledger representation."""

    spans = []
    for request in requests:
        arrival = float(request.arrival.time)
        finish = None if request.finish is None else float(request.finish)
        spans.append(RequestSpan(
            request_id=int(request.id),
            arrival=arrival,
            finish=finish,
            cohort=_cohort_for(arrival, windows),
        ))
    return spans


def cohort_denominators(spans: Iterable[RequestSpan]) -> dict[int, int]:
    counts = defaultdict(int)
    for span in spans:
        counts[span.cohort] += 1
    return dict(counts)


def residence_integral(
    spans: Iterable[RequestSpan],
    window: CohortWindow,
    start: float,
    end: float,
) -> float:
    """Return integral N_p(t)dt over [start,end)."""

    if end < start:
        raise ValueError("Credit interval reversed")
    if not end > start:
        return 0.0
    total = 0.0
    for span in spans:
        if span.cohort != window.period:
            continue
        left = max(float(start), span.arrival)
        right = min(float(end), window.deadline)
        if span.finish is not None:
            right = min(right, span.finish)
        if right > left:
            total += right - left
    return float(total)


def deadline_count(
    spans: Iterable[RequestSpan],
    window: CohortWindow,
    start: float,
    end: float,
    *,
    include_end: bool = False,
) -> int:
    """Count requests whose cohort deadline falls in this credit interval."""

    due = window.deadline
    inside = start <= due <= end if include_end else start <= due < end
    if not inside:
        return 0
    return sum(
        span.cohort == window.period
        and (span.finish is None or span.finish > due + 1e-12)
        for span in spans
    )


def _direct_cohort_cost(
    spans: Sequence[RequestSpan],
    window: CohortWindow,
    *,
    penalty_seconds: float,
    delay_weight: float,
) -> CohortCost:
    members = [span for span in spans if span.cohort == window.period]
    residence = 0.0
    deadline_requests = 0
    for span in members:
        right = min(window.deadline, span.finish) if span.finish is not None else window.deadline
        residence += max(0.0, right - span.arrival)
        if span.finish is None or span.finish > window.deadline + 1e-12:
            deadline_requests += 1
    punished = residence + penalty_seconds * deadline_requests
    normalized = delay_weight * punished / len(members) if members else 0.0
    return CohortCost(
        period=window.period,
        requests=len(members),
        residence_seconds=float(residence),
        deadline_requests=int(deadline_requests),
        punished_delay_seconds=float(punished),
        normalized_delay_seconds=float(normalized),
    )


def penalized_delay_identity(
    spans: Iterable[RequestSpan],
    windows: Sequence[CohortWindow],
    *,
    penalty_seconds: float = 10.0,
    delay_weight: float = 10.0,
) -> list[CohortCost]:
    """Reference cohort delay computed directly from each request's finish time."""

    values = tuple(spans)
    return [
        _direct_cohort_cost(
            values,
            window,
            penalty_seconds=penalty_seconds,
            delay_weight=delay_weight,
        )
        for window in windows
    ]


def _interval_cost(
    spans: Sequence[RequestSpan],
    windows: Sequence[CohortWindow],
    counts: dict[int, int],
    start: float,
    end: float,
    *,
    penalty_seconds: float,
    delay_weight: float,
    include_end: bool,
) -> tuple[float, float, dict[int, float], dict[int, int]]:
    residence_by_cohort = {}
    deadline_by_cohort = {}
    punished = 0.0
    normalized = 0.0
    for window in windows:
        residence = residence_integral(spans, window, start, end)
        due_count = deadline_count(spans, window, start, end, include_end=include_end)
        if residence or due_count:
            residence_by_cohort[window.period] = residence
            deadline_by_cohort[window.period] = due_count
            punished_now = residence + penalty_seconds * due_count
            count = counts.get(window.period, 0)
            punished += punished_now
            if count:
                normalized += delay_weight * punished_now / count
    return float(punished), float(normalized), residence_by_cohort, deadline_by_cohort


def build_system_credit(
    decisions: Sequence[DecisionEvent],
    spans: Sequence[RequestSpan],
    windows: Sequence[CohortWindow],
    *,
    start_time: float,
    final_time: float,
    penalty_seconds: float = 10.0,
    delay_weight: float = 10.0,
) -> tuple[list[SystemTransition], CreditAudit]:
    """Build ordered global transitions and audit the complete cohort identity."""

    ordered = sorted(decisions, key=lambda item: (item.time, item.sequence))
    if len({item.sequence for item in ordered}) != len(ordered):
        raise ValueError("Duplicate global decision sequence")
    if ordered and (ordered[0].time < start_time or ordered[-1].time > final_time):
        raise ValueError("Decision outside the credited episode")
    if windows and final_time + 1e-9 < max(window.deadline for window in windows):
        raise ValueError("System-credit final time precedes a cohort deadline")
    counts = cohort_denominators(spans)
    decision_times = [item.time for item in ordered]
    events = defaultdict(lambda: defaultdict(int))
    points = {float(start_time), float(final_time), *(window.deadline for window in windows)}
    for span in spans:
        window = next((item for item in windows if item.period == span.cohort), None)
        if window is None:
            raise ValueError(f"Unknown cohort {span.cohort}")
        finish = window.deadline if span.finish is None else min(span.finish, window.deadline)
        points.add(span.arrival)
        points.add(float(finish))
        if finish > span.arrival:
            events[float(span.arrival)][span.cohort] += 1
            events[float(finish)][span.cohort] -= 1
    points.update(decision_times)
    timeline = sorted(points)
    active = defaultdict(int)
    residence_by_bucket = defaultdict(lambda: defaultdict(float))
    deadline_by_bucket = defaultdict(lambda: defaultdict(int))
    for index, now in enumerate(timeline[:-1]):
        for cohort, change in events.get(now, {}).items():
            active[cohort] += change
            if active[cohort] < 0:
                raise AssertionError("Negative active cohort ledger")
        bucket = bisect_right(decision_times, now) - 1
        duration = timeline[index + 1] - now
        for cohort, count in active.items():
            if count:
                residence_by_bucket[bucket][cohort] += count * duration
    for window in windows:
        due = deadline_count(spans, window, -math.inf, math.inf)
        bucket = bisect_right(decision_times, window.deadline) - 1
        deadline_by_bucket[bucket][window.period] = due

    def bucket_cost(bucket):
        punished = normalized = 0.0
        for window in windows:
            residence = residence_by_bucket.get(bucket, {}).get(window.period, 0.0)
            due_count = deadline_by_bucket.get(bucket, {}).get(window.period, 0)
            punished += residence + penalty_seconds * due_count
            count = counts.get(window.period, 0)
            if count:
                normalized += delay_weight * (residence + penalty_seconds * due_count) / count
        return punished, normalized, dict(residence_by_bucket.get(bucket, {})), dict(
            deadline_by_bucket.get(bucket, {})
        )

    pre_punished, pre_normalized, _, _ = bucket_cost(-1)

    transitions = []
    transition_punished = transition_normalized = 0.0
    for index, decision in enumerate(ordered):
        end = ordered[index + 1].time if index + 1 < len(ordered) else final_time
        finished = index + 1 == len(ordered)
        punished, normalized, residence, deadlines = bucket_cost(index)
        transition_punished += punished
        transition_normalized += normalized
        if finished:
            next_obs = np.zeros_like(decision.obs)
            next_mask = np.ones_like(decision.mask, dtype=bool)
        else:
            next_obs = np.array(ordered[index + 1].obs, copy=True)
            next_mask = np.array(ordered[index + 1].mask, dtype=bool, copy=True)
        transitions.append(SystemTransition(
            obs=np.array(decision.obs, copy=True),
            mask=np.array(decision.mask, dtype=bool, copy=True),
            action=int(decision.action),
            reward=float(-normalized),
            next_obs=next_obs,
            next_mask=next_mask,
            terminated=bool(finished),
            truncated=False,
            version=int(decision.version),
            collection_step=int(decision.collection_step),
            request_id=int(decision.request_id),
            stage_index=int(decision.stage_index),
            window_index=int(decision.window_index),
            elapsed_seconds=float(end - decision.time),
            decision_index=int(decision.sequence),
            decision_time=float(decision.time),
        ))
        transitions[-1].residence_by_cohort = residence
        transitions[-1].deadline_by_cohort = deadlines

    direct = penalized_delay_identity(
        spans,
        windows,
        penalty_seconds=penalty_seconds,
        delay_weight=delay_weight,
    )
    total_punished = sum(row.punished_delay_seconds for row in direct)
    total_normalized = sum(row.normalized_delay_seconds for row in direct)
    covered_punished = pre_punished + transition_punished
    covered_normalized = pre_normalized + transition_normalized
    identity_error = max(
        abs(total_punished - covered_punished),
        abs(total_normalized - covered_normalized),
    )
    if identity_error > 1e-9:
        raise AssertionError(
            f"System credit identity failed: direct={total_punished}/{total_normalized}, "
            f"covered={covered_punished}/{covered_normalized}"
        )
    audit = CreditAudit(
        cohort_costs=tuple(direct),
        pre_decision_punished_delay=float(pre_punished),
        pre_decision_normalized_delay=float(pre_normalized),
        transition_punished_delay=float(transition_punished),
        transition_normalized_delay=float(transition_normalized),
        total_punished_delay=float(total_punished),
        total_normalized_delay=float(total_normalized),
        identity_error=float(identity_error),
        decision_count=len(ordered),
        first_decision_time=ordered[0].time if ordered else None,
        final_time=float(final_time),
    )
    return transitions, audit


SYSTEM_TRANSITION_FIELDS = tuple(field.name for field in fields(SystemTransition))


def replay_snapshot(replay) -> dict:
    """Serialize the independent global-transition replay without losing dt."""

    if not replay.data:
        return {"position": replay.position, "columns": {}}
    return {
        "position": replay.position,
        "columns": {
            name: torch.as_tensor(np.asarray([getattr(item, name) for item in replay.data]))
            for name in SYSTEM_TRANSITION_FIELDS
        },
    }


def restore_replay(replay, state: dict) -> None:
    replay.data.clear()
    columns = state["columns"]
    if columns:
        arrays = {name: columns[name].numpy() for name in SYSTEM_TRANSITION_FIELDS}
        for values in zip(*(arrays[name] for name in SYSTEM_TRANSITION_FIELDS), strict=True):
            replay.data.append(SystemTransition(*values))
    replay.position = int(state["position"])

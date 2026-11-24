"""Read-only adapter that records global Agent-S decisions and their real states."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from comparison.monolithic_service.environment import MeasuredMultistageSimulator
from .credit_ledger import (
    CohortWindow,
    DecisionEvent,
    build_system_credit,
    request_spans,
)


class SystemCreditSimulator(MeasuredMultistageSimulator):
    """Preserve the original physical engine while replacing only S accounting."""

    def __init__(
        self,
        env,
        *args,
        cohort_windows: tuple[CohortWindow, ...],
        credit_enabled: bool = True,
        **kwargs,
    ):
        if not hasattr(env, "reference_work"):
            env.reference_work = float(np.min(env.w.stages[:, 0]))
        self.cohort_windows = tuple(cohort_windows)
        self.credit_enabled = bool(credit_enabled)
        self.global_decisions: list[DecisionEvent] = []
        self.credit_audit = None
        self.credit_build_seconds = None
        super().__init__(env, *args, **kwargs)

    def ready(self, request, previous=None):
        if not self.credit_enabled or not self.collect:
            return super().ready(request, previous)
        base_scheduler = self.scheduler

        def recording_scheduler(obs, mask):
            action = int(base_scheduler(obs, mask))
            if not bool(mask[action]):
                raise AssertionError("System-credit scheduler returned an illegal action")
            self.global_decisions.append(DecisionEvent(
                sequence=len(self.global_decisions),
                time=float(self.now),
                obs=np.array(obs, copy=True),
                mask=np.array(mask, dtype=bool, copy=True),
                action=action,
                version=int(self.version),
                collection_step=int(self.collection_step),
                request_id=int(request.id),
                stage_index=int(request.index),
                window_index=int(self.window_index),
            ))
            return action

        self.scheduler = recording_scheduler
        try:
            return super().ready(request, previous)
        finally:
            self.scheduler = base_scheduler

    def end_replay(self):
        super().end_replay()
        if not (self.credit_enabled and self.collect):
            return
        tick = perf_counter()
        spans = request_spans(self.requests, self.cohort_windows)
        transitions, audit = build_system_credit(
            self.global_decisions,
            spans,
            self.cohort_windows,
            start_time=self.start,
            final_time=self.now,
            penalty_seconds=self.env.failure_penalty,
        )
        self.transitions = transitions
        self.credit_audit = audit
        self.credit_build_seconds = perf_counter() - tick
        if len(self.transitions) != self.decisions:
            raise AssertionError(
                f"Global transition count {len(self.transitions)} != decisions {self.decisions}"
            )
        if any(not item.elapsed_seconds >= 0 for item in self.transitions):
            raise AssertionError("Negative global decision interval")


def cohort_windows(
    *,
    start: float,
    periods: int,
    period_seconds: float,
    tail_seconds: float,
) -> tuple[CohortWindow, ...]:
    return tuple(
        CohortWindow(
            period=index + 1,
            start=start + index * period_seconds,
            end=start + (index + 1) * period_seconds,
            deadline=start + (index + 1) * period_seconds + tail_seconds,
        )
        for index in range(periods)
    )

"""Physical action ranking, prior residual and state-similarity analysis."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .offline_predictor import compare as offline_compare


def known_exp039(path=Path("runs/results/EXP039")):
    rows = []
    for name in ("gs38", "natural", "pressure", "favorable"):
        data = json.loads((Path(path) / f"{name}.json").read_text(encoding="utf8"))
        flips = {tuple(item["key"]): item for item in data["cells"]["full"]["flips"]}
        for probe in data["probes"]:
            event = flips[tuple(probe["key"])]
            legal = event["legal"]
            full_action = int(event["full_node"])
            alternative = int(probe["alternative"])
            if alternative == full_action:
                continue
            actions = [full_action, alternative]
            actual = [0.0, -float(probe["cohort_reward_delta"])]
            q_values = [float(event["q"][legal.index(node)]) for node in actions]
            prior_values = [float(event["prior"][legal.index(node)]) for node in actions]
            best = int(np.argmax(actual))
            rows.append({
                "case": name,
                "q_top1": int(actions[int(np.argmax(q_values))] == actions[best]),
                "prior_top1": int(actions[int(np.argmax(prior_values))] == actions[best]),
                "q_corr": _safe_spearman(q_values, actual),
                "prior_corr": _safe_spearman(prior_values, actual),
                "q_regret": float(actual[best] - actual[int(np.argmax(q_values))]),
                "prior_regret": float(actual[best] - actual[int(np.argmax(prior_values))]),
            })
    return {
        "states": len(rows),
        "q_top1": float(np.mean([row["q_top1"] for row in rows])) if rows else 0.0,
        "prior_top1": float(np.mean([row["prior_top1"] for row in rows])) if rows else 0.0,
        "q_corr": float(np.mean([row["q_corr"] for row in rows])) if rows else 0.0,
        "prior_corr": float(np.mean([row["prior_corr"] for row in rows])) if rows else 0.0,
        "q_regret": float(np.mean([row["q_regret"] for row in rows])) if rows else 0.0,
        "prior_regret": float(np.mean([row["prior_regret"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def _safe_spearman(values, targets):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(values) < 2 or np.ptp(values) < 1e-12 or np.ptp(targets) < 1e-12:
        return 0.0
    return float(spearmanr(values, targets).statistic)


def q_ranking(groups):
    rows = defaultdict(list)
    for group in groups:
        for state in group["states"]:
            actions = state["actions"]
            actual = [action["reward_sum"] for action in actions]
            prior = [action["prior"] for action in actions]
            q = [action["q"] for action in actions]
            best = max(range(len(actions)), key=lambda index: actual[index])
            rows[(group["stage"], group["profile"])].append({
                "prior_top1": int(actions[int(np.argmax(prior))]["action"] == actions[best]["action"]),
                "q_top1": int(actions[int(np.argmax(q))]["action"] == actions[best]["action"]),
                "prior_corr": _safe_spearman(prior, actual),
                "q_corr": _safe_spearman(q, actual),
                "prior_regret": float(actual[best] - actual[int(np.argmax(prior))]),
                "q_regret": float(actual[best] - actual[int(np.argmax(q))]),
            })

    def summarize(values):
        return {
            "states": len(values),
            "prior_top1": float(np.mean([row["prior_top1"] for row in values])),
            "q_top1": float(np.mean([row["q_top1"] for row in values])),
            "prior_corr": float(np.mean([row["prior_corr"] for row in values])),
            "q_corr": float(np.mean([row["q_corr"] for row in values])),
            "prior_regret": float(np.mean([row["prior_regret"] for row in values])),
            "q_regret": float(np.mean([row["q_regret"] for row in values])),
        }

    return {f"{stage}.{profile}": summarize(values) for (stage, profile), values in sorted(rows.items())}


def similar_state_pairs(groups, threshold=0.25, max_pairs=30):
    candidates = []
    for group in groups:
        for state in group["states"]:
            candidates.append((group, state, np.asarray(state["state_signature"], dtype=float)))
    matrix = np.stack([item[2] for item in candidates])
    mean, std = matrix.mean(0), matrix.std(0)
    std[std < 1e-8] = 1
    normalized = (matrix - mean) / std
    pairs = []
    records = [
        (item[0], item[1], normalized[index])
        for index, item in enumerate(candidates)
    ]
    for (left_group, left, left_vector), (right_group, right, right_vector) in combinations(records, 2):
        if (
            left_group["id"] == right_group["id"]
            or left_group["stage"] != right_group["stage"]
            or left_group["profile"] != right_group["profile"]
            or left["decision"]["stage"] != right["decision"]["stage"]
        ):
            continue
        distance = float(np.sqrt(np.mean((left_vector - right_vector) ** 2)))
        pairs.append({
            "left": f"{left_group['id']}:p{left['period']}",
            "right": f"{right_group['id']}:p{right['period']}",
            "distance": distance,
            "left_best": left["best_action"],
            "right_best": right["best_action"],
            "different_best": left["best_action"] != right["best_action"],
            "left_prior": left["decision"]["prior_node"],
            "right_prior": right["decision"]["prior_node"],
            "left_full": left["decision"]["full_node"],
            "right_full": right["decision"]["full_node"],
            "same_legal_mask": left["decision"]["legal"] == right["decision"]["legal"],
        })
    pairs.sort(key=lambda item: item["distance"])
    qualifying = [
        item for item in pairs
        if item["same_legal_mask"] and item["distance"] <= threshold
    ]
    return {
        "standardization": "z-score over all collected state signatures",
        "distance": "RMS Euclidean distance",
        "threshold": threshold,
        "eligible_states": len(candidates),
        "eligible_pairs": len(pairs),
        "qualified_pairs": qualifying,
        "qualified_different_best": sum(
            item["different_best"] for item in qualifying
        ),
        "nearest_broader_pairs": pairs[:max_pairs],
    }


def evaluate(groups):
    return {
        "known_exp039": known_exp039(),
        "q_ranking": q_ranking(groups),
        "similar_state_pairs": similar_state_pairs(groups),
        "offline": offline_compare(groups),
    }

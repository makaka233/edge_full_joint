"""Offline action-return prediction with strict trajectory-level splits."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Sample:
    group: str
    state: int
    action: int
    features_a: np.ndarray
    features_b: np.ndarray
    reward: float
    centered_reward: float
    profile: str
    stage: str


def load_groups(directory):
    groups = []
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf8"))
        groups.append(record["group"])
    if not groups:
        raise FileNotFoundError("No state groups collected")
    return groups


def samples(groups):
    output = []
    for group in groups:
        for state_index, state in enumerate(group["states"]):
            rewards = [branch["reward_sum"] for branch in state["actions"]]
            center = float(np.mean(rewards))
            for branch in state["actions"]:
                output.append(Sample(
                    group=group["id"],
                    state=state_index,
                    action=int(branch["action"]),
                    features_a=np.asarray(branch["features_a"], dtype=np.float32),
                    features_b=np.asarray(branch["features_b"], dtype=np.float32),
                    reward=float(branch["reward_sum"]),
                    centered_reward=float(branch["reward_sum"] - center),
                    profile=group["profile"],
                    stage=group["stage"],
                ))
    return output


def split_groups(values):
    train = [group for group in values if group["seed"] in (20261113, 20261115)]
    validation = [
        group for group in train
        if group["seed"] == 20261115 and group["profile"] == "sustained070"
    ]
    train = [group for group in train if group not in validation]
    test = [group for group in values if group["seed"] == 20261117]
    return train, validation, test


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, value):
        return self.network(value).squeeze(-1)


def _stack(values, key):
    return np.stack([getattr(item, key) for item in values])


def fit_representation(train, validation, test, *, key, seed, epochs=400):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x_train = _stack(train, key)
    x_validation = _stack(validation, key)
    x_test = _stack(test, key)
    y_train = np.asarray([item.centered_reward for item in train], dtype=np.float32)
    y_validation = np.asarray([item.centered_reward for item in validation], dtype=np.float32)
    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = float(y_train.mean()), float(y_train.std() or 1.0)
    xt = torch.as_tensor((x_train - x_mean) / x_std)
    xv = torch.as_tensor((x_validation - x_mean) / x_std)
    xte = torch.as_tensor((x_test - x_mean) / x_std)
    yt = torch.as_tensor((y_train - y_mean) / y_std)
    yv = torch.as_tensor((y_validation - y_mean) / y_std)
    model = MLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xt)
        loss = nn.functional.mse_loss(prediction, yt)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.mse_loss(model(xv), yv))
        if value < best:
            best = value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(xte).numpy() * y_std + y_mean
    return predicted


def state_metrics(samples_subset, predictions):
    by_state = {}
    grouped = defaultdict(lambda: {"top1": [], "regret": [], "direction": []})
    for sample, prediction in zip(samples_subset, predictions, strict=True):
        by_state.setdefault((sample.group, sample.state), []).append((sample, float(prediction)))
    top1 = []
    regrets = []
    errors = []
    direction = []
    pressure_top1 = []
    for (_, _), rows in by_state.items():
        actual = np.asarray([sample.reward for sample, _ in rows])
        centered = np.asarray([sample.centered_reward for sample, _ in rows])
        predicted = np.asarray([value for _, value in rows])
        best = float(actual.max())
        selected = int(np.argmax(predicted))
        top1.append(bool(np.isclose(actual[selected], best)))
        group = rows[0][0].group
        grouped[group]["top1"].append(top1[-1])
        regrets.append(best - float(actual[selected]))
        grouped[group]["regret"].append(regrets[-1])
        errors.extend(np.abs(predicted - centered))
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                if abs(actual[left] - actual[right]) > 1e-8:
                    correct = (
                        (actual[left] > actual[right]) == (predicted[left] > predicted[right])
                    )
                    direction.append(correct)
                    grouped[group]["direction"].append(correct)
        if rows[0][0].profile == "sustained070":
            pressure_top1.append(top1[-1])
    return {
        "states": len(by_state),
        "actions": len(samples_subset),
        "mae_reward": float(np.mean(errors)),
        "direction_accuracy": float(np.mean(direction)) if direction else 0.0,
        "top1_accuracy": float(np.mean(top1)),
        "mean_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "sustained070_top1_accuracy": float(np.mean(pressure_top1)) if pressure_top1 else 0.0,
        "by_group": {
            group: {
                "top1": float(np.mean(values["top1"])),
                "regret": float(np.mean(values["regret"])),
                "direction": float(np.mean(values["direction"])) if values["direction"] else 0.0,
            }
            for group, values in grouped.items()
        },
    }


def compare(groups, repeats=5, epochs=400):
    started = time.perf_counter()
    train_groups, validation_groups, test_groups = split_groups(groups)
    train = samples(train_groups)
    validation = samples(validation_groups)
    test = samples(test_groups)
    result = {}
    for representation, key in (("original", "features_a"), ("extended", "features_b")):
        runs = [
            state_metrics(
                test,
                fit_representation(
                    train,
                    validation,
                    test,
                    key=key,
                    seed=seed,
                    epochs=epochs,
                ),
            )
            for seed in range(1, repeats + 1)
        ]
        result[representation] = {
            metric: {
                "mean": float(np.mean([run[metric] for run in runs])),
                "std": float(np.std([run[metric] for run in runs])),
            }
            for metric in (
                "mae_reward", "direction_accuracy", "top1_accuracy",
                "mean_regret", "max_regret", "sustained070_top1_accuracy",
            )
        }
        result[representation]["runs"] = runs
    result["split"] = {
        "train_groups": [group["id"] for group in train_groups],
        "validation_groups": [group["id"] for group in validation_groups],
        "test_groups": [group["id"] for group in test_groups],
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
    }
    group_ids = sorted(test_groups, key=lambda item: item["id"])
    rng = np.random.default_rng(20261124)
    deltas = {name: [] for name in ("top1", "regret", "direction")}
    for _ in range(2000):
        selected = rng.integers(len(group_ids), size=len(group_ids))
        for name in deltas:
            extended_value = np.mean([
                result["extended"]["runs"][run]["by_group"][group_ids[index]["id"]][name]
                for run in range(repeats)
                for index in selected
            ])
            original_value = np.mean([
                result["original"]["runs"][run]["by_group"][group_ids[index]["id"]][name]
                for run in range(repeats)
                for index in selected
            ])
            deltas[name].append(extended_value - original_value)
    result["paired_bootstrap_extended_minus_original"] = {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        }
        for name, values in deltas.items()
    }
    result["paired_seed_delta_extended_minus_original"] = {
        metric: {
            "mean": float(np.mean([
                extended[metric] - original[metric]
                for extended, original in zip(
                    result["extended"]["runs"],
                    result["original"]["runs"],
                    strict=True,
                )
            ])),
            "std": float(np.std([
                extended[metric] - original[metric]
                for extended, original in zip(
                    result["extended"]["runs"],
                    result["original"]["runs"],
                    strict=True,
                )
            ])),
        }
        for metric in ("mae_reward", "direction_accuracy", "top1_accuracy",
                       "mean_regret", "sustained070_top1_accuracy")
    }
    result["wall_seconds"] = time.perf_counter() - started
    return result

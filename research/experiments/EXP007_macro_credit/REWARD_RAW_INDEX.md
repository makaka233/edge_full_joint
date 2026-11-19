# EXP007 · Reward、时延和宏观动作

| 训练种子 | 原始训练全部 240 期 | 固定验证 | 独立 Holdout 逐期对照 | 需求—动作—Reward |
|---|---|---|---|---|
| 20261113 | [Reward / E2E](../../../evidence/macro_credit/EXP007/figures/train_20261113.png) | [固定 Reward / E2E](../../../evidence/macro_credit/EXP007/figures/fixed_20261113.png) | [独立 Reward / E2E](../../../evidence/macro_credit/EXP007/figures/holdout_20261113.png) | [期内对齐](../../../evidence/macro_credit/EXP007/figures/actions_20261113.png) |
| 20261114 | [Reward / E2E](../../../evidence/macro_credit/EXP007/figures/train_20261114.png) | [固定 Reward / E2E](../../../evidence/macro_credit/EXP007/figures/fixed_20261114.png) | [独立 Reward / E2E](../../../evidence/macro_credit/EXP007/figures/holdout_20261114.png) | [期内对齐](../../../evidence/macro_credit/EXP007/figures/actions_20261114.png) |

独立图的 1–8、9–16、17–24 属于三个不同的 8 期连续 episode，episode 之间边界不是一条连续物理队列。训练图按提交顺序画出配对分支的所有点，不把强制反事实当作实际在线门的动作频率。完整 [指标](../../../evidence/macro_credit/EXP007/metrics.json)、[原始日志](../../../evidence/macro_credit/EXP007/raw)、[历史尖峰后续期](../../../evidence/macro_credit/EXP007/historical_continuations.json)。

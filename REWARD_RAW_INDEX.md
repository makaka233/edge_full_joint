# 原始 Reward 图索引

基线图由 EXP001–003 原始数据生成，见 [原始索引](evidence/INDEX.md#双种子总览)。以下对照来自 EXP005，64 GS/臂；绝不以均值替代逐次提交。周期 arm 仅展示原始基线/重放，本轮未将周期自适应作为修复目标。

| 模式 | 20261113 | 20261114 |
|---|---|---|
| D-only | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_d_13.png) | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_d_14.png) |
| S-only | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_s_13.png) | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_s_14.png) |
| BUILD 联合 | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_joint_13.png) | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_joint_14.png) |
| 周期联合 | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_periodic_13.png) | [原始 Reward](evidence/spike_study/EXP005/figures/reward_raw_placement_periodic_14.png) |

[对照原始 JSONL](evidence/spike_study/EXP005/raw) · [全部原始 E2E 和固定验证图](evidence/spike_study/EXP005/figures) · [训练及两组 Holdout 指标](evidence/spike_study/EXP005/metrics.json) · [逐例反事实](evidence/spike_study/EXP005/counterfactual_summary.json)。重绘：`python -m scripts.spike_results --redraw evidence/spike_study/EXP005 --out runs/replot_exp005`。

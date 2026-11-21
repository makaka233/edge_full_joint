# EXP023 · 五种子 5000 GS 固定重部署

在 EXP019 瞬时矩阵切换基线上验证更长训练及跨种子稳定性。库存协调高斯 Agent-D、Adam、原 Masked Dueling Double DQN、业务负载和物理设置沿用基线。五个种子分别从随机初始化开始。每个种子训练 625 个 episode，每个 episode 包含 8 个 24 秒业务期，共 5000 个在线 GS。固定验证每 8 个 episode 执行一次，末次 GS5000 再执行一次。基线源码见 [EXP019 双种子长训提交](https://github.com/makaka233/edge_full_joint/commit/7681e015f0755f8dbbfd44b490f10d495f7cf130)，实际运行提交记录在各种子的 `manifest.json` 中。

## 启动与恢复

在项目根目录的 PowerShell 中执行以下一行。五个种子依次训练，全部达到目标后，程序自动完成各自 `best`、`last` 和相同 `last` 权重的固定动作对照，使用 EXP019 的三个自然负载 Holdout 场景。

```powershell
python -m edge_d.cli adaptive train --experiment EXP023 --seeds 20261113 20261114 20261115 20261116 20261117 --macro-mode redeploy --episodes 625 --periods 8 --period-seconds 24 --eval-every 8 --evaluate-after --prefix ".\runs\results\EXP023\instant.redeploy.long5000"
```

中断后在同一指令末尾追加 `--resume`。程序跳过已完成的训练，继续未完成的训练，再执行独立评估。

## 证据位置

所有产物写到 `runs/results/EXP023/`。公共前缀后加 `.seed<seed>`，例如 `instant.redeploy.long5000.seed20261113.train.jsonl` 和 `instant.redeploy.long5000.seed20261114.holdout.jsonl`。每种子的 `train.jsonl`、`updates.jsonl`、`fixed.jsonl`、`holdout.jsonl`、`best.pt`、`last.pt` 和 `resume_latest.pt` 均使用自己独立的前缀。

正式结果待本地运行完成后审核。

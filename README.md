# 协同边缘计算：Agent-D / Agent-S 研究工程

[从这里开始](docs/00_START_HERE.md) · [当前状态](docs/CURRENT_STATUS.md) · [研究交接](docs/RESEARCH_HANDOVER.md) · [复现](docs/REPRODUCTION_GUIDE.md) · [实验索引](research/INDEX.md) · [图表与数据](evidence/INDEX.md)

当前唯一训练主线为 `python -m edge_d.cli placement-train`。直接整矩阵 Agent-D 与原 Masked Dueling DDQN、原事件仿真器共同运行。本次整理没有改变部署、调度、物理容量或 Reward 的学习机制；历史 EDIT / REINFORCE 与 R12 单独归档。

尖峰研究（EXP005）：[根因](SPIKE_ROOT_CAUSE.md) · [受控反事实](SPIKE_COUNTERFACTUALS.md) · [修复与验收](SPIKE_FIX_REPORT.md) · [原始图](REWARD_RAW_INDEX.md)。默认仍为原基线；`--inventory-routing` 与 `--queue-residual` 是独立可追溯的实验开关。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m edge_d.cli placement-train --mode d-only --seed 20261113 --out runs/my_exp_d13
python -m edge_d.cli plot --out runs/figures_regenerated
```

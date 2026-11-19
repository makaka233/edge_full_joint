# 协同边缘计算：原始物理系统 + 原始 Agent-S + 改进 Agent-D 的完整研究工程

**本工程完整保留原始资源仿真和原始 Agent-S 算法，实现后来研究的 ADD/REMOVE/RELOCATE/FINISH Agent-D 与同一仿真器上的 D/S 联合训练。**
它是能够运行的研究工程，**不是**已经证明收敛的最终算法，也不是原始 R12 联合训练在信用设计上的完全等价实现。

## 开始

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.verify_allocator_modes
python -m edge_d.cli smoke
python -m edge_d.cli joint-smoke
```

**主入口** `python -m edge_d.cli`：

```powershell
# 原始 S（完整版 DDQN）+ 冻结当前 D 的训练
python -m edge_d.cli s-train --seed 20261113 --updates 64 --out runs/s
# 后续改进 D 的 BUILD-only 训练
python -m edge_d.cli train --seed 20261113 --updates 64 --out runs/d
# 改进 D 与原始 S 的真实联合训练
python -m edge_d.cli joint-train --seed 20261113 --updates 64 --out runs/joint
# KEEP/BUILD + D/S 长期研究性训练（不保证收敛）
python -m edge_d.cli periodic-train --seed 20261113 --updates 64 --periods 3 --out runs/periodic
```

`--updates` 是优化器更新次数，**不是**逻辑天/GS。请在正式训练前检查物理预算与构造终止风险。

**源代码导航：** `edge_d/` 改进后的部署构造与策略、`edge_s/agent.py` 原始 S 薄适配层、`edge_d/_engine/edge_rl/agents.py` 原始 S 网络与 DDQN 核心（逐字节保留）、`edge_d/_engine/edge_rl/replay.py` 原始年龄过滤回放（逐字节保留）、`edge_d/_engine/edge_rl/physics.py` 原始事件仿真/KKT、`edge_d/_engine/edge_rl/cpu_allocation.py` 原始 CPU 调度模式、`edge_joint/` 新联动训练编排、`reference/original_project/` 完整原始可追溯代码、`evidence/` 原始实验记录。

**请首先阅读：** [完整交接](docs/MASTER_HANDOVER.md)、[逐模块来源](docs/MODULE_PROVENANCE.md)、[物理分配合同](docs/RESOURCE_CONTRACT.md)、[限制与风险](docs/KNOWN_LIMITATIONS.md)、[运行与复现](docs/REPRODUCE.md)。

当前 benchmark CPU 分配默认 `finish_pf`（来自交接包配置），而不是 KKT。逐跳有线带宽始终使用原始 `sqrt(KKT)`；要验证 CPU KKT，可在 `configs/experiment.json` 将 `cpu_allocator` 改为 `kkt`，**注意这构成新的物理实验条件，不得与旧默认结果混为一谈**。

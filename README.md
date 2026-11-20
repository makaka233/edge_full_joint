# 协同边缘计算：原始物理系统 + 原始 Agent-S + 改进 Agent-D 的完整研究工程

库存协调 Agent-D 与原始 DDQN 的 BUILD-only 联合训练见 [EXP016 单文件交接](research/EXP016/AI_RESEARCH_REVIEW.md) 与 [连续训练复现](research/EXP016/REPRODUCE.md)。训练需求分布对照见 [EXP015](research/EXP015/AI_RESEARCH_REVIEW.md)。

初始调度探索与影子排空见 [EXP017 单文件交接](research/EXP017/AI_RESEARCH_REVIEW.md)。

连续部署的本地入口为 `python -m edge_d.cli adaptive`。见 [EXP019 运行指南](research/EXP019/README_RUN.md)及[项目与实验管理](docs/RESEARCH_WORKFLOW.md)。

低约束资源感知探索、双种子验收与 EXP016 续训见 [EXP018 单文件交接](research/EXP018/AI_RESEARCH_REVIEW.md)及[复现命令](research/EXP018/REPRODUCE.md)。

## 2026-11-19 研究交接：有界整矩阵策略

本轮新增 `placement-train`：随机初始化的条件部署策略直接生成完整合法矩阵，继续使用原始事件引擎、Reward 与 Masked Dueling DDQN。原 EDIT、原 S 独立入口和两个历史联合入口仍可运行。参见 [本轮实验报告](docs/PLACEMENT_RESEARCH.md)、[算法与训练协议](docs/PLACEMENT_METHOD.md)、[物理与 R12 审计](docs/TAKEOVER_AUDIT.md) 和 `evidence/research_20261119/`。下文的旧命令及非声明是交接基线说明。

```powershell
python -m edge_d.cli placement-train --seed 20261113 --gs 64 --mode d-only --out runs/placement_d_13
python -m edge_d.cli placement-train --seed 20261113 --gs 64 --mode s-only --out runs/placement_s_13
python -m edge_d.cli placement-train --seed 20261113 --gs 64 --mode joint --out runs/placement_joint_13
python -m edge_d.cli placement-train --seed 20261113 --gs 64 --mode periodic --periods 4 --fixed-scenes 2 --holdout-scenes 4 --out runs/placement_periodic_13
```

对种子 `20261114` 重复，输出路径后缀改为 `_14`。新实验的 `--gs` 是真实训练矩阵提交数（周期模式含 KEEP），不是参数更新数、连续物理周期或天。全部算法对照使用默认 CPU `finish_pf`。

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

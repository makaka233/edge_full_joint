# 源码来源与功能保真清单

| 范围 | 现行位置 | 来源与保真状态 |
|---|---|---|
| 原始事件仿真、资源事件时钟、无线、逐跳有线 KKT | `edge_d/_engine/edge_rl/physics.py` | 原始源码逐字节保留，active |
| 原 CPU 分配模式 `kkt/ps/finish_pf/fcfs/srpt` | `edge_d/_engine/edge_rl/cpu_allocation.py` | 原始源码逐字节保留，active |
| 原 S Dueling DDQN、可选结构化/辅助分支 | `edge_d/_engine/edge_rl/agents.py` | 原始源码逐字节保留，active |
| 原 SReplay 年龄过滤 | `edge_d/_engine/edge_rl/replay.py` | 原始源码逐字节保留，active |
| 原网络世界、节点与阶段、合法部署 | `_engine/edge_rl/{model,config,deployment}.py` | 原始源码逐字节保留，active |
| 原动态需求环境与 Reward | `_engine/{environment,dynamic}.py` | 原始算法，import path only 改动，active |
| 原部署切换成本 | `_engine/migration_cost.py` | 原始源码逐字节保留，但现行配置未必启用完整 migration |
| 改进后 D 编辑构造与 CPU FINISH | `edge_d/{policy,actions,rollout,structure,training}.py` | 后期研究版，active，未收敛 |
| 原 S 薄适配层 | `edge_s/agent.py` | 仅参数/观测校验/动作审计；不重新定义 DDQN 训练 |
| D/S BUILD-only 联合训练 | `edge_joint/training.py` | 新协调实现，调用原 S，非原 R12 算法无损迁移 |
| KEEP/BUILD 周期联合训练 | `edge_joint/periodic.py` | 新基线协调实现，尚未证明长期收敛 |
| 原始 R12 联合训练与宏观信用源码 | `reference/original_project/` | 隔离保留原始源码；不是当前主入口的调用依赖 |
| 历史学习曲线与失败样本 | `evidence/` | 保留关键原始日志与负结果，不混合为新训练数据 |

文件 SHA256 见 `SOURCE_PROVENANCE.json` 与 `ORIGINAL_PHYSICS_AND_S_SHA256.json`。`reference/original_project` 的原始代码依赖原本脚本的路径/配置，作为研究溯源而不是当前主运行路径，未承诺所有历史实验命令都可以无需历史模型运行。

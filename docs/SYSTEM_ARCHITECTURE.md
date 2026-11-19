# 当前源码分工

| 目录 | 职责 |
|---|---|
| [configs](../configs) | 原世界拓扑、需求、物理评价配置 |
| [edge_d](../edge_d) | 唯一主 CLI、环境适配、需求状态、整矩阵解码策略 |
| [edge_d/_engine](../edge_d/_engine) | 事件仿真、资源分配、动态环境和原始 S 核心 |
| [edge_s](../edge_s) | 原调度器适配；不替换核心 DDQN |
| [edge_joint](../edge_joint) | 统一四模式训练、GS 和评价数据采集 |
| [scripts](../scripts) | 只读重绘、物理/学习对照、发布和检验 |
| [research/history](../research/history) | 停用的部署/旧协调源码、原 R12 参考、历史文档及负结果 |

一次物理提交：部署/KEEP → 原始环境提交 → S 逐阶段收集转移 → 原事件引擎完成/截尾 → 原始 Reward → JSONL。配对/轨迹批结束才更新优化器；仅独立固定验证选模。见 [交接](RESEARCH_HANDOVER.md)。

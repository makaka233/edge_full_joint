# 独立研究员交接

最新研究：[EXP007 冻结 D/S 的连续队列宏观信用](../research/experiments/EXP007_macro_credit/README.md)。原物理、Reward 系数、DDQN 和默认开关保持冻结；EXP007 的加速周期是单独的连续评价，不重定义原短窗实验。上阶段 [EXP006 四组消融](../research/experiments/EXP006_generalization/README.md) 保留逐场景反例。

## 模型与主线

32 个异构节点、4 区、52 条无向物理链路、10 类串行服务（共 28 阶段）、12,000 个固定接入用户；随机请求与需求制度变化来自环境而非策略。D 决定阶段在节点上的二值部署；S 根据每次就绪阶段及实时资源竞争选择合法执行副本。队列时间、计算时间、AP 与多跳有线流量均由原事件引擎推进。系统目标是长期降低端到端时延、尾部和切换/持有代价，不是只让代理损失下降。见 [物理模型](PHYSICAL_MODEL.md)。

`edge_d/placement.py` 定义随机初始化的条件策略和 LP/MILP 解码器；直接交付覆盖且满足 RAM/SSD 的矩阵。BUILD-only 采用配对共享事件外生随机数的高斯得分函数更新，不拟合未验证的 Critic。周期模式使用 return-to-go 与 KEEP/BUILD 门，让旧部署和需求跨期延续，但不跨窗延续队列。原 S 由 `edge_s/agent.py` 调用 `edge_d/_engine/edge_rl/agents.py` 和 `replay.py`，保留在线/目标网络、合法掩码、经验过滤和 Polyak。协调器为 `edge_joint/placement_training.py`，唯一当前运行入口 `python -m edge_d.cli placement-train`。

## 如何读证据

先看 [当前状态](CURRENT_STATUS.md)，然后从 [实验登记表](../research/INDEX.md) 找每个问题的控制变量、结论和配置。逐次训练与逐次评价是不同日志；[证据索引](../evidence/INDEX.md) 可追到八条独立原图、固定验证和独立 Holdout。历史 EDIT 构造中止、信用泛化失败、尖峰/截尾和 R12 差分各自见 [历史主题](../research/history/INDEX.md)，无需先阅读旧交接。

## 下一位研究员

按 [复现指南](REPRODUCTION_GUIDE.md) 先验证归档、重画和逐模式烟测，再做新的独立实验。优先在更长时间窗和不同需求制度上预注册配对 KEEP/BUILD 对照；不改当前物理配置、Reward 或最终 Holdout 划分来掩盖退化。待确认真实适应后才比较原 R12 D80 信用、S 系统信用及持续队列；完整启用辅助 S 前先验证采集、回放和学习闭环。过去的多份交接原文仅作档案；阶段 ZIP 在本地 `dist/archive/`，不嵌入主交付。

# 完整项目交接（从未参与历史对话的研究员）

## 1. 研究目标与用户要求

构建异构协同边缘计算双时间尺度深度强化学习系统：Agent-D 长期决定全部分阶段服务副本的部署矩阵；Agent-S 对用户每一个就绪阶段选择实际执行边缘节点；确定节点后，由真实离散事件仿真分配计算与链路带宽，处理多跳传输、队列和最终端到端时延。关注长期累计 Reward 上升、平均请求 E2E 下降、尾部风险减少，最终双种子、未见需求场景、至少 64 个明确定义的宏观决策 GS 验证收敛。

当前研究重点是 **Agent-D 收敛问题**，不允许因此删除原 Agent-S、资源分配器、物理网络或 D/S 联合训练。严禁 PPO、预生成完整部署候选池、人为固定副本上限、事后按 Reward 选取提交矩阵、将启发式结果冒充强化学习结果、删除负尖峰。

## 2. 物理与问题规模

- 32 个异构边缘节点，4 区域，52 条无向物理边（按原模型派生有向逐跳带宽资源）；10 类串行服务，共 28 个阶段，12,000 用户，固定接入节点；请求按随机需求生成。
- D 输出 28×32 二值矩阵，要求全部阶段覆盖且 RAM/SSD 合法。后续新增 CPU 理想承载 LP 下界的 FINISH 门控作为**可选研究约束**，非通用充分可行条件。
- 现行配置 CPU 总计约 366.9185 CU/s，AP 上行速率约 60.7–118.1 Mbps。历史提出过 7200 CU/s、每 AP 300 Mbps 等目标，但 **不属于当前这套已验证 benchmark**；请勿混写。
- 计算/链路工作量与流量单位以 `configs/world.json` 和原始 `edge_rl/model.py` 为准。物理统计含 2×6s 观测、20s 真实预热与 8s 尾部观察；`failed` 实际指窗口**截尾**，不意味着永久丢包。

## 3. 真实模块构成

**D（研究版本）：** `edge_d/{policy,actions,rollout,coverage,structure,world,training}.py`，覆盖阶段 ADD，编辑阶段 ADD/REMOVE/RELOCATE/FINISH，CPU 可行性 FINISH 门控，终局 REINFORCE。已知对长构造信用/终止/独立泛化无法保证；尚未通过收敛验收。

**S（原始实现）：** `edge_d/_engine/edge_rl/agents.py` 内原始 `DuelingNetwork` / `SchedulingAgent` 和扩展分支，包括可选竞争资源特征、结构化阶段节点编码、局部辅助网络、double DQN、Polyak。`edge_d/_engine/edge_rl/replay.py` 是原始有界/年龄过滤 SReplay。`edge_s/agent.py` 只提供统一参数和调度调用适配，**没有重新实现 Q 网络和 TD 更新**。现行配置使用 24 维竞争资源特征。

**连续物理：** `edge_d/_engine/edge_rl/physics.py` 原始事件引擎、`cpu_allocation.py` 原始 `kkt/ps/finish_pf/fcfs/srpt` 选择。逐跳 wired 使用原始 `sqrt(KKT)`，无线是固定总容量共享，计算调度模式默认 finish_pf，可切换 KKT。`migration_cost.py` 原始成本核算可选；当前配置主要启用旧矩阵的 holding/switch 成本，不可宣称全部真实镜像迁移机制已建模。

**联合训练：** `edge_joint/training.py` 使用当前 D 与原始 S 对同一次真实物理评估分别更新；`edge_joint/periodic.py` 实现可运行 KEEP/BUILD 跨部署周期路径。新协调器不是原始 R12 连续 D80 信用算法的等价移植；原始完整源码留在 `reference/original_project/`。

## 4. 历史验证与负结果

- 历史原 ActionQ 同状态 FINISH/CONTINUE 独立动作排序存在错误；额外配对信用不能在双种子上产生可靠一致的 Reward 改善。
- 192 张合法完整矩阵中，34 张在理想 CPU LP 下界下满足 rho>=1，15 张在短窗没有观察截尾；严重负尖峰和持续承载不足有关，但截尾统计/固定惩罚也会放大 Reward 下降。
- CPU 可行性 FINISH 门控使小样本初始结果显著改善，**不等于学会了部署**。
- Seed13 的历史 EDIT PG 记录到更新 60 后构造不能自然 FINISH；Seed14 记录到 57，未完成 64 更新验收。构造步数增长是重要结构风险。
- 本次整合后，原 S 独立训练 1 更新得到 holdout Reward -12.0725（初始 -12.1569）；D/S 联合 1 更新得 -32.9209（初始 -12.1569）且 25 截尾。这些仅是 smoke，不能宣称统计显著。周期性训练 1 更新已进行，完整 terminal holdout 评价执行限时内未结束，详见 runs/archive/legacy/checks/。
- CPU 分配器同一请求流、同一矩阵验证：默认 finish_pf Reward -9.6413，原始 sqrt-KKT CPU Reward -11.8431，均无截尾，工作守恒误差约 1e-12。**不要选择较优模式冒充训练改善。**

## 5. 模块完整性与保真界限

保留了原始 `edge_rl/agents.py`、`replay.py`、`physics.py`、`cpu_allocation.py`、`model.py`、`config.py`、`deployment.py`、`migration_cost.py` 原始源码文件。`environment.py`、`dynamic.py` 仅将 import 调整为包内相对路径，实际算法不变。文件 SHA 与来源映射见 `docs/SOURCE_PROVENANCE.json`。

**注意：** 原始 R12 主训练/宏观信用/多场景信用、原始所有辅助算法，并没有被宣称在主入口无损迁移；其原始源代码作为独立 `reference/original_project` 保留，供后续研究员明确核对迁移目标。当前主入口运行的联合训练为“原始 S + 新 D”的联合训练协调器，不是原始旧 D/S 训练所有语义的逐字重现。也不能保证所有可选 S 信用模式可仅通过切换 JSON 激活完整训练：例如 local auxiliary 可能需要额外的局部经验更新编排；原始算法实现已保留，请在使用该模式之前单独测试对应的数据链。

## 6. 需要新研究员优先解决的问题

1. 保持原 S、物理/KKT 及数据语义不被再次简化；用独立哈希、守恒和 S TD 目标单测守护。
2. 重新研究能有效学习完整 28×32 矩阵的 D，缩短构造步数，研究有限步安全终止，不靠扼杀探索制造初始 Reward 优势。
3. 纠正当前 2×6s 观测的长期负载代表性与截尾惩罚解释；比较持续长请求轨迹下真实排队与短窗评价。
4. 核对原 S 扩展信用和新 D 的全局奖励目标；分别验证 S-only、D-only、固定 S 的 D、联合训练。
5. 恢复真正日级长期动态需求、KEEP/BUILD 成本与差分迁移，保证训练的宏观步/更新数/物理评价数分别计数。
6. 至少两个种子、>=64 有明确定义的宏观 GS，固定验证/完全独立 Holdout，逐点原始 Reward/E2E/截尾/真实失败、矩阵、副本、CPU LP、构造步数与墙钟，禁止删负峰。每运行仅保存 best/last 模型，交接 ZIP 不包含模型权重。

## 7. 研究交接输出要求

每轮无模型独立 ZIP 必须包含完整源代码、环境配置/依赖、原始 S 和物理/KKT、原始模型/研究历史可追溯源码、D/S 联合入口、完整项目说明、研究报告、负结果、原始图与日志、测试/哈希及复现命令。不得依赖本次以外的压缩包和聊天内容。必须标明实际训练/测试与未验证推测。项目文件结构改变必须不影响原始物理固定轨迹结果。

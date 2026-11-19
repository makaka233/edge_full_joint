# Agent-S 与 D/S 联合训练合同

## 调度动作

- 一个真实请求到达，先经历共享 AP 无线上传，然后按所属服务的阶段顺序进入 `Simulator.ready`。
- S 的输入为当前阶段针对 **32 个节点** 的特征矩阵（当前配置 `(32,24)`）、合法副本节点布尔掩码 `(32,)`；输出一个实际执行节点整数 `0..31`。
- 非法动作由仿真拒绝；Agent-S 采样与 Double DQN 计算均在掩码空间中执行。即使只有 1 个副本也能正常选择。
- 无线侧固定共享；逐跳传输、CPU 分配与带宽竞争继续由原物理引擎处理。

## 真实转移、Q 与经验回放

`edge_s.agent.SchedulingAgent` 继承并调用原始 `edge_rl.agents.SchedulingAgent`，使用原始两个 Dueling Q 网络：在线 Q 和 target Q；下一状态用在线网络在合法节点上选 `argmax`，用 target 网络计算该动作价值；真终止不 bootstrap；Huber 损失更新；Polyak 调整 target。Replay 直接使用原始 `edge_rl.replay.SReplay`，记录原生 `STransition`，同时有容量和版本年龄过滤，采样时不把 `censored` 等同真实被丢弃。S 的 `reward` 是仿真原生逐阶段耗时目标，并非 D 的宏观 Reward。

## 联合训练边界

BUILD-only：在同一次 `sample_layout` 物理运行上，D 采集完整部署构造 log_prob/终局 Reward，S 采集真实逐阶段转移；所有 rollout 采集完成才分别更新 D 和 S，避免同批内部策略漂移。`fixed` 只能选 best；最终 `holdout` 仅用于初始和末尾检验。

周期性：同一 `Env` 跨期保留 `old` 矩阵；第一期必须 BUILD；后续宏策略自己选择 KEEP/BUILD，BUILD 由 D 完整构造新矩阵，KEEP 不修改旧矩阵并继续处理下一期真实请求。每期持有/切换费用仍由引擎计算。D 采用整段周期折扣回报的 REINFORCE；S 仍消费原生阶段转移。

## 已知研究限制

1. CPU `rho*<1` 为长期平均需求下**必要条件**，不构成真实随机服务 SLA 保证；过载短时也不等于立即队列爆炸。
2. 20s 真实预热、2×6s 观测与 8s 尾部是当前基线的物理抽样协议，不可说是完整 86400s 天级模拟。
3. 长 D 构造可能循环并在 `max_actions` 后真实中止；训练明确失败，不伪造 Reward；需要研究有限步可达与安全 FINISH。
4. S 的 DDQN 只有局部阶段信用，单独优化可能使全局 Reward 下降；需要认真审查联合目标对齐。
5. 当前 READMe 命令是**运行/接口验收**，必须另做 ≥64 宏观学习检查点、双种子、独立 Holdout、原始 Reward/E2E/截尾全记录才能宣称收敛。

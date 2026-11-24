# EXP037 · Proposed 资源先验 DDQN 冷启动验证

## 研究问题

旧 EXP029 前 64 GS 五种子无极差 Reward。新 Proposed 请求率和系统容量沿用原口径，每请求 CPU 工作量翻倍后，随机 S 的局部派发会触发长队列。EXP034 的均匀探索仍留下反例。EXP036 提前优化缓解了多数种子，但无法改变 GS1，且 Seed16 在 GS3 的 Reward 为 −112.457。[EXP036 迁移归因](../EXP036/AI_RESEARCH_REVIEW.md)

## 预登记候选

使用原有 24 维 S 观测。以 (t_n = t_{mathrm{CPU,余量}}+t_{mathrm{CPU,在途}}+w/F_n+t_{mathrm{路径,独占}}+t_{mathrm{路径,积压}}) 估计候选节点的资源时间。所有量均可在派发之前从当前真实状态获得，单位为秒。该量没有模拟任务的实际完成时间。用原阶段回报的时延参考 (	au) 将它变为无量纲 (g_n=log(1+t_n/	au))。对合法动作取均值，最终 (Q_	heta=Q_{	heta,mathrm{网络}}+overline g_{mathrm{合法}}-g_n)。合法动作的先验均值为零，单一合法节点的先验为零。网络权重从原随机种子初始化，可通过残差超越物理估计。

在线贪婪、固定验证、Double DQN 的在线选动作和目标网络估值使用同一 Q 表达。原 ε 的均匀探索、合法动作集合、实际训练 Reward、Bellman 目标形式与 Adam 均沿用。与 EXP018 的行为混合不同，这一候选把当期资源估计写入值函数参数化，在训练前和之后都有效。没有直接调节物理容量、算量或到达率。

对照复现 EXP036 原 Q，候选只开启 `s_resource_q_prior`。随机初始化、逐 GS 更新、D、v2 新 Proposed 物理、到达率及同种子外生请求保持一致。五种子 20261113–20261117，各臂各 16 个真实在线 GS，2 次 D、128 次 S 优化。分 GS1–8 与 GS9–16，强筛选要求两段极差期和截尾均为零。匹配 GS16 `last` 冻结权重，另用同流的独立自然和持续 0.70 各一条 8 GS 连续轨迹，严格非劣沿用 Reward 差 ≥−0.03、完成 E2E 增 ≤1%、P99 和失败计数不增加。每档仅一条独立流，早期或平均收益不能证明长期可靠。

## 复现

从项目根目录运行。中断后重用同命令可续完成 16 GS 对照和短窗评估。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m scripts.exp037_run --workers 5
```

所有原始日志、模型、结果和图保存于 `runs/results/EXP037/`。来源 SHA 写入 `protocol.json` 和各训练清单。这里只运行 16 GS 有界对照。

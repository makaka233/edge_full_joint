# EXP040 · Agent-S 同状态动作价值审计

正式父提交为 EXP039 的 `123f2e4d383f0a4f1bdc6b91fcd4741ca8af9934`。本轮保持 Proposed 物理、原系统 Reward、Gaussian Agent-D 与带资源先验的 Masked Dueling Double DQN。EXP039 的 24 个探针中，13 个完整 Q 偏好动作损害群体 Reward。其中 7 个连事后 TD 也偏好该动作，另 6 个的目标方向已与当前 Q 相反。[EXP039 报告](../EXP039/AI_RESEARCH_REVIEW.md)

## 审计预登记

- Seed 20261113、20261115、20261117 各使用全新的一条自然需求与一条持续 0.70 请求流。GS64 冻结 D/S 生成真实事件前缀，在第二个 24 秒期初分叉，锁定目标矩阵和外生请求。
- 每种子负载组合选三个不同请求。候选必须在目标期前 12 秒、拥有 2–3 个合法节点。依次选最早的决策、拥有最高可见 CPU 工作压力的决策、最早出现 GS64 顶位翻转的决策，后一种无法选取时用时间最早者。选取只使用当时的观测及 Q。
- 逐一物理执行每个合法节点，其余调度使用同一 GS64 冻结策略。记录原 Reward、焦点多阶段回报、其他在途请求等待和下一状态 TD。用相同反事实标签衡量仅先验、确定性重建的同随机种子 GS0、真实 GS16/64 的 Top-1、Spearman 和 Reward regret。无可辨 Reward 的标签不进入 Top-1/Spearman。
- 审核完整 GS64 ring 的阶段、年龄、采集版本、同请求相邻阶段与选中节点队列覆盖。仅为诊断，以选中节点 CPU 待清空时间 0.5 秒分层，采样规则与正式训练无关。用同一 DDQN 复算组内 TD 误差。
- 若证据可操作，优先测试一种训练修正。审计与训练后的独立验收使用不同随机源。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m scripts.exp040_audit --workers 3
```

本地工件统一保存于 `runs/results/EXP040/`。

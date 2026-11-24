# EXP043 · 五种子 504 GS 长期联合训练验收

## 正式协议

源码父提交 `dd4fcfbe2412ad14a244950e72cf54614ed6f968`。基线为 EXP041 与 EXP042 A 的一步资源先验 Masked Dueling Double DQN。资源先验启用，均匀 Replay，逐 GS 更新 S，每 GS 八次梯度更新。保持现有 D Gaussian Actor、链路定价解码、滞后优势校准、Proposed v2 物理、原 Reward、24 秒业务期、瞬时 REDEPLOY 和探索时钟。两步 Watkins、LaBER 均关闭。训练新增代码只记录真实 GS0 快照、调度直方图与逐资源工作量。

- 训练使用 `20261113`–`20261117`，各自从真正随机初始化开始，63 个 8 GS episode，每种子 504 GS、预期 D/S 更新为 63/4032。恢复只接受本轮同配置 Checkpoint。`initial.pt` 只封存训练真实 GS0 权重、优化器、Replay 和 RNG 作为审计证据，正式模型仍为 `best.pt`、`last.pt` 和 `resume_latest.pt`。
- 固定验证沿用场景 `19001`、`19002`，在 GS0/64/128/192/256/320/384/448/504 评价，按原净 Reward 选 best。逐种子保留全部在线 Reward、E2E、极差、队列、物理守恒、部署汉明距离与节点派发分布。派发分布变化为相邻业务期的总变差，业务及可行节点变化也会影响该统计。best/last 的固定场景差值和参数距离另外报告。
- 独立冻结采用自然、突发、持续 0.55、持续 0.70 各两条全新外生轨迹，每条 8 GS，五种子共同评价。A 为 GS0 完整 D/S，B 为 GS504 完整 D/S，C 为 GS504 D 与仅资源先验 S，D0 为 GS0 D 与仅资源先验 S。A/D0、B/C 分别逐期共享矩阵。四臂均共用各自配对的请求流，D 用均值动作，S 为贪婪动作。B−A 为联合收益，B−C 为相同 D 下 S 的独立贡献，C−D0 用相同资源先验隔离 D 贡献。

## 预登记判据

**A 长期稳定性**要求配置及更新、逐期物理审计齐全，GS0 固定统计与 EXP042 A 一致且无 `Reward ≤ −20` 或截尾，五种子 GS504 固定均值相对 GS0 同时提高 Reward 和降低完成 E2E，且独立 B−A 至少 4/5 种子双改善。

后期持续退化定义为某一种子 GS448 和 GS504 在同一项指标上均较其 GS256 恶化，Reward 减少超过 0.30 或完成 E2E 增加超过 5%。尾部扩张比较等长的 GS1–120 与 GS385–504。严重负峰率上升超过 1 个百分点，或截尾、终局未完成、丢弃任一按请求计数的发生率上升超过 0.1 个百分点，或每期最大 CPU/有线积压的 P95 增幅超过 `max(1 CU 或 Mbit, 20%)`，并同时出现在至少三个种子及五种子均值上，判定为系统性扩大。

**B 双智能体有效学习**要求 A 通过，B−A 的独立平均 Reward/E2E 双改善，且 B−C 与 C−D0 分别达到平均 Reward 增益至少 0.03、完成 E2E 至少降低 1%。每项消融还须在至少 3/5 种子、3/4 负载、5/8 条独立轨迹发生双改善，并在两个持续压力档无平均惩罚后 P99、截尾、终局未完成或丢弃增加。轨迹跨模型共享，五种子 × 八轨迹的单元不可作为 40 条独立请求流。GS0 与 GS504 之外同时检查固定趋势，报告保留全部失败点和 best/last 差距。

## 一键复现

从项目根目录运行。默认串联正式训练、独立评价、汇总和绘图，失败后用同一命令自动从本轮已提交恢复状态继续。也可用 `--stage train`、`--stage evaluate`、`--stage report` 分步。运行总墙钟按 `timing.jsonl` 保存，阶段每种子训练计时保存在 Checkpoint 与汇总。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m scripts.exp043_run --workers 5
```

原始训练为 `runs/results/EXP043/baseline.seed<seed>.*`，逐独立配对为 `independent.<profile>.<ordinal>.seed<seed>.json`。本地保留全部大体积账本与模型。命令完成后正式报告的结果段自动更新。

交付前只做 2 GS 小对照。启用记录前后的请求、原始 Reward、完整矩阵、完成时延及优化账本一致，各完成一次 D 与 16 次 S 更新。新增资源账本通过守恒核验，GS0 快照含真实随机权重、空优化器历史和 RNG。该烟测与正式随机种子结果使用不同文件前缀。

### 从已完成的 GS504 恢复独立评价

首次正式运行完成了五种子训练。冻结评估覆盖了调度回调，导致训练用派发直方图未收到记录，原守恒检查因此报错。修复将冻结评估的派发统计接到实际执行节点，并分别核对逐期决策与含排空期的决策总数。训练 Checkpoint、物理、选节点逻辑与预登记协议保持原样。直接执行以下命令从独立阶段继续，成功后生成正式结论。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
python -m scripts.exp043_run --stage evaluate --workers 5
if ($LASTEXITCODE -eq 0) { python -m scripts.exp043_run --stage report }
```

## 实测结果

<!-- EXP043_RESULTS_START -->

五个种子的正式训练已完成。待独立评价与报告阶段结束后生成收敛结论。

<!-- EXP043_RESULTS_END -->

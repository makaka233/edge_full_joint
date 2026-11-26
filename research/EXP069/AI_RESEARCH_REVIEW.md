# EXP069 · 策略一致的相对价值学习

工程源码 `56fdc61fc58f691856dc5a1b002bb414773444fa`，父版本 `46d8c53b737c4aa5c3ea954335e3592e87d171f0`，登记 `00862022b8da51dbe4e090ec48df960db870537b673ab936c33f37670c3ac64d`。 [预登记](PROTOCOL.md)

EXP068 本地正式结果包含 9 个 72 GS 场景和 17 个物理起点。自然样本中 12/24 槽同向的 KEEP/REDEPLOY 各 2/2 个。旧固定八小时分叉只作历史诊断。

A 用固定两小时；B 用普通离散 SAC；C 的唯一区别是物理配对辅助。冻结 EXP061 D/S，保留 24 秒物理窗、原请求、部署、成本和 Reward。同种子的主轨迹外生键、Gate 初始化、优化器、三步软 Retrace、Replay 和更新预算配对。

C 对首动作分别 KEEP/REDEPLOY。随后按两个真实后继状态使用同一冻结 Actor 权重及配对随机数。有限期标签为跨槽修正折扣物理差加后续动作的熵收益差。最后计分槽再推进一槽至截止，首动作不计熵、不带终点 Q。该标签仍是截断且带版本滞后的 Monte Carlo 目标，长期 Q 可受到终点、采样方差和 Actor 后续变化影响。

已完成训练臂 0/6，独立场景 0/9。正式结果按已完成阶段展示。


三种子训练及独立评价完成后才能作出晋级判断。

## 复现与数据

```powershell
python -m scripts.exp069_run smoke
python -m scripts.exp069_run train
python -m scripts.exp069_run evaluate
python -m scripts.exp069_run report
```

原始文件在 `runs/results/EXP069/`。每臂保留 `best.pt`、`last.pt`、`resume_latest.pt` 与完整分叉、更新账本。

## 本轮工程核验

- 核对 EXP068 远端分支头和本地正式账本。9 个 72 槽场景、17 个配对点及两个方向各 2 个稳定自然点均已对账。历史标签不进入本轮训练。
- Seed20261113 隔离烟测真实推进 B/C 各 10 槽、各更新 18 次。两组在配对释放前具有相同事实执行轨迹。C 的分叉从 GS2 开始，七槽计分到 GS8，GS9 补结算后释放。GS10 的第 11 次更新首次使用，累计抽样 64 次。主轨迹共 85,685 个事件，16 个额外分叉槽共 136,531 个事件。
- 原始、修正物理、后续熵和软有限期 Δ 分别为 +0.015317、+0.017134、−0.000051 和 +0.017083。这是近似持平的烟测点。两臂原候选、初始队列及配对随机数均核验，保存恢复后的下一次动作和 Critic 更新逐项一致。
- 三臂各 6 槽独立烟测完成，B/C 各有一组附加物理诊断，原始曲线和三张图通过生成及读取。12 项定向回归覆盖原物理、请求信用和训练恢复。烟测主运行时钟有异常跳跃，墙钟原值保留，不以它估算正式作业耗时。
- [烟测完整报告](../../runs/results/EXP069/smoke-validated/AI_RESEARCH_REVIEW.md) · [训练原始曲线](../../runs/results/EXP069/smoke-validated/fig1-raw-training.png) · [优化器更新](../../runs/results/EXP069/smoke-validated/fig2-optimizer-history.png)

正式预算已经登记，三种子 B/C 每臂 144 槽与 286 次优化器更新，C 每种子额外 116 个分叉槽。独立评价包含 9 个场景且各臂 72 槽。本地尚无任何正式训练检查点或评价。由用户依次执行 `train`、`evaluate`，任一阶段结束后会自动更新本报告，也可手动运行 `report`。

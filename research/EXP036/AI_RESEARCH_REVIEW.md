# EXP036 · Proposed 五种子逐 GS 更新长训

## 研究问题

在 `comprised_exp1` 的较高计算负载下，将 Agent-S 的 64 次 episode 末 DDQN 更新分布到八个 GS 边界，能否缩短随机调度造成的局部拥塞，同时保持长期 Reward、平均完成 E2E 及独立持续负载的尾部可靠性？

EXP035 的 Seed14 同流小试中，首 8 GS 的极差期从 7 降到 1，截尾从 486 降到 0。首期仍由随机初始化 S 调度。EXP024 在旧环境也改善过初期风险，但持续压力的独立 P99 退化。EXP036 的长期评价分别核对两个现象。[EXP035](../EXP035/AI_RESEARCH_REVIEW.md) · [EXP024](../EXP024/AI_RESEARCH_REVIEW.md)

## 冻结协议

| 项目 | EXP036 |
|---|---|
| 训练对象 | 原 Proposed 多阶段服务，`compute_with_compact_payload` |
| 对照 | 只读的 `comparison/runs/proposed/comprised_exp1/seed<seed>/`，正式训练源码 `c108d7453bf740d8088ae701fa6f13a0684ba68e` |
| 候选 | `s_update_timing='period'`，设为 EXP036 默认 |
| 初始状态 | 两个智能体从各自种子的随机参数开始 |
| 种子 | 20261113、20261114、20261115、20261116、20261117 |
| 训练长度 | 每种子 625 个八期 episode，共 5000 GS。每 GS 24 秒 |
| 优化预算 | 每种子 625 次 D 更新，40,000 次 S 梯度更新预算 |
| S 采样 | 原 ε 与全局 GS 衰减时钟。每个 GS 内有效转移达 replay 门槛后使用本期预算，缺样时按原逐期机制延迟 |
| D 和物理 | 原链路定价、高斯 Actor、EXP029 滞后优势校准、合法矩阵、瞬时切换、原 v2 物理和 Reward |
| 固定验证 | GS0、每 64 GS、GS5000，按平均净 Reward 选 `best` |
| 独立评价 | 两个新外生随机源，各设自然、突发、持续 0.55 和持续 0.70，共 8 条 24 GS 连续队列。突发为第 9–12 期目标负载 0.70。两臂 `best`、`last` 分开冻结比较，S 为贪婪合法调度，D 用 Actor 均值 |

候选复用已存在的逐期 DDQN 实现。普通 GS 边界只消费已完成并可形成有效信用的转移，跨期任务继续推进。期末未完成的合成终止样本留在账本，逐期臂不会把它们用于学习。除了更新时间改变引起的回放可用性差异，两臂共享同一训练算法。EXP034 的均匀冷启动探索关闭。现有 MSP、历史 Proposed 和其他套件继续使用各自的协议。

主比较以同种子原始 Reward、请求加权完成 E2E 为先，同时检查前 8/64 GS 的极差与截尾、后期趋稳、最差 1%/5% 和独立 P95/P99、积压、终局未完成。查看 `last` 和 `best`，逐种子和独立请求轨迹保留反例。报告实际 S 更新次数，若回放不足造成预算未用完，部分结果需按实际预算解释。独立评价逐期核对请求哈希和工作量。每种负载两个独立外生随机源，同一轨迹由五个权重重复评价不能增加轨迹样本量。

## Windows 入口

在当前项目根目录，使用能运行原 Proposed 的 Python 环境。执行同一命令会自动从已有 `resume_latest.pt` 恢复，通过完整性核对后评估并绘制逐 GS 原始图。历史 Proposed 日志和 best/last 文件用于配对，保持本地可读。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m comparison.run --suite exp036 --workers 5
```

阶段可单独执行。独立评价需要正式训练完整结束。

```powershell
python -m comparison.run --suite exp036 --stage register
python -m comparison.run --suite exp036 --stage train --workers 5
python -m comparison.run --suite exp036 --stage evaluate --workers 5
python -m comparison.run --suite exp036 --stage report
```

轻量检查使用独立 `proposed.smoke.*` 前缀，候选只运行首个真实八期 episode。冻结部分仅用自然和持续 0.70 的四期窗口测试入口，历史参照仍是长训模型。

```powershell
python -m comparison.run --suite exp036 --smoke --workers 2
```

## 结果文件与版本证据

`runs/results/EXP036/` 保存 `proposed.protocol.json` 和 `proposed.seed<seed>.*`。每种子有 `train.jsonl`、`updates.jsonl`、`fixed.jsonl`、`manifest.json`、`best.pt`、`last.pt`、`resume_latest.pt`、`raw.png` 和逐轨迹独立评估。汇总写入 `proposed.summary.json`，含固定验证及每个种子、每个负载、每个检查点的配对评分。正式运行的实际 Git SHA 保存在协议和清单的 `source_history` 中，从本地结果读取。

## 已执行验证

在源码提交 `52335e3c95f66f82b6667743eaf705b6edcbd3eb` 上完成真实烟测。Seed20261113 跑完 8 GS，训练请求逐期匹配历史 Proposed，实际完成 1 次 D 和 64 次 S 优化，首次更新在 GS1 结束。自然和持续 0.70 的短窗口冻结评估通过同流和物理工作量核对，并成功生成原始图与汇总。再次执行烟测复用完整检查点和已有评估。两项跨期转移、预算与恢复回归和 21 项原对比模块测试通过。[烟测汇总](../../runs/results/EXP036/proposed.smoke.summary.json)

正式训练和长队列结果待用户运行后审核。现阶段只检验工程路径与短时物理推进。

# EXP035 小试 · Proposed 的 Agent-S 逐 GS 更新

## 假设与协议

Proposed 的 `comprised_exp1` 在默认训练中每 8 GS 集中做 64 次 Agent-S 优化。检验按每个 24 秒 GS 的边界推进同样预算，能否使随机初始化的调度器更早利用经验，减少初始 CPU 局部拥塞。复用 `edge_joint.adaptive_training.PeriodUpdates`。每个 GS 提供 8 次更新预算，回放足够才使用。普通业务边界不终止在途请求，按已完成的有效转移入库。D 仍每八期更新一次。两臂均使用原 ε，无首次更新前强制均匀探索。

运行源码为 `8d7e805b1d3db9c3a03aa1690d8af8603e7d38f9`。选 Seed20261114 作为已知严重启动案例，小试 16 GS。从随机参数训练逐期臂，复用同种子 EXP034 原探索基线。逐期外生请求哈希与 ε 相同。首 8 GS 矩阵全部相同。首个期末获得 729 条有效转移，实际执行 8 次 S 更新，GS2 用更新后的网络调度。16 GS 内无待用更新预算，两臂实际均为 D 2 次、S 128 次。没有更改物理、奖励或训练算法源码。[EXP034 基线及物理配置](../../comparison/proposed/cold_start_exploration/AI_RESEARCH_REVIEW.md)

## 真实训练观察

时延单位为秒。完成 E2E 按请求加权，P99 用惩罚后时延。CPU 和有线为各时段最大单资源期末积压。

| 区间与 S 更新 | Reward | 完成 E2E | 惩罚后 E2E | P99 | Reward ≤ −20 | 截尾 | 终局未完成 | CPU（CU） | 有线（Mbit） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GS1–8，原每 8 GS | −72.223 | 3.754 | 7.581 | 38.923 | 7 | 486 | 149 | 235.723 | 2.650 |
| GS1–8，每 GS | −11.195 | 1.051 | 1.051 | 6.804 | 1 | 0 | 0 | 21.872 | 2.650 |
| GS9–16，原每 8 GS | −11.416 | 1.138 | 1.138 | 3.366 | 0 | 0 | 0 | 9.793 | 3.050 |
| GS9–16，每 GS | −11.251 | 1.121 | 1.121 | 3.185 | 0 | 0 | 0 | 11.598 | 5.821 |

候选的唯一极差期是 GS1，Reward −23.44。第一次更新发生在 GS1 的物理边界后，GS1 的派发、节点和链路队列、部署都与基线相同。GS1 的请求评分窗口延长到第 32 秒，24 秒以后的调度仍可能改变首批请求的完成时延和 Reward。GS7 的 Reward 为 −105.02 → −8.58，截尾由 117 → 0。首次 D 更新后，GS9–16 的部署有 7/8 期不同，第二轮结果包含 D/S 联合学习效应。

同流、资源守恒和固定验证 ε=0 已检查。跨期经验 bootstrap 与派发版本归属、优化预算、恢复一致性的两项现有回归均通过。

## 判断

本次结果支持在 Proposed 中用逐期 S 更新缩短冷启动拥塞的持续时间。单种子是已知异常，16 GS 小试无法覆盖跨种子或独立压力表现。历史 EXP024 的旧环境曾出现相同的启动改善，但 512 GS 后在持续 0.55 和 0.70 的独立 P99 配对中全部退化，本次也观察到第二轮有线积压最大值增加。下一步用 Proposed 其余四个种子复查首 16 GS，将自然及持续压力的冻结 P99 保留为后续晋级条件。[EXP024 结果](../EXP024/AI_RESEARCH_REVIEW.md)

## 本地证据与复现

本地前缀为 `runs/results/EXP035/period.seed20261114.*`，含协议、原始训练、更新、固定验证、检查点和逐 GS 摘要。基线来自 EXP034 的 Seed14。以下命令在仓库根目录恢复或复现候选，不改写 EXP034 文件。[协议](../../runs/results/EXP035/period.seed20261114.protocol.json) · [逐 GS 摘要](../../runs/results/EXP035/period.seed20261114.summary.json)

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
@'
from dataclasses import asdict, replace
from pathlib import Path
import json
import torch
from comparison.proposed.cold_start_exploration.run import ROOT, MeasuredMultistageSimulator, configuration, environment
from edge_joint.adaptive_runner import train
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp027_run import fingerprint

seed = 20261114
original = json.loads((ROOT/'protocol.json').read_bytes())['plan']
cfg = replace(configuration(seed, 'baseline'), s_update_timing='period')
protocol = dict(experiment='EXP035', seed=seed, episodes=2,
                historical_protocol=fingerprint(original), config=asdict(cfg))
out = RunFiles(Path('runs/results/EXP035/period.seed20261114'))
torch.set_num_threads(1)
with run_lock(out):
    train(cfg, out, 2, resume=(out/'resume_latest.pt').exists(),
          world_factory=environment(original), simulator_factory=MeasuredMultistageSimulator,
          environment_id=fingerprint(protocol))
'@ | python -
```

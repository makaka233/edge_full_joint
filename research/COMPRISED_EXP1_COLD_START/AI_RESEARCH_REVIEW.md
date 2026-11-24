# Proposed 冷启动拥塞诊断与修复交接包

## 交接结论

`proposed` 是 `comprised_exp1` 的分阶段服务架构。新配置前 64 GS 有 24 次单期 Reward ≤ −20，其中 23 次发生在首次学习更新前。旧 EXP029 同期为 0 次。在两个重灾种子上，已精确重放旧配置和新主场景的首轮，并通过受控对照定位到**每请求计算量翻倍与随机初始调度器的节点集中偏好共同导致局部 CPU 过载**。固定原请求和部署后，关闭随机初始调度器的探索进一步恶化结果；提高首轮合法动作探索或替换为成熟调度器可大幅缓解这两例。成熟权重仅用于诊断。

建议第一项修复实验只改 `proposed` 的训练冷启动策略。S 在成功完成第一次优化器更新前使用均匀合法动作探索，首次更新后恢复原按全局 GS 计时的 ε 调度。本交接包包含实测证据、短重放入口和验证门槛。候选尚未接入正式训练，当前训练代码和结果仍是原基线。

## 1. 项目和版本

| 项目 | 当前协议 |
|---|---|
| 仓库 | `makaka233/edge_full_joint`，分支 `comparison-monolithic-service` |
| 原正式训练源码 | `c108d7453bf740d8088ae701fa6f13a0684ba68e`，见 `comprised_exp1.protocol.json` |
| 冷启动重放前代码 | `3ff13e036599201b19f7680c44bb9c31c97277ad` |
| 已记录的诊断摘要 | `094824c97c93322d5f459007b02f640d668261e7` |
| 实验 | `comprised_exp1`，架构内部标识 `multistage`，文件夹 `proposed` |
| 主场景 | `compute_with_compact_payload`，逐服务 v2 载荷目录 |
| 拓扑与资源 | 32 个异构节点、52 条物理连边、104 个独立有向队列、32 条无线接入队列。CPU 总容量 366.9185 CU/s，RAM/SSD、无线和有线容量沿用原配置 |
| 服务部署 | 10 类服务、28 个串行执行阶段。D 每 24 秒输出一张 28×32 合法部署矩阵，REDEPLOY 瞬时生效；已有任务按原指派节点推进 |
| 请求过程 | 3,500 个固定接入端点；全网名义 20 请求/秒，按同期旧世界的期望强度逐期校准。旧世界为 12,000 端点，两边的逐请求标识、位置和服务标记可能不同 |
| 实际负载变化 | 每阶段计算工作量 ×2，部署 RAM/SSD 需求保持原值。v2 的 18 条阶段间消息各按应用字段计算，大小 64–65,568 B。无线首段输入沿用原配置 |
| 资源分配 | CPU `finish_pf`，参考工作量 0.579849186388 CU；有线按剩余工作量平方根分配容量；同节点阶段不占用有线；无线对当前活跃请求等分 |
| 随机性 | 正式训练固定种子 `20261113`–`20261117`，各架构独立初始化；两架构在同种子同 GS 的外生请求哈希一致。跨 EXP029 与本轮只匹配期望强度和场景种子 |
| 训练与验证 | 每种子 5000 GS，625 个 episode，每轮 8 GS。625 次 D 更新，40,000 次 S 更新；每 8 个 episode 做固定验证，`best` 按固定验证净 Reward 选择 |
| 冻结评价 | 两条共同外生轨迹，各连续 24 期；两个架构各自的 `best`、`last` 均有记录。旧 EXP029 权重来自不同训练环境 |
| 软件 | 已登记 Windows、Python 3.11.7、PyTorch 2.11.0+cpu、NumPy 2.4.6 |

### 算法与代码位置

| 模块 | 实现与定位 |
|---|---|
| 实验入口 | `comparison/run.py` 的 `protocol()`、`run_files()` 与 `train_one()`，以配置指定世界，用原训练器运行分阶段分支 |
| v2 环境 | `comparison/edge_inference_profile.json`、`comparison/profile_audit.py`、`comparison/monolithic_service/environment.py`。`configure_workload()` 编译消息字节；`match_period_rate()` 保持期内期望到达强度 |
| D 部署 | `edge_d/placement.py` 的 `PlacementPolicy` 与 `decode()`。57 维原观测，57→32→8，标准差 0.5 的高斯策略，使用链路定价与容量协调。Adam 0.003，`γ_D=0.99`，每轮一更 |
| D 滞后优势校准 | `edge_joint/adaptive_training.py` 的 `update()` 使用前轮偏移，更新速率 1/16；`edge_joint/adaptive_runner.py` 从零初始化偏移，并在调用 `episode()` 采集完 8 期后执行更新 |
| S 调度 | `edge_d/_engine/edge_rl/agents.py`、`edge_joint/adaptive_training.py` 的 `episode()`。Masked Dueling Double DQN，观测 32×24，Adam 0.0003，`γ_S=0.99`，目标网 `τ=0.01`。每 GS 等价预算 8 个 minibatch，每轮末集中执行，batch 128、回放容量 25,000、最大龄 128 GS |
| S 原探索率 | 训练中 `ε = max(0.05, 0.25 - 0.20 × (GS-1)/511)`。首轮其余约 75% 的动作依赖随机初始化的 Q；冻结评价 `ε=0` |
| 计分 | `edge_joint/adaptive_training.py` 的 `request_metrics()` 和 `episode()`，每期 Reward 为 `−10 × 平均惩罚后时延（秒） − 持有成本 − 切换成本`。期末加 8 秒为观察截止，未完成请求按 `截止 − 到达 + 10 秒` 计分，截尾不等同于物理丢弃 |

## 2. 问题证据

阈值 `Reward ≤ −20` 仅用于诊断计数。两环境五种子首轮平均期望到达率均为 16.77 请求/秒，全网 CPU 提供负载从 15.62% 升到 31.23%。前 64 GS 均为 19.91 请求/秒，CPU 提供负载为 18.50% 和 37.10%。新环境 927 个首轮截尾集中在 Seed13、14、17，五种子都出现首轮极差点。剔除 Seed13 后，剩余四种子首轮仍有 18 个极差期。

| 种子 | EXP029 首轮最差 GS / Reward | proposed 首轮最差 GS / Reward | proposed 极差期 GS1–8 / GS1–64 | proposed 首轮截尾 |
|---|---|---|---:|---:|
| 20261113 | GS5 / −10.829 | GS5 / −61.588 | 5 / 6 | 86 |
| 20261114 | GS6 / −15.843 | GS7 / −105.018 | 7 / 7 | 486 |
| 20261115 | GS7 / −8.347 | GS2 / −21.543 | 2 / 2 | 0 |
| 20261116 | GS7 / −9.507 | GS2 / −20.511 | 1 / 1 | 0 |
| 20261117 | GS2 / −6.657 | GS2 / −108.242 | 8 / 8 | 355 |

EXP029 在五种子前 64 GS 最低为 Seed15 GS15 的 −17.063，所以这份证据表明旧配置的尖峰低于本次极差判定阈值。它并不建立跨配置的冷启动安全保证。全部新环境初期最差点均在 GS1–8，初次优势校准和 S 更新尚未发生。

## 3. 受控诊断

对两个严重种子重建随机 D/S，只重放 GS1–8 连续队列，未调用优化器。旧配置和新主场景逐期匹配训练日志的请求哈希、部署矩阵、Reward、完成 E2E 与截尾。四个 3,500 端点场景之间保持外生请求流和随机源一致，D 可响应当前算量与载荷观测，因此下表是从环境参数到物理结果的总效应。

| 场景 | Seed14 首轮平均 Reward / 截尾 | Seed17 首轮平均 Reward / 截尾 |
|---|---:|---:|
| EXP029 原配置 | −8.74 / 0 | −5.66 / 0 |
| 仅改 3,500 端点 | −8.66 / 0 | −5.99 / 0 |
| 3,500 端点，仅翻倍算量 | −67.22 / 456 | −70.60 / 360 |
| 3,500 端点，仅 v2 载荷 | −11.15 / 5 | −5.20 / 0 |
| v2 主场景 | −72.22 / 486 | −73.71 / 355 |

### 资源瓶颈

| 指标 | Seed14 GS7，节点 21 | Seed17 GS2，节点 25 |
|---|---:|---:|
| 原始 Reward / 惩罚后平均时延 | −105.018 / 10.489 s | −108.242 / 10.800 s |
| 已完成任务平均时延 | 4.333 s | 6.174 s |
| 当期节点已入 CPU 工作 / 24 秒处理能力 | 375.62 / 187.43 CU（2.00 倍） | 554.58 / 387.38 CU（1.43 倍） |
| CPU 利用率 / 期末待处理 | 100% / 235.72 CU | 95.65% / 186.61 CU |
| 队列首末在途请求 | 215 → 232 | 109 → 169 |
| 体现集中派发的阶段 | 阶段 5 的 58 个任务入节点 21，共 162.98 CU，部署候选为 1、21、26 | 阶段 8 的 49 个任务入节点 25，共 182.77 CU，该阶段有 19 个合法节点 |
| 部署覆盖回退 | `false` | `false` |

两期最高有线期末待处理工作分别为约 0.56 和 0 Mbit。节点工作量按 `compute` 事件进入实际 CPU 队列计量，含按上期选择继续完成的在途阶段。本次诊断支持局部 CPU 拥塞为主导机制。

### 固定请求和部署，只替换 S 行为

复用实际训练的八张矩阵及其请求序列，改变 S 的打分权重或 ε，连续推进物理队列。成熟 S 取同种子的正式 best 在线网络，保持原探索率。

| 启动行为 | Seed14 首轮平均 Reward / 截尾 | Seed17 首轮平均 Reward / 截尾 |
|---|---:|---:|
| 随机初始化，原 ε 规则 | −72.22 / 486 | −73.71 / 355 |
| 随机初始化，纯贪婪 `ε=0` | −103.69 / 746 | −115.30 / 719 |
| 随机初始化，均匀合法 `ε=1` | −12.93 / 4 | −11.12 / 0 |
| 同种子成熟 S，原 ε 规则 | −8.21 / 0 | −7.85 / 0 |

参考两例固定部署的结果，随机 Q 的偏好本身足够触发过载。成熟 S 在相同部署下能分散任务，说明这两例存在可利用的合法路由。仅均匀探索时 Seed14 仍有 4 个截尾和单期 Reward −24.01，效果边界需如实保留。以上均为首轮、两个已知重灾种子的重放，不能直接推断训练后的效果。

## 4. 下一轮修复方案与验收

**候选只改训练时 S 的冷启动动作选择。** 在 `edge_joint/adaptive_training.py` 中把它做成默认关闭的显式选项。训练处于在线、权重未冻结且配置启用时，只要 `s.updates == 0` 就设 `ε=1`，用现有 scheduler 在合法节点中均匀采样。计数首次增加后，ε 从当前全局 GS 继续原公式。测试/固定/独立评价继续使用原贪婪规则。每 GS 记录实际 ε、S version、优化器步数。候选沿用端点、请求、拓扑、阶段算量、部署合法性、Reward、截止时间、训练预算和原 D/S 损失函数。

### 建议执行顺序

1. **复现基线。** 使用第 5 节短重放，验证两个种子的旧配置和新主场景逐期对账。确认原 23 次初期极差统计可由五份原始日志重算。
2. **测试策略边界。** 选项关闭时，原五种子首轮请求、布局、奖励和调度行为与基线一致。选项开启时，更新前每个动作都合法，固定验证和独立评价 `ε=0`，恢复运行读取持久化的 `s.updates`。再检查首次更新前若训练采样数尚不够，均匀探索仍有效且不会阻止累计转移。
3. **做五种子首两轮同流对照。** 固定由现协议登记的五个种子和前两轮场景。基线和候选均允许实际 D/S 学习，保持初始权重、外生请求、种子及更新预算一致，记录 GS1–8 和 GS9–16 的极差计数、截尾、已完成平均时延及节点过载。把两例固定矩阵重放留作归因证据，独立于实训通过判定。
4. **预登记冷启动门槛。** 建议针对五个既有种子的首轮，将 23 个极差周期降至不超过 11 期，并使 927 个截尾显著下降。分种子核对已完成时延与截尾，检查是否把问题推迟到 GS9–16。阈值应在候选运行前登记，再预选独立初始场景复核泛化。
5. **做有界学习验证。** 通过冷启动门槛后再由用户运行同预算的五种子 512 GS，从随机初始化开始。核对原奖励/队列账本，并冻结检查实际选出的 `best` 与 `last`。重点衡量首轮风险、后续改善趋势和独立完成时延是否一起受益。候选跑独立实验标识和输出路径，保留原 `comprised_exp1` 及 MSP 基线。

**停止条件。** 五种子或新初始状态下表现不稳定、出现相当或更严重的积压、初期改善随后反弹或固定验证/独立评价退化，都应停留在诊断阶段。若均匀探索无法稳定过关，再用节点候选数、当前排队量和工作注入证据决定是否另立按可观测队列分散的候选。候选保留原 CPU 时序、Reward 标尺和全部样本。正式长训只有在用户授权后启动。

## 5. 自带短重放命令

从项目根目录运行下列 PowerShell 段。默认选择 Seed20261114，程序只做首轮 8 GS 重放并打印统计。复现 Seed20261117 时替换 `--seed` 末尾参数。`--scheduler-checks` 追加三组同部署 S 对照，其中成熟权重检查要求本地有正式 `best.pt` 和原协议。阶段模型和训练模型通过仓库源码创建，前两种环境对账需要原训练日志。命令不会调用优化器或写入原结果。

本文件内嵌脚本已用 Seed20261114 和 `--scheduler-checks` 执行，逐期一致性断言通过，8 组统计与第 3 节相符。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
@'
import argparse
import json
from itertools import islice
from pathlib import Path
from statistics import mean

import torch

from comparison.monolithic_service.environment import MeasuredMultistageSimulator, world
from comparison.profile_audit import registered_profiles
from comparison.run import load_projection
from edge_d.world import stable_seed
from edge_joint.adaptive_training import RolloutControl, components, episode
from scripts.exp019_report import unpack
from scripts.exp029_long import config


def head(path):
    with path.open('rb') as file:
        return list(islice(map(json.loads, file), 8))


def show(name, rows):
    worst = min(rows, key=lambda row: row['reward_raw'])
    print(name, 'mean_reward=', round(mean(row['reward_raw'] for row in rows), 3),
          'censored=', sum(row['censored'] for row in rows), 'worst_gs=', worst['gs'],
          'worst_reward=', round(worst['reward_raw'], 3), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, choices=(20261114, 20261117), required=True)
parser.add_argument('--scheduler-checks', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
seed = args.seed
cfg = config(seed)
scene = stable_seed('EXP019', seed, 'train-scene', 1)
root = Path('comparison/runs')
proposed = root/'proposed'/'comprised_exp1'/f'seed{seed}'/'train.jsonl'
exp029 = Path('runs/results/EXP029')/f'formal.calibrated.gs5000.seed{seed}.train.jsonl'
archives = {'old': head(exp029), 'compute_with_compact_payload': head(proposed)}
_, profiles = registered_profiles()
fields = ('trace_sha256', 'active_matrix', 'reward_raw', 'e2e_completed', 'censored')
for name, profile in [('old', None), *profiles.items()]:
    env = world(seed, scene, 'train', profile=profile)
    d, s, _, _, gate, _ = components(cfg)
    rows = episode(cfg, env, d, s, gate, scene, 'train', 1, d_audit={},
                   simulator_factory=MeasuredMultistageSimulator)[0]
    if name in archives:
        for row, saved in zip(rows, archives[name], strict=True):
            assert all(row[field] == saved[field] for field in fields), (name, row['gs'])
    show(name, rows)

if args.scheduler_checks:
    candidates = (root/'comprised_exp1'/'comprised_exp1.protocol.json', root/'comprised_exp1.protocol.json')
    path = next(path for path in candidates if path.is_file())
    plan = json.loads(path.read_bytes())['protocol']
    _, best = load_projection(plan, 'multistage', seed, 'best')
    name = 'compute_with_compact_payload'
    actual = archives[name]
    for label, epsilon in [('mature_S', None), ('greedy_untrained', 0.), ('uniform_legal', 1.)]:
        env = world(seed, scene, 'train', profile=profiles[name])
        d, s, _, _, gate, _ = components(cfg)
        if label == 'mature_S':
            s.online.load_state_dict(best['s_online'])
        layouts = [unpack(env, row['active_matrix']) for row in actual]
        rows = episode(cfg, env, d, s, gate, scene, 'train', 1, recorded_matrices=layouts,
                       simulator_factory=MeasuredMultistageSimulator,
                       control=RolloutControl(epsilon=epsilon))[0]
        for row, saved in zip(rows, actual, strict=True):
            assert row['trace_sha256'] == saved['trace_sha256']
            assert row['active_matrix'] == saved['active_matrix']
        show(label, rows)
'@ | python -X utf8 - --seed 20261114 --scheduler-checks
```

## 6. 原始产物和边界

核心数值和复现方法已含在本文件。下列本地记录用于进一步追查，每个 `seed<seed>` 含原训练日志、固定验证、更新账本、独立流日志及 `best.pt`、`last.pt`、`resume_latest.pt`。

- `comparison/runs/proposed/comprised_exp1/seed<seed>/` 中的 `train.jsonl`、`manifest.json`、`updates.jsonl`、正式检查点。
- `comparison/runs/comprised_exp1/comprised_exp1.initial_diagnosis.json` 中的逐期重放、部署矩阵、负载、截尾及调度对照数据。
- `comparison/runs/comprised_exp1/comprised_exp1.protocol.json` 与 `comprised_exp1.review.json` 中的正式实验协议和汇总。
- `runs/results/EXP029/formal.calibrated.gs5000.seed<seed>.train.jsonl` 中的历史基线。

项目环境机制、完整资源口径与比较结果见 [原实验综述](../COMPARISON_EDGE_INFERENCE/AI_RESEARCH_REVIEW.md)。正式训练代码见 [源码版本](https://github.com/makaka233/edge_full_joint/commit/c108d7453bf740d8088ae701fa6f13a0684ba68e)。

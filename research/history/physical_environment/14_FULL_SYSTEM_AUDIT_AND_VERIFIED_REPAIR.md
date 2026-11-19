# Agent-D 全链路工程审阅、隐藏问题与已验证修复（2026-11-19）

## 0. 严格结论

**本轮没有发现已经收敛的 Agent-D 算法，也没有把训练尖峰变小当作成功。** 发现并重现实证一个会使“不同训练种子/独立人口泛化”判断失真的底层实验协议缺陷：`previous_experiment.world(seed,scene)` **实际忽略了 `seed` 对人口生成及需求状态初始化的影响**。在相同 `scene` 下，两个训练种子的用户接入、需求、上下文完全一致；`seed+101` 的所谓独立验证也没有独立人口。前几轮实验的 Actor 初始化和请求仿真随机流仍有随机差别，因此不能说所有重复试验完全相同。

已新增**可选择、可重现的训练/验证/留出三路独立人口与需求流**，保留原 `world()` 及所有历史工程，物理资源/调度/Reward/合法部署完全不动。修复后在两种子 4 次更新对照中，独立 Holdout 只在一颗种子改善，未通过 RL 收敛验收。该修复提升实验真实性，**不是声称训练已修复**。

## 1. 本轮代码审阅范围与证据路径

| 审阅域 | 关键源码/证据 | 结论 |
| --- | --- | --- |
| 环境、需求、人群 | `previous_experiment.py:world`, `dynamic.py`, `train.py:environment` | **P0 实验种子缺陷，已新增修复路径** |
| 拓扑与物理参数 | `configs/world.json`, `configs/experiment_r0_joint.json` | 模型与最初研究规格存在数值/机制差异，**不擅改** |
| 事件仿真、资源竞争 | `edge_rl/physics.py`, `edge_rl/cpu_allocation.py` | CPU 是 finish_pf，链路 sqrt-work KKT；当前只有边资源，没有独立 NIC 总量池 |
| 短窗和 Reward | `environment.py`, `audit/r2a_sampling.py` | 两个 6 秒观测窗，6 秒预热、8 秒尾；失败是尾部截尾加惩罚，不是完整日级时延 |
| 合法部署 | `environment.Env.legal`, `native_legal_edit_prototype.py` | 覆盖、RAM、SSD 与原子 RELOCATE 保持；原始 BUILD 不事后投影 |
| Actor、FINISH、Critic | `edit_policy_study.py`, `finish_global_study.py`, `action_cv_study.py`, `paired_policy_repair.py` | Q 主要用作方差控制；终局信用难以区分具体编辑；历史正负结果在包内 |
| 验证 | `hierarchical_results`, `world_protocol_results`, `deep_physics_audit.json` | 过去主要检验同一人口、改变 Actor 和请求种子；现在新增未见人口和需求 |
| 联合 D/S | `r12/joint.py`, `edge_rl/agents.py` | 本轮是 BUILD-only，固定启发式 S，不是联合收敛验证 |
| 独立交接完整性 | 全 ZIP 源码、日志、测试、SHA256 | 新包自包含；历史 SHA 账本归档；源包旧 `PACKAGE_FILE_SHA256.json` 实测对 `00_START_HERE.md`、`edit_policy_study.py` 有 2 处哈希不一致，新的根目录账本完整重新生成并独立验收 |

## 2. P0：实验世界种子未独立

历史 `world(seed,scene)` 先通过 `setup(seed,'A')` 读入**固定人口种子 5101**，再调用 `e.reset(840000+scene*103,0)`；重置的随机源与传入的 `seed` 无关。示例直接检查 `scene∈{0,7,20,21,70,71}`：

- 不同 `seed` 的 `user_access`、`node_demand`、`context` 全部逐元素相同。
- `seed+101` 对以上三类也毫无改变。
- 这不会让两个算法在不同 `policy_seed` 下生成相同矩阵，也不会取消不同 `physical_seed` 的到达噪声；问题是**需求分布与人口泛化被高估**。

### 具体修复

`independent_world_protocol.py:make_world(seed,scene,split)` 通过 SHA256 稳定生成与种子、用途、场景绑定的随机源：

- `train`：训练人口+训练 Markov 状态；
- `fixed`：同一训练人口，但全新的需求状态；
- `holdout`：独立的人口接入分布、用户分组、Markov 需求状态；
- 相同输入可复现；不同训练种子、`train/holdout` 不会共享同一人口；
- 所有网络节点、链路、CPU、RAM、SSD、无线、服务工作量、物理 Reward、合法性规则保持不变；
- 绝不选择 Reward 较好的随机世界或完整部署作训练动作。

`versioned_edit_rollout.py` 是只替换世界构造的一份**明确版本化 BUILD rollout**。历史 `edit_policy_study.py` 不被改写，旧结果仍可按旧协议重放。运行 `world_protocol_training.py` 是显式启用新实验协议，而不是静默修改所有现存脚本。

测试：`python -m unittest -q test_independent_world_protocol` 共 **7 个**定向测试；顶层全部 `test_*.py` 合计 **73 个**测试通过。

## 3. P0/P1：物理目标与真实模型不一致／应区分模型选择与缺陷

**当前实际物理模型**：32 节点、52 无向边/104 有向链路资源、10 服务、28 阶段、12,000 用户、总 CPU **366.9185 CU/s**、AP 容量 **60.7076–118.0697 Mbps**、每节点 RAM/SSD 硬约束、CPU **finish_pf**、有线链路**平方根 KKT**。链路不按独立 NIC 共享总容量再次竞争。

原研究规格的 **7200 CU/s、每 AP 300 Mbps、CPU 平方根 KKT** 与当前数值/机制不同。CPU 单位可能已重标定，**不能直接按数值比判定过载或偷偷改成原值**；必须在下一次物理模型合同中确认单位与模式。`cpu_rates([1,4,9],12)` 在 `finish_pf` 下分配 `[7.5,3,1.5]`，平方根 KKT 则为 `[2,4,6]`，分配规律具有本质差别。

`regime_evolution=true` 会将当前需求的日内振幅设为 **0**，虽然原世界配置 `daily_amplitude=0.7`。因此，此版本有随机 Markov 跨期场景变化，但**没有原本预期的日内正弦到达曲线**，同时状态上下文仍含日内 sin/cos。模型没有同时实现两个机制。

本轮没有改动以上任何物理选择，以防把物理模型替换伪装成 RL 的学习收益。所有实测配置可由 `audit_model_contract.py` 生成的 `deep_model_contract.json` 复核。

## 4. P1：训练样本、短窗与 Reward 有效信号

该 BUILD-only 环境每次物理评价只使用**两个 6 秒 cohort 窗口**，另有每窗 6 秒预热、8 秒观察尾部；并非连续跑完 7200 秒长期区间或保持真实跨窗积压。未在尾部完成的请求按截尾年龄加失败惩罚计入 `penalized_delay`。`latency_objective=period_mean`，原 `Reward≈-10 × penalized_e2e - 小额持有/切换费用`，不是独立的“服务部署收益评分”。这解释了大多数正常样本 Reward 密集及严重失败造成尺度压缩，不能通过截尖峰解决。

原先的 Actor 每更新只从单一需求状态得到 6 个完整轨迹的终局物理 Reward，虽有数百个内部动作的 `log_prob`，却不等于获得几百个独立动作价值标签。早前训练包含真实尖峰、Actor 也确实改变矩阵，但多种信用方法未形成两种子一致的 Reward+E2E 改善；这一负结果应继续保留。

## 5. 原始轨迹：资源必要承载下界与灾难性 Reward

上轮 `hierarchical_results/train.jsonl` 的 **192** 条完整真实物理轨迹（原扁平 96、阶段分解 96）全部保留：

| 项目 | 原扁平 96 | 阶段分解 96 |
| --- | ---: | ---: |
| `cpu_stage_pair_bound` 与 Reward 的样本相关 | -0.8887 | -0.8470 |
| 出现失败请求的轨迹 | 8 | 11 |
| 总失败请求 | 411 | 929 |
| 无失败轨迹平均 Reward | -6.933 | -7.376 |
| 有失败轨迹平均 Reward | -46.819 | -64.918 |

必要 CPU 下界大于 1 不被用于硬掩码，而是用于审计；相关性不是因果证明。极端失败与布局承载瓶颈高度相关，说明不能把尖峰仅解释为图表尺度或纯随机噪声。

## 6. 本轮实际新增物理验证

### 6.1 同一完整矩阵在不同需求环境的真实评价

固定一张**事先取自历史 Holdout 的合法部署矩阵**，不筛选结果、不给 Actor 作为候选动作，在两个种子下使用相同物理种子分别重新执行旧世界与独立 Holdout：

| 程序配置 | Reward | 惩罚 E2E | 失败请求 |
| --- | ---: | ---: | ---: |
| Seed13 旧世界 | -6.4871 | 0.6374 s | 0 |
| Seed13 新独立 Holdout | -39.8943 | 3.9782 s | 31 |
| Seed14 旧世界 | -7.5088 | 0.7396 s | 0 |
| Seed14 新独立 Holdout | -9.7901 | 0.9677 s | 0 |

每个配置原位重复执行两次，验证相同输入奖励和请求轨迹哈希一致。跨世界比较**不是同一真实请求轨迹的动作因果比较**，只说明环境外推有实际风险。全部 6 个世界配置及重复检验在 `deep_physics_audit.json`。

### 6.2 等物理预算双种子短训（真实实验）

**对照目标只检验世界随机化修复的影响，不宣称 Actor 算法升级。** 两组都是相同 `EditActor + REINFORCE + LoadAwareHeuristic`，相同初始化、优化器、合法性、物理开销；唯一训练差别是世界工厂：历史固定人口世界 vs 独立训练人口/需求世界。最终两组都在**同一个独立 Holdout 人口/需求**上用相同的采样种子、物理随机种子比较。

双种子 20261113/20261114，每种子每组 4 次 Actor 更新，每次 4 条完整训练物理轨迹，共计 **64 条训练轨迹**；在更新 0/2/4 做独立 Holdout，每点 4 条轨迹，共 **48 条评价轨迹**。无模型文件。注意这不是 GS；每个检查点仅 4 条物理评价，统计置信度低。

| 种子 | 训练世界协议 | Holdout Reward：更新0 → 更新4 | Holdout E2E：更新0 → 更新4 | 更新4失败请求 |
| --- | --- | ---: | ---: | ---: |
| Seed13 | 历史固定世界 | -17.9364 → -20.1668 | 1.7811 → 2.0039 s | 29 |
| Seed13 | 新独立世界 | -17.9364 → -11.4439 | 1.7811 → 1.1318 s | 0 |
| Seed14 | 历史固定世界 | -20.4998 → -31.0259 | 2.0368 → 3.0881 s | 157 |
| Seed14 | 新独立世界 | -20.4998 → -30.5260 | 2.0368 → 3.0380 s | 163 |

**结论**：初始 Holdout 完全相同，真实训练后矩阵和结果会发生改变；但 Seed14 的两组都出现明显退化与大量失败，修复协议不能直接带来可靠收敛。任何种子特定的大幅改善均不能替代独立复现。完整无平滑/无尖峰删改原始 Reward 和 E2E 曲线在 `world_protocol_results/figures`；逐条物理事实在 `train.jsonl`、`eval.jsonl`、`updates.jsonl`。

## 7. 审阅中识别但尚未修复的开放风险

1. **独立环境泛化**：上一轮双种子的世界相同，本轮已修正环境生成协议；仍需更多需求种子与完整长期测试才能估计可靠性。
2. **资源模型合同**：CPU 资源量、AP Mbps、CPU 分配算法与最初设计不同；若要求恢复原规格，必须单独版本化并开展物理重新标定、验证，不应与 Actor 修复混在同一实验。
3. **单动作信用**：原 ActionQ 曾在 12 组独立 FINISH 配对中排序失败；拟合已见状态并不等于 Q 泛化。当前新实验并没有悄悄恢复这个不可信 Q。
4. **高方差的长动作链**：几十到几百步的构造仅从 4–6 个终局评价学习，罕见失败可能主导更新；不能把 `log_prob` 总数误算为独立物理反馈数。
5. **终止和可编辑循环**：FINISH 可自主采样，但偏好监督曾出现 1100 步不终止；应在不强迫 FINISH、不修改 Reward 的前提下研究有限序列策略或有限步动作语义。
6. **长期物理未闭合**：当前独立 BUILD 旧布局为零，既没训练真实 KEEP/BUILD，也没模拟镜像预热、请求连续性、跨窗积压；迁移成本开关缺省为空。缺少 NIC 汇聚共享约束。
7. **调度联合学习未覆盖**：固定启发式 S 能帮助定位 D，但不能证明 D/S 联合可从零学习。
8. **评价样本不足**：4 次独立 Holdout 物理轨迹/检查点只能做烟测，不足以判定奖励总体上升、E2E 总体下降、尾部风险收敛。

## 8. 下一轮优先任务（可证伪的修复顺序）

A. 先冻结当前物理参数合同，升级评价到跨人口/跨 Markov 场景的分层独立抽样，保证固定 Holdout、训练集和测试集隔离；所有原始 Reward、E2E、失败、矩阵和训练梯度原样记录。

B. 再做**动作价值排序与资源可承载性结合**：分层采样正常、边界、高风险部分状态；按同一人口和请求物理流配对检验具体 ADD/FINISH/RELOCATE 的条件回报。保留正负优势、无标签泄漏，训练集和未见场景均需正确排序。CPU 下界是状态特征/审计，而非硬约束或 Reward 替代。

C. Critic 若过不了独立排序验收，不可作为 Actor 主更新的唯一依据；比较等预算的真正 Q 驱动离散策略改进、原 REINFORCE 和物理配对梯度，记录动作概率变动、布局变化、原始 E2E/Reward；禁止 PPO、候选完整布局池、任何人工副本上限或基于 Reward 替换执行布局。

D. 通过双种子 32/64 次以上真实 Actor 更新与独立分层评价后，才能准备完整日级 S/D 联合训练。禁止把 BUILD-only Actor update 误称 GS。

## 9. 复现命令（仅依赖本 ZIP）

在解压根目录：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -q
python audit_model_contract.py
python deep_physics_audit.py --out deep_physics_audit_replay.json
python world_protocol_training.py --out independent_world_replay --seeds 20261113 20261114 --updates 4 --rollouts 4 --eval-scenes 2 --eval-draws 2 --eval-every 2
# 已有全部数据如需重画，默认读取 world_protocol_results
python analyze_world_protocol.py
python verify_package_files.py
python validate_handover.py
```

**保留历史重放**：`python hierarchical_training.py --help` 和之前报告的 8 更新命令仍使用旧 `world()`，其结果不可与新世界结果直接视为同分布对照。原 `PACKAGE_FILE_SHA256.json` 在 `HISTORICAL_PACKAGE_FILE_SHA256.json` 中留存，新的 SHA256 对应本轮完整交接内容。

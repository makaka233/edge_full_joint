# 阶段 A/B：无损接手与物理审计

## 来源与边界

2026-11-19 在工作目录获得解包后的交接树，未找到同名 ZIP。首次运行前逐个核对 `SHA256_MANIFEST.json`，216/216 字节、大小均匹配。最初无 `.git`；远端无引用。原始参考代码、历史负结果独立保留。原始 `edge_rl/{agents,replay,physics,cpu_allocation,config,model,deployment}.py` 由测试逐文件比对原始哈希。

| 路径 | 类型 | 输入 → 输出 / 保留边界 |
| --- | --- | --- |
| `configs/`, `world.py`, `_engine/dynamic.py` | 原模型 + 实验适配 | 拓扑、固定用户接入、分群偏好、按区域和服务变动的需求 → 场景状态与分离的请求随机流。`population_seed` 文件值不是 `make_world` 实际传入值，后者以 seed/split 派生。 |
| `_engine/edge_rl/physics.py` | 原样保留 | 到达 → AP 上行 → S 选择阶段副本 → 固定路由逐跳有线 → CPU → 下一阶段/完成。Registry 变动前结算旧速率，以 generation 跳过陈旧完成事件。 |
| `edge_s/agent.py`, `_engine/edge_rl/{agents,replay}.py` | 薄适配 + 原样保留 | 当前请求、候选点资源竞争等 24 列特征、合法掩码 → Masked Dueling DDQN 选择；回放中保留合法动作、下一状态、终止与版本，在线网络选动作，目标网络估值，Polyak 更新，按采集 GS 剔除旧经验。原始辅助分支保留，默认关闭，不声称已逐项验证。 |
| `_engine/environment.py`, `physical.py` | 原 Reward + 现行采样协议 | 矩阵和请求流 → 两次独立预热、观测、尾随 → Reward、原 S 转移、更新旧部署并推进需求周期。新增可选遥测，使用原 `Simulator.telemetry`，不用于学习输入。 |
| `edge_d/{policy,actions,rollout,structure,coverage,training}.py` | 后续历史改进 | 覆盖阶段 + EDIT + REINFORCE，保留独立历史入口和 CPU LP FINISH 门控；不当作当前推荐训练策略。 |
| `edge_joint/{training,periodic}.py` | 交接协调层 | 原 S + 后期 D，保留 S-only、BUILD 联合、周期联合；不是 R12 等价迁移。 |
| `edge_d/placement.py`, `edge_joint/placement_training.py` | 本轮研究 | 从零初始化的条件策略直接解码单张合法矩阵；固定启发式、冻结 D、同时学习、KEEP/BUILD 共享实际物理测量。 |

## 容量、分配和事件

当前配置：32 个节点、4 区、52 条无向物理边、10 类服务合计 28 阶段、12,000 用户；CPU 总量约 366.9185 CU/s、AP 上行约 60.7–118.1 Mbps。7200 CU/s 和 300 Mbps 是历史目标，不是现行值。节点 RAM/SSD 只约束持久化阶段副本；AP 上行按活跃用户均分；逐跳有线每个方向各自独立执行 KKT，不含全网共享 NIC 联合约束。

对固定活跃集合和正工作量，最小化 `sum(W_i/f_i)`，约束 `sum(f_i)<=F`，KKT 给出 `f_i ∝ sqrt(W_i)`；这是固定速率下剩余完成时间之和的局部代理，不是动态流量下最小化整网 E2E 的证明。`finish_pf` 用 `w_i=1/(1+W_i/W_min_catalog)`，对应加权对数效用 `sum(w_i log f_i)`，强调较短剩余工作；`ps` 均分，`fcfs/srpt` 把当前容量给单个任务。五种模式单元检查工作守恒，事件层另外核查已服务+剩余与提交量守恒。CPU LP 只检查理想可任意分配的平均工作量，不保证真实 S 可调度，更不检查链路；`rho>=1` 是可持续性风险，不等于一定在当前短窗截尾。

本轮未改 CPU 默认 `finish_pf`。交接脚本的单张矩阵、同请求流配对探针：`finish_pf` 0.9490 s，`kkt` 1.1692 s，均零截尾且守恒误差 < 2.4e-12；不是学习收益，也不能据此估计总体优劣。若以 CPU KKT 为主问题，须预注册独立物理条件并重新训练两种算法，而不是复用本轮模型作学习结论。

## 时间尺度、Reward 与可靠性

- `experiment.json` 的基础 warm 为 6 s；`commit_warm` 提交时暂时覆盖为 20 s，两个 6 s 观测窗口，各有最多 8 s 后续，窗口间不携带事件队列。窗口选自同一需求状态下的嵌套请求流，背景请求可继续到达，后续不是清空系统。逻辑周期为 7200 s；KEEP/BUILD 跨期延续 `Env.old` 与需求状态，不延续物理事件队列。任何模式均未仿真完整天。
- 默认 `Reward = -(10 × 窗口惩罚后平均 E2E + holding + switching)`；当前未启用可选 migration 额外项。未完成请求计到最后观察时的时延下界，再加 10 s；`failed` 是截尾，没有永久丢弃语义。完成 E2E 和完成 p95 只统计完成者，有幸存者偏差；统一报告截尾、惩罚后 E2E 与 `>1 s 或截尾` 的复合尾部率，其中 1 s 是诊断阈值，不是验证过的 SLA。
- 总需求按指数间隔与场景率生成，固定/训练共享人口生成规则和对应种子，但其需求和请求流分离；最终 holdout 使用独立人口种子、需求和请求流。种子对内配对，共享请求轨迹；不是将同一组请求当作多个独立样本。固定与 holdout 的场景数小，不能据此证明总体稳定性。
- 原始 S 默认局部转移的时延成本针对被调度阶段，和 D 的整体 cohort Reward 不等价。新联合流程把原始 `STransition` 写入原始回放；配对分支在更新前共享同一 S 版本。S 的策略采样随访问状态与时序分叉，共享 RNG 种子并不意味着每个阶段经历相同探索动作。

## R12 与当前实现的差分

1. R12 主环境每个 GS 提交一次，刷新/KEEP 策略与需求持续推进，原 S 只采主分支窗口转移。辅助分支从主状态拷贝，固定 D/S 版本并匹配请求、期限和样本数，不写入主 Replay。
2. R12 对随机阶段做压力分层和逆概率权重；用 KEEP/BUILD 与目标阶段 refresh/不 refresh 的反事实，通过三期 `0.95^h × 0.1` 回报给门和 stage head 赋信用。`long_branch` 调用 R9 D80：一条连续资源轨迹、分桶独立截尾。交接的短窗加普通 EDIT 终局 Reward 缺少这些设计。
3. R12 两 GS 一批校验 D/S 版本，再做受约束的 Adam、熵与锚点 KL 诊断，随后做原 S 更新、递增其版本，并保存环境和回放以续跑。新 BUILD 以两张实际矩阵共享底层随机数，不是一个主分支加隔离辅助；周期版按配对完整轨迹的 return-to-go 更新，不恢复 R12 的阶段反事实、D80 训练目标、锚点信任域或精确续训。
4. 保留 R12 作为对照而不是删除。实验尚未单独量化 D80 信用、S 系统信用和 episode 长度的因果效益；本轮不切换默认设置。下一步应以同一部署参数化、调度器、种子和计算预算，只改变信用协议，对比独立长轨迹与 HOLDOUT。

## 历史负结果

`runs/archive/legacy/evidence/pg64_seed13` 与 `pg64_seed14` 保留训练、固定验证与未完成 64 次更新记录（约 60、57 次）；不能将中止视为无学习信号。`runs/archive/legacy/evidence/original_action_cv/summary.json` 中两种 credit arm 的固定验证均下降；训练集拟合不代表未见状态方面的正确性。`runs/archive/legacy/evidence/spike_causality_results/all_192_exact_lp.json` 保留 192 张矩阵中 34 张理想 CPU 承载不足的调查。过往 `runs/archive/legacy/checks/` 的短训不是收敛证据。

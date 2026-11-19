# 独立研究员完整交接说明：R12-A 稳定性修复版（待长训验收）

## 一、交付定位与研究结论

本交付依据 2026-11-17 完成的 R12 正式工程，**选择 R12-A 而非 R12-B**，因为在同一组新 Markov 需求的 GS32 评价中，时间尺度预热 B 未获得双种子优势。此包仅保留从零启动所需源代码、原物理配置、训练及独立测试入口；**不附带先前已训练的网络参数、历史 Replay、海量原始实验记录，也不附带旧实验压缩包**。

主要研究目标不是追求离线全局最优，而是得到真实事件物理仿真中 D/S 两智能体从随机参数开始可训练、Reward 总体改善与 E2E 下降、原始结果可复现的闭环。要区分整个联合系统学习收益、S-only 贡献、D 额外增量和 D120 持续负载风险。强固定 R30 仅是性能上界附近的物理参考，**不是训练先验或动作替代**。

此前 R12 原项目完成 Seed14/Seed13×A/B/S-only 六条 32GS 实训，A 每种子 16 次 D 更新、256 次 S 更新，B 12 次 D 更新；新独立 GS32 A 的 Reward/E2E：Seed14 `-4.9513 / 0.4764s`，Seed13 `-5.1690 / 0.4988s`，同期 S-only 分别 `-4.9585 / 0.4773s`、`-5.2193 / 0.5031s`。D 的独立增量仍小，固定共同成熟 S 下有些状态甚至回退，不得夸称 D 已学会可靠最优部署。本包没有这些训练权重；上述数字只作为来源背景，不是本包新增实测。

## 二、系统物理与目标

- 32 异构节点：4 区×8；52 无向连接映射到 104 个有向链路带宽资源；固定最短路径；每 AP 共享 300Mbps 上行。
- 10 类服务 / 28 串行阶段 / 12,000 固定接入用户；服务阶段依赖前一阶段输出；Poisson 到达，由三状态 Markov、AR 业务热点及区域/服务需求转移驱动；随机外生请求与环境状态有可复现种子。
- CPU、RAM、SSD、NIC 和逐跳有线链路竞争；计算采用原 Finish-PF，链路采用原 KKT；事件驱动任务生命周期、排队及未完成请求惩罚。总资源量和各节点实际值以 `configs/world.json` 为唯一权威，不使用旧讨论中的替代 CPU 单位。
- Agent-D 只在部署周期边界决定整个服务阶段副本可用性；每 GS=7200 逻辑秒。Agent-S 对每个真实阶段就绪任务选择执行副本；无线 AP 300Mbps 固定共享，不是智能体优化变量。
- 原始训练主环境 2×6 秒样本窗口，6 秒预热+8 秒尾部（不是连续7200秒），D 信用的 30 秒预热/80 秒采样由 `r9/continuous_credit.py` 定义，D120 是更长的独立物理回归。不可将不同 cohort 分母下的未完成数量直接互比。
- Reward = `-(delay_weight×penalized_delay + holding_cost + switch_cost)`，配置延迟权重10、holding0.08、switch0.04、removal ratio0.25。holding/switch 的详细原规范见 `environment.py` 和 `configs/experiment_r0_joint.json`；不允许仅为了曲线改善改系数。

## 三、策略和更新闭环

**D 结构：**原 R5.1 条件网络，包含基础 embedding/交互 residual 和宏观 KEEP/BUILD、阶段 Refresh、Count、ordered-Node 分布。每阶段节点概率输出连续 logits，采样离散动作并使用相应 log probability 的策略梯度；完整潜在矩阵经原 `p23_global.Decode` 全局 MILP 解码，确保覆盖及 RAM/SSD 合法。没有在线布局候选池、副本配额、CPU 硬掩码、人工 KEEP 触发，也不用 PPO。当前 D 梯度是带正确宏观概率加权的条件分层 Policy Gradient 和温和熵项，使用 KL 有界 Adam 进行参数更新，**不是 Discrete SAC / PPO**，请勿把历史预期算法写成已实现算法。

**S 结构：**Masked Dueling Double DQN，每真实自然主 GS 后收集有效阶段转移；每 2GS 做 16 次 S DDQN 更新，即每 GS 均值8次。D 辅助 H3 值由同一冻结版 S 和后续 D 当前版本产生，但 D80 评价的虚拟请求不允许进入 S Replay。S 的探索 epsilon 及 target 机制沿用原配置。

**D 信用：**每 GS 形成原主自然状态上的阶段对照；阶段 k 以正概率 `q(k)` 被采样，乘以 `1/(28 q(k))`；两 GS 一次 D Adam。连续 D80 使用 R9-B 正确逐6秒 cohort 截止；辅助物理分支不改变自然主环境、不筛选差布局。

**最佳模型：**预登记固定 seeds `975101`,`975103` 各 4 GS、合计8GS验证，选择平均真实 Reward 更高的模型；GS0 先验证，此后每8GS及最终请求步验证。独立需求 `160003,160029,160071,160109` 不得参与 best 选择。`latest.pt` 储存 D/S 网络、D Adam、Markov 环境、RNG 和经 gzip 压缩的最新 S Replay；`best.pt` 只保存 D/S 网络及哈希、GS元信息。仅有这**两个模型文件**，没有其他版本/步号检查点。请用 `latest.pt` 正常续训，不得从 best.pt 续训冒充完整 Adam 和 S Replay 连续训练。

## 四、历史研究要点（R5–R12）

- R5：双种子 64GS 旧联合基线，主要收益来自 S；D 长期学习与可靠 BUILD 不足。
- R8：自然 Markov 需求下有限合法布局对照证明动态成本增益空间相对强固定参考不大；但不能称全可行空间全局上界。
- R9：原2×6秒短窗在持续过载时会漏报 D120 失败，建立30秒预热/连续80秒物理信用。
- R10：逆概率多保真 HT 数学期望无偏，但重尾物理差值造成极端梯度与实际性能退化；未升级为正式算法。
- R11：合法旧布局与独立需求交叉状态确实增加严重CPU过载覆盖，但未提升双种子独立 BUILD 质量；早期不成熟 S 在部分状态下会对有效 BUILD 给出反向信用。
- R12：尝试前8GS冻结D、只训练S的时间协调B；相对原A独立双种子没有稳定提升。早期曾将原A定位为阶段性可用项目；但随后用户上传的独立新100GS日志证明 A 可在GS40后出现严重D策略退化，因此不能再直接宣称旧A稳定收敛。

## 五、部署、启动、复现和恢复

工程已保留源文件的原模块名称，避免引入新物理模型。首选 Linux + Python3.13；从项目根目录 `bash run.sh train --seed 20261114 --steps 32`、`bash run.sh train --seed 20261113 --steps 32`。运行产物在 `r12/experiments/A/r51/results/<seed>_D/`。

- **从零：**不传 `--resume`，输出目录必须不存在。模型初始化、Adam、S Replay、Markov 状态、RNG 都重新产生。
- **继续：**命令 `bash run.sh train --seed 20261114 --steps 64 --resume`，目标 GS 为累计64，不是再加64。模型 /优化器/ Replay 等均从 `latest.pt` 恢复。恢复时若存在超出最新完整偶数 GS 的未提交尾部日志，会隔离到 `aborted_*.jsonl` 后重新物理执行。
- **独立验证：**`bash run.sh eval --seed 20261114 --arm A --model latest`，固化四个独立 Markov 需求、每个8GS；结果不用于选 best。
- **D120：**`bash run.sh d120 --seed 20261114 --arm A --model best --periods 2`，对另一个需求执行原物理 D120，切勿把2个GS无失败解释为全分布可靠。
- **对照：**`bash run.sh s-only --seed 20261114 --steps 32`，在独立子目录保存 S-only 的自身 best/latest；不影响联合模型。
- **原始曲线：**`bash run.sh plot --seed 20261114 --arm A`，输出每GS原始点和固定验证点；不做滑动平均替代。可以用独立评价 JSONL 做后续曲线。

运行入口可通过 `bash run.sh train --help` 检查；Windows 使用 `run.ps1`（仅入口提供，尚未跨平台验证一致性）。不要手动把旧 `R12_models_new.zip` / Replay 放进最小工程，它们不是从零训练的依赖。

## 六、最小化变更、真实测试与限制

**原物理未修改：**`edge_rl`, `dynamic.py`, `environment.py`, `p23_global.py`, `r0_joint.py`, `r2b_joint.py`, `configs` 原样保留。R9 corrected continuous-credit 函数保持原样。辅助 cohort 的 `r2a_sampling.py` 去除了旧审计脚本的导入，`hasharr` / `summarize` / `agg` 函数体与原版相同；`r12/credit.py` 合并原 `r4_stratified` 数学；`r12/joint.py` 内嵌原 `r9.long_branch` 并改为 best/latest 保存。不修改离散动作采样规则、Reward、随机流、D80 物理或 S Replay 更新逻辑。针对后续真实100GS失败，稳定性修正版修改条件策略梯度的联合概率权重、D学习率与参数更新方式，具体见 `DIAGNOSIS_AND_FIX.md`。

**已实测确认：**原始最小工程曾通过GS1–2的真实执行回归；它后来在用户Seed14实际100GS中退化，不能沿用原版的长期成功声明。**本修正版**通过新GS0双验证轨迹、Count KL/梯度数学、Adam完整回退、模型与Replay存取单元测试；利用用户上传的冻结D80信用重放了前2GS主物理与D/S更新（D参数变化、16次S更新）。也执行GS16/100 D/S固定请求交叉实测。**修正版尚未重新跑完双种子32/100GS，不能声称已经证实收敛。**

**边界：**完整32GS是在源R12发行版执行的，不是本最小包重新长训的结果。短窗主物理不覆盖整个7200秒；D80与D120非同一评价分母；Agent-D相对S-only的增量小。严重随机初期物理过载可能导致单个GS墙钟较长；不可通过降低负载/删除失败动作加速。MILP跨机器数值结果可能变化，Linux/Python3.13 CPU环境为可复现起点；Windows/CUDA未实测同位一致。

研究员如果需要升级 D 网络、改变物理或抽样协议，应在独立分支中进行并明确标注与 R12-A 基线的对照，不能污染本包。

## 七、R12-A 100GS失败后的正式修正与后续研究边界

2026-11-17 新增 Seed14 100GS 原始日志与 `latest.pt`，实测发现 GS40–100 固定验证 68 个未完成、Reward 约 -26.5、E2E 约 2.63 秒，D 的 Count 提案为896/896全选，经原MILP投影成相同的299副本布局。原CPU线性规划显示这张布局在四个验证周期的全局CPU理想压力均超过1，而单阶段必要压力均小于1；这是跨阶段全局竞争。因外生请求哈希跨GS相同，不能把固定验证持续失败归咎于需求重新抽样。

固定 S 的互换实验发现 D16+S100 明显好于 D100+S16；S 尚能继续学习，D 策略自身已退化。源码确认原最小版硬编码 `Adam(lr=.01)`，但原配置为 `.0003`，相差33.33倍。更重要的是训练入口使用 `Policy(...,'D')`，而旧辅助损失使该分支条件建设策略梯度不乘宏观BUILD概率，和真正联合期望Reward目标不一致。现在两处均已经修复；加上连续可微熵正则及只在真实自然训练状态实施的按头KL回退，不添加副本配额、人工触发或CPU硬约束。

**完整源代码审计路径**：D动作 `r0_joint.py`/`r2b_joint.py`，特征与条件网络 `r41/interaction.py`，R9持续信用 `r9/continuous_credit.py`、D分层目标 `r12/credit.py`、主循环与S replay `r12/joint.py`，KKT/事件物理 `edge_rl/`、`dynamic.py`、`environment.py`，合法投影 `p23_global.py`，原配置 `configs/experiment_r0_joint.json`，新增约束单元 `r12/stability.py` 与 `tests/`。保留后续的负样本与未完成记录。

**对接下一位研究员：**请先运行 `python -m unittest discover -s tests -p test_stability.py -v`，再双种子从零GS0/8/16/32/64/100训练与冻结验证；关注 `adam_audit.jsonl` 中 `trust_step`/Count KL、宏观 `macro_p`、投影比率、实际CPU全局压力、S-only的独立收益。若新版本在 GS32 已出现持续性错配，不要通过更换随机种子、截断Reward或修改负载来隐藏。若32/100GS双种子验证仍未改善，诚实标记失败并进一步检查宏观/条件信用估计与投影多对一的学习可辨识性。

**历史资产依赖：**本最小工程不依赖旧模型。旧 `R12_models_new.zip`、R11/R10/R9研究包仅用于历史诊断或额外交叉对照。可运行训练继续只有 `best.pt`、`latest.pt` 两个模型文件；Replay 已压缩进 latest，无另存独立Replay。

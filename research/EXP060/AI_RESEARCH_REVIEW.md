# EXP060 研究交接 · 结构化 Gaussian Agent-D

本版交接记录实现、预登记和工程验收。正式实验由独立 Windows 进程执行，分析以两种子四臂全部完成 GS504 且独立评价账本齐全为前提。工程烟测的指标只用于验证数据和更新链路。目前没有关于 Reward、E2E、尾部风险、跨种子有效性或候选晋级的正式结论。

## 研究起点

已核验 GitHub 父版本 `20f9ac8b3939c55b698d0c59edbc07f250d16345`。EXP055/056 记录了从随机初始化得到的联合学习收益。EXP057 的冻结替换将 Seed13、Seed15 的晚期回退主要归因于 D。EXP058 的半学习率与 EXP059 的旧请求成本校正未取得跨种子独立优势。EXP022 的直接偏好方案以新放置路径生成布局，在正式评价中出现 28 副本附近的稀疏布局和持续压力下的失败。本轮保留 EXP055 的 D/S 训练、CPU/链路资源机制、24 秒物理窗、固定两小时部署时机及原始 Reward，只检验阶段条件的 Gaussian 动作表达。[EXP022](../EXP022/AI_RESEARCH_REVIEW.md) · [EXP055](../EXP055/AI_RESEARCH_REVIEW.md) · [EXP056](../EXP056/AI_RESEARCH_REVIEW.md) · [EXP057](../EXP057/AI_RESEARCH_REVIEW.md) · [EXP058](../EXP058/AI_RESEARCH_REVIEW.md) · [EXP059](../EXP059/AI_RESEARCH_REVIEW.md)

## 算法与概率模型

对原 `pair_features` 的 28×32×19 张量保留每个阶段和节点的原始关系。19 个特征覆盖阶段计算/载荷/内存等资源需求、节点 CPU/RAM/SSD、区域和服务需求、上一部署矩阵、节点库存占用。B 的全局分支与 A 都使用原有 57 维均值/标准差/最大值统计和 57→32→8 网络。B 的阶段分支对每个阶段的 32 个节点共享 `Linear(19,16)` 和 Tanh，再拼接均值与最大值，最后用共享 `Linear(32,3)` 输出三个控制量。节点集合归约对列重排不变，阶段随输入资源关系保持响应。新增阶段权重独立随机初始化，全局 D 与 S 在线、目标网络同种子公平配对，Replay 都从空开始。D 参数为 A 2,120、B 2,539，Agent-S 在线网络为 18,306。[输入特征](../../edge_d/world.py) · [部署策略](../../edge_d/placement.py)

将 8 个全局均值和 84 个阶段均值拼接后，从共同标准差 `0.5` 的独立 Gaussian 一次采样。`log π(z|s)` 是全部 92 维的对数概率和，样本在 score 梯度中视为固定，原折扣和优势乘完整的 score。解码不参与求导。已用解析偏导 `∂logπ/∂μ=(z−μ)/σ²` 核对全部联合维度。梯度裁剪和 Adam 信用规则与 A 相同。

每阶段的采样值先经 Tanh，依次对应本地性、偏向高 CPU 容量节点和冗余接受强度。前两项修正原覆盖运输的阶段—节点优先级，本地性同时作用于可选副本的位置得分，第三项只作用于原冗余评分。链路定价、容量协调、结构覆盖回退及随机噪声均来自同一个原解码过程。原运输解只有满足每阶段覆盖与节点 RAM/SSD 才能作为布局；结构覆盖回退在相同合法约束下求解，随后每个可选副本仅在剩余资源足够时进入矩阵。阶段控制为零时同键重现 A 的矩阵。每次实际部署只采样一次并提交一张合法完整矩阵。旧审计的更新前后同键解码仅记录位变化，其矩阵不进入物理队列。[Actor 训练与解码接入](../../edge_joint/adaptive_training.py)

## P1 工程证据

- 使用同一真实环境和解码键，在阶段 0、9、18 分别单独扰动三个控制通道。九格均改变了对应阶段的 1–4 个节点位，完整矩阵改变 1–12 位。固定键的冗余三格从 1→5、2→3、2→3 个该阶段副本。更换全局场景后，84 维阶段均值的 L2 变化为 0.0754。
- 12 对真实合法解码中，A/B 总副本中位数为 68.5/69。平均单次解码约 10.0/14.0 ms。这组少量、单环境计时提示可能的解码开销，实际验收用所有正式训练物理解码的计时和预登记 1.25 门槛。副本检查仅为诊断与验收。
- Seed13 B 跑两条随机初始化物理烟测。一条 GS8 恢复到 GS16，另一条连续到 GS16。逐槽请求、部署、Reward、E2E/P99、梯度和 D/S 更新一致，`last.pt` SHA256 相同。两个候选 episode 的阶段分支梯度范数约 889.37 和 216.81（裁剪前），其实际参数步长 0.06141 和 0.03327。两臂还在 0.55 连续压力下通过短冻结评价，真实请求指纹配对、资源守恒误差小于 `3e−12`。9 项原与新增回归通过。正式验收不使用这些早期表现。[烟测原始证据](smoke.json)
- Seed13 A 的计时版从随机参数真实复跑 GS1–8。请求、矩阵、行动分布、CPU/链路、Reward、时延和更新账本与 EXP055 所核字段逐槽/逐更新完全一致。正式 A 仍会各复跑 504 GS，以比较同主机开销。[基线烟测](baseline-smoke.json)

## 正式协议与完成凭证

第一阶段 Seed20261113、20261115 各从随机开始运行 A/B 两套，每臂 63 episode、504 GS、63 D 更新和 4,032 S 更新。部署和调度产生各自物理后继，外生流保持配对。四种原/持久热点自然/脉冲，加原环境 0.55、0.70 八槽持续压力，每种子六个独立新流，对比各臂随机 GS0 和 GS504 `last`。原固定验证含 GS0、128、256、384、504；它只用于 `best`。通过第一阶段才由用户确认启动 Seed14，之后仍按同一门槛评价。本轮不启动五种子长训。[完整阈值与随机协议](PROTOCOL.md)

后台入口结束四条正式训练及 12 组独立评价后原子写入 `runs/results/EXP060/initial-complete.json`。`launch.initial.json` 记录 PID、启动时间、标准输出和错误日志；未出现完成标记时，以错误日志判断是否中断。检查点、逐 GS 日志和独立账本留在 `runs/results/EXP060/`，不覆盖任何历史实验。输出包括每臂 `best.pt`、`last.pt`、`resume_latest.pt`、`train.jsonl`、`updates.jsonl`、`fixed.jsonl`、独立 `holdout.seed*.json`。汇总入口在唤醒分析时读取这些原始文件，生成八张图和 `summary.initial.json`。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pytest -q tests/test_exp060.py tests/test_exp059.py tests/test_exp058.py tests/test_exp056.py
python -m scripts.exp060_run register
python -m scripts.exp060_run smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\exp060_start.ps1 -Workers 4
# 出现 initial-complete.json 后执行分析
python -m scripts.exp060_run report
# 只有 summary.initial.json 判定通过后才启动 Seed14
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\exp060_start.ps1 -Workers 2 -Third
```

若运行中断，重新执行同一后台启动命令，会恢复已提交检查点并跳过已经核验的端点与场景；每次启动使用独立的控制台日志。复现依赖本机 EXP055 联合训练源账本、EXP056–059 既有随机键登记与 EXP043 负载登记。代码及轻量交接在 GitHub 备份。[EXP060 分支](https://github.com/makaka233/edge_full_joint/tree/exp060-structured-placement-actor)

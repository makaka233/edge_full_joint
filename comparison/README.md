# 多阶段服务与整体服务对比

多阶段组使用 EXP029 的实际联合训练入口。整体组在 monolithic_service 内构造十个原子服务单元，复用原事件引擎与更新公式。两组分别从随机参数训练，用同一请求流、业务期和 D/S 梯度更新预算。原 EXP029 权重和日志作为历史证据。

## 运行

从仓库根目录运行。并行任务之间独立保存，使用同一条命令可以继续已注册的对应预算。

```powershell
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python -m comparison.run --suite pilot --workers 6
```

`pilot` 对 Seed13、16、17 各训练两组 512 GS（64 次 D、4,096 次 S 更新），然后从冻结的 best 和 last 出发跑独立评价并生成图。本轮已经完成，结论见交接。命令的 `--stage train`、`--stage evaluate`、`--stage report` 可单独运行对应阶段。Windows 本地多进程管道不可用时可改为 `--workers 1`。

### 只长训整体服务

```powershell
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python -m comparison.run --suite mature --stage train --workers 5
```

整体服务的 Seed13/16/17 从 pilot 的完整 GS512 状态接续，Seed14/15 从随机初始化开始，五个种子各自达到 **总计 GS5000**。中断后重跑同一条命令。产物前缀为 `mature.monolithic.seed<seed>`，用 `mature.learning.review.json` 和 `mature.learning.fixed.png` 查看学习与后期稳定性。原始 `pilot.*` 和 `runs/results/EXP029/` 照常保留。旧 EXP029 文件缺少 EXP030 可选负载条件信用字段，读取时按旧算法的 `false` 解释；其余配置严格匹配。

确认长期学习表现后，再在全新共同请求流上冻结对比。多阶段直接读取历史 EXP029 的五组 best/last，整体服务使用本次训练的五组 best/last。先完成配对评价，再单独汇总。

```powershell
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python -m comparison.run --suite mature --stage evaluate --workers 5
python -m comparison.run --suite mature --stage report --workers 1
```

```powershell
python -m unittest comparison.monolithic_service.test_comparison -v
python -m comparison.run --suite smoke --workers 1
```

真实短测使用 smoke 前缀，pilot 和 formal 的训练各自随机初始化。

## 预注册物理设定

|轴|水平|在同一外生需求下改动|
|---|---|---|
|A 请求|自然、0.40、0.55、0.70|每期用原工作量与 CPU 算目标提供负载，两组逐期请求哈希相同|
|B CPU|0.7×、1×、1.3×、1.6×|各节点 CPU 容量统一同比例缩放，保持请求过程|
|C 有线|0.5×、1×、1.5×、2×|双向独立容量同比例缩放，保持原路由权重与路径|
|D 内部数据|0.5×、1×、2×、4×|只缩放原多阶段相邻的输出、输入对|

水平、种子和预算登记在 [experiment_matrix.json](experiment_matrix.json)。原自然水平和 B/C/D 的 1× 共享同一个物理基准结果。两个未参与训练的独立人口和请求流连续运行 24 期，各训练种子共享这两条流。模型权重冻结，D 仍可在每期按当前条件重新解码，S 使用合法贪婪动作。主图报告最后模型，汇总保留 best 和 last 的全部配对。best 只用两个相同固定验证场景的平均净 Reward 选出。

在 `runs/results/monolithic_service_comparison/` 查看协议、架构训练与梯度账本、原始物理日志、`best.pt`、`last.pt`、`resume_latest.pt`。`pilot.review.json` 按 seed、物理条件、外生流和检查点提供指标，每条结果同时保留资源用量和尾部。四张 `pilot.holdout.*.png` 为独立冻结对比，`pilot.seed*.raw.png` 为原始训练轨迹。GS、优化器步骤、实际调度次数和到达请求数分别记录。

[实现细节](monolithic_service/README.md) · [单文件研究交接](../research/COMPARISON_MONOLITHIC/AI_RESEARCH_REVIEW.md)

## 高计算、紧凑载荷新场景

`python -m comparison.run --stage profile-audit` 使用 3,500 个端点并匹配原环境逐期总期望到达率。当前 v2 按十类服务和 18 条内部边分别定义载荷，审计写入 `profile-audit.v2.json`。该入口只运行物理审计。阶段依赖、字节预算及来源见 [环境设计与交接](../research/COMPARISON_EDGE_INFERENCE/AI_RESEARCH_REVIEW.md)。

### comprised_exp1 双架构完整运行

从项目根目录执行。命令自动训练两种架构的 20261113–20261117，各从随机参数完成 5,000 GS，然后执行配对独立评价并生成图表。再次执行同一条命令会恢复训练和补全未写完的独立评价；内存不足时可将 `--workers 3` 改为 `--workers 1`。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python -m comparison.run --suite comprised_exp1 --workers 3
```

训练在 v2 主场景进行，3,500 个端点、保留原全网到达率、2 倍算量、逐服务内部消息。每个架构和种子计划 625 次 D、40,000 次 S 更新；训练与固定验证的外生请求流逐期核对。独立评价使用两条新的共同轨迹，每条连续 24 期，冻结各自的 best 和 last。四个工作负载包括原算量/原载荷、2 倍算量/原载荷、原算量/v2 载荷、2 倍算量/v2 载荷。其余三格属于冻结策略的单因素或组合对照。

| 产物 | 目录 |
|---|---|
| 分阶段服务（proposed） | `comparison/runs/proposed/comprised_exp1/seed<seed>/` |
| 完整服务部署（MSP） | `comparison/runs/MSP/comprised_exp1/seed<seed>/` |
| 共享协议、汇总及 best 对比图 | `comparison/runs/comprised_exp1.protocol.json`、`comparison/runs/comprised_exp1.review.json`、`comparison/runs/comprised_exp1.holdout.profile.png` |

种子目录包含 `manifest.json`、`train.jsonl`、`fixed.jsonl`、`updates.jsonl`、架构专属的 `holdout.*.json`、`train.raw.png`，以及 `best.pt`、`last.pt`、`resume_latest.pt`。新环境检查点绑定实验协议，旧环境权重不能直接用于续训或评价。根目录的 Git 忽略规则保留本地运行结果，备份源码和说明。

正式运行完成后，`python -m comparison.plot_means` 从已有逐 GS 训练记录和独立评估汇总生成 `comparison/runs/comprised_exp1.training.mean5.png` 和 `comparison/runs/comprised_exp1.holdout.mean5.png`。训练图同时保留完整原始五种子均值和 128 GS 滚动趋势；四场景图比较两种架构各自的 best。

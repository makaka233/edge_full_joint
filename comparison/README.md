# 多阶段服务与整体服务对比

多阶段组使用 EXP029 的实际联合训练入口。整体组在 monolithic_service 内构造十个原子服务单元，复用原事件引擎与更新公式。两组分别从随机参数训练，用同一请求流、业务期和 D/S 梯度更新预算。原 EXP029 权重和日志作为历史证据。

## 运行

从仓库根目录运行。并行任务之间独立保存，使用同一条命令可以继续已注册的对应预算。

```powershell
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python -m comparison.run --suite pilot --workers 6
```

`pilot` 对 Seed13、16、17 各训练两组 512 GS（64 次 D、4,096 次 S 更新），然后从冻结的 best 和 last 出发跑独立评价并生成图。命令的 `--stage train`、`--stage evaluate`、`--stage report` 可单独运行对应阶段。Windows 本地多进程管道不可用时可改为 `--workers 1`。确认两组训练闭环正常且决定扩展预算后，再用 `--suite formal` 启动五种子 5,000 GS 批次。

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

# EXP019 源码变更

## 瞬时矩阵切换

| 文件 | 改动 |
|---|---|
| `edge_joint/adaptive_physics.py` | 合法目标在决策时刻成为有效矩阵。直接按新矩阵计持有成本，保留已有资源作业和工作守恒验证。 |
| `edge_joint/adaptive_training.py` | 移除等待动作与准备特征，保留 D/S 及宏观信用。记录即刻激活、矩阵和成本。 |
| `edge_joint/adaptive_runner.py`、`edge_joint/adaptive_cli.py` | 同一主入口支持历史受阻期复算。格式 2 区分瞬时版，旧检查点不进入新训练和评价。 |
| `scripts/exp019_report.py` | 继续读历史账本并保留历史阻塞诊断，审计新日志的合法矩阵、切换时点和成本。 |
| `tests/test_adaptive_training.py` | 验证重叠不可行时的双向切换、成本和在途任务、新掩码、三组更新、恢复与旧检查点保护。 |

本轮通过 14 项聚焦测试、三模式 CLI 烟测、学习组评价和旧数据两例重放。数据与边界见[单文件交接](AI_RESEARCH_REVIEW.md)。

## 历史容量阻塞版记录

| 文件 | 改动 |
|---|---|
| `edge_joint/adaptive_physics.py` | 继承原事件引擎，增加旧副本引用计数、容量重叠规划、实际镜像传输、准备完成切换、资源与请求守恒检查。 |
| `edge_joint/adaptive_training.py` | 连续多周期 rollout、长期队列评分、宏观策略和库存协调解码路径梯度、原 DDQN 回放、逐点资源证据。 |
| `edge_joint/adaptive_storage.py` | 运行前缀锁、日志落盘、原子模型及恢复状态、随机状态序列化。 |
| `edge_joint/adaptive_runner.py` | 训练、恢复、固定验证、权重初始化、Git 来源登记、独立评价与物理验证。 |
| `edge_d/cli.py`、`edge_joint/adaptive_cli.py` | 为现有主程序增加 adaptive 子命令。 |
| `tests/test_adaptive_training.py` | 验证容量规划、镜像传输、跨切换的阶段连续性、有影响范围的回报及可精确续跑。 |

本轮在当前 Windows 项目目录交付。EXP018 原源码与 GS1024 产物在既有工作树和运行目录中继续保留。正式训练、独立评价和结果研究分别由后续显式调用触发。

## 已执行的轻量验证

- `python -m edge_d.cli adaptive --help` 验证主程序接口。
- `python -m unittest discover -s tests -p test_adaptive_training.py -v` 验证四项测试。完整运行和中断续跑的 D、门、S 参数及关键逐点物理量精确相同。
- Seed13 命令行烟测由 1 个 episode 启动，再恢复到 2 个 episode，共 4 GS。产生 2 次 D 更新、2 次门更新、32 次 DDQN 更新，best/last/恢复状态均存在。独立评价生成 24 条业务期记录并通过各比较对象同流检查。
- Seed14 加载冻结的 EXP018 GS1024 权重，运行 2 个各 1 秒的连续物理期。烟测轨迹存入共享 `runs/results/EXP019/`。

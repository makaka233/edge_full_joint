# DeepSeek Agent-S 状态充分性审计

## 结论

本审计不修改正式 Agent-S。共采集 96 个真实事件状态和 258 个合法动作反事实，
覆盖 GS0、GS16、EXP038 GS64、失败的系统信用 GS64，以及自然/持续 0.70 两类。

主要结果：

- 当前 Agent-S Q 在枚举动作上的 Top-1 命中率从 GS0 的 `50.0%` 降到 GS64 的
  `41.7%`，平均排序相关从 `+0.229` 降到 `-0.146`。
- GS64 持续 0.70 的 Q 排序相关为 `-0.375`。
- EXP039 已知困难状态中，当前完整 Q 的 Top-1 为 `45.8%`，仅先验为 `54.2%`。
- 原 24 维观测与增加系统级摘要的离线预测没有稳定显著差异，95% bootstrap
  区间均跨零。
- 没有找到满足预定义相似性阈值且合法动作集合相同的状态对。

因此，本轮不支持直接增加一批新状态特征，也不支持修改在线 Agent-S。下一优先级
应是调查 DDQN 的价值学习稳定性和动作排序保持。详见
[`AI_RESEARCH_REVIEW.md`](AI_RESEARCH_REVIEW.md) 与
[`CODEX_HANDOVER.md`](CODEX_HANDOVER.md)。

## 目录

- `source_audit.md`：24 维观测逐列源码审计。
- `collect_states.py`：真实状态采集与冻结动作分叉。
- `counterfactual.py`：决策策略、事件模拟器和反事实运行。
- `features.py`：原观测特征、真实系统摘要和无未来信息约束。
- `offline_predictor.py`：轨迹分组的数据划分及 A/B 监督比较。
- `evaluate.py`：Q 排序、已知 EXP039 状态和相似状态分析。
- `report.py`：统计图。
- `run.py`：总入口。
- `results/`：本地状态组、评价、汇总和图表，不提交大文件。

## 复现

从仓库根目录执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s deepseek_workspace/state_value_audit/tests -v
python -m deepseek_workspace.state_value_audit.run --stage collect --workers 1
python -m deepseek_workspace.state_value_audit.run --stage evaluate
python -m deepseek_workspace.state_value_audit.run --stage report
```

Windows 沙箱不允许进程管道时，采集脚本自动串行执行同一物理实验。

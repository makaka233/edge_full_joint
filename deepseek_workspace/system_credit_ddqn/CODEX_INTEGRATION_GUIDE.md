# DeepSeek to Codex 集成指南

> 本文档仅描述候选原型及其边界。DeepSeek 不会修改主工程。是否集成由 Codex
> 根据 `AI_RESEARCH_REVIEW.md` 的真实结果决定。

## 新增模块

- `credit_ledger.py`
- `simulator_adapter.py`
- `scheduling_agent.py`
- `training.py`
- `evaluate.py`
- `report.py`
- `run.py`

## 最小接口

1. 原 `episode()` 需要接受一个可替换的 simulator factory，当前已支持。
2. Agent-S 需要接受全局转移，并以 `elapsed_seconds` 计算每样本折扣。
3. 训练日志需要记录信用恒等误差、事件数、信用构建时间和 DDQN 更新耗时。
4. 评价需要比较原局部信用、系统信用及同 D 的仅资源先验。

## 可能需要修改的主工程位置

- `edge_joint/adaptive_training.py`：仅为正式训练器接入新的 S 更新和计时。
- `edge_s/agent.py`：仅在正式集成时加入事件时间目标，当前原型通过子类实现。
- `edge_joint/adaptive_runner.py`：保存新增训练指标和实验类型。
- `tests/`：迁移本目录测试到正式回归。

如果现有 simulator factory 和 S 子类足以完成集成，应尽量减少主工程改动。

## 保持 EXP038 行为

- 不改变 `s_system_credit` 默认值。
- 不改变原 `STransition`、原 replay 和原 episode 更新路径。
- 不改变任何物理配置或 Reward。
- 新行为必须由显式实验配置启用。
- 原 EXP038 检查点必须继续按原格式加载，不能静默迁移。

## 集成前门槛

- 13 项 DeepSeek 原型测试和原工程全量测试通过。
- 真实短训完成 D/S 更新且无异常。
- 16 GS 基础训练检查通过。
- 64 GS 独立评价显示系统信用相对原局部信用有真实 Reward/E2E 收益。
- 系统信用相对同 D 的仅先验显示学习残差独立贡献。
- P99、截尾和终局失败不系统性恶化。

具体通过、失败与未证明项在最终 `AI_RESEARCH_REVIEW.md` 中更新。

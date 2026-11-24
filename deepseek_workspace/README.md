# DeepSeek 工作区

本目录用于标识和记录 DeepSeek 在本项目中的独立研究工作，避免与其他研究员的
分支、实验协议和运行记录混淆。

## 工作边界

- 唯一工作分支：`deepseek-workspace`
- 创建起点：`6953236577fc1e8cca6be6ec2870ff164b7d75c9`
- 建立日期：2026-11-24
- 当前状态：已完成项目初始审阅；尚未启动新的训练或对比实验

本工作区只保存协作说明、研究计划和指向正式证据的索引，不复制主程序、实验源码
或运行结果。项目仍使用唯一的源码工作树和既有主入口。

## 正式产物位置

后续每一轮实验继续遵守仓库根目录 `AGENTS.md`：

- 实验协议、复现说明和单文件交接：
  `research/<EXPERIMENT_ID>/AI_RESEARCH_REVIEW.md`
- 日志、检查点、评价结果和图表：
  `runs/results/<EXPERIMENT_ID>/`
- 源码、配置和脚本修改：
  现有源码模块及 `scripts/`，在 `deepseek-workspace` 分支提交
- 每次正式运行必须记录实际源码提交 SHA，并在交接中引用对应结果

## 当前接手基线

初始审阅确认：

- EXP037 的资源时间先验显著改善 Proposed 冷启动。
- EXP038 的 GS64 验证未通过“学习残差应超过仅先验”的门槛。
- EXP039 已完成共同状态和单动作审计，但结果尚未写回正式交接。
- 下一轮应先收口 EXP039，再决定是否开展独立的多轨迹、同预算对比实验。

详细审阅结论由主分支研究文档和各实验的 `AI_RESEARCH_REVIEW.md` 维护。

DeepSeek 对 EXP039 的完整接手、证据汇总和研究判断见
[`DEEPSEEK_HANDOVER.md`](DEEPSEEK_HANDOVER.md)。

系统级时延信用 DDQN 原型位于
[`system_credit_ddqn/`](system_credit_ddqn/)，包括独立代码、测试、结果和交接。

Agent-S 状态充分性与动作价值审计位于
[`state_value_audit/`](state_value_audit/)，包括 24 维状态源码审计、真实物理
反事实、Q 排序和离线可预测性比较。

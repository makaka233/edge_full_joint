# 00 · 从这里开始

目标是在异构边缘网络中联合学习部署与逐阶段调度，真实降低 E2E 及尾部风险。新 D 每次生成一张完整矩阵，原始 S、事件引擎和资源分配器未替换。

- **已完成**：四种模式、两粒种子、每组 64 次真实物理提交。D-only 独立 E2E 下降约 19.9%/26.5%；BUILD 联合独立 Reward/E2E 改善，但仍未超过成熟启发式 S。
- **未完成**：长期自适应 KEEP/BUILD，原 R12 D80 信用及可选 S 信用的完整验证。周期门在独立对照等同于首期 BUILD 后一直 KEEP。
- **运行与重画**：[PowerShell 命令](REPRODUCTION_GUIDE.md) · [源码分工](SYSTEM_ARCHITECTURE.md)。默认入口不加载历史训练算法。
- **证据**：[八条完整原始曲线与逐次数据](../evidence/INDEX.md)；[实验结果](../research/INDEX.md)；[历史失败](../research/history/INDEX.md)。
- **继续研究**：[当前状态](CURRENT_STATUS.md) 与 [独立研究员交接](RESEARCH_HANDOVER.md)。

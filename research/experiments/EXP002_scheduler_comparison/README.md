# EXP002 — 原始 Agent-S 学习与 BUILD 联合

S-only 冻结随机初始化的 D **策略**，矩阵随需求和解码随机种子变化，并非整程固定一张矩阵。BUILD 联合同时更新新 D 和原 DDQN；每个配对期间 S 版本固定。每个 arm 双种子各 64 次提交、128 次 S 更新；联合各 32 次 D 更新。独立 E2E：S-only 为 0.729 → 0.691 s、6.890 → 0.608 s；联合为 0.729 → 0.522 s、6.890 → 0.536 s；联合 14 的截尾 556 → 0。联合优于各自初始化但未超过 EXP001 的启发式 S。S-only 13 最终尾部率反增；新场景权重交换对照显示已学 D 与随机 S 的组合可退化。

[原始日志/权重互换](../../../evidence/raw/EXP002) · [图表](../../../evidence/figures/EXP002) · [配置](../../../evidence/manifests/EXP002) · [结果/更新](../../../evidence/summaries/EXP002) · [逐文件清单与指标](manifest.json) · [共同协议](../../../docs/REPRODUCTION_GUIDE.md)

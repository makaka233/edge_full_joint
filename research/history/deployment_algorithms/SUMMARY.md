# HIST002 · ADD、EDIT 与 CPU FINISH 门控

ADD-only 被覆盖+可逆 EDIT 替代，EDIT+REINFORCE 后出现动作长度增长和自然终止中止，[种子13](pg64_seed13) 与 [种子14](pg64_seed14) 保留原始失败记录。CPU LP 门控能阻止部分承载不足布局过早 FINISH，却无法避免编辑循环：[门控报告](20_CPU_FEASIBILITY_FINISH_GATE_REPORT.md)、[原始结果](cpu_viability_finish_results)。现行整矩阵解码替代构造，不更改原结构约束；LP 仍用作承载诊断。

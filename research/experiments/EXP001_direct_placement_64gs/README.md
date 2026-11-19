# EXP001 — 直接完整矩阵生成

问题：固定负载启发式 S 后，新 D 能否从随机初始化学习部署？与历史 EDIT 不同，单次解码提交一张合法矩阵。双种子各 64 次提交、32 次 D 更新；未达到数学收敛证明，但通过初步学习验收。独立 Reward：13 为 −5.564 → −4.660，14 为 −5.981 → −4.556；E2E 为 0.554 → 0.444 s、0.594 → 0.436 s。训练负峰 −122.71/−51.42 全保留，holdout 两者均零截尾。

[原始日志](../../../evidence/raw/EXP001) · [图表](../../../evidence/figures/EXP001) · [配置](../../../evidence/manifests/EXP001) · [结果/更新](../../../evidence/summaries/EXP001) · [逐文件清单与指标](manifest.json) · [共同协议](../../../docs/REPRODUCTION_GUIDE.md)

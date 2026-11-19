# 本轮独立接手说明（CPU 可行性 FINISH 约束）

从 `00_START_HERE.md`、`01_FULL_PROJECT_DESCRIPTION.md` 了解系统研究目标、物理环境、Agent-D/S 与运行方式。从 `14_FULL_SYSTEM_AUDIT_AND_VERIFIED_REPAIR.md`、`17_SPIKE_CAUSALITY_AND_EVALUATION_REPAIR.md` 阅读已验证底层问题和历史负结果。`20_CPU_FEASIBILITY_FINISH_GATE_REPORT.md` 是本轮实际训练/代码/数据的唯一最新结论，保留所有原始负尖峰与历史记录。历史研究分支仍可复现，**不要将其中旧的负结果覆盖**。

## 项目合同

- 32 个异构节点、52 条无向物理链路、10 类服务、28 个串行阶段、12000 用户；一个 28×32 完整二值副本矩阵，服务部署时钟远慢于逐阶段调度和连续资源分配。
- 当前用于改进验证的是独立 BUILD-only 子问题；S 用固定启发式，暂未恢复最终 D/S 联合学习；物理配置目前与早期目标（CPU 总量和 AP 容量、CPU 分配算法）存在差异，参见历史合同审计文件。
- 无 PPO、无预构造完整矩阵候选池、无按 Reward 优选/替换部署、无人工副本限额、无强制投影。Actor 自主选择 ADD、REMOVE、RELOCATE、FINISH。CPU FINISH 必要条件是明确可选约束，而非 Reward 代替。
- 当前主要风险：CPU 受限条件下可能存在无安全可达完成布局的草稿；在有效 CPU 限界内真实链路/随机到达/尾部延迟可能仍造成失败；不能将四更新改善解释为收敛。

## 快速运行

Python 3.11+ / Linux 或 Win11，安装 `python -m pip install -r requirements.txt`，运行：

```powershell
python -m unittest test_cpu_viability_contract test_cpu_viable_finish_policy -v
python -m unittest discover -s . -p 'test_*.py' -q
python cpu_viability_finish_training.py --out cpu_viability_finish_repeat --seeds 20261113 20261114 --updates 4 --rollouts 3 --eval-scenes 2 --eval-draws 2 --eval-every 2
python plot_cpu_viable_finish.py
python verify_package_files.py
python validate_handover.py
```

`cpu_viability_finish_results` 已包含真实历史原始数据；复现时必须使用新输出目录，不能覆盖历史。对于 `cpu_viability_finish_guard=True` 的新动作条件分布，训练记录每条原始轨迹与 CPU 检查次数；若新场景不能自然 FINISH，原代码抛错并保存错误，不会伪造 Reward 或为策略选择修复矩阵。

## 当前验收

双种子各自 4 更新的受控真实物理短训，48 条训练 + 48 条独立评价；总 96 条物理轨迹。CPU 限定组训练 24 张、Holdout 24 张完整矩阵，均结构合法、CPU 理想 rho<1 且自然 FINISH；未证明更长期训练、更广泛环境或联合收敛。全测试 80 项通过。额外两组训练前矩阵 120 秒持续物理压力复验脚本见 `audit_cpu_finish_120s.py`；真实结果位于 `cpu_viability_finish_results/sustained_120s.json`，使用完全一致请求流完成配对，不代表全面稳态证明。完整 SHA256 文件清单及独立解压验收命令包含在交接 ZIP。

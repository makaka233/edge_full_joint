# 协同边缘计算双智能体从零训练：R12-A 稳定性修正版

**性质：针对用户真实 Seed14 100GS 退化日志的源码修复版；不是已经证明 100GS 收敛的模型。**

本工程复用原 32 节点、52 条无向链路/104 有向资源、10 服务/28 阶段、12000 用户、三状态 Markov + Poisson 到达、原 KKT/Finish-PF/事件物理、全局 MILP 合法解码及 `Reward=-(10×惩罚E2E+持有成本+差分切换成本)`。不修改物理容量、Reward、到达率或请求轨迹。

- Agent-D：自主 KEEP/BUILD + 28×32 完整离散部署（Refresh、1..32 Count、节点有序抽样），条件策略梯度及原 80 秒物理信用、阶段概率修正；不使用 PPO、CPU 硬掩码、副本配额或在线候选布局池。
- Agent-S：Masked Dueling Double DQN，每请求逐阶段选择合法副本执行节点；仅主自然轨迹进入 Replay。
- R12-A 稳定性修复：恢复配置 D 学习率 0.0003；修正宏观 BUILD 与条件部署梯度相乘关系；适度熵正则与 KL 审计/Adam 回退（只在训练状态上）；验证改为固定 2 条轨迹各 4GS。

**先看 [完整失败诊断与修复依据](DIAGNOSIS_AND_FIX.md)**、[研究员交接](RESEARCHER_HANDOVER.md)。没有打包任何预训练模型、历史测试原始大文件，也不修改 R12 原始实验数据。

## Windows PowerShell

确保 Python 安装包含 PyTorch、NumPy、SciPy；如果原本已正确安装 CUDA PyTorch 2.11，**不必强行降级**。缺失依赖时运行 `python -m pip install -r requirements.txt`。

```powershell
cd C:\Users\1\Desktop\edge_joint_R12A_stability_fix
Set-ExecutionPolicy -Scope Process Bypass -Force

# 从零开始：两个种子建议分别运行
.\run.ps1 train --seed 20261114 --steps 32
.\run.ps1 train --seed 20261113 --steps 32

# 若 32GS 结果通过独立评价，继续相同轨迹至 100GS
.\run.ps1 train --seed 20261114 --steps 100 --resume
.\run.ps1 train --seed 20261113 --steps 100 --resume

# 原始 Reward/E2E 趋势（无平滑）
.\run.ps1 plot --seed 20261114 --arm A
.\run.ps1 plot --seed 20261113 --arm A

# 完全独立自然需求评价，禁止用于选择 best
.\run.ps1 eval --seed 20261114 --arm A --model best
.\run.ps1 eval --seed 20261113 --arm A --model best

# 同轮次仅 S 学习参考（单独的输出目录）
.\run.ps1 s-only --seed 20261114 --steps 32
```

Linux 等价命令是 `bash run.sh train --seed 20261114 --steps 32`。`--steps` 是累计 GS，不是追加数量。每 GS 对应 7200 逻辑秒但主物理是预登记随机短窗；D 信用为每周期 80 秒连续物理（H3）。实际运行的墙钟和请求数由日志审计。建议先检查 GS0/8/16/32，**不要因为单个异常点重选种子或删除失败**。

## 文件与恢复

每个训练种子唯一输出目录：
`r12/experiments/A/r51/results/<seed>_D/`

**仅保存 `latest.pt` 与 `best.pt` 两个模型**：`latest.pt` 含当前 D/S、D Adam、物理环境、RNG、压缩 Replay 和训练态 KL 审计锚点，用于 `--resume`；`best.pt` 仅含评估用 D/S 权重。其余 `train.jsonl`、`aux.jsonl`、`eval.jsonl`、`updates.jsonl`、`adam_audit.jsonl`、`wall.jsonl`、`best_metadata.json` 都是非模型数值日志，全部保留，不能作为额外的模型文件。

`best` 使用冻结需求种子 **975101、975103** 各4GS的平均真实 Reward，8GS。训练或独立测试不得读取 `best` 的成绩来挑选在线主动作。旧版本 `latest.pt` 不与新算法兼容；把旧工程完整备份后**在新解压目录从零训练**，不要覆盖旧模型。

独立冻结评价位于 `r12/independent/`，原生 D120 回归位于 `r12/d120/`。参考已跑通的 `run.ps1 eval` 和 `run.ps1 d120`。Windows 与 Linux 的 MILP 浮点求解路径可能不逐位一致，需比较请求哈希、合法性及统计，不宣称完全字节复现。

## 自检

```powershell
python -m unittest discover -s tests -p "test_stability.py" -v
```

完整工程/学术语义详见 `RESEARCHER_HANDOVER.md`；代码变更及原始根因详见 `DIAGNOSIS_AND_FIX.md`。**本版仅经过单元、真实主短窗及冻结 D80 标签回放验证，尚未经过两种子新 32/100GS 收敛验证。**

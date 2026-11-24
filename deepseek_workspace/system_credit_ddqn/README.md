# DeepSeek 系统级时延信用 DDQN

> **最终状态：有效负结果。** 信用数学和全局转移语义通过，但 64 GS 训练和
> 24/24 独立评价单元均未超过原局部信用或仅资源先验。详见
> [`AI_RESEARCH_REVIEW.md`](AI_RESEARCH_REVIEW.md)。

本目录是 DeepSeek 独立研究原型。主工程中的 `edge_d/`、`edge_s/`、
`edge_joint/`、`scripts/`、`tests/`、`research/` 和 `runs/` 均只读。

## 候选

Agent-S 不再使用“同一请求下一阶段”的局部奖励，而是：

1. 按真实业务期记录每个 cohort 的请求到达、完成和评分截止。
2. 计算所有尚未完成请求在相邻全局 S 调度决策之间的驻留面积。
3. 在 cohort 到达数确定后，以 `1 / n_p` 归一化。
4. 在 cohort 截止时计入一次 10 秒惩罚。
5. 将动作后的下一状态连接到下一次全局 S 决策的真实观测。
6. 使用 `gamma_S ** elapsed_seconds` 的 Double DQN 目标。

该系统奖励等于原请求级惩罚后时延的成本：

```text
r_j = -10 * sum_p (
    residence_integral(p, j) + 10 * deadline_count(p, j)
) / n_p
```

部署持有成本和切换成本仍只属于 Agent-D 的原环境 Reward。

## 代码结构

- `credit_ledger.py`：精确 cohort 恒等、事件扫描、全局转移与回放序列化。
- `simulator_adapter.py`：只读包装原物理仿真器，记录真实全局决策。
- `scheduling_agent.py`：保持 Dueling DDQN 和资源先验，使用事件时间折扣。
- `training.py`：系统信用训练、检查点、恢复和逐 episode 账本。
- `evaluate.py`：局部、系统信用和同 D 仅先验的冻结评价。
- `run.py`：注册、训练、评价和报告入口。
- `tests/`：信用、目标、物理、恢复和真实短训测试。

## 命令

从仓库根目录执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s deepseek_workspace/system_credit_ddqn/tests -v
python -m deepseek_workspace.system_credit_ddqn.run --stage register
python -m deepseek_workspace.system_credit_ddqn.run --stage train --arms system local_period --target-episodes 2
```

继续到 64 GS：

```powershell
python -m deepseek_workspace.system_credit_ddqn.run --stage train --arms system local_period --target-episodes 8 --resume
python -m deepseek_workspace.system_credit_ddqn.run --stage evaluate --arms system local_period --target-episodes 8 --resume
python -m deepseek_workspace.system_credit_ddqn.run --stage report
```

所有运行结果位于 `deepseek_workspace/system_credit_ddqn/results/`。大型模型和
原始结果不提交 Git。

## 严格边界

- 系统信用优化的是事件时间折扣目标，不声称与原未折扣周期 Reward 完全等价。
- 在线动作不使用候选矩阵池、真实未来模拟或反事实物理优化。
- 物理、容量、请求和原 Agent-D Reward 不允许改变。
- 16 GS 未通过基础检查时不推进 64 GS。
- 64 GS 独立评价未通过时，不得以 TD loss 下降或 Q 值变小替代真实收益。

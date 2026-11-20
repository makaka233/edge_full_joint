# 全工程架构（当前活跃版本）

## 数据流和单一物理后端

1. `edge_d.world.make_world(seed, scene, split)` 创建随机人口和需求状态，三个 split 隔离：`train`、`fixed`、`holdout`。
2. Agent-D (`edge_d.policy.EditActor`) 在覆盖阶段进行 ADD，在编辑阶段进行 ADD/REMOVE/RELOCATE/FINISH，从全零草稿构造完整 `(28,32)` 二值矩阵。无预生成完整候选池，无基于 Reward 的事后矩阵替换。
3. `edge_d.rollout.sample_layout(...)` 是完整矩阵提交入口，支持注入 Agent-S 调度器、采集原生调度转移，并允许在同一 Env 中连续多周期使用；旧 D-only 默认依然走固定 `LoadAwareHeuristic`。
4. `edge_d.physical.commit_warm(...)` 是共享的单一物理评价接口，调用隔离的 `_engine.environment.Env.commit` 和 `_engine.edge_rl.physics.Simulator`。实时选择节点执行每个就绪服务阶段；没有重新实现第二套物理环境。
5. `Simulator` 在 `collect=True` 下输出 `STransition(obs, mask, action, reward, next_obs, next_mask, terminated, ...)`；`edge_s.agent.SchedulingAgent` 用真实轨迹做 Masked Double DQN 更新。
6. `edge_joint.training` 负责 s-only 和 BUILD-only 联合训练；`edge_joint.periodic` 负责真实跨宏观周期持有旧矩阵，宏策略决定 KEEP 或 BUILD，D 构造或直接复用旧矩阵，S 在每周期真实请求事件中调度。

## 算法隔离

| 代码 | 状态与算法 | 更新目标 |
|---|---|---|
| `edge_d/training.py` | 原 D-only 失败基线，REINFORCE | 真实完整部署终局 Reward |
| `edge_s/agent.py` | Masked Dueling Double DQN；双网络、合法掩码、Huber TD、Polyak | 真实逐阶段局部时延回报 |
| `edge_joint/training.py::mode=s-only` | 随机初始化 D 权重冻结，仅训练 S | S 的真实调度转移 |
| `edge_joint/training.py::mode=joint` | D REINFORCE + S DDQN，同一次物理评价分别取信用 | D 整体 Reward；S 逐阶段回报 |
| `edge_joint/periodic.py` | 宏 Bernoulli KEEP/BUILD + D REINFORCE + S DDQN | 跨周期折扣的物理 Reward + S 局部回报 |

不同训练器不互相依赖，且不依赖旧 `r0/r2b/r12/r41` 脚本。已验证历史负结果存 `runs/archive/legacy/evidence/`；未将废弃网络和策略代码混入当前运行时。

## 外部合同与未解决缺陷

- D 对 S 只有通过真实物理 Reward 的间接反馈；尚无理论保证的一致双时间尺度 off-policy 更新。
- S 的每阶段 TD 更新使用固定 `gamma_s`，没有显式模拟动作间时间间隔折扣；局部回报不严格等同全局 E2E。
- 现行短窗口、尾部截尾和 CPU 门控可能显著影响 Reward 方差，不能视为已完善的系统 SLA 指标。
- KEEP/BUILD 已能真实执行和持有旧部署，但没有可学习的自适应部署时长；固定 `--periods` 是本参考协议的宏观运行次数。

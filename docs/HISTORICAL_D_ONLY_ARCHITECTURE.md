# 软件架构与模块边界

## 当前唯一执行链

```text
python -m edge_d.cli train
  └─ edge_d.training.train (保存 train/eval/updates JSONL, best.pt / last.pt)
       ├─ edge_d.world.make_world(seed, scene, split)
       │    └─ edge_d._engine (Config + Env + DynamicWorld)
       ├─ edge_d.policy.EditActor / choose_action
       │    ├─ coverage: 逐步 ADD 直到覆盖所有阶段
       │    └─ editing: 自主 ADD / REMOVE / RELOCATE / FINISH
       ├─ edge_d.actions.NativeLegalEdits (RAM/SSD + coverage)
       ├─ edge_d.world.cpu_lower_bound (理想 CPU min-max LP)
       └─ edge_d.physical.evaluate_layout (一次完整矩阵提交)
            ├─ 20s 原始请求流预热 + 2×6s 观测 + 8s tail
            ├─ 固定 LoadAwareHeuristic 替代 Agent-S
            └─ _engine.edge_rl.physics (原事件引擎与原始 Reward)
```

唯一活跃配置入口：`configs/world.json`（物理拓扑、服务、人群），`configs/experiment.json`（需求过程、窗口、成本和 CPU allocator）。旧的训练命令和历史论文代码都**不作为运行依赖**。

## 模块职责（替换算法时保持接口）

| 路径 | 唯一职责 | 后续可更换 |
|---|---|---|
| `world.py` | 产生可复现的独立需求世界；20维观测特征；CPU LP | 需求合约经对照后才改 |
| `policy.py` | 当前长序列 EDIT Actor、条件离散分布和精确 `log_prob` | **优先替换** |
| `actions.py` | ADD/REMOVE/RELOCATE 的矩阵合法性 | 应保留硬 RAM/SSD 规则 |
| `rollout.py` | 整张矩阵的策略构造、可选 FINISH CPU 门控、日志 | 可用整体动作生成器替换 |
| `physical.py` | 只接受合法完整矩阵并评估真实原始 Reward | 保留，直到物理协议独立审核 |
| `_engine/` | 从原工程提取的事件驱动物理后端代码 | 不是任何历史训练器 |
| `training.py` | 当前 REINFORCE + LOO、固定验证选模、独立 holdout | **优先替换** |
| `cli.py` | 唯一命令行入口 | 保持稳定 |
| `plotting.py` | 原始 Reward/E2E/构造步，未平滑未删除 | 保留 |

## 清理原则

1. 没有通配导入、旧 Rxx 训练模块或测试场景绑死的绝对路径；
2. 旧方法的研究证据只保存在 `docs/HISTORY.md` / `evidence/`，不能从主路径导入；
3. `_engine/` 的物理实现优先保持算法对照时稳定；未来必须通过相同矩阵、相同请求流的回归测试才允许修改；
4. `sample_layout()` 返回真实物理评价字典与可导动作 `log_prob` 列表。现有编码将整个内部构造视为一个完整部署物理样本。
5. 不存在自动强制 FINISH、行动合法化投影或根据 Reward 重新选部署；达到 1100 个内部动作仍未结束时显式异常。
6. 并发请求的 CPU 和有线通信竞争发生在 `_engine` 的连续时间离散事件物理层，Agent-D 内部构造步不推进时间。

## 关键内部数据合同

- `X`: `np.ndarray`，shape `(28,32)`，dtype bool。
- `state`: `(base, resource_dynamic, structural_context, draft, legal_mask, phase, finish_index)`；`base` shape `(28,32,20)`。
- `Action`: 四元组 `(kind,stage,node,destination)`；REMOVE 不允许移除最后一个副本；RELOCATE 原子完成。
- `sample_layout`: record 至少包含 `reward, penalized_e2e, censored, cpu_rho_lower_bound, decisions, kind_counts, matrix_packed_hex, physical_trace_sha256`。
- `train.jsonl`: 每一条代表**一张实际提交并得到真实物理评价的完整矩阵**。不需要模型文件即可用 packed matrix 恢复实际部署。
- `updates.jsonl`: 每次优化器更新一个记录；`update` 不是逻辑天数。
- `fixed`: 仅做学习跟踪与 best 模型选择；`holdout`: 独立 population，起点与结束评价。

## 与目标最终系统的差距

当前工程不包含 Agent-S 的 DDQN 训练、D/S 从随机初始化联合训练、长时间 KEEP/REDEPLOY 策略与差分迁移。**这些不能因为物理后端仍支持多阶段任务就声称已实现**。后续可通过新增清晰模块接入，不能把旧 R12A 的复杂训练器重新堆到主入口。

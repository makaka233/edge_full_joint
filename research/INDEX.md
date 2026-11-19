# 研究实验索引

| ID | 研究问题 / 算法 | 训练长度 | 结论 |
|---|---|---|---|
| [EXP001](experiments/EXP001_direct_placement_64gs/README.md) | 直接矩阵 D，固定启发式 S | 双种子各 64 GS | 独立 E2E 与 Reward 初步改善 |
| [EXP002](experiments/EXP002_scheduler_comparison/README.md) | 原 S、BUILD 联合、启发式对照及权重交换 | 两臂双种子各 64 GS | 联合胜初始化，未胜启发式；S-only 有尾部反例 |
| [EXP003](experiments/EXP003_periodic_keep_build/README.md) | 四期 KEEP/BUILD 与门对照 | 双种子各 64 GS | 有收益，但门未通过自适应验收 |
| [EXP004](experiments/EXP004_allocator_and_censoring/README.md) | 80 s 连续分配器与截尾/排空 | 8 正常 + 2 压力配对，非训练 | 不切换默认；截尾不等于丢弃 |
| [EXP005](experiments/EXP005_reward_spikes/PROTOCOL.md) | Reward 尖峰、库存协调解码与 DDQN 时延先验 | 双种子 64 GS 对照；周期仅审计 | [根因](../SPIKE_ROOT_CAUSE.md) · [修复结论](../SPIKE_FIX_REPORT.md) |
| [EXP006](experiments/EXP006_generalization/README.md) | 四组消融、未见需求、连续积压和周期归因 | 64 GS × 双种子 × 两模式；3 个未见需求种子 | [原图](experiments/EXP006_generalization/REWARD_RAW_INDEX.md) · [周期审计](experiments/EXP006_generalization/PERIODIC_AUDIT.md) |
| [EXP007](experiments/EXP007_macro_credit/README.md) | 冻结 D/S、跨期连续队列、配对 KEEP/BUILD 信用 | 240 期 × 双种子；独立 3×8 期 | [原始图](experiments/EXP007_macro_credit/REWARD_RAW_INDEX.md) · [信用差分](experiments/EXP007_macro_credit/CREDIT_AUDIT.md) |
| [EXP008](experiments/EXP008_agent_d/README.md) | 冻结宏观门和 S、完整矩阵资源残差 Actor | 每学习臂 128 GS × 双种子 | [Actor 审计](experiments/EXP008_agent_d/ACTOR_AUDIT.md) · [预注册](experiments/EXP008_agent_d/PROTOCOL.md) |
| [HIST001](history/credit_assignment/SUMMARY.md) | FINISH / ActionQ / 配对信用 | 历史物理对照 | 跨状态与固定验证未通过 |
| [HIST002–005](history/INDEX.md) | 部署、尖峰、物理、联合历史 | 见主题档案 | 历史、非主训练路径 |

[登记表](registry.json) 是制图与归档清单的机器入口；[当前基线](CURRENT_BASELINE.md) 定义实验比较约束。编号是研究管理 ID，不是算法命名。

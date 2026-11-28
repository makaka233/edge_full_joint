# EXP081 · EXP061 B 组结构化 Actor 五种子重训

源代码起点 `d749c1039b1a26776a3e23e4834e171da89ff404`。报告生成版本 `82f3fea733ab495db60c3eaf0eba3b258a54cb40`。
完成至 GS5040 的训练 **0/5**，独立评价 **0/30**。
使用 EXP061 B 组 92 维结构化 Gaussian Actor、资源先验 DDQN、固定两小时部署与原始 Reward。起点、恢复及评估协议见同目录 `PROTOCOL.md`。

| 种子 | GS / D / S | 最低原始 R | 严重 / 截尾 / 未完成 | best GS | 训练秒 |
|---|---|---:|---|---:|---:|

## 独立评价

| 种子/场景 | Reward GS0 → 末次 | 完成 E2E（s） | P99（s） | 末次 CPU CU/有线 Mbit | 末次严重/截尾/未完成 |
|---|---|---|---|---|---|

B 组烟测完成 GS2 保存及 GS4 恢复。核对了 92 维动作、结构化阶段分支实际更新、2 次 D 更新、32 次 S 更新及一组同流独立评价。

原始记录和每种子的 `best.pt`、`last.pt`、`resume_latest.pt` 保存在 `runs/results/EXP081/B/seed<seed>/`。

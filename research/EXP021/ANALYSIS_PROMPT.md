# EXP021 离线分析提示词

请从主工程读取 `runs/results/EXP021/randomness.seed20261113.jsonl` 与 `randomness.seed20261114.jsonl`，并按 `research/EXP021/PROTOCOL.md` 复算归因。使用 `python -m scripts.exp019_report --d-randomness-review` 只复算现有数据。核对六个状态的同流、期初状态和 A/B/C/D 成对扰动，分开报告高斯扰动、解码随机源及两者交互，保留每一个真实物理结果。历史热点只能用已存的成熟策略重建。其对照与自然和压力场景可核对的期初证据应分别说明。

如果我明确提供了新训练前缀，再审核对应训练和独立评价。核对随机初始化、双种子在线提交预算、原 Reward 和相同需求流。逐点画未平滑的 Reward、真实完成 E2E、惩罚后 E2E、P95/P99 和截尾。比较最差 1%/5%、热点积压及固定验证选出的 best 和末次 last。用 `python -m scripts.exp019_report --s-update-review --prefix "<提供的前缀>"` 核对每次优化的回放 Reward 与实测阶段时延，检查 S 更新计数、批次来源 GS、未完成和缺计时样本。完整请求 E2E 独立报告。只分析已有数据，将实际结果写入 `research/EXP021/AI_RESEARCH_REVIEW.md`，图写回 `runs/results/EXP021/`。

# R12-A 稳定性修复版：工作入口

1. 阅读 `DIAGNOSIS_AND_FIX.md`（真实100GS负结果）、`README.md`（启动）、`RESEARCHER_HANDOVER.md`（完整项目与历史）。
2. `python -m unittest discover -s tests -p test_stability.py -v` 运行梯度/KL/模型恢复单元测试。
3. 在新的、未包含任何旧 `latest.pt` 的工程目录运行 `bash run.sh train --seed 20261114 --steps 32`，再运行种子 `20261113`。Windows 用 `./run.ps1 train --seed ... --steps 32`，PowerShell执行策略临时改为 Bypass。
4. `bash run.sh plot --seed 20261114 --arm A` 生成未平滑的在线/固定验证 Reward/E2E；独立评价用 `bash run.sh eval --seed ... --arm A --model latest`。此外用 `bash run.sh s-only --seed ... --steps 32` 评估S-only学习贡献。
5. 如果从初始到GS32的两种子原始和固定验证指标有真实进展、没有系统性崩溃，再 `--resume --steps 100`；否则保留失败并报告。不能因为结果不佳就改变需求种子。
6. 训练仅保留 `best.pt`、`latest.pt`，其中 latest 含最新 Replay/Adam/env/rng/自然训练状态KL锚点。其他原始日志完整保留。

> 此发行版修复了已证实的梯度权重与学习率失配，**没有完成新的32/100GS长训验收**。用户上传的旧100GS负结果不得转称新版本结果。

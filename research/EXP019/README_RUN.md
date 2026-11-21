# EXP019 瞬时矩阵切换运行指南

## 双种子固定 REDEPLOY 基线

当前训练、物理、配置、测试和 EXP019 报告入口以 [EXP019 双种子长训审计提交](https://github.com/makaka233/edge_full_joint/commit/7681e015f0755f8dbbfd44b490f10d495f7cf130) 为源。正式基线使用原 Masked Dueling Double DQN、库存协调高斯 D 和 Adam。两个种子各从随机初始化执行 375 个 episode，每个 episode 8 个 24 秒业务期，共 3000 GS。固定验证每 8 个 episode 执行一次。CLI 的历史默认宏观模式为 `learned`，固定重部署须显式指定 `--macro-mode redeploy`。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
foreach ($seed in 20261113, 20261114) {
    $prefix = ".\runs\results\EXP019\instant.redeploy.baseline-rerun.seed$seed"
    python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --episodes 375 --periods 8 --period-seconds 24 --eval-every 8 --prefix $prefix
    if ($LASTEXITCODE -ne 0) { throw "EXP019 baseline training failed for $seed" }
}
```

恢复时对相同训练命令追加 `--resume`。评价使用本次运行前缀，宏观模式由检查点读取。以下是种子 20261113 的评价示例。

```powershell
python -m edge_d.cli adaptive evaluate --seed 20261113 --prefix ".\runs\results\EXP019\instant.redeploy.baseline-rerun.seed20261113"
```

历史正式结果使用 `instant.redeploy.long3000.seed<seed>` 前缀。可用下列命令离线复算当时的双种子结果。

```powershell
python -m scripts.exp019_report --long-redeploy
```

后续候选按原历史分支复现。分位数 DDQN 和尖峰诊断见 [EXP019 后续对照](https://github.com/makaka233/edge_full_joint/blob/exp019-adaptive-deployment/research/EXP019/README_RUN.md)，其余见 [EXP020](https://github.com/makaka233/edge_full_joint/tree/exp020-deployment-kl/research/EXP020)、[EXP021](https://github.com/makaka233/edge_full_joint/tree/exp021-deployment-randomness/research/EXP021) 和 [EXP022](https://github.com/makaka233/edge_full_joint/tree/exp022-resource-placement/research/EXP022)。

## 环境与轻量验证

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pip install -r requirements.txt
python -m edge_d.cli adaptive --help
python -m unittest discover -s tests -p test_adaptive_training.py -v
```

如需复算已验证的历史受阻案例，运行下行。它读取旧日志，用随机初始化的原 Agent-S 重建首个 episode，并在 GS2 或 GS4 单次替换目标矩阵。输出为 `instant.historical.seed20261113.replay.jsonl`。

```powershell
python -m edge_d.cli adaptive verify --seed 20261113 --replay-from ".\runs\results\EXP019\from_zero.seed20261113" --replay-gs 2 4 --prefix ".\runs\results\EXP019\instant.historical.seed20261113"
```

## 由用户启动的有界对照

两种子、三个宏观模式各跑 8 个 episode，每个 episode 八个 24 秒业务期，合计每组 **64 个在线 GS**。每组 D/S 和门均从随机参数初始化，前 8 次更新在同种子下收到逐期相同的外生请求流。两固定组只更新 D/S 与宏观价值头，学习组同时更新宏观策略。固定验证在 GS0、16、32、48、64 执行。该预算用于观察机制和初始学习。

```powershell
foreach ($seed in 20261113, 20261114) {
    foreach ($mode in 'keep', 'redeploy', 'learned') {
        $prefix = ".\runs\results\EXP019\instant.$mode.seed$seed"
        python -m edge_d.cli adaptive train --seed $seed --macro-mode $mode --episodes 8 --eval-every 2 --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "EXP019 training failed for $mode / $seed" }
    }
}
```

首次运行保留这套前缀。重新开展独立运行时可另选带 `instant.` 的文件名前缀，并对所有比较组采用同一配置。旧阻塞版模型使用格式 1，瞬时切换使用格式 2。回放历史版本请用训练清单记录的 Git 提交。瞬时切换训练需从全新前缀开始。

## 继续中断的同一运行

以 Seed13 的学习门为例。恢复时传入与首次运行相同的 seed、宏观模式、固定验证间隔和前缀。绝对目标 `--episodes 8` 表示最多补到 GS64。

```powershell
python -m edge_d.cli adaptive train --seed 20261113 --macro-mode learned --episodes 8 --eval-every 2 --prefix ".\runs\results\EXP019\instant.learned.seed20261113" --resume
```

## 独立评价

每个训练运行在三个未见场景评价 best、last，以及用该运行 last 的相同 D/S 做固定 KEEP 与固定 REDEPLOY 对照。评价从训练检查点读取宏观模式。

```powershell
foreach ($seed in 20261113, 20261114) {
    foreach ($mode in 'keep', 'redeploy', 'learned') {
        $prefix = ".\runs\results\EXP019\instant.$mode.seed$seed"
        python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "EXP019 evaluation failed for $mode / $seed" }
    }
}
```

## 本地结果

所有训练、评价和重放写入 `runs/results/EXP019/`。各组通过文件名前缀区分。历史文件 `from_zero.seed20261113.*`、`no_keep.seed20261113.*` 继续用于旧版分析。

| 后缀 | 内容 |
|---|---|
| `.manifest.json` | 物理模型、初始化、有效配置、Git 提交、工作树状态和场景。 |
| `.train.jsonl`、`.updates.jsonl` | 逐期物理轨迹和逐 episode 优化记录。 |
| `.fixed.jsonl`、`.holdout.jsonl` | 固定验证和同流独立评价。 |
| `.best.pt`、`.last.pt`、`.resume_latest.pt` | 固定最优、末次模型和完整恢复状态。 |
| `.replay.jsonl` | 历史受阻状态的物理对照。 |

正式实验结束后，把[离线审核提示词](ANALYSIS_PROMPT.md)交给 Codex。上下文及已观察结果见[单文件交接](AI_RESEARCH_REVIEW.md)。

## 固定 REDEPLOY 的冻结算法基线

以下入口只读取指定运行的 best 和 last 模型。它在两场固定验证、三个自然 Holdout 及两个独立压力场景评价相同请求流。基线包含随机初始化与 best 的 D/S 交叉、末次模型、固定库存协调解码与负载感知启发式。两个压力场景的首期目标全网 CPU 提供负载为 0.55 和 0.70，此后继续按原需求过程演化。

```powershell
python -m edge_d.cli adaptive baseline --seed 20261113 --prefix ".\runs\results\EXP019\instant.redeploy.long3000.seed20261113"
```

命令输出 `*.baseline.gs<N>.jsonl`。使用该文件名报告的 GS 复算指标和图表。

```powershell
$gs = [int](Read-Host 'Baseline output GS')
python -m scripts.exp019_report --prefix instant.redeploy.long3000.seed20261113 --baseline-gs $gs
```

计算的物理界只放宽竞争与部署限制，用于衡量可能的改善空间。各算法的可实现表现来自连续事件仿真。

复算已有 GS64 权重的双种子早期归因，只需跑两场固定验证。

```powershell
foreach ($seed in 20261113, 20261114) {
    python -m edge_d.cli adaptive baseline --seed $seed --fixed-only --prefix ".\runs\results\EXP019\instant.redeploy.seed$seed"
    if ($LASTEXITCODE -ne 0) { throw "Fixed baseline failed for seed $seed" }
    python -m scripts.exp019_report --prefix "instant.redeploy.seed$seed" --baseline-gs 64 --baseline-fixed
    if ($LASTEXITCODE -ne 0) { throw "Fixed baseline report failed for seed $seed" }
}
```

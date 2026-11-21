# EXP019 瞬时矩阵切换运行指南

## 固定 REDEPLOY 尾风险对照

先前长训使用原 DDQN。新增的分位数 DDQN 只有显式设置 `--s-algorithm quantile` 才会使用。双种子、两算法各从随机初始化运行 16 个 episode，即 128 GS。两组的 D、需求、奖励、物理条件、探索及更新预算相同。固定验证间隔为两个 episode，固定验证净 Reward 决定 best。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
foreach ($seed in 20261113, 20261114) {
    foreach ($algo in 'ddqn', 'quantile') {
        $prefix = ".\runs\results\EXP019\instant.tail.$algo.seed$seed"
        python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --s-algorithm $algo --episodes 16 --eval-every 2 --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "Tail trial training failed for $algo / $seed" }
    }
}
```

中断后只恢复相应种子和算法，保留所有参数和绝对 episode 目标。下例恢复 Seed13 分位数训练。

```powershell
python -m edge_d.cli adaptive train --seed 20261113 --macro-mode redeploy --s-algorithm quantile --episodes 16 --eval-every 2 --prefix ".\runs\results\EXP019\instant.tail.quantile.seed20261113" --resume
```

独立评价先运行三场自然 Holdout，再运行四个从未参与训练和选模的持续压力场景。19301/19302 每期保持全网 CPU 提供负载 0.55，19303/19304 为 0.70，用原需求过程演化区域和服务组成。所有比较共享请求流。持续压力只用于独立评价。

```powershell
foreach ($seed in 20261113, 20261114) {
    foreach ($algo in 'ddqn', 'quantile') {
        $prefix = ".\runs\results\EXP019\instant.tail.$algo.seed$seed"
        python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "Natural Holdout failed for $algo / $seed" }
        python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix --sustained
        if ($LASTEXITCODE -ne 0) { throw "Sustained Holdout failed for $algo / $seed" }
    }
}
python -m scripts.exp019_report --tail-trial
```

报告入口对账真实训练提交、同流、部署、截尾、最差 1%/5% 并生成完整原始图。结果统一位于 `runs/results/EXP019/`，使用 `instant.tail.*` 前缀。已有长训权重和日志仍使用原前缀。请在训练和评价完成后明确告知 Codex 进行离线审核。

如需重做两个历史诊断，只运行以下冻结入口。它读取已有长训文件，写入单独的 `.spike.gs*.json`。

```powershell
foreach ($seed in 20261113, 20261114) {
    python -m edge_d.cli adaptive spike-audit --seed $seed --prefix ".\runs\results\EXP019\instant.redeploy.long3000.seed$seed"
    if ($LASTEXITCODE -ne 0) { throw "Historical spike audit failed for $seed" }
}
python -m scripts.exp019_report --spike-review
```

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

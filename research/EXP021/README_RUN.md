# EXP021 本地复现

从主工程运行。输入权重和原始对照日志位于 `runs/results/EXP019/` 与 `runs/results/EXP020/`。归因和图表输出统一写入 `runs/results/EXP021/`。请在同一个已备份的源码版本上保留两种子的输入。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pip install -r requirements.txt
python -m edge_d.cli adaptive --help
python -m unittest discover -s tests -p test_adaptive_training.py -q
python -m unittest discover -s tests -p test_original_fidelity.py -q
```

两条诊断命令复算已登记的焦点。每个种子共 111 次焦点提交，并为每次提交复放相同的队列前序。它们读取已存权重和日志，不进行参数更新。复跑会以原子写入替换同种子的 EXP021 归因输出。

```powershell
foreach ($seed in 20261113, 20261114) {
    python -m edge_d.cli adaptive d-randomness --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "EXP021 diagnosis failed for seed $seed" }
}
python -m scripts.exp019_report --d-randomness-review
```

已有归因 JSONL 时，可以直接执行最后一行，生成逐点 Reward、E2E 和 P99 图及统计。具体归因结果、候选启动判定和证据链接见[单文件交接](AI_RESEARCH_REVIEW.md)。

## 原 Adam 运行及逐 S 更新图

EXP021 未达到状态条件方差候选的启动门槛。以下命令用于需要独立复跑原 Adam 512 GS 并采集新 S 日志时。每个种子从零训练，64 episode × 8 期 = 512 GS。训练完成后分别执行自然与持续负载评估。

```powershell
foreach ($seed in 20261113, 20261114) {
    $prefix = ".\runs\results\EXP021\instant.adam.seed$seed"
    python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --s-algorithm ddqn --d-update adam --episodes 64 --periods 8 --period-seconds 24 --eval-every 8 --prefix $prefix
    if ($LASTEXITCODE -ne 0) { throw "EXP021 training failed for $seed" }
    python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix
    if ($LASTEXITCODE -ne 0) { throw "EXP021 natural Holdout failed for $seed" }
    python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix --sustained
    if ($LASTEXITCODE -ne 0) { throw "EXP021 sustained Holdout failed for $seed" }
    python -m scripts.exp019_report --s-update-review --prefix $prefix
    if ($LASTEXITCODE -ne 0) { throw "EXP021 S-update report failed for $seed" }
}
```

恢复时单独执行对应组的同一训练命令，保留 `--episodes 64` 和原前缀，再追加 `--resume`。逐 S 优化记录在 `<prefix>.s_updates.jsonl`，绘图写在同目录的 `<prefix>.s_updates.png`。秒数为从阶段就绪到完成的实测物理时长。旧检查点继续运行时，从其真实 S 更新号之后开始记录，标明没有计时的旧回放样本。原始请求级 E2E 见 `<prefix>.train.jsonl` 与评价日志。

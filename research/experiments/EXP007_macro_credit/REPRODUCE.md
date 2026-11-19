# EXP007 · PowerShell 复现

先按 [EXP005 复现](../EXP005_reward_spikes/REPRODUCE.md) 生成 `runs/spike_batch` 中两种子的 `residual_placement_joint_*`（固定 D/S）及 `baseline_placement_periodic_*`（原宏观门）固定最佳模型。仅调度敏感性需按 [EXP006 复现](../EXP006_generalization/REPRODUCE.md) 生成 `scheduler_placement_joint_*`，将其训练目录整理到 `runs/exp006/training`，或提供 `--batch`。

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp007 --seed $seed --out "runs/exp007/residual_$seed"
    if ($LASTEXITCODE -ne 0) { throw "Macro training failed: $seed" }
    python -m scripts.exp007 --seed $seed --arm scheduler --gate "runs/exp007/residual_$seed/gate.json" --out "runs/exp007/scheduler_$seed"
    if ($LASTEXITCODE -ne 0) { throw "Frozen sensitivity failed: $seed" }
    python -m scripts.exp007 --seed $seed --followup --out "runs/exp007/followup_$seed"
    if ($LASTEXITCODE -ne 0) { throw "Historical continuation failed: $seed" }
}
python -m scripts.exp007_report --work runs/exp007 --out runs/exp007_rebuilt
```

不训练直接从包内原始数据重画：先复制 `evidence/macro_credit/EXP007/raw` 至 `runs/exp007_replot/raw`，再执行 `python -m scripts.exp007_report --out runs/exp007_replot`。包不含冻结的 D、S 或门的参数；可按所附协议重训。结果由逐点轨迹产生，不平滑、删减或替换原分数。

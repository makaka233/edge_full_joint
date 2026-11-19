# EXP009 · PowerShell

先按 [EXP005](../EXP005_reward_spikes/REPRODUCE.md) 生成两种子原 BUILD 联合的固定最佳 `runs/spike_batch/baseline_placement_joint_13/14/best.pt`。固定 S 权重不上传；可为每种子显式传 `--checkpoint`，哈希写入运行配置。所有输出目录用新路径。

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
python -m unittest discover -s tests -p test_exp009.py -v
python -m scripts.exp009 audit --out runs/exp009/audit
# 不需长训的闭环烟测
python -m scripts.exp009 fit --seed 20261113 --arm spatial --gs 2 --out runs/exp009/smoke
# 正式原版（两个种子各重复一次，输出相应更名）
python -m scripts.exp009 fit --seed 20261113 --arm original --gs 128 --out runs/exp009/original_20261113
```

正式库存与空间臂各启动独立终端运行 `python -m scripts.exp009 fit --seed 20261113 --arm inventory --gs 256 --out runs/exp009/inventory_20261113` 和相同命令的 `--arm spatial --out runs/exp009/spatial_20261113`。对于种子 20261114 重复。两臂在 128 等待协同证据；以下代码按预注册标准决定是否延长：

```powershell
foreach ($seed in 20261113,20261114) {
    $inv = Get-Content -Raw "runs/exp009/inventory_$seed/gs128.json" | ConvertFrom-Json
    $spatial = Get-Content -Raw "runs/exp009/spatial_$seed/gs128.json" | ConvertFrom-Json
    $gate = @{ continue=($spatial.worst -ge ($inv.worst-10) -and $spatial.spikes -le ($inv.spikes+2)); inventory=$inv; spatial=$spatial }
    $target = "runs/exp009/continuation_$seed.json"
    $gate | ConvertTo-Json | Set-Content -LiteralPath "$target.tmp" -Encoding ASCII
    Move-Item -LiteralPath "$target.tmp" -Destination $target
}
python -m scripts.exp009_report --work runs/exp009
python -m scripts.research_review --experiment EXP009 --out AI_RESEARCH_REVIEW.md
```

如果只重画原图，直接运行 `python -m scripts.exp009_report`，读 GitHub 归档，不需要本地权重。审阅文档可用 `--verify` 核对。训练可生成 `best.pt`、`last.pt`，不进入提交。

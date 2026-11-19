# EXP008 · PowerShell 复现

先按 [EXP005 命令](../EXP005_reward_spikes/REPRODUCE.md) 生成两种子的原 BUILD 联合 `baseline_placement_joint_13/14/best.pt`。该权重只用于独立交叉评价，训练 D 均从随机参数开始。也可用 `--checkpoint` 指定等价的本地文件，哈希记录在 EXP008 配置。原始 D/S/KKT/KEEP 接口保留。

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
foreach ($seed in 20261113,20261114) {
    foreach ($arm in 'original','inventory','resource','heuristic') {
        python -m scripts.exp008 --seed $seed --arm $arm --gs 128 --out "runs/exp008/${arm}_$seed"
        if ($LASTEXITCODE -ne 0) { throw "EXP008 failed: $arm $seed" }
    }
}
python -m scripts.exp008_report --work runs/exp008 --out runs/exp008_rebuilt
python -m scripts.research_review --experiment EXP008 --out dist/AI_RESEARCH_REVIEW.md
```

不重训直接从归档重画：先复制 `evidence/deployment_learning/EXP008/raw/` 到新 `runs/exp008_replot/raw/`，再执行 `python -m scripts.exp008_report --out runs/exp008_replot`。ZIP 不附任何 D/S 权重；同协议重训可得到 `best.pt` 和 `last.pt`。物理评估全部原样归档。训练原图使用不同标记分别展示同状态的正/负分支；两者均真实提交并用于估计梯度，绝非跨时演进的在线经验。

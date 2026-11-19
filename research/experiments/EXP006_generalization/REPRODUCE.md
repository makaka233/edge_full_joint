# EXP006 · 复现与数据路径

[预注册协议](PROTOCOL.md)。源代码不替换原算法，`scripts.exp006` 只组织冻结模型、独立压力、宏观反事实及缺失消融。模型权重不交付。先按 [EXP005 复现](../EXP005_reward_spikes/REPRODUCE.md) 在本地重训/生成三组固定最佳文件；同义臂训练数据以 EXP005 原始归档为准。

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$runs = 'placement_joint_13','placement_joint_14','placement_s_13','placement_s_14'
foreach ($run in $runs) {
    foreach ($arm in 'baseline','candidate','residual') {
        foreach ($seed in 20261201,20261202,20261203) {
            python -m scripts.exp006 pressure --run $run --arm $arm --demand-seed $seed --out "runs/exp006/pressure/${run}_${arm}_$seed.jsonl"
            if ($LASTEXITCODE -ne 0) { throw "Frozen evaluation failed" }
        }
    }
}
foreach ($run in $runs) {
    python -m scripts.exp006 train --run $run
    if ($LASTEXITCODE -ne 0) { throw "Scheduler-only training failed" }
    foreach ($seed in 20261201,20261202,20261203) {
        python -m scripts.exp006 pressure --run $run --arm scheduler --demand-seed $seed --out "runs/exp006/pressure/${run}_scheduler_$seed.jsonl"
        if ($LASTEXITCODE -ne 0) { throw "Scheduler-only evaluation failed" }
    }
}
foreach ($run in 'placement_periodic_13','placement_periodic_14') {
    python -m scripts.exp006 periodic --run $run --out "runs/exp006/$run"
    if ($LASTEXITCODE -ne 0) { throw "Periodic replay failed" }
}
python -m scripts.exp006_report --work runs/exp006 --out runs/exp006_rebuilt
```

仅重新绘图（不需要权重或再次训练）：

```powershell
New-Item -ItemType Directory -Force runs/exp006_replot
Copy-Item -Recurse evidence/generalization/EXP006/raw runs/exp006_replot/raw
python -m scripts.exp006_report --out runs/exp006_replot
```

逐点评价有 `short.trace_hash`、`continuous.trace_hash`、位图、检查点 GS；同需求比较自动校验哈希一致。短窗原 Reward 与连续 E2E 分开，容量不变。所有时延分位数在请求层合并，图按场景合并三个未见种子，不对每场景的分位数再取均值。墙钟列报告原 run 的实际耗时，含训练、固定评价以及相应基线的历史审计成本；比较算法时额外看训练提交物理耗时。

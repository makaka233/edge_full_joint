# EXP005 · Windows 复现

工作目录为解压根目录；创建新环境并安装 `requirements.txt`。先看 [固定协议](PROTOCOL.md) 和 [调度消融协议](SCHEDULER_ABLATION.md)。每个输出目录必须为空；权重只存在于本地运行目录，不加入交付。重放由实时 S 训练状态生产，不能用最终权重冒充历史逐 GS 决策。

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$names = 'd_13','d_14','s_13','s_14','joint_13','joint_14','periodic_13','periodic_14'
foreach ($name in $names) {
    $run = "placement_$name"
    python -m scripts.spike_experiment $run --out "runs/spike_batch/baseline_$run"
    if ($LASTEXITCODE -ne 0) { throw "Baseline failed: $run" }
    if ($name -notlike 'periodic*') {
        python -m scripts.spike_experiment $run --candidate --out "runs/spike_batch/candidate_$run"
        if ($LASTEXITCODE -ne 0) { throw "Inventory arm failed: $run" }
    }
    if ($name -like 's_*' -or $name -like 'joint_*') {
        python -m scripts.spike_experiment $run --queue-residual --out "runs/spike_batch/residual_$run"
        if ($LASTEXITCODE -ne 0) { throw "Residual arm failed: $run" }
    }
}
python -m scripts.spike_results runs/spike_batch --out runs/spike_rebuilt
python -m scripts.spike_results --redraw evidence/spike_study/EXP005 --out runs/replot_exp005
```

`baseline` 保持原解码和原 DDQN。`candidate` 只调解库存与运输支持。`residual` 在同一解码上加工作量时延先验。训练 GS 计数不包括额外审计、固定评价、Holdout。原 Holdout 为历史同协议场景；新 Holdout 为未参与选模的场景 40–47，依然用独立人口分组。多个进程同时运行的墙钟受竞争影响，不能作为单线程性能比较；独立报告每次物理评价耗时和总流程墙钟。

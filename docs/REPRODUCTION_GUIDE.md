# Windows PowerShell 复现

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& $py -m scripts.verify_package
& $py -m unittest discover -s tests -v
& $py -m scripts.verify_allocator_modes
& $py -m edge_d.cli plot --out runs/figures_regenerated
foreach ($seed in 20261113,20261114) {
    foreach ($mode in 'd-only','s-only','joint','periodic') {
        $extra = @()
        if ($mode -eq 'periodic') { $extra = @('--periods','4','--fixed-scenes','2','--holdout-scenes','4') }
        & $py -m edge_d.cli placement-train --mode $mode --seed $seed --out "runs/${mode}_${seed}" @extra
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $mode $seed" }
    }
}
& $py -m edge_d.cli plot --run-dir runs/d-only_20261113 --out runs/replot_d13
```

每个目录必须为空；训练器只保存 `best.pt` 和 `last.pt`，按固定 Reward 选最佳。最终 Holdout 只评估初始、固定最佳、最终，不反向选模。固定 Reward 按物理提交取算术均值；固定/holdout 惩罚后 E2E 按请求加权，完成 E2E 按完成请求加权。BUILD 使用 4 固定/8 holdout 场景，周期使用 2 固定/4 holdout 四期轨迹，不能把一个轨迹的四期当作四个独立样本。图名中 `raw`、`fixed_eval`、`holdout_eval` 不可互换。

新正式实验：分配未使用的 EXP ID，更新 [登记表](../research/registry.json) 和 [索引](../research/INDEX.md)，每臂用唯一输出目录名作为 run ID。保留原始 JSONL，用 gzip **字节无损压缩**写入 `evidence/raw/<ID>/<run>/`，将配置/训练清单与汇总分置 `manifests/` 和 `summaries/`，执行 `python -m scripts.research_artifacts` 生成图和实验清单；只重画到外部路径时使用 `--out`。已有源日志不得覆盖。更详细的字段语义见 [证据索引](../evidence/INDEX.md)。

发布检验：`python -m scripts.package_research --out dist/edge_full_joint_normalized.zip`，然后 `python -m scripts.accept_release dist/edge_full_joint_normalized.zip dist/new_acceptance`。验收在新目录中执行哈希、逐点对账、图表重绘、19 项回归及四模式短训，旁置带 ZIP 哈希的独立验收记录。PNG 字节重绘对比要求与交付绘图环境一致（Windows / Matplotlib 3.11.1）；不同环境可比较 JSONL 数值和显示内容。

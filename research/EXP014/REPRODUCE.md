# EXP014 复现

从仓库根目录运行以下 PowerShell 命令。用于绘图的旧训练与固定验证原始日志已随仓库保留，见 [EXP010](../EXP010/raw/) 与 [EXP012](../EXP012/raw/)。独立对照需要两组旧 Agent-D 权重和两种子的冻结原始 Agent-S。若本地尚无权重，按 [EXP010](../EXP010/REPRODUCE.md) 与 [EXP012](../EXP012/REPRODUCE.md) 的训练命令生成。冻结 Agent-S 读取 EXP005 原始 BUILD 联合实验固定验证最佳文件中的 `s` 键，探索率零。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp01[234].py' -v
New-Item -ItemType Directory -Force runs/exp014 | Out-Null
python -m scripts.verify_allocator_modes --out runs/exp014/allocator_modes.json
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp014 mechanism --seed $seed --out "runs/exp014/mechanism_$seed"
}
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp014 fit --seed $seed --registered runs/exp014 --out "runs/exp014/resource_$seed"
}
python -m scripts.exp014_batch --work runs/exp014 --inventory runs/exp010 --factorized runs/exp012 --schedulers runs/spike_batch --workers 7
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp014 marginal --seed $seed --source runs/exp014 --out "runs/exp014/marginal_$seed"
}
python -m scripts.exp014_report --work runs/exp014 --out research/EXP014
python -m scripts.exp014_review --out research/EXP014
```

只重画图表时运行 `python -m scripts.exp014_report --redraw --out research/EXP014`。已归档的 `train/fixed/short/probe/continuous` 为逐点评价，`updates` 为训练参数更新账本。库存协调训练与固定验证分别取 [EXP010](../EXP010/raw/) 的 `train.jsonl.gz` 与 `eval.jsonl.gz`，后者只读取 `split=fixed`。原因子化策略训练与固定验证读 [EXP012](../EXP012/raw/) 对应文件。图表自动生成脚本为 [exp014_figures.py](../../scripts/exp014_figures.py)。逐场景保留完整请求延迟向量，P95/P99 可重新计算。

新策略训练日志的 `path_stages` 和 `path_nodes` 列出按轮展开的已考察位置，`path_accepted` 按位编码其接受与拒绝，`path_masks` 每次按阶段行优先编码当时合法集合，28 位补齐为 4 字节。静态 rollout logits 可由确定性训练复现，另在每次更新存储焦点节点 logits 与库存响应权重。已归档参数账本含更新前和更新后系数。从零初始化权重出发，可用前一次更新后的系数校验当前更新前值。

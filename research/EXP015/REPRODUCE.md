# EXP015 复现

在仓库根目录运行。冻结 DDQN 来自 [EXP010 复现说明](../EXP010/REPRODUCE.md) 对应的 EXP005 原始 BUILD 联合最佳模型，评价使用其中的 `s` 键，探索率为零。原分布的训练记录已随 EXP010 与 EXP014 保留，权重可从原实验目录读取或按下列命令重训。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp01[2-5].py' -v
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp015 --original --seed $seed --arm inventory --out "runs/exp015/original/inventory_${seed}"
    python -m scripts.exp015 --original --seed $seed --arm resource --out "runs/exp015/original/resource_${seed}"
    python -m scripts.exp015 --seed $seed --arm inventory --out "runs/exp015/mixed/inventory_${seed}"
    python -m scripts.exp015 --seed $seed --arm resource --out "runs/exp015/mixed/resource_${seed}"
}
python -m scripts.exp015_evaluate batch --distribution original --out runs/exp015 --inventory runs/exp015/original --resource runs/exp015/original --checkpoint runs/spike_batch --workers 7
python -m scripts.exp015_evaluate batch --distribution mixed --out runs/exp015 --checkpoint runs/spike_batch --workers 7
```

若未通过升级验收，按协议追加对最终矩阵的离线关键阶段单副本诊断。

```powershell
foreach ($seed in 20261113,20261114) {
    foreach ($tag in 'original','mixed') {
        python -m scripts.exp015_evaluate marginal --distribution $tag --seed $seed --source "runs/exp015/$tag" --out "runs/exp015/diagnostic_${tag}_${seed}"
    }
}
python -m scripts.exp015_report --work runs/exp015 --out research/EXP015
python -m scripts.exp015_review --out research/EXP015
```

训练中的两条同流物理分支分别保留 `branch=positive/negative`，此标签不代表 Reward 的正负。两个分支各计一次 GS，调度动作与优化器更新另计。相同需求下的真实请求向量、矩阵、守恒、P99 和资源竞争全部可从 [原始记录](raw/) 核对。库存原分布的原始训练和固定验证见 [EXP010 原始记录](../EXP010/raw/)，节点并行原分布见 [EXP014 原始记录](../EXP014/raw/)。

只重画图表可运行 `python -m scripts.exp015_report --redraw --out research/EXP015`。

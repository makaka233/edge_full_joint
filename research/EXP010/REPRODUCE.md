# EXP010 复现

从仓库根目录在 PowerShell 运行。以下命令会生成实验文件和本地权重。输出路径需使用新目录。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp0[01][90].py' -v
foreach ($seed in 20261113,20261114) {
    python -m edge_d.cli placement-train --seed $seed --mode joint --gs 64 --out "runs/spike_batch/baseline_placement_joint_$($seed % 100)"
}
python -m scripts.exp010_batch --out runs/exp010 --checkpoints runs/spike_batch
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp010_diagnostics --work runs/exp010 --seed $seed --case counts --out "runs/exp010/counts_$seed"
    python -m scripts.exp010_diagnostics --work runs/exp010 --seed $seed --case continuous --out "runs/exp010/continuous_$seed"
}
python -m scripts.exp010_report --work runs/exp010 --out research/EXP010
python -m scripts.exp010_review --out research/EXP010
```

正式运行的冻结 S 来自 EXP005 原始 BUILD 联合实验相同种子的固定最佳权重，两个种子的最佳均为 GS56。提取 `best.pt` 中的 `s` 键。没有队列先验，评价探索率为零。原冻结权重若可用，可直接指定其两个目录的共同父目录。要同时复现已保留的 EXP009 回顾性证据，在生成 EXP010 汇总时加 `--exp009 <EXP009 原始运行目录>`。短窗测试用原版 `make_world` 和 `commit_warm`。连续队列用同一个原版 Simulator 处理 130 秒连续业务，前 20 秒预热，观察 [20,110) 秒到达请求至 130 秒。到达一直持续到 130 秒。

归档 `raw/` 保留每次完整矩阵、请求延迟向量、原始 Reward/成本、资源峰值和截尾，训练还保留完整合法掩码及同序 float32 logit。矩阵和掩码为行优先按位打包，logit 是小端 float32 的 Base64，采样肯定与否可从最终矩阵在合法掩码的位置恢复。用于诊断的详细资源时间积分保存在本地运行的 `windows`。原始 Torch 权重及运行临时文件保存在忽略的 `runs/`，便于重算与回滚。

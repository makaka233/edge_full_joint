# EXP012 复现

从仓库根目录运行。短窗回报维持双 6 秒观测、20 秒预热和最多 8 秒随访的原设置。连续队列采用同一个原版 Simulator 处理全部到达流，并以连续请求 E2E 和积压评价。两个协议没有混合计算 Reward。

旧组的真实训练和固定评估直接读取 `research/EXP010/raw/<arm>_<seed>/`，筛选固定分割。跨调度器重新评估需要之前训练好的本地 Agent-D 权重和 EXP005 原始 BUILD 联合的固定最佳 Agent-S 权重。若未保存，先按 [EXP010 复现命令](../EXP010/REPRODUCE.md) 生成。旧组权重均来自其 GS0、128、256 和固定最佳。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp012.py' -v
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp012 mechanism --seed $seed --out "runs/exp012/mechanism_$seed"
}
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp012 fit --seed $seed --out "runs/exp012/factorized_$seed" --registered runs/exp012
}
python -m scripts.exp012_batch --out runs/exp012/evaluation --baselines runs/exp010 --trained runs/exp012 --checkpoints runs/spike_batch
python -m scripts.exp012_report --work runs/exp012 --out research/EXP012
python -m scripts.exp012_review --out research/EXP012
```

`raw/` 中保留新组的原训练数据、每次参数更新、独立短窗、采样、持续队列数据、机制验证和双头反事实。旧组的原训练与固定验证从上述 EXP010 原始记录读取。三个模型重新评估使用共同物理请求流。本地运行的短窗原始日志另保留详细每阶段资源时间积分。模型权重留在本地 `runs/`。

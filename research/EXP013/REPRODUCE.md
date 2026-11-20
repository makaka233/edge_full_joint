# EXP013 复现

从仓库根目录运行。先保留 [EXP010](../EXP010/REPRODUCE.md) 的库存基线末次权重和 [EXP012](../EXP012/REPRODUCE.md) 的因子化末次权重。EXP012 的原始短窗、持续队列、训练点和固定验证数据已随仓库保留。

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp013.py' -v
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp013_audit --seed $seed --weights runs/exp012 --out "runs/exp013/audit_$seed"
    python -m scripts.exp013_audit --seed $seed --weights runs/exp012 --out "runs/exp013/order_$seed" --order-only
    python -m scripts.exp013 mechanism --seed $seed --exp010 runs/exp010 --exp012 runs/exp012 --out "runs/exp013/mechanism_$seed"
}
python -m scripts.exp013_report --work runs/exp013 --out research/EXP013
python -m scripts.exp013_review --out research/EXP013
```

机制比较使用冻结末次权重。数量和位置机制探针在每个配对前恢复同一初始权重，新建一次优化器，执行一次原 Reward 更新。它们互不形成正式训练轨迹。正式 `fit` 命令要求两个 `decision.json` 通过。本轮正式训练门槛未过。图上参考的双种子完整原始训练、固定验证、短窗和持续队列均标明 EXP012。

`raw/audit_*` 存原 GS256 对账、逐次拒绝时的共享资源、固定矩阵需求反事实和逐请求事件重放。`raw/order_*.jsonl.gz` 存不改变原候选前缀的逆序成对测试。 `raw/mechanism_*` 存所有新物理行、实际概率、两个头的初始 logits、动态掩码和逐阶段执行节点序列。原预热参数和两窗口请求流不变。实验权重留在本地运行目录。

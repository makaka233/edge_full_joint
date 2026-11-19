# EXP011 复现

从仓库根目录运行。先按 [EXP010](../EXP010/REPRODUCE.md) 生成同种子的库存、仅高斯、仅空间和联合本地权重，分别使用 GS0、128、256 所需检查点。运行输出使用新目录。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp0[01][19].py' -v
foreach ($seed in 20261113,20261114) {
    foreach ($part in 'fit','gate') {
        python -m scripts.exp011 --weights runs/exp010 --seed $seed --part $part --out "runs/exp011/${part}_$seed"
    }
    python -m scripts.exp011_gate --work runs/exp011 --weights runs/exp010 --seed $seed --out "runs/exp011/rank_$seed"
    foreach ($arm in 'inventory','gaussian','spatial') {
        python -m scripts.exp011_gate --work runs/exp011 --weights runs/exp010 --seed $seed --preference $arm --out "runs/exp011/${arm}_$seed"
    }
}
python -m scripts.exp011_trace --work runs/exp011 --recorded research/EXP011/raw/replay_inputs.jsonl.gz --out runs/exp011/trace
python -m scripts.exp011_report --work runs/exp011 --out research/EXP011
python -m scripts.exp011_review --out research/EXP011
```

场景 170/166 的原始矩阵、随机流和基准结果随 `raw/replay_inputs.jsonl.gz` 保存。初次手持 EXP009 目录时也可传 `--exp009 runs/exp009`。两种方式均用原 Simulator 开启追踪重放，核对同流与 E2E 后导出实际走过的有向链路、每阶段有线驻留和 CPU 驻留。短窗队列到达持续到随访截止。节点机会成本记录为其剩余资源内还能装下的其他阶段数。

训练参照图来自 EXP010 已完成的真实训练，其全部逐点数据可从 [EXP010 审核报告](../EXP010/AI_RESEARCH_REVIEW.md) 获取。EXP011 的每次物理提交、负样本、截尾和请求延迟向量位于本目录 `raw/`。原始矩阵为 28×32，按行打包。压力需求只作用于单次场景实例，容量、Reward 和原引擎保持原配置。

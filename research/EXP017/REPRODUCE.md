# EXP017 复现

从仓库根目录运行。取 [EXP016 原始账本](../EXP016/raw/) 中登记的矩阵、潜变量和随机流，由相同随机初始化重建相应版本的 S。诊断权重只从真实训练前缀得到。批处理使用独立进程。受限的 Windows 进程管道下，联合训练自动回落为逐个执行同一对物理评估。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp01[67].py' -v
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp017_forensics prepare --seed $seed --out "runs/exp017/weights_$seed"
}
python -m scripts.exp017_batch forensics --work runs/exp017 --workers 6
python -m scripts.exp017_batch pilot --work runs/exp017 --workers 6
python -m scripts.exp017_report --work runs/exp017 --out research/EXP017
python -m scripts.exp017_review --out research/EXP017
```

`raw/forensics/case_<seed>_<gs>_<arm>` 下每个决策均记录合法节点编号、利用的节点、进度、活动请求、32 节点 CPU 作业数和稀疏的有向链路在途数。`q_f32le` 是原 Q 的小端 float32 数组的 Base64，按 0–31 节点排序。完整请求文件记录每项服务的阶段就绪、开始计算、完成事件及是否观察截尾。正式截止与影子排空的数据有独立字段，事件按时刻区分。

训练的 `train` 和 `antithetic` 分别存在线和实际物理比较。每个组、每种子有 64 在线和 64 配对点。六次固定情景重复于 GS0 和 GS64。断点权重与回放保存在运行目录。重新绘图可运行 `python -m scripts.exp017_report --redraw --out research/EXP017`。

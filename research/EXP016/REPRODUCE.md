# EXP016 复现与连续训练

在仓库根目录运行。在需要限制本机进程管道的运行环境中，可给每条 `scripts.exp016` 命令加 `--serial`。并行方式把配对物理执行委托给独立进程，用主进程的同一版 Online Q 计算相同调度策略。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -p 'test_exp016.py' -v
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp016 --seed $seed --out "runs/exp016/seed$seed" --gs 64
    foreach ($gs in 128,256,512) {
        python -m scripts.exp016 --seed $seed --out "runs/exp016/seed$seed" --gs $gs --resume
    }
    python -m scripts.exp016_resume_audit --source "runs/exp016/seed$seed" --gs 64 --out "runs/exp016/resume_$seed.json"
}
python -m scripts.exp016_evaluate batch --out runs/exp016 --workers 6
python -m scripts.exp016_report --work runs/exp016 --out research/EXP016
python -m scripts.exp016_review --out research/EXP016
```

检查点 GS0、64、128、256、512 均生成一个独立 `gsNNNN.pt`，保存 D、S Online／Target、两边 Adam、回放环位置与全部样本、Torch RNG、已提交的账本偏移和固定最佳记录。每次探索与回放采样以绝对场景派生随机种子，恢复后不重启 ε 曲线。若进程意外中断，恢复时保留超过最近完整检查点的日志尾部到 `.interrupted-*`，重跑这段在线训练。完整检查点留本地以支持真实恢复。

`train.jsonl` 记录在线提交并用于 S 回放。`antithetic.jsonl` 是逐条真实执行但只用于 D 配对的物理分支。`updates.jsonl` 记录实际更新、参数差、挂接回放和总仿真时长。`fixed.jsonl` 记录固定验证。独立 `short` 与 `queue` 目录按 D/S 检查点配对。最佳只由固定 Reward 选择。

从归档重新绘图可运行 `python -m scripts.exp016_report --redraw --out research/EXP016`。

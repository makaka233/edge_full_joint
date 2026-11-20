# EXP018 复现

在仓库根目录运行。先按 [EXP016 复现](../EXP016/REPRODUCE.md)重建两个完整 GS512 检查点。已有对应权重和账本时可直接使用。按 [EXP017 复现](../EXP017/REPRODUCE.md)生成四个原始 S 快照。以下目录以 `runs/` 为例。所有权重留在运行目录。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$cases = @(20261113,1),@(20261114,2),@(20261114,3),@(20261114,7)
foreach ($case in $cases) {
    $seed,$gs = $case
    python -m scripts.exp018 case --seed $seed --gs $gs --source "runs/exp017/weights_$seed" --out "runs/exp018/case_${seed}_${gs}_R"
}
python -m scripts.exp018_report cases --work runs/exp018 --weights runs/exp017 --out research/EXP018
python -m unittest discover -s tests -p 'test_exp01[678].py' -v
```

四例因果门槛通过后，从随机参数启动唯一候选。

```powershell
python -m scripts.exp018_batch candidate --work runs/exp018/candidate --gs 64
python -m scripts.exp018_report pilot --work runs/exp018/candidate --out research/EXP018
```

如果两个种子的 64 GS 条件均通过，续训相同轨迹并做原独立集对照。

```powershell
foreach ($gs in 128,256,512) {
    python -m scripts.exp018_batch candidate --work runs/exp018/candidate --gs $gs
}
python -m scripts.exp016_evaluate batch --out runs/exp018/candidate --workers 6
python -m scripts.exp018_main_report candidate --work runs/exp018/candidate --out research/EXP018
```

如候选未通过，续训 EXP016 主线。首次命令复制 GS512 与对应原始账本。后续命令恢复最新完整检查点。

```powershell
foreach ($gs in 768,1024) {
    python -m scripts.exp018_batch continue-main --reference runs/exp016 --work runs/exp018/main --gs $gs
}
foreach ($seed in 20261113,20261114) {
    python -m scripts.exp016_resume_audit --source "runs/exp018/main/seed$seed" --gs 512 --out "runs/exp018/main/resume_$seed.json"
}
python -m scripts.exp018_evaluate batch --out runs/exp018/main --gs 1024 --workers 6
python -m scripts.exp018_main_report main --work runs/exp018/main --out research/EXP018
```

从仓库已归档的实测原始数据重新生成图表和交接文档。

```powershell
python -m scripts.exp018_figures
python -m scripts.exp018_review
```

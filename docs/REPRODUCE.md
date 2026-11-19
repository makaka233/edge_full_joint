# 从零复现与真实训练命令

## 1. 环境

Windows 11 PowerShell（无需执行 `run.ps1`）：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS 对应 `python3 -m ...`。依赖只有 numpy/scipy/torch/networkx/matplotlib；不需要任何外部 ZIP/模型。

## 2. 四类实验（每个实验分别跑双种子）

```powershell
# (A) Agent-D-only，仍是未收敛参考基线
python -m edge_d.cli train --seed 20261113 --updates 64 --out runs/d13
python -m edge_d.cli train --seed 20261114 --updates 64 --out runs/d14

# (B) 独立 Agent-S（冻结随机 D）
python -m edge_d.cli s-train --seed 20261113 --updates 64 --out runs/s13
python -m edge_d.cli s-train --seed 20261114 --updates 64 --out runs/s14

# (C) BUILD-only 的 D/S 联合学习
python -m edge_d.cli joint-train --seed 20261113 --updates 64 --out runs/j13
python -m edge_d.cli joint-train --seed 20261114 --updates 64 --out runs/j14

# (D) 跨周期 KEEP/BUILD + Agent-S 联合学习（探索性）
python -m edge_d.cli periodic-train --seed 20261113 --updates 64 --periods 3 --out runs/p13
python -m edge_d.cli periodic-train --seed 20261114 --updates 64 --periods 3 --out runs/p14
```

**注意：`--updates 64` != 真正模拟了 64 个完整逻辑天。** 周期性训练每次更新会采样 `episodes_per_update × periods` 个宏观真实物理抽样窗口。一次训练更新是否相当于论文意义 GS，必须在研究协议内重新定义。`

单步、低成本测试命令：

```powershell
python -m edge_d.cli smoke
python -m edge_d.cli joint-smoke
python -m edge_d.cli s-train --updates 1 --rollouts 2 --fixed-scenes 1 --fixed-draws 1 --holdout-scenes 1 --holdout-draws 1 --out runs/test_s
python -m edge_d.cli joint-train --updates 1 --rollouts 2 --fixed-scenes 1 --fixed-draws 1 --holdout-scenes 1 --holdout-draws 1 --out runs/test_joint
python -m edge_d.cli periodic-train --updates 1 --periods 2 --fixed-episodes 1 --holdout-episodes 1 --out runs/test_periodic
```

`s-train`, `joint-train`, `periodic-train` 对随机初始差策略可能非常耗时，严重积压仍按原事件引擎处理。不能以截尾或人工补 Reward 假装训练顺利。上面长训练是研究命令，不是已运行完成的结果。

## 3. 评价、绘图、模型

```powershell
python -m edge_d.cli joint-evaluate --checkpoint runs/j13/last.pt --split holdout
python -m edge_d.cli periodic-evaluate --checkpoint runs/p13/last.pt --split holdout
python -m edge_d.cli plot --run-dir runs/j13
python -m edge_d.cli plot --run-dir runs/p13
```

文件夹内只有 `best.pt` 和 `last.pt` 两个模型文件，和 `train.jsonl`, `updates.jsonl`, `eval*.jsonl`, `manifest.json`；周期性训练会额外有 `eval_periodic*.jsonl`。`best` 仅由 fixed 选，holdout 仅在初始和末尾出报告。输出指标原始不平滑，`failed` 对应 `censored`，不直接等同真正被丢弃。

## 4. 独立验收

`python scripts/verify_package.py` 校验 ZIP 提供的 SHA256 清单；`checks/` 中只保留本轮已经真实跑过的短烟测结构摘要，不包含任何训练模型权重。查看 `docs/HANDOVER.md` 中的失败结论及复现限制。

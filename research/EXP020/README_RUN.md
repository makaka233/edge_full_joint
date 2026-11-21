# EXP020 本地运行

项目根目录仍是 `C:\Users\1\Desktop\edge_full_joint`。本轮使用原 Adaptive 主程序，所有新运行和中间产物写入 `runs/results/EXP020/`。

## 准备与轻量测试

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pip install -r requirements.txt
python -m edge_d.cli adaptive --help
python -m unittest discover -s tests -p test_adaptive_training.py -v
python -m unittest discover -s tests -p test_original_fidelity.py -v
```

## 已完成的冻结诊断复算

输入是 EXP019 双种子的固定 REDEPLOY GS3000 运行。命令读取旧权重和原日志，输出新目录下的 `sampling.seed*.jsonl`。

```powershell
foreach ($seed in 20261113, 20261114) {
    python -m edge_d.cli adaptive d-sampling --seed $seed --prefix ".\runs\results\EXP019\instant.redeploy.long3000.seed$seed"
    if ($LASTEXITCODE -ne 0) { throw "Sampling diagnosis failed for seed $seed" }
}
python -m scripts.exp019_report --d-sampling-review
```

## 正式训练由用户启动

两种子、两更新方式均随机初始化，用原 Masked Dueling Double DQN 和固定 REDEPLOY。与 EXP019 长训清单一致，每组 **375 episode × 8 期 = 3000 个在线 GS**，每期 24 秒。固定验证含 GS0、每 64 GS 及末次 GS3000。GS512 仍是中途检查点。探索衰减仍用原来的 512 GS 时钟。写出首个完整检查点后可以按原命令恢复。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
foreach ($seed in 20261113, 20261114) {
    foreach ($dUpdate in 'adam', 'kl') {
        $prefix = ".\runs\results\EXP020\instant.$dUpdate.seed$seed"
        python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --s-algorithm ddqn --d-update $dUpdate --episodes 375 --periods 8 --period-seconds 24 --eval-every 8 --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "EXP020 training failed for $dUpdate / $seed" }
    }
}
```

中断后只恢复对应分组，保留原 seed、算法、前缀和所有训练参数。按旧指令已完成 GS512 的组也可以直接续到 GS3000。示例恢复 Seed13 KL 组。

```powershell
python -m edge_d.cli adaptive train --seed 20261113 --macro-mode redeploy --s-algorithm ddqn --d-update kl --episodes 375 --periods 8 --period-seconds 24 --eval-every 8 --prefix ".\runs\results\EXP020\instant.kl.seed20261113" --resume
```

## 独立评价与绘图

四组训练都结束后，在 PowerShell 单独执行这一行。它读取新运行的恢复点，补齐自然 Holdout 和持续高负载 Holdout 并生成报告。已完整写出的评价会跳过。重复运行时，只需要重做未写完的场景套件。

```powershell
python -m scripts.exp019_report --d-kl-trial --evaluate-missing
```

自然 Holdout 为 19101–19103，持续压力为 19301/19302 的每期 0.55 和 19303/19304 的每期 0.70。动态需求组成沿原过程变化，两算法评价同流。正式训练继续使用原自然需求分布。

评价齐全时，`python -m scripts.exp019_report --d-kl-trial` 只复算已有结果。运行目录保存逐期原始日志、更新记录、best、last 和最新恢复点。总体数值保存在 `runs/results/EXP020/instant.comparison.json`，完整原始、固定验证和独立评价图在同一目录。

如果在续跑前已经对 GS512 做过 Holdout，请先归档该前缀下的 `holdout.jsonl` 和 `holdout-sustained.jsonl`，在原结果目录内给文件名加上 `gs512` 标记。GS3000 的报告命令会补齐原文件名下的新评价。

如需在相同物理状态和同一个冻结 S 下分离**训练后 D** 的采样风险，可在 GS3000 完成后补做以下诊断。每组复用与前期相同的 24 个标准化扰动。命令只读取旧 EXP019 参考状态和 S 及新 EXP020 的 D。

```powershell
foreach ($seed in 20261113, 20261114) {
    foreach ($dUpdate in 'adam', 'kl') {
        $source = ".\runs\results\EXP019\instant.redeploy.long3000.seed$seed"
        $models = ".\runs\results\EXP020\instant.$dUpdate.seed$seed"
        python -m edge_d.cli adaptive d-sampling --seed $seed --prefix $source --sampling-models $models
        if ($LASTEXITCODE -ne 0) { throw "EXP020 sampling failed for $dUpdate / $seed" }
    }
}
python -m scripts.exp019_report --d-sampling-trained-review
```

运行完成后，将[离线分析提示词](ANALYSIS_PROMPT.md)发送给 Codex。

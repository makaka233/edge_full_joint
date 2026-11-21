# EXP022 Windows 复现

从原工程根目录运行。新文件统一保存在 `runs/results/EXP022/`，旧实验只作输入。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pip install -r requirements.txt
python -m edge_d.cli adaptive --help
python -m unittest discover -s tests -p test_resource_placement.py -v
python -m unittest discover -s tests -p test_adaptive_training.py -q
python -m unittest discover -s tests -p test_original_fidelity.py -q
```

冻结归因从 EXP019 `instant.redeploy.long3000.seed<seed>` 和 EXP020 `instant.adam.seed<seed>` 读取已存权重与日志。

```powershell
foreach ($seed in 20261113, 20261114) {
    python -m edge_d.cli adaptive d-resource-audit --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "EXP022 frozen audit failed for $seed" }
}
python -m scripts.exp019_report --d-resource-review
```

## 正式 512 GS 试验

以下是用户启动的研究对照。每组 64 个 episode × 8 期，不加载历史权重。固定验证随训练在 GS0、每 64 GS 和末次自动执行。

```powershell
foreach ($seed in 20261113, 20261114) {
    foreach ($policy in 'gaussian', 'resource') {
        $prefix = ".\runs\results\EXP022\instant.$policy.seed$seed"
        python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --s-algorithm ddqn --d-update adam --d-policy $policy --capacity-diagnostics --episodes 64 --periods 8 --period-seconds 24 --eval-every 8 --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "EXP022 training failed for $policy / $seed" }
    }
}
```

中断后只恢复对应组，保持配置和最终目标。

```powershell
$seed = 20261113
$policy = 'resource'
$prefix = ".\runs\results\EXP022\instant.$policy.seed$seed"
python -m edge_d.cli adaptive train --seed $seed --macro-mode redeploy --s-algorithm ddqn --d-update adam --d-policy $policy --capacity-diagnostics --episodes 64 --periods 8 --period-seconds 24 --eval-every 8 --prefix $prefix --resume
```

四组都结束后执行以下命令。它自动补齐自然和持续压力 Holdout，核对固定验证和同流账本，绘制完整原始 Reward、E2E、P99 和逐 S 更新曲线。结果保存在同目录的 `instant.comparison.json`、`instant.holdout.png` 和每个前缀的 `training.png`、`fixed.png`、`s_updates.png`。已经完成 Holdout 时可移除 `--evaluate-missing` 仅做复算。

```powershell
python -m scripts.exp019_report --d-resource-trial --evaluate-missing
```

也可以只评价一个完成的前缀。

```powershell
$seed = 20261113
$prefix = ".\runs\results\EXP022\instant.resource.seed$seed"
python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix
python -m edge_d.cli adaptive evaluate --seed $seed --prefix $prefix --sustained
python -m scripts.exp019_report --s-update-review --prefix $prefix
```

正式结束后发送[离线分析提示词](ANALYSIS_PROMPT.md)，以既有日志完成审核。512 GS 结果用于第一次正式对照，无法验证中期 GS1308 的发生概率。

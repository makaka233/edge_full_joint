# 项目与实验管理

## 源码和备份

GitHub `makaka233/edge_full_joint` 是源码版本记录。主程序入口在 `edge_d.cli`。每轮从当前经过验证的提交开一条短分支，写一组同主题代码、协议和测试，完成轻量验收后提交并推送。用实验编号标注分支、提交和研究文档。发布时同样记录分支和完整 SHA，便于从任意轮次复现。候选验证完成后依据结果决定是否合并或成为默认。分支管理参考 [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)。

本机在项目根目录保留一套源码工作区。各轮在同一目录切换 Git 分支，从明确的提交启动主程序。Git 管理源版本，运行目录用于保存产物。

## 统一存放结果

新日志、检查点、评估、烟测和图表统一写入 `runs/results/<实验编号>/`。种子和新运行标识写在文件名中。每个运行的 manifest 记录代码 SHA、配置、训练权重来源、场景分割和软件环境。训练日志逐条追加，固定验证与 Holdout 分账，模型保存 best、last 和最近完整状态。大体积结果留本地。每轮的源码、配置、复现脚本、交接文档及其变更均备份到 GitHub。

每轮的设计、复现说明及交接统一放在 `research/<实验编号>/`，对外交接文件名为 `AI_RESEARCH_REVIEW.md`。文档和结果目录使用完全相同的实验编号。已迁移的旧产物集中于 `runs/archive/legacy`，`runs/archive/PATH_INDEX.csv` 记录旧位置。

归档前先盘点 Git 跟踪文件、本地运行文件以及源码、测试、说明对各路径的引用。让测试必需的固定输入留在 `tests/fixtures/`。归档旧运行产物前，在 GitHub 核对原提交和归档标签。迁移完成后验证新路径、轻量测试和远端提交。已推送的改动用新提交回滚，可保留各轮历史。参见 [Git 恢复和回滚命令](https://git-scm.com/docs/git#_reset_restore_and_revert)。

推送分支、核对提交和保留归档标签用于本项目的版本回档。完整仓库容灾副本另有独立镜像流程。未跟踪的本地运行产物需要由运行目录另行保存。参见 [GitHub 仓库备份文档](https://docs.github.com/en/repositories/archiving-a-github-repository/backing-up-a-repository)。

## 历史产物恢复

原 `evidence/` 存放训练记录、曲线和审计数据，原 `checks/` 存放烟测、分配器检查和短训记录。这批旧交付产物没有统一实验编号，现保存在 `runs/archive/legacy/` 下。原交付包清单 `SHA256_MANIFEST.json` 一起归档。GitHub 标签 [archive/pre-cleanup-evidence-checks](https://github.com/makaka233/edge_full_joint/tree/archive/pre-cleanup-evidence-checks) 保留原始版本。

需要在新克隆项目中恢复这批文件时，从项目根目录运行。已存在的本地归档请先按路径索引查看。

```powershell
git fetch origin tag archive/pre-cleanup-evidence-checks
$legacy = 'runs/archive/legacy'
New-Item -ItemType Directory -Path $legacy -Force | Out-Null
if ((Test-Path "$legacy/evidence") -or (Test-Path "$legacy/checks") -or (Test-Path "$legacy/SHA256_MANIFEST.json")) {
    throw 'Archive destination already contains legacy files'
}
$tar = Join-Path ([IO.Path]::GetTempPath()) ("edge-legacy-" + [guid]::NewGuid() + ".tar")
git archive --format=tar --output="$tar" archive/pre-cleanup-evidence-checks evidence checks SHA256_MANIFEST.json
if ($LASTEXITCODE -ne 0) { throw 'Git archive failed' }
tar.exe -xf $tar -C $legacy
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }
Remove-Item -LiteralPath $tar
```

该命令只还原指定历史产物。原始研发资料仍可通过提交或标签访问。参见 [git archive 文档](https://git-scm.com/docs/git-archive)。

## 研究验收

1. 在正式运行前写清假设、对照、种子、请求场景、物理设置、提交与优化预算、评价口径和验收条件。
2. 提交前用轻量测试验证连续性、守恒、恢复和命令入口。烟测按实际长度标注。
3. 用户从指定提交手动训练。固定集选择 best，独立集一次性比较 best、last 和同流对照。
4. 审核时同时列出每个种子、原始尖峰、完成及截尾、Reward、E2E 和尾部、实际更新数及墙钟。独立样本量所支持的结论范围应写入报告。

配置、代码、数据分割及计算成本的记录参考 [NeurIPS 研究清单](https://neurips.cc/public/guides/PaperChecklist)。少量种子的结果需要保留分种子表现并说明不确定性，参考 [Henderson 等](https://ojs.aaai.org/index.php/AAAI/article/view/11694)及 [Agarwal 等](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。

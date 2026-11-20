# 项目与实验管理

## 源码和备份

GitHub `makaka233/edge_full_joint` 是源码版本记录。主程序入口在 `edge_d.cli`。每轮从当前经过验证的提交开一条短分支，写一组同主题代码、协议和测试，完成轻量验收后提交并推送。用实验编号标注分支、提交和研究文档。发布时同样记录分支和完整 SHA，便于从任意轮次复现。候选验证完成后依据结果决定是否合并或成为默认。分支管理参考 [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)。

本机在项目根目录保留一套源码工作区。各轮在同一目录切换 Git 分支，从明确的提交启动主程序。Git 管理源版本，运行目录用于保存产物。

## 统一存放结果

新日志、检查点、评估、烟测和图表统一写入 `runs/results/<实验编号>/`。种子和新运行标识写在文件名中。每个运行的 manifest 记录代码 SHA、配置、训练权重来源、场景分割和软件环境。训练日志逐条追加，固定验证与 Holdout 分账，模型保存 best、last 和最近完整状态。大体积结果留本地。每轮的源码、配置、复现脚本、交接文档及其变更均备份到 GitHub。

每轮的设计、复现说明及交接统一放在 `research/<实验编号>/`，对外交接文件名为 `AI_RESEARCH_REVIEW.md`。文档和结果目录使用完全相同的实验编号。已迁移的旧产物集中于 `runs/archive/legacy`，`runs/archive/PATH_INDEX.csv` 记录旧位置。

## 研究验收

1. 在正式运行前写清假设、对照、种子、请求场景、物理设置、提交与优化预算、评价口径和验收条件。
2. 提交前用轻量测试验证连续性、守恒、恢复和命令入口。烟测按实际长度标注。
3. 用户从指定提交手动训练。固定集选择 best，独立集一次性比较 best、last 和同流对照。
4. 审核时同时列出每个种子、原始尖峰、完成及截尾、Reward、E2E 和尾部、实际更新数及墙钟。独立样本量所支持的结论范围应写入报告。

配置、代码、数据分割及计算成本的记录参考 [NeurIPS 研究清单](https://neurips.cc/public/guides/PaperChecklist)。少量种子的结果需要保留分种子表现并说明不确定性，参考 [Henderson 等](https://ojs.aaai.org/index.php/AAAI/article/view/11694)及 [Agarwal 等](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。

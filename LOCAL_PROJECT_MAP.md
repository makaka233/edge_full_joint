# 本地项目路径

| 路径 | 用途 |
|---|---|
| [source](source/) | 当前源码工作树。进入这里运行主程序，以 Git 分支及提交识别版本。 |
| [runs/results](runs/results/) | 每轮一个同名目录，例如 `runs/results/EXP019/`。日志、检查点、评价和图表统一存放在该轮目录内。 |
| [runs/archive/PATH_INDEX.csv](runs/archive/PATH_INDEX.csv) | 旧文件的迁移表。历史运行、验收副本、旧压缩包和图表保存在这里索引的归档中。 |

[EXP019 运行指南](source/research/EXP019/README_RUN.md)给出当前训练、恢复和评价命令。单文件交接按轮次保存在 `research/<实验编号>/AI_RESEARCH_REVIEW.md`。[研究流程](source/docs/RESEARCH_WORKFLOW.md)说明 GitHub 标注和证据验收。

归档已保留 10,367 个文件，约 8.29 GB。两个历史 EXP018 GS1024 检查点仍在原内部层级。旧运行命令或聊天链接若使用搬迁前路径，可用迁移表定位。

项目外层 checkout 是旧版分支，仍保留其既有源码和未提交文件。当前实验命令从 `source/` 执行。

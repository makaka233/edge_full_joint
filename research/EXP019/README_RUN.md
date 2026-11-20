# EXP019 本地运行

从 Windows PowerShell 进入项目根目录。以下各命令可直接复制到新的 PowerShell 会话中使用。旧权重通过统一归档读取，新产物写入本轮结果目录。当前分支为 `exp019-adaptive-deployment`。

```powershell
Set-Location 'C:\Users\1\Desktop\edge_full_joint'
python -m pip install -r requirements.txt
python -m edge_d.cli adaptive --help
python -m unittest discover -s tests -p test_adaptive_training.py -v
```

## 可选物理烟测

下面的命令冻结已验证的 EXP018 D/S，运行两个各 1 秒的连续物理期。

```powershell
python -m edge_d.cli adaptive verify --seed 20261114 --periods 2 --period-seconds 1 --prefix ".\runs\results\EXP019\smoke.seed20261114" --init-checkpoint ".\runs\archive\legacy\runs\exp018\main\seed20261114\gs1024.pt"
```

## 正式训练

各运行 128 个 episode，每个 episode 八个业务期，对应每种子 1,024 个在线 GS。每个 episode 正常结束后刷新账本与恢复点。初始、每 16 个 episode 和最终执行固定验证。这两条命令从同种子 EXP018 GS1024 初始化 D/S 权重，继续训练 D/S，宏观门随机初始化。

```powershell
python -m edge_d.cli adaptive train --seed 20261113 --episodes 128 --init-checkpoint ".\runs\archive\legacy\runs\exp018\main\seed20261113\gs1024.pt"
python -m edge_d.cli adaptive train --seed 20261114 --episodes 128 --init-checkpoint ".\runs\archive\legacy\runs\exp018\main\seed20261114\gs1024.pt"
```

省略 `--init-checkpoint` 可随机初始化 D/S。显式 `--freeze-ds` 会固定 D 的均值解码和 S 的贪婪调度，只训练宏观层。另一个独立运行需采用另一个前缀，同一前缀只能由一个进程占用。

## 中断恢复

重新打开 PowerShell 后进入项目根目录。用相同 seed、原训练参数和绝对目标 episode 数恢复。若首次训练指定了自定义前缀，恢复时也传入该前缀。

```powershell
python -m edge_d.cli adaptive train --seed 20261113 --episodes 128 --resume
python -m edge_d.cli adaptive train --seed 20261114 --episodes 128 --resume
```

## 独立评价

训练完成后执行。每种子在三个同流未见场景比较 best、last、固定 KEEP 和每期请求 REDEPLOY，原子保存结果。

```powershell
python -m edge_d.cli adaptive evaluate --seed 20261113
python -m edge_d.cli adaptive evaluate --seed 20261114
```

## 文件后缀

所有文件集中在项目根目录的 `runs/results/EXP019/`。正式前缀分别为 `seed20261113` 和 `seed20261114`。

| 后缀 | 内容 |
|---|---|
| `.manifest.json` | 初始化来源、有效配置、Git 提交、工作树状态、软件版本与场景。 |
| `.train.jsonl`、`.updates.jsonl` | 逐期物理采样和逐 episode 优化更新。 |
| `.fixed.jsonl`、`.holdout.jsonl` | 固定选模和同流独立评价。 |
| `.best.pt`、`.last.pt` | 固定最优和末次模型。 |
| `.resume_latest.pt` | 完整训练恢复状态及账本提交边界。 |
| `.errors.log` | 如有，异常及堆栈。 |

正式训练和独立评价结束后，将[离线审核提示词](ANALYSIS_PROMPT.md)交给 Codex。单文件交接位于 `research/EXP019/AI_RESEARCH_REVIEW.md`。

# 2026-11-19 实验报告与研究员交接

**结论：D-only 与 BUILD-only 双智能体在两种子上通过本轮独立改善检查；不能宣称已达到长期联合收敛或已学会需求自适应 KEEP/BUILD。** 这是原合成 benchmark 的真实离散事件仿真，不是生产数据。方法见 [有界整矩阵策略](PLACEMENT_METHOD.md)，物理、S 与 R12 差分见 [接手审计](TAKEOVER_AUDIT.md)。

## 实验协议

- 固定 `finish_pf`、逐跳 wired KKT、AP 等分、原 Reward、有效 warm 20 s。训练两种子 `20261113/20261114`，每 arm 64 次训练提交；GS0/8/16/24/32/40/48/56/64 做固定验证。只以固定 Reward 选最佳，其后才运行 holdout 初始/最佳/末次对照。BUILD 每种子每次 4 个固定或 8 个独立 holdout 场景；周期每次 2 个固定或 4 个独立四期 episode；周期同 episode 的 4 个时期不是 4 个独立样本。对照不能用 holdout 反向挑选。
- 主结果 `e2e` 为合计请求加权的窗口惩罚后 E2E；零截尾时等于实测完成 E2E，其他时候不可称作实测完成时延。`tail_fraction` 为超过 1 s 或窗口截尾的请求比例；完成 p95/完成均值、持有/切换、实际副本、队列、链路与资源数据另存。Reward 为按提交平均，和请求加权 E2E 不可直接线性换算。
- 8 个正式 runs 共 **512 次训练物理提交**，加验证共 **1112 次物理提交**（6×124+2×184，不含专项探针和复现重跑）；每次真机运行数见 `runs/archive/legacy/evidence/research_20261119/summary.json`。D-only 与 BUILD 联合各 32 次 D 更新，S-only 为零，周期各 8 次；含学习 S 的 6 次运行各 128 次原始 DDQN 更新。训练仿真约 3477–3636 秒/次，单次实测主进程墙钟 82–272 秒（并行运行，不能作为单线程同机性能比较）。
- 异常样本全部保留：例如 D-only 训练最差 Reward `-122.71/-51.42`，BUILD 联合 `-123.34/-81.72`，周期 `-135.56/-134.18`。8 份 `raw_training.png` 每个 GS 一点，未平滑、剪裁或排除。单样本的 JSONL 含完整矩阵、小窗请求哈希、截尾、资源有限容量/活跃数/利用率、动作和版本；权重不在归档内。

## 训练与独立对照

箭头为 GS0 → GS64。此表列末次，不用最佳 holdout 倒推挑模型。最佳由固定选择，完整结果也在归档内。

| 模式 / 种子 | 固定 Reward | 独立 Reward | 独立请求加权 E2E (s) | 独立截尾 | 独立尾部率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-only / 13 | -18.633 → -4.558 | -5.564 → -4.660 | 0.554 → 0.444 | 0 → 0 | 5.96% → 1.38% |
| D-only / 14 | -24.272 → -4.459 | -5.981 → -4.556 | 0.594 → 0.436 | 0 → 0 | 8.98% → 0.48% |
| S-only / 13 | -19.059 → -18.993 | -7.120 → -6.837 | 0.729 → 0.691 | 0 → 0 | 14.78% → 15.72% |
| S-only / 14 | -49.904 → -24.477 | -63.069 → -6.119 | 6.890 → 0.608 | 556 → 0 | 55.77% → 9.66% |
| D/S BUILD / 13 | -19.059 → -5.594 | -7.120 → -5.298 | 0.729 → 0.522 | 0 → 0 | 14.78% → 4.36% |
| D/S BUILD / 14 | -49.904 → -5.138 | -63.069 → -5.457 | 6.890 → 0.536 | 556 → 0 | 55.77% → 3.81% |
| D/S KEEP/BUILD / 13 | -5.813 → -5.923 | -6.957 → -6.528 | 0.736 → 0.678 | 9 → 0 | 14.67% → 14.00% |
| D/S KEEP/BUILD / 14 | -18.220 → -5.031 | -82.751 → -7.072 | 8.774 → 0.731 | 1672 → 0 | 61.27% → 18.82% |

- D-only 的最终 holdout E2E 降幅为 19.9%/26.5%，末次两个种子均为固定最优，holdout 尾部下降不是由删除截尾样本产生。独立完成请求 1813/2071。两个种子的训练覆盖 MILP 回退 27/16 次，最终提交理想 CPU `rho>=1` 2/1 次；不可认为候选始终有 CPU 证书。原 64 更新 EDIT 历史与本轮 64 GS 非等预算，不能比较曲线作算法因果归因。
- BUILD 联合固定最佳均为 GS56，holdout E2E 0.529/0.545 s，截尾均为零。GS64 固定 Reward 略低于最佳；两种子改善代表学习证据，不证明已收敛。联合在这里未超过固定启发式 S 的 D-only（分别约 0.444/0.436 s），说明随机启动的联合学习仍需改进，不宣称学习 S 超越成熟启发式。
- 四个另外的需求场景，使用固定选出的 checkpoint，做 D0/S0、D0/S*、D*/S0、D*/S* 同轨迹互换。13 的 E2E 分别为 1.407/0.758/0.619/0.580 s；14 为 8.639/2.496/**11.378**/0.502 s。Seed14 学习后 D 遇到随机 S 可变差；报告交互，不将完整改善归给当前策略中的单个网络。
- 四期联合训练实际执行 KEEP 34/37 次，但两个验证最佳的门在原 holdout 和另一组四场景均为首期 BUILD、之后全部 KEEP；在新场景，与强制 KEEP 相同（0.813/0.589 s）。同权重强制每期 BUILD 为 2.885/0.621 s，截尾 353/0 个（KEEP 均零）。**学会需求自适应切换未通过。** 该结果只能说明在当前短 episode 中保留良好初始布局可能优于每期重建，不代表长期永远 KEEP 最优。

## 物理对照：CPU KKT 与截尾

重放两个 D-only 固定最佳的 8 张真实矩阵：80 s 连续队列，30 s 预热，每 6 s 桶独立 8 s 截尾，背景流继续到最后判定；固定矩阵、启发式 S 和请求流，只切换 CPU。8 场景的均值 `finish_pf=0.4210 s` / `kkt=0.4443 s`，两者均零截尾，最大守恒误差小于 `1e-11`。此结果只涵盖该协议/样本，**保留 CPU 默认 finish_pf，未训练 CPU-KKT arm**。

另外对两种子各重放训练最低 Reward 的压力样本。`finish_pf` 在 80 s 观测中的截尾分别是 800/412；停止新到达后再给 30 s，先前截尾对象有 275/287 个已实际完成。CPU KKT 的对应值是 800/412、随后完成 6/108，同请求流下不同出清不等于改变抽样。30 s 后仍未完成者依旧是未观察到完成，不是永久丢弃。压力场景选取刻意偏尾，只用于诊断。

## 复现与继续研究

Windows PowerShell，建议新环境，不改变已有机器全局环境。已使用 Python 3.11 / PyTorch 2.11.0 / NumPy 2.4.6 / SciPy 1.17.1 / NetworkX 3.6.1 / Matplotlib 3.11.1。为新的复现准备一个实验编号。

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& $python -m unittest discover -s tests -v
& $python -m scripts.verify_allocator_modes
& $python -m edge_d.cli smoke
& $python -m edge_d.cli joint-smoke
$round = Read-Host 'Experiment ID'
$study = "runs/results/$round"
New-Item -ItemType Directory -Path $study -ErrorAction Stop | Out-Null
foreach ($suffix in 13, 14) {
    $seed = 20261100 + $suffix
    & $python -m edge_d.cli placement-train --seed $seed --mode d-only --out "$study/placement_d_$suffix"
    & $python -m edge_d.cli placement-train --seed $seed --mode s-only --out "$study/placement_s_$suffix"
    & $python -m edge_d.cli placement-train --seed $seed --mode joint --out "$study/placement_joint_$suffix"
    & $python -m edge_d.cli placement-train --seed $seed --mode periodic --periods 4 --fixed-scenes 2 --holdout-scenes 4 --out "$study/placement_periodic_$suffix"
}
& $python -m scripts.probe_physics --out "$study/allocator_continuous.jsonl" "$study/placement_d_13" "$study/placement_d_14"
& $python -m scripts.probe_physics --stress --out "$study/allocator_stress.jsonl" "$study/placement_d_13" "$study/placement_d_14"
& $python -m scripts.probe_learning --out "$study/joint_checkpoint_swaps.jsonl" "$study/placement_joint_13" "$study/placement_joint_14"
& $python -m scripts.probe_learning --out "$study/periodic_gate_controls.jsonl" "$study/placement_periodic_13" "$study/placement_periodic_14"
$runs = @(Get-ChildItem $study -Directory -Filter 'placement_*' | ForEach-Object FullName)
$probes = @(Get-ChildItem $study -File -Filter '*.jsonl' | ForEach-Object FullName)
& $python -m scripts.archive_results --out "$study/summary" @runs --probes @probes
```

首次复现用空输出目录；不要把新日志混入交接证据，也不要用同一批最终 holdout 反复调参。仓库 `.gitignore` 及 ZIP 排除运行权重。打开 `runs/archive/legacy/evidence/research_20261119/` 内每个 run 的 `raw_training.png`、`validation.png`、`updates.png`；`.jsonl.gz` 是无损压缩的未删减原始行。

验收记录见 `runs/archive/legacy/evidence/research_20261119/validation.json`：A/B 与 C/D 快照均独立解压校验并运行单元测试；完整版在独立解压目录执行 17 项测试、原始分配器探针、BUILD/JOIN 两个原始 smoke 和新周期 8 GS smoke。原 R12 使用原入口的模块路径与工作目录，7 项稳定性测试通过。D-only 20261113 重跑的 64 个 GS 的矩阵、请求哈希与 Reward 逐点相同。

**下一轮验收顺序**：先在不改变 Reward 的情况下，以不同负载制度和更长的连续队列复验当前策略，再比较部署门的现有 LP 压力特征和显式配对 KEEP/BUILD 信用。预注册固定集与全新人口/需求 holdout，统一物理预算比较 64/128 GS 与固定 KEEP/BUILD；之后才比较 R12 D80 信用和 S 系统信用，并检验 S 辅助分支的真实 replay/更新闭合。不用变更 CPU 默认、惩罚或规模掩盖坏结果。

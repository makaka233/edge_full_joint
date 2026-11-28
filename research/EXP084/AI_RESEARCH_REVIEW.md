# EXP084 物理反事实净收益驱动的事件式部署

状态 **ready**；更新时间 2026-11-28T14:34:18；报告代码 `54ba9e148c16a48571bee8b62341bd8be91d832c`。
EXP081 B五种子GS5040 last D/S冻结；本轮不训练D/S、不恢复SAC。EXP082 complete 45/45已在54ba9e1归档。
每个GS一次候选、每轨迹一个预登记固定扰动场。92维D输出、资源解码、原Reward、CPU/链路KKT均不改。可选fixed_decoder_key默认None，历史逐GS键保持不变。
回归目标是指定KEEP续策的4/8GS有限期动作净收益差，不是无限期最优Q、SAC收敛或SMDP收敛。在线按8GS预测净收益>0选择REDEPLOY，否则KEEP；无强制动作或保持期限。
请求窗口24秒/原8秒自然结算；两个分支推进9槽，4/8槽作为真实前缀，额外一槽用于成熟信用。此前未结算cohort的影响以−delay_weight·ΔE2E_previous/γ加入差值，旧沉没成本相消。所有失败罚值保留。
行为访问为每轨迹独立Beta(0.7,1.3)抽取概率后自然Bernoulli动作，与标签无关；预登记开口不按收益重抽。训练/验证/独立诊断/最终测试按完整轨迹、人口、需求、解码键隔离。
区域/服务迁移只根据外生热点序列选取首个满足预登记前后稳定4槽的原生成器随机流；不改转移概率、不看物理收益。自然条件并不等于72槽恒定需求；真实热点时间线保留。
已采集轨迹 0，成对起点 0，已训练网络 0/3，完整连续评价组 0。
默认主网络8401；8402/8403仅检查初始化一致性，不根据最终测试选网络。best仅由独立验证MAE选择。

| 模型种子 | EXP081 last SHA256 |
|---|---|
| 20261113 | 2dc6c0bb30c67a4cdaa0a2a4c666000389f850401f0bdcb4fe732ed57b40ea6c |
| 20261114 | 88d35dc7062f14fda52beb2a8352027fde80d76862ae76a97a0da65c282fa6aa |
| 20261115 | f01cb1d113629be63f3e62c7e53e8809c878c316afe14d52a4521f4cecfa77f4 |
| 20261116 | 818c960853ed6c8e8b9bff6bc6d2549e71a97c17eeda9dd0cadc73380b5e3809 |
| 20261117 | 1bc5e1328780588d7d471da6b2a6eb977873bb083bf03915a5fab0245d6dc6f8 |

工程回归：{'default_exp081_8gs_exact': True, 'observer_and_moving_candidate_exact': True, 'restored_adam_rng_update_exact': True, 'updated_parameter_sha': '3d0d81b7454f1a11aa4cfac94c4e4ece6c0ddcb526a622bd59fe44a725023d6b', 'work_error': 2.2851054382044822e-11, 'source': {'commit': '54ba9e148c16a48571bee8b62341bd8be91d832c', 'dirty': True, 'tracked_dirty': True, 'python': '3.11.7', 'os': 'Windows-10-10.0.26200-SP0', 'torch': '2.11.0+cpu', 'numpy': '2.4.6'}}。短训不提供收敛证明。

| 分区/种子/场景/开口 | 4GS 原始/边界修正差 | 8GS 原始/边界修正差 | 8GS时延收益/增量持有/切换 | 4/8终点未完成 KEEP/REDEPLOY | 观察8GS末CPU KEEP/REDEPLOY |
|---|---|---|---|---|---|

| 网络 | 训练/验证点数 | best更新 | 8GS验证MAE/常数MAE | 有效方向命中 | KEEP/REDEPLOY识别 |
|---|---|---|---|---|---|

| 网络 | 参数量/输入维度 | 实际更新 | 拟合秒数 | 标签4/8尺度 |
|---|---|---|---|---|

| 场景/扰动场 | 策略 | Reward/E2E/P99 | 最低R | 严重/截尾/未完成 | CPU/链路 | 重部署/矩阵变化 | 持有/切换 | 完整间隔/右删失(h) |
|---|---|---|---|---|---|---|---|---|

## 研究判断
尚无完整正式验收结论；烟测和部分回归误差不能证明在线可靠性或长期学习成功。

原始状态/请求/资源账本/成本分解：`runs/results/EXP084/data/`；冻结连续评价：`evaluation/`；模型best/last/resume_latest：`models/`。
实际物理事件：访问 0，额外分叉 0。
已完成物理采集任务秒数之和 0.0；评价任务秒数之和 0.0。多进程总墙钟另见execution.json。
命令：`python -m scripts.exp084_run smoke`；`python -m scripts.exp084_run all --workers 5`；可分别collect、fit、evaluate、report。
宏观2h与24秒业务窗口压缩局限保留；固定KEEP续策标签不等于自适应循环策略价值。所有原始严重负峰保留，模型未按测试Reward挑选。

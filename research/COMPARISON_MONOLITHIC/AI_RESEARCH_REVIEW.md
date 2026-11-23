# EXP029 多阶段与整体服务架构对比

## 研究口径

以 EXP029 五种子正式执行提交 `9c4034f9b7154a592c0849eafa449d23c360814d` 及报告为参照，在现有工作树建立 `comparison-monolithic-service`。本对比的改动位于新增比较模块。多阶段端直接调用原 EXP029 形态的 adaptive_runner.train。标准算法为 Gaussian REINFORCE、链路定价、滞后优势基线、Masked Dueling Double DQN、原 Replay 与原训练时钟。两组各自从随机参数开始。

实际物理配置为 32 节点、52 条无向链路、10 种服务、28 个原串行阶段、12,000 位用户，基准全网 CPU 从配置逐节点相加为 **366.9185127 CU/s**。整体服务的 10×32 动作满足全覆盖与原 RAM/SSD 容量约束。原目录字段依次为 CPU CU、输入 Mbit、输出 Mbit、RAM、SSD，CPU 和 RAM/SSD 分别求和，无线入口取首段输入。末段输出量在目录中保留，完成时遵循原引擎语义。结构差异包括原子 CPU 工作粒度、一请求一次节点选择、完整副本共址和内部有线通信消失。两侧使用相同的 finish_pf 正则参考、逐有向链路 KKT、原拓扑、容量、最短路与费用。资源等价矩阵的实际 RAM/SSD、持有和切换费用已逐项对账。

两组训练用同种子、同原自然分布、同请求、同真实 GS 及 D/S 更新时钟。相同形状的 D 可获得相同随机初值，张量、优化器和 replay 互相独立。整体服务 S 的输入适配为 32×19，原多阶段为 32×24。移除的是无适用语义的下一阶段字段。评价冻结权重，在共同独立需求下改变单一物理因素，按 last 出主图，另存 best 对照。固定权重下允许 D 在每期重新解码部署。 [EXP029 原报告](../EXP029/AI_RESEARCH_REVIEW.md) · [注册因素与预算](../../comparison/experiment_matrix.json)

## 实际验证

**15/15 项专项测试通过。** 已逐项执行 CPU 和 RAM/SSD 求和、原子落点、超容量和未覆盖拒绝、拆分矩阵拒绝、恰好一次合法调度、入口/末段字段、无内部通信、相邻数据同步缩放、CPU/有线/无线守恒、资源加权 Reward、四轴同流、独立参数优化器和 replay、精确恢复、32×19/32×24 与 checkpoint 区分。断续与连续训练的 D、S、目标网络、D/S 优化器、价值头、回放和滞后偏移逐项相同。

Seed13 的两组均实际训练 4 GS。短测每期 2 秒、batch=1，正式运行采用原 24 秒期和 batch=128。

|实际烟测|多阶段|整体服务|
|---|---:|---:|
|到达请求|153|153|
|D 更新|2|2|
|S 更新|32|32|
|实际调度次数（含排空）|441|153|
|在线平均 Reward|−6.19779|−5.49856|
|在线完成 E2E（ms）|633.29|571.24|
|截尾|0|0|
|best／last GS|0／4|4／4|

真实冻结短测共 20 个架构／检查点／条件单元，四类图和两组在线逐 GS 图已生成。各条件的配对请求、CPU 工作量与原费用通过复核。中间数据 4× 下整体组的矩阵、原始 Reward 和逐请求时延与自然基准逐期相同。表格给出 **last** 的短测结果，用于核对输出与统计口径。

|冻结短测条件|请求数（每组）|Reward 多阶段／整体|完成 E2E（ms）多阶段／整体|
|---|---:|---:|---:|
|自然|58|−5.24469／−5.90912|517.33／590.41|
|目标负载 0.70|291|−15.39022／−9.92373|1523.96／965.00|
|CPU 1.6×|58|−4.01542／−3.33962|392.91／321.85|
|有线 2×|58|−4.54793／−5.35504|446.80／537.61|
|中间数据 4×|58|−8.93310／−5.90912|893.81／590.41|

本地并行短测遇到 Windows 沙箱管道权限错误，在串行模式完成。512 或 5,000 GS 正式训练未运行。单种子 4 GS 和两期冻结样本不足以判断架构优势、训练收敛或压力泛化。架构结论应由独立冻结评价逐条件、逐种子和跨流支撑。完成 E2E 必须与惩罚后分位数、截尾和可完成比例一并解读。

## 输出、复现与下一步

在项目根目录运行：

```powershell
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python -m comparison.run --suite pilot --workers 6
```

按原命令续跑。独立评价可用 `--stage evaluate`，已有结果可用 `--stage report`。pilot 为 Seed13/16/17，每个架构每种子 512 GS、64 次 D 和 4,096 次 S 更新。主评价包括两个独立未来、24 个连续期。16 个因素水平中四个自然基线等价，复用同一冻结结果。两个外生轨迹对训练种子共享，汇总按种子和流分别输出。结果包括请求量、训练更新、墙钟、P95/P99、最差 1%/5%、实际存储、CPU/有线利用与积压。副本分别报告原每阶段副本或每服务整体副本，也报告多阶段全链共址位置数作为存储结构描述。

原数据保留在 `runs/results/monolithic_service_comparison/`。协议保存物理配置和因素表哈希，每次训练与评价的记录含实际源码 SHA，冻结结果校验模型文件指纹。正式冻结结果由程序出图。四张短测图每个因素只有一个干预水平，图中的数值仅验证工程接口。

[短测账本](../../runs/results/monolithic_service_comparison/smoke.review.json) · [原始在线短测](../../runs/results/monolithic_service_comparison/smoke.seed20261113.raw.png) · [因素 A](../../runs/results/monolithic_service_comparison/smoke.holdout.load.png) · [因素 B](../../runs/results/monolithic_service_comparison/smoke.holdout.cpu.png) · [因素 C](../../runs/results/monolithic_service_comparison/smoke.holdout.wired.png) · [因素 D](../../runs/results/monolithic_service_comparison/smoke.holdout.intermediate.png) · [运行文档](../../comparison/README.md)

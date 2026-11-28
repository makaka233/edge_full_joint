# EXP088：冻结成熟 D/S 的时机 DDQN（预登记 v1）

基线 adf92e9787fb51d71f14440b168b66c2a433e3fb；只开放 T。EXP087 的 8 点为历史机制诊断，不是 T 标签或独立泛化结果。EXP086 原 [True,False] 保留。EXP081/B seed20261113、20261117 的 GS5040 last 整体配对，读取并核对 proof SHA256；D、S online/target 冻结且逐世界验哈希。

唯一环境为 configs/exp086_long_demand.json。macro=7200s、micro=24s、tail=8s；沿用实际 profile、拓扑、容量、Reward 和截止规则。24 秒不是完整两小时队列演化。每世界使用独立登记的 scene/population/decoder 固定扰动场；所有候选仍重新根据当前状态合法解码。不得调整成本、强制动作或搜索候选。每个边界一次候选；当前动作先确定，反事实不参与该动作选择。

## 状态（0 起始索引）

复用 EXP084 features → EXP085 compact。前48维均 sign(x)log(1+abs(x))，不拟合数据标准化；年龄单位为 GS（2小时），另记真实宏观秒数。

|维度|意义/原单位|
|---|---|
|0–2|总到达率、相对上一槽的需求绝对差之和、相对最近实际布局变化时需求的绝对差之和，request/s|
|3–4|距实际矩阵变化、距 R 指令的 GS|
|5–8|改变 bit 数、新增/删除 footprint、原 switch_cost|
|9–11|旧/候选 holding_cost、实际在途请求数|
|12–15、16–19、20–23|旧压力、新压力、新减旧阶段压力的 p50/p90/p99/max（阶段提供工作率/部署节点CPU和）|
|24–26、27–29、30–32|旧/新节点名义提供负担及其差的 mean/p90/max|
|33–35|实际节点 CPU 未完成工作/容量的 mean/p90/max（秒）|
|36–38|有线未完成工作/容量的 mean/p90/max（秒）|
|39–41|阶段候选减旧的预计通信秒数 mean/p90/max|
|42–44|阶段 CPU 未完成工作 mean/p90/max（CU）|
|45–47|旧/候选总副本数、阶段最大在途CPU工作（CU）|
|48–49|当前宏观日相位 sin/cos（不变换）|
|50–52|上一24秒到达cohort的总数、此刻尚未完成数、此刻已完成任务的实际时延和（秒），同 sign-log|

上一cohort按真实到达时间索引，包括BOOTSTRAP cohort；空历史显式记录。不得包含未来完成时间、remaining、迁移内部答案或人工采样段剩余长度。53维是受限观测，不声称严格Markov。

## 学习、信用与采样

T种子由 stable_seed('EXP088-v1',DSseed,'network') 固定；53→32→16 ReLU、V+二动作 A−mean(A)，无动作偏置手工初始化。复用工程 DDQN target/terminal helper、SReplay、Adam和原子checkpoint。gamma=.99、lr=3e-4、Huber beta=1、clip=5、batch64、uniform buffer20000；至少64条成熟transition；每8主GS最多2批；64次更新硬同步target。epsilon .5→.05线性覆盖4008实际主决策的前75%，此后.05。无PER、无辅助Actor、无Reward裁剪。

原账本 R_t=−c_t−d_t 不改。TD r_t=−c_t−d_(t−1)/gamma，前cohort截止为原槽末+8秒。完整有限终点最后一步再加−d_t，可逐路径证明折扣和守恒：每个非末延迟从 gamma^t d_t 移到 gamma^(t+1)d_t/gamma。世界起点无前cohort。BOOTSTRAP用于物理初始化和原始账本，不作为T选择经验。

每世界168主GS是采样截断：terminated=False，次状态在GS169需求已揭示且动作前取得。每8GS只更新/记录，不drain。额外GS169按段末epsilon行为续行，使用行为RNG的副本，不计决策/epsilon/更新/分叉；只结算第168槽原Reward，不存replay。其后停止该训练样本，不将新世界串成旧队列。评估168GS统一真实停流+8秒尾段，末端残留单列。

每32个主T决策 fork一次，计数跨世界，正常125对/种子，安全上限128。复制完整env/sim及外置previous/anchor/changed/executed；两臂共享首候选，各跑一个24秒普通窗口，无tail；独立得到下一需求与候选/53维状态。只学已确定的因果信用，后续由DDQN bootstrap，不使用固定KEEP长标签。两臂一起进入同一均匀transition池，pair_id和动作保留；单条均匀抽样在期望上两臂等权，不按收益或标签重采样。每批恒64条，记录两臂各自曝光数。

两模型各24新世界×168=4032主GS，含24BOOTSTRAP、4008T决策；最多24补充观测GS、250实际成对GS（上限256）。D/S更新0。全部训练预算与验证必须完成，不能因初期动作固定或性能较差停止。

## 数据隔离、选模和条件最终评价

所有键由 EXP088-v1 + seed + train/validation/test + 世界索引 + key类别登记。各训练24世界、验证2世界、最终3世界；不基于需求/Reward重抽。只注册最终键，整组条件未满足不生成最终世界。与EXP086/087及彼此键碰撞时明确失败。

验证GS0/2016/4032，各世界168GS；固定2h、24h、KEEP各在同世界跑一次缓存。任务加权完成E2E相对2h≤1%是本轮工程容忍度，不是统计非劣证明。额外可靠性登记：惩罚任务均值和P99均≤2h的1.01倍；截尾率、严重槽、末端残留均不增加。基线已有失败仍明确报告，不能称可靠。所有验证世界分别满足才算该checkpoint可行。

先筛可行checkpoint，再依次按两世界实际变化总数、后续切换费、含初始化的持有+切换总费、任务加权E2E排序。无可行者不保存伪best；统一last和resume_latest照存，best缺失说明原因。打开最终测试还要求两个模型均有可行checkpoint，且其每个验证世界实际变化和后续切换费严格减少、总费不增加；这定义“整组基本目标验证候选”，不对可靠性事后让步。

最终只开一次，登记锁定best/last哈希；每个世界比较2h/24h/KEEP/best/last，所有请求轨迹相同。相同权重结果允许通过身份缓存复用，明确标签。A=真实更新且验证相对初始化有物理改善；B=相对2h满足性能/可靠性并减少实际变化/后续切换费，总费不增；C=还须对可行24h/KEEP有额外净收益。固定周期更优则额外价值未证。不能以变化次数减少冒充总资源成本同比减少。

## 恢复、证据与停止

完整世界边界保存online/target/optimizer/replay/两个RNG/计数/曝光。中途异常回到上个已提交世界，覆写该未提交世界证据并重放，不能保留更新又重复计预算；无进程托管框架。源依赖覆盖edge_d、edge_joint、comparison和实际导入scripts、fork/snapshot及配置；显示函数独立于执行签名，报告改动不触发重跑。模型、场景、解码、截止和软件版本绑定缓存。

烟测：旧默认路径与分块路径同流同状态；有限信用折扣恒等式；单步两臂无tail、同开场/同候选/同流、资源守恒；BOOTSTRAP历史非空；第169次状态不被续行覆盖；64+成熟样本真实更新与保存恢复精确；报告链路。烟测不参与正式指标。异常输出原因并停止，不自动调参。独立测试未满足条件时保持未使用，不等于测试失败。

命令：`python -m scripts.exp088_run all --workers 2`；`smoke`仅烟测，`report`只读结果重画。长任务在本地独立进程，程序按世界保存status并自动汇总，无额外监视器。只保留必要best/last/resume_latest；原始完整ledger本地，紧凑逐GS和更新日志/图/Git哈希备份，不ZIP。

方法来源：[Double DQN](https://arxiv.org/abs/1509.06461)、[Dueling](https://proceedings.mlr.press/v48/wangf16.html)。论文方法不构成本项目收敛保证；因果移位、数据增广和预算为待验证研究设计。

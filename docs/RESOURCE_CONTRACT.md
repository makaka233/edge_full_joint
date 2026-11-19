# 原始资源分配器、KKT 与真实物理合同

- **链路**：原始 `physics.Resource.allocate` 在 `wired` 分支按剩余传输工作量 `sqrt(work_i)` 比例分配每条有向链路容量；发生任务加入、完成或资源成员变化时先 settle 原速率、再计算新速率，保持事件间恒定。固定路由为原始 World 多跳路径。不是独立全网 NIC 的新联合凸优化器，不得夸大。
- **CPU**：`cpu_allocation.cpu_rates` 提供 `kkt`、`ps`、`finish_pf`、`fcfs`、`srpt`。`kkt` 使用 sqrt(remaining_work) 归一分配；`finish_pf` 是另一工作守恒优先级规则。当前 `configs/experiment.json` 默认 `finish_pf`；若论文需要 CPU KKT，修改配置后重新复验，不能沿用先前 PF 实验结论。
- **无线**：每个接入 AP 的上行总容量由原世界节点字段给出，所有正在上传的用户共享；当前实现用无线均分，非学习的无线优化。
- **离散执行**：原始 `Simulator` 根据部署副本掩码调用 S，逐跳传输和逐阶段计算在事件时钟推进，记录 `STransition` 和任务完成。
- **安全性**：覆盖/RAM/SSD 为结构约束；D 后期增加 `cpu_lower_bound` 理想持续承载 LP 门控。这个必要检查不是随机排队 SLA 的充分保证；短时 CPU 超载不等于立即无限时延。
- **评价**：当前 20s 预热、2×6s 观测、8s 尾部；`failed` 是超过统计窗口被截尾的请求，并非物理永久丢弃。Reward 与惩罚后平均时延相关；必须单独报告完成、截尾和真实持续积压。

### 可复验分配器对照

```powershell
python -m scripts.verify_allocator_modes
```

固定一张 Actor 真实生成的完整矩阵，固定需求场景和请求随机流，分别用默认 CPU `finish_pf` 和原始 CPU `kkt` 模式评价，打印 JSON（可用 `--out` 另存），验证轨迹哈希相同和资源工作量守恒。结果不是算法训练曲线。

### 新实验选择 CPU 分配器

在 `configs/experiment.json` 改 `cpu_allocator` 为 `kkt`，其余配置保持不变，再运行正常 `python -m edge_d.cli ...`。请把这个物理配置差异保存在实验记录。严格控制实验中不同策略采用相同 CPU 模式。

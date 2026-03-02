# MTSAD：基于 GDN 的架构重构

## 核心动机：原版 GDN 的三大致命缺陷

在真实的工业控制系统（ICS）中，原版 GDN 的设计存在不可调和的矛盾：

1. **静态图误报：** 采用静态全局嵌入计算图结构，无法适应工业系统正常的工况切换（如启停、阀门动作）。正常的图结构改变会被强行当作异常。
2. **滞后性与突变抹杀：** 强依赖残差与 SMA（滑动平均）平滑，导致对持续时间短、破坏力强的“突发脉冲异常”响应极慢，检测延迟（TTD）极高。
3. **扩展性灾难：** 全局相似度计算带来 $O(N^2d)$ 的复杂度，面对上千维传感器网络时显存直接溢出。

------

## 模块一：特征提取与工况感知 (Condition Sensing)

**目标：** 抛弃容易抹杀局部突变的全局均值池化，敏锐捕捉局部子系统（如单条回路）的工况切换。

- **时序编码：** 对输入 $\mathbf{X}_t$ 提取特征 $\mathbf{h}_{i,t}$。

- **局部敏感注意力读出 (Subsystem-level Readout)：**

  $$e_{i,t} = \mathbf{w}_{attn}^T \text{LeakyReLU}(\mathbf{W}_h \mathbf{h}_{i,t})$$

  $$\beta_{i,t} = \text{Softmax}(e_{i,t}) \quad \Rightarrow \quad \mathbf{h}_{sys,t} = \sum_i \beta_{i,t} \mathbf{h}_{i,t}$$

------

## 模块二：防 OOM 与防塌缩的图路由生成 (Sparse MoE Routing)

**目标：** 用极小的代价生成动态图，同时解决离散路由的“梯度断裂”和“模式塌缩”陷阱。

- **低秩原型图 (Low-Rank Adapters)：** 不学 $M$ 个完整的稠密图，而是共享 Base Embedding，各工况仅学极小的低秩残差：

  $$\mathbf{E}^{(m)} = \mathbf{E}_{base} + \mathbf{U}^{(m)} \mathbf{V}^{(m)}$$

- **Gumbel-TopK 与 ST 梯度直通 (打破路由死亡)：**

  注入 Gumbel 噪声以鼓励探索，并在反向传播时给予全部分支梯度：

  $$g_t \sim -\log(-\log(\mathcal{U}(0,1)))$$

  $$\tilde{\boldsymbol{\pi}}_t = \text{Softmax}((\mathbf{z}_t + \mathbf{g}_t) / \tau) \quad \text{(Soft Gate)}$$

  $$\boldsymbol{\pi}^{hard}_t = \text{Top2}(\tilde{\boldsymbol{\pi}}_t) \quad \text{(硬截断并重归一化)}$$

  $$\boldsymbol{\pi}_t = (\boldsymbol{\pi}^{hard}_t - \tilde{\boldsymbol{\pi}}_t)\text{.detach()} + \tilde{\boldsymbol{\pi}}_t \quad \text{(前向用 Hard，反向传 Soft)}$$

- **二次稀疏化 (Post-merge Re-sparsification)：**

  将 Top-2 图加权合成后，强制对每个节点保留 $\text{Top-}K'$ 条边，严格锁死后续 GAT 的计算复杂度为 $O(T \cdot N \cdot K' \cdot d_h)$，绝不膨胀。

------

## 模块三：长尾数据驱动与联合优化 (Joint Optimization)

**目标：** 工业正常工况极度不平衡（稳态占 90%），禁止用 KL-to-Uniform 强迫模型抖动。

- **数据驱动先验与 Warm-up：**

  前 $T_w$ 个 Epoch 不加平衡 Loss。在 $T_w$ 结束时，对 $\mathbf{h}_{sys}$ 跑一次 K-means 聚类，统计真实的工业长尾分布 $\mathbf{p} = [p^{(1)}, \dots, p^{(M)}]$。

- **负载均衡损失：**

  计算 Batch 软路由均值与先验 $\mathbf{p}$ 的 KL 散度：

  $$\mathcal{L}_{balance} = \mathrm{KL}(\bar{\boldsymbol{\pi}} \| \mathbf{p})$$

- **Stop-Gradient 残差门控平滑：**

  只在预测准（稳态）时平滑图结构，预测差（异常/突变）时放开图结构。**必须 detach** 以防优化器作弊（故意改坏预测来降低平滑惩罚）：

  $$g_t = \exp(-\eta \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|_2^2)\text{.detach()}$$

  $$\mathcal{L}_{smooth} = \sum_t g_t \|\mathbf{A}^*_t - \mathbf{A}^*_{t-1}\|_F^2$$

------

## 模块四：抗误报的异常评分体系 (Anomaly Scoring)

**目标：** 把“结构断裂”当成强烈异常信号，但严格区分“合法的工况切换”与“被攻击导致的回路解耦”。

- **基础残差分数：** $\|\mathbf{x}_t - \hat{\mathbf{x}}_t\|_1$

- **工况一致性门控 (Condition-Consistency Gate)：**

  如果路由分布大变，说明是合法切换，压制结构报警分数：

  $$\Delta_\pi(t) = \|\boldsymbol{\pi}_t - \boldsymbol{\pi}_{t-1}\|_1 \in [0, 2]$$

  $$\text{Gate}_t = 1 - \frac{1}{2}\Delta_\pi(t)$$

- **归一化局部结构漂移 (Normalized Local Drift)：**

  防止高连接总线节点吸走所有分数，改为节点级归一化，并取漂移最大的 Top-q% 节点均值（对应局部回路断裂）：

  $$d_i(t) = \frac{\|\mathbf{A}^*_{t, i:} - \mathbf{A}^*_{t-1, i:}\|_1}{\|\mathbf{A}^*_{t-1, i:}\|_1 + \epsilon}$$

- **最终异常分数 $s_t$：**

  $$s_t = \alpha \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|_1 + \gamma \cdot \text{Gate}_t \cdot \text{TopqMean}(\{d_i(t)\})$$

------

## 模块五：CCF-B 投稿评测护城河

在实验设计 (Experiments) 章节必须建立的防御工事：

1. **彻底抛弃单纯的 Point Adjustment (PA) F1：** 在 2024-2025 年的顶会，用 PA 会被直接打上“进步幻觉”的标签。
2. **主打指标：** 采用 **VUS-PR** (对容忍度积分，反映整体鲁棒性) 和 **PATE** (延迟敏感，证明你的架构解决了 GDN 的滞后性)。
3. **消融实验表格预留位置：** * w/o Gumbel ST (证明防塌缩必要性)
   - w/o Data-driven Prior (证明长尾分布适应性)
   - w/o Condition Gate (证明消除工况切换误报的能力)
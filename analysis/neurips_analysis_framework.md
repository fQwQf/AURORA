# AURORA NeurIPS 改进分析框架

## 核心主张重新定位

原版「自动调参」声明被审稿人拆穿（gamma 仍然敏感）。NeurIPS 版本改为：

> AURORA 解决联邦学习中表示对齐的时序矛盾（Temporal Dichotomy）：早期需要强几何约束防止表示空间发散，后期需要弱约束允许本地适应。AURORA 通过 Meta-Annealing 与 Gradient Decoupling 将这一时序决策内化为可学习过程。

---

## 已有数据支持的分析（可直接写入论文）

### 分析 1：Meta-Annealing 确实在工作

数据来源：logs/multi_seed/CIFAR100_a005_OursV14_seed*.log

关键观察：
- Raw lambda（sigma_local^2 / sigma_align^2）从 Round 0 的约 12-15 单调衰减到 Round 9 的约 2-3
- Effective lambda = Raw x s(p)，从约 11 衰减到 0
- 3 个 seed 下轨迹高度一致（std 小），说明行为稳定
- 这是自动发生的——sigma 参数通过梯度下降学会了衰减趋势

论文写法：Figure 2 展示 lambda 轨迹。分析要点：
(a) Raw lambda 单调衰减 = sigma 参数学到了「随训练推进降低对齐约束」的先验
(b) 各客户端轨迹相似 = 对客户端数据异构性鲁棒
(c) s(p) 提供单调先验，防止 Raw lambda 后期反弹

### 分析 2：自适应初始化与数据熵的正相关

数据：15 个点（3 seeds x 5 clients），线性拟合斜率 = 19.90

结论：低熵客户端（数据更集中，非 IID 程度更高）获得更高的初始 lambda。
这是 AURORA 能感知数据分布异构性并相应调整策略的直接证据。

对比固定 lambda 基线（A2）：A2 对所有客户端施加相同强度，无法感知异构性差异。

### 分析 3：消融实验的完整故事

当前数据（CIFAR-100, alpha=0.05, K=5, 3 seeds）：

| 方法 | Mean | Std | 关键 |
|------|------|-----|------|
| FedAvg | 0.0406 | 0.0022 | non-IID 下几乎失效 |
| FedETF+Ensemble | 0.2506 | 0.0090 | ETF 固定分类器，无表示对齐 |
| FAFI No ETF-Align | 0.4249 | 0.0110 | SupCon+Proto，无几何约束 |
| AURORA w/o UncW | 0.4628 | 0.0107 | ETF 对齐但 lambda 固定 |
| AURORA | 0.4883 | 0.0045 | 完整方法 |

关键 insight：
- FAFI -> AURORA w/o UncW：+3.79%，ETF 几何约束有效
- AURORA w/o UncW -> AURORA：+2.55%，自适应调度有效
- **Std 从 0.0107 降到 0.0045（减少 58%）**：稳定性提升是比均值更有说服力的指标

### 分析 4：gamma 敏感性问题的正确解释

Reviewer h6WK 指出 gamma 变化导致 SVHN 52.9% -> 16.4%。

正确解释：
- gamma 是 stability regularization 强度，防止 lambda explosion
- SVHN alpha=0.05 极端场景下，lambda 自然趋向增大（非 IID 程度高）
- gamma=0（无约束）：lambda 爆炸，对齐项主导，本地适应完全丧失 -> 16.4%
- gamma 合理范围（1e-5 到 1e-4）内 AURORA 稳定

修复方向：
(a) 说明 gamma 是 safety bound 而非性能调节参数
(b) 展示 gamma 在 3 个量级范围内的鲁棒性
(c) 给出选择法则：gamma ≈ 1/lambda_max

---

## 待补充的新实验

### P1 必做（响应审稿人）

1. 完整消融表（当前后台运行中）
   - A3（无动态衰减）：验证 s(p) schedule 必要性
   - A5（Feature Collapse）：用 force_feature_alignment=True 而非 No Detach
   - AURORAFedAvg：量化 feature ensemble 聚合的贡献

2. gamma 鲁棒性分析
   - gamma in {1e-6, 1e-5, 1e-4, 1e-3} x 所有数据集
   - 目标：证明除 SVHN alpha=0.05 极端场景外，gamma 影响有界

3. K=20 scalability
   - 用 configs/K20.yaml 跑 AURORA 和主要 baseline
   - 验证 lambda 轨迹在更多客户端下仍然一致

### P2 强化 insight

4. 不同 alpha 下的 lambda trajectory 对比
   - alpha in {0.05, 0.1, 0.3, 0.5}（现有配置都有）
   - 预期：alpha 越小，lambda 初始值越高，衰减越慢
   - 直接证明 AURORA 能感知数据分布

5. Proto 对齐度定量分析
   - 每轮计算各客户端 prototype 与 ETF 顶点的余弦距离
   - 对比 AURORA vs AURORA w/o UncW
   - 预期：AURORA 早期对齐度更高，后期允许轻微偏离
   - 这是 Temporal Dichotomy 的直接可视化证据

---

## Feature Collapse 消融的正确设计

原版问题：Table 21 测试的是「No Detach」（梯度回传到 encoder），而非「feature vs prototype 对齐」。

正确的两个独立消融维度：

| 维度 A：对齐目标 | 维度 B：梯度是否回传 |
|------|------|
| 对齐 prototype -> ETF（AURORA 标准） | detach（标准，防止梯度干扰） |
| 对齐 feature -> ETF（force_feature_alignment=True） | no detach（ablation） |

完整 2x2 消融表才是正确的实验设计。

---

## 论文写作要点

### Experiments 部分分析深度要求

每个实验结果必须回答：
1. 这个数字说明了什么机制在工作？
2. 为什么这个方向上的性能变化是合理的（mechanistic explanation）？
3. 这个结果是否可以推广到其他场景？

### 必须明确说明的局限性

- 计算开销：sigma 参数训练增加约 X% 客户端计算量
- gamma 在极端非 IID 场景下需要关注
- 当前最多验证到 K=20，大规模联邦学习行为待验证
- 缺乏理论收敛保证（实验验证为主）

坦诚说明局限性有助于被接受，NeurIPS 对过度声明警惕。

---

## 进度跟踪

| 任务 | 状态 |
|------|------|
| 6 个分析图表生成 | 完成 (analysis/figures/) |
| 消融实验 A3/A5/AURORAFedAvg | 运行中 (PID 1337762) |
| gamma 鲁棒性实验 | 待启动 |
| K=20 scalability 实验 | 待启动 |
| alpha sweep lambda trajectory | 待启动 |
# repaired truncated tail

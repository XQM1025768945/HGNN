# 结构超图重构神经模型：脉冲权重节点与四阶段闭环框架

**摘要**  
传统神经网络以“层与张量”的堆叠为核心范式，其表达力受限于成对关系建模。本文提出一种以高阶关系结构为基座的神经模型重构框架，将超图关联矩阵的每个元素视为具有膜电位、不应期与发放行为的脉冲节点，使稀疏性发生在关系层而非表征层。该框架包含四阶段闭环：软超边构造、结构诊断、结构重构与一致性精炼，并建立微观–中观–宏观三层映射。核心模块SSHA（Spiking Structural Hypergraph Activator）通过事件驱动机制实现超图的高效稀疏计算。本文形式化分析了退化一致性、表达力条件、过平滑上界与泛化边界，并给出可证伪的实验设计。该框架为高阶关系建模与脉冲神经计算的融合提供了新路径。

**关键词**：超图神经网络；脉冲神经网络；结构重构；高阶关系；事件驱动计算

---

## 1 引言

当前深度学习模型主要基于层与张量的堆叠，如CNN、RNN与Transformer。这类范式在处理成对关系时表现优异，但面对现实世界中普遍存在的高阶交互（多模态融合、社交群体、分子复合物、脑网络）时存在根本局限：卷积核仅捕获局部成对关系，自注意力本质仍是节点对关联，图神经网络的1-WL表达力有限。超图允许一条超边连接任意数量节点，为高阶关系提供自然表示。

然而，现有超图神经网络（HGNN、HGNN+、UniGNN等）多集中于静态超图结构，缺乏动态建模能力，且未将超图结构与脉冲计算深度融合。现有SNN+超图工作通常将SNN置于节点侧，超图仅负责拓扑聚合。本文的差异化贡献在于：**将关联矩阵元素本身脉冲化**，使关系层具备事件驱动的稀疏性。

本文主要贡献：
1. 提出四阶段闭环框架，实现超图结构与神经算法的深度融合；
2. 设计SSHA脉冲权重节点模块，将 \(H_{ve}\) 建模为脉冲节点；
3. 建立微观–中观–宏观三层映射与对偶超图耦合；
4. 给出结构诊断指标体系与结构手术回调机制；
5. 形式化退化一致性、表达力条件与过平滑边界，并明确待证命题。

---

## 2 相关工作

**超图神经网络**：HGNN[1]首次将超图拉普拉斯引入GNN，HGNN+[2]扩展了多模态融合，UniGNN[3]统一了超图与图的消息传递。Sheaf Hypergraph Networks[4]引入纤维结构增强归纳偏置。

**脉冲神经网络与超图结合**：已有工作如Hybrid Spiking Hypergraph Network、事件流超图分类、spiking GNN on manifolds，均将SNN作为节点激活单元，超图仅做聚合拓扑。本文的“关联元素脉冲化”与“关系层稀疏”尚未被覆盖。

**沃尔夫勒姆超图理论**：其多路系统与因果不变性为结构手术的收敛性检验提供概念框架，但其无监督、纯生成、抽象语义的设定与本文有监督重构目标正交。

---

## 3 形式化底座

### 3.1 结构超图

定义五元组 \(\mathcal{H}=(V,E,\mathbf{H},w_v,w_e)\)，其中软关联矩阵 \(\mathbf{H}\in[0,1]^{|V|\times|E|}\) 可学习，满足
\[
\sum_e \mathbf{H}_{ve}\le 1,\qquad \sum_v \mathbf{H}_{ve}=1.
\]
顶点度 \(d(v)=\sum_e \mathbf{H}_{ve}w_e\)，超边度 \(\delta(e)=\sum_v \mathbf{H}_{ve}\)。归一化超图拉普拉斯：
\[
\Delta = I - D_v^{-1/2}\mathbf{H}W_eD_e^{-1}\mathbf{H}^\top D_v^{-1/2}.
\]
当所有 \(|e|=2\) 时退化为图拉普拉斯。

### 3.2 三层映射

| 层级 | 顶点 | 超边 | 用途 |
|---|---|---|---|
| 微观 | token/patch/通道/神经元 | 注意力分配、卷积感受野、gating簇 | 谁与谁在同一计算事件 |
| 中观 | 模块/层 | 残差/跳跃/多输入融合路径（有向） | 前向不可逆 |
| 宏观 | 样本/批次/任务 | 共享机制子网络的实例集合 | 元学习、长尾重加权 |

三层通过对偶超图 \(\mathcal{H}^*\) 耦合，对应节点↔超边双向消息传递。

### 3.3 结构不变量

谱性质：\(\lambda_2\)、谱间隙、\(\rho(\Delta)\)；拓扑：Betti数、闭包缺陷；曲率：Ollivier–Ricci分布；组合：超图模体丰度；脉冲：发放率、同步指数、脉冲熵、Fano因子、能耗。

---

## 4 四阶段闭环框架

### 4.1 阶段0：软超边构造

\[
\mathbf{H}^{(l)} = \mathrm{Sinkhorn}\!\left(\exp\left(-\frac{1}{\tau}\mathrm{Dist}(f_\theta(X^{(l)}),C)\right)\right)
\]
其中 \(C\) 为可学习原型，\(\tau\) 为温度。Sinkhorn双随机化给出参与强度且梯度可穿过。

### 4.2 阶段1：结构诊断

对每层计算谱、曲率、模体与脉冲指标，触发三类动作：
- \(\rho(\Delta)\to 1\)：过平滑边缘 → 残差扩散或降层；
- 负曲率集中 + 模体异常 → 超边细化（split）；
- 超边度重尾 + 有效秩低 → 超边合并/剪枝（prune/merge）。

### 4.3 阶段2：结构重构

前向传播重写为拉普拉斯扩散：
\[
X^{(l+1)}=\sigma\big([I-\gamma_l\Delta^{(l)}]X^{(l)}W^{(l)}+\alpha_l X^{(l)}\big)
\]
实现为两阶段消息传递：
\[
y_e=\phi\Big(\sum_v \mathbf{H}_{ve}\psi(x_v)\Big),\qquad
x_v'=\sum_e \mathbf{H}_{ve}\varphi(y_e)
\]
关键改进：\(\mathbf{H}\) 逐层动态更新；对偶重建约束 \(\mathcal{L}_{\text{dual}}=\|X-\hat X\|^2+\|Y-\hat Y\|^2\)；跨尺度多阶谱通道；可选sheaf结构。

### 4.4 阶段3：一致性精炼

自监督超边重建、超边级对比判别、结构正则 \(\lambda_1\|\mathbf{H}\|_1+\lambda_2\mathrm{tr}(X^\top\Delta X)\)、熵正则与温度退火防止塌缩。

---

## 5 脉冲权重节点模块（SSHA）

### 5.1 核心思想

将 \(\mathbf{H}_{ve}\) 视为脉冲节点，具备膜电位 \(u_{ve}\)、不应期 \(r_{ve}\)、低通迹 \(a_{ve}\)。三轨严格分离：连续态 \(u\)、二值脉冲 \(s\)、迹 \(a\)。稀疏性发生在关系层。

### 5.2 单步无环动力学

**输入**：\(s_v(t-1)\)、\(y_e(t-1)\)、\(u_{ve}(t-1),a_{ve}(t-1),r_{ve}(t-1)\)。

1. 输入电流：
\[
I_{ve}(t)=w_{\text{pre},v}s_v(t-1)+w_{\text{post},e}\|y_e(t-1)\|_2+b_{ve}-\lambda r_{ve}(t-1)
\]
2. 不应期衰减：\(r_{ve}(t^-)=\max(r_{ve}(t-1)-1,0)\)，\(\alpha_{ve}=\mathbb{1}[r_{ve}(t^-)\le 0]\)
3. 膜电位：\(u_{ve}(t)=\beta u_{ve}(t-1)\alpha_{ve}+I_{ve}(t)\)
4. 自适应阈值：\(V_{\text{th}}^{ve}(t)=\max(\mathrm{softplus}(V_0^{ve})+\eta(\hat a_{ve}-a^\star),0.1)\)
5. 发放：\(s_{ve}(t)=\Theta(u_{ve}(t)-V_{\text{th}}^{ve}(t))\alpha_{ve}\)
6. 状态更新：\(r_{ve}\leftarrow r_{ve}+ \tau_{\text{ref}}s_{ve}\)，\(u_{ve}\leftarrow u_{ve}(1-s_{ve})\)，\(a_{ve}=\rho a_{ve}+s_{ve}\)
7. 有效关联：
\[
\tilde H_{ve}=
\begin{cases}
\mathrm{softmax}(H^{\text{logits}}/T)_{ve}\cdot\sigma(3(u_{ve}-V_{\text{th}}^{ve})), & \text{train}\\
\mathrm{softmax}(H^{\text{logits}}/T)_{ve}\cdot s_{ve}, & \text{eval}
\end{cases}
\]

### 5.3 消息传递

V→E：\(y_e=\phi(W_e\frac{\sum_v \tilde H_{ve}x_v}{d_e+\epsilon})\)，\(d_e=\sum_v \mathrm{softmax}(H^{\text{logits}}/T)_{ve}+\epsilon\)  
E→V：\(x_v^{\text{agg}}=\frac{\sum_e \tilde H_{ve}y_e}{d_v+\epsilon}\)，\(x_v\leftarrow x_v+g_\theta(x_v)\odot(x_v^{\text{agg}}-\gamma(x_v-x_v^{\text{agg}}))\)

顶点LIF：\(u_v=\beta_v u_v\alpha_v+R_\theta(x_v)\)，\(s_v=\Theta(u_v-V_{\text{th}}^v)\alpha_v\)

高阶同步事件：\(s_e=\Theta(\sum_v s_{ve}-\theta_e)\)

### 5.4 激活规则

- k-WTA竞争：训练用sparsemax/entmax松弛，推理用硬top-k；
- 不应期抑制重复激活；
- 自适应阈值自动稀疏化；
- 同步检测形成高阶事件 \(s_e\)。

---

## 6 理论分析

### 6.1 退化一致性（命题1）
- 当 \(|e|\equiv 2\) 且硬门时，SSHABlock退化为spiking GCN两阶段传播；
- 当 \(\mathbf{H}\) 为one-hot时，退化为spiking attention pooling；
- 当 \(V_{\text{th}}\to 0\) 且 \(T\to\infty\)，收敛到连续HGNN前向。

### 6.2 表达力（命题2）
在注入超图级结构编码（拉普拉斯特征向量、随机游走转移、离散Ricci曲率、局部度概况）且满足非退化条件时，本框架可区分标准超图消息传递无法区分的某些正则超图。

### 6.3 过平滑上界（命题3）
由 \(\rho(I-\gamma\Delta)\) 给出Dirichlet能量的逐层衰减上界，\(\gamma\) 可据此设定。

### 6.4 泛化讨论（猜想1）
沿均匀稳定性路线，因 \(\mathbf{H}\) 也参与学习，稳定性常数需重算；对偶重建一致性收紧后，堆叠层数泛化上界不再松弛。

### 6.5 结构手术收敛性（猜想2）
若不同手术顺序（split/prune/merge）最终收敛到同一超图同构类，则框架具有因果不变性级别的鲁棒性。此可作为多路系统视角下的收敛性检验。

---

## 7 实验设计

### 7.1 验证矩阵

| 数据集 | 编码 | 对照 | 主指标 |
|---|---|---|---|
| DVS-Gesture / SHD | TTFS, T=16–32 | Spiking GNN / HSHGN | 精度、延迟、pJ/样本 |
| Cora/Citeseer转脉冲 | 泊松率, T=8 | HGNN / HSL | acc + 超边稀疏度 |
| 分子/蛋白 | 静态率 | UniGNN / Sheaf HN | 精度 vs 参数量 |
| 消融 | — | 关掉 \(s_e\) | 精度跌幅 = 高阶贡献下界 |

### 7.2 可证伪实验

1. **H0**：模体丰度 vs 随机超图 permutation test，无显著差异则不用超图。
2. **H1**：软超边能否复现已知结构（F1/ARI）。
3. **H2**：诊断指标能否指导压缩（剪层后精度保持）。
4. **H3**：动态结构+对偶重建是否优于静态（消融）。

### 7.3 Sanity check

- \(|e|\equiv2\) 退回spiking GNN；
- \(\mathbf{H}\) one-hot退回spiking attention；
- 两条退化路径给出实验曲线。

### 7.4 能耗与显存

能耗按每样本spike数×ops×pJ/spike建模（Loihi≈23.6 pJ/spike）。显存主项 \(T\cdot N\cdot M\cdot 4\text{B}\)，配合梯度检查点可降约1/3。默认锚点：Cora转脉冲 N=2708,M=256,T=8；DVS-Gesture N=1156,M=128,T=16；分子 N≤64,M=32,T=4。

---

## 8 边界与限制

- **T3不可逆**：从成对观测重建高阶结构不唯一，只能输出后验分布。
- **超边爆炸**：\(|E|\) 不得随 \(|V|\) 二次增长，需原型上限、截断k-NN、变分生成控制。
- **噪声敏感**：软关联对 \(\tau\) 与距离度量敏感，需报告温度扫描。
- **不要为超图而超图**：模体丰度不显著则不强上。
- **替代梯度有偏**；时间步增加开销；软硬门梯度冲突。

---

## 9 结论与展望

本文提出结构超图重构神经模型框架，以脉冲权重节点实现关系层稀疏与事件驱动计算，建立四阶段闭环与三层映射。理论分析给出退化一致性、表达力条件与过平滑边界，并明确待证命题。未来方向包括：超图与单纯复形结合、自适应基数控制、可解释性、神经形态硬件加速、结构迁移学习。该框架为高阶关系建模与脉冲计算的融合提供了新路径。

---

## 参考文献

[1] Feng Y, You H, Zhang Z, et al. Hypergraph Neural Networks. AAAI 2019.  
[2] Gao Y, Feng Y, Ji S, Ji R. HGNN+: General Hypergraph Neural Networks. TPAMI 2022.  
[3] Huang J, et al. UniGNN: A Unified Framework for Graph and Hypergraph Neural Networks. IJCAI 2021.  
[4] Sheaf Hypergraph Networks. NeurIPS 2023.  
[5] Wolfram S. The Physicalization of Metamathematics and Its Implications for the Foundations of Mathematics. 2022.  
[6] Zhuravlev A. Hypergraph Observers and Natural Gradient Learning. 2026.  
[7] CHRE: Causal Hypergraph Rewrite Engine. 2023.  
[8] Martins A, Astudillo R. From Softmax to Sparsemax. ICML 2016.  
[9] Wu Z, et al. A Comprehensive Survey on Graph Neural Networks. TNNLS 2021.  
[10] Davies M, et al. Loihi: A Neuromorphic Manycore Processor. IEEE Micro 2018.

---

**附录A：代码实现**  
开源仓库包含 `ssha/` 核心模块（`surrogate.py`, `weight_node.py`, `block.py`, `model.py`, `surgery.py`, `diagnosis.py` 等）、`tests/` 退化与梯度测试、`scripts/` 训练诊断脚本。运行 `pip install -e . && pytest tests/ -v` 即可验证。

**附录B：关键实现纪律**  
三轨分离；单步V→E→V交错；训练软门+sparsemax，推理硬top-k；期望度归一化；Sinkhorn仅初始化；LIF真正接入；读出层补齐；BPTT与STDP分离（结构走手术回调）。

# SSHA 技术文档：核心算法、Bug 修复与测试验证

> 结构超图重构神经模型（Structural Soft-Hypergraph with Spiking Association）
> 版本 0.2.0 · 2026-10-13

***

## 目录

1. [项目定位](#1-项目定位)
2. [核心算法原理](#2-核心算法原理)
3. [关键接口契约](#3-关键接口契约)
4. [Bug 修复详情](#4-bug-修复详情)
5. [测试验证结果](#5-测试验证结果)
6. [运行方式](#6-运行方式)
7. [已知边界与后续工作](#7-已知边界与后续工作)

***

## 1. 项目定位

把神经网络的「层 + 张量」范式换成「高阶关系结构」范式：用软超图 **H** ∈ R^{N×M} 表达节点间的高阶交互，用脉冲权重节点（Spiking Weight Node）在**关系层**（而非表征层）实现事件驱动稀疏。

三条主线：

| 主线     | 定位    | 内容                |
| :----- | :---- | :---------------- |
| T2（主）  | 结构即算法 | SSHA + 四阶段闭环定义新前向 |
| T1（子）  | 结构诊断  | 压缩已有模型，剪 20–30% 层 |
| T3（边界） | 结构反演  | 只输出后验分布，不承诺唯一解    |

***

## 2. 核心算法原理

### 2.1 脉冲权重节点（SpikingWeightNode）

关联张量 **H** 的**每个元素**是一个有膜电位、有不应期、会发放的脉冲节点。文件：[weight\_node.py](ssha/weight_node.py)。

**可学参数**：

| 参数          | 形状      | 作用                                   |
| :---------- | :------ | :----------------------------------- |
| `H_logit`   | \[N, M] | 软关联 logits（经 softmax/temp → H\_base） |
| `w_pre`     | \[N]    | 突触前脉冲驱动权重（**非零初始化，见 4.5 Bug 7**）     |
| `w_post`    | \[M]    | 突触后超边特征驱动权重（**非零初始化**）               |
| `b`         | \[N, M] | 逐元素偏置（零初始化）                          |
| `v_th_base` | \[N, M] | 阈值基底（经 softplus 保证为正）                |
| `lam_ref`   | 标量      | 不应期抑制系数                              |

`w_pre` / `w_post` 由 `drive_init`（默认 1.0）缩放的高斯初始化。它们**必须非零**：
驱动项 `I = w_pre ⊙ s_v_prev + …` 是乘法，`w_pre ≡ 0` 会让 `∂I/∂s_v_prev = 0`，
切断顶点脉冲到关系层的全部梯度通路（详见 4.5 Bug 7）。

**状态变量**（buffer，`reset_state()` 重建）：膜电位 `u`、脉冲 `s`、活跃度迹 `a`、不应期计时 `ref`、发放率估计 `rate_est`、跨时间步累计脉冲 `s_acc`（供诊断使用）。

**单步动力学**（对应 `forward`）：

```text
I        = (w_pre ⊙ s_v_prev) ⊗ 1 + 1 ⊗ (w_post ⊙ ‖y_e_prev‖) + b − λ_ref · ref
ref      = max(ref − 1, 0)                      # 不应期计时衰减
active   = 1[ref ≤ 0]                           # 不应期门控
u_pre    = β · u · active + I                   # 漏电积分得到**复位前**膜电位（β=0.85）
v_th     = clamp(softplus(v_th_base) + 0.2·(rate_est − target), min=0.1)
s        = spike_fn(u_pre − v_th) · active      # ATan 替代梯度发放
ref     += s · τ_ref                            # 发放后进入不应期
u        = u_pre · (1 − s)                      # 发放复位（此后 u 不再是 u_pre）
a        = ρ · a + detach(s)                    # 活跃度迹（供手术判据）
s_acc   += detach(s)                            # 跨时间步累计（供诊断）
rate_est = (1−homeo_rate)·rate_est + homeo_rate·detach(s)
```

注意 `u_pre` 是**复位前**膜电位，必须单独留存：复位会把恰好发放的条目清零，
软门若用复位后的 `u` 就会与脉冲反相关（详见 4.5 Bug 5）。

`v_th` 是**有效**静息阈值（默认 0.2）：`v_th_base = log(expm1(v_th))`，使 `softplus(v_th_base) == v_th`。
若直接把 0.5 填进 softplus 空间，实际阈值会是 0.974，小 N 配置下会导致关系层死启动（详见 4.5 Bug 8）。

发放率估计 `rate_est` 以 `homeo_rate=0.01` 做指数滑动平均，并通过 `0.2·(rate_est − target)` 项反馈调节阈值——这是**内稳态机制**：发放过频的节点阈值升高，反之降低。

**软/硬两分支输出**：

```text
soft (hard=False):  H_eff = H_base ⊙ sigmoid(3·(u_pre − v_th))  # 可微门，训练用
hard (hard=True):   H_eff = hard_topk(H_base ⊙ s, k=8, dim=0)   # 事件驱动，推理用
```

### 2.2 SSHA 块：V→E→V 交错

文件：[block.py](ssha/block.py)。单个 `SSHABlock` 完成一次「顶点 → 超边 → 顶点」传播：

```text
H_eff = SpikingWeightNode(s_v_prev, y_e_prev)          # 关系层脉冲稀疏
x_v   = W_v · x                                        # 顶点侧线性变换（4.5 Bug 6）
y_e   = GELU( W_e · (H_effᵀ x_v / d_e) )               # V→E 聚合（d_e 为超边度）
x_agg = (H_eff y_e) / d_v                              # E→V 回传（d_v 为顶点度）
x_new = x + g ⊙ (x_agg − γ·(x − x_agg))                # 门控残差 + 拉普拉斯平滑项
```

`W_v` 此前只被构造、从未参与前向（梯度恒为 `None`），与「前 30% epoch 只学 (W\_v, W\_e)」
的训练纪律直接矛盾；现已接入 V→E 聚合，残差路径仍走原始 `x` 以保留 skip connection。

随后顶点侧 LIF 发放 `s_v = spike_fn(u_v − 0.5)`，超边侧同步事件 `s_e = 1[col_sum(s) > 2]`（一条超边内 ≥3 个权重节点同时发放，视为一次高阶同步）。

### 2.3 替代梯度（ATan Surrogate）

文件：[surrogate.py](ssha/surrogate.py)。前向为 Heaviside 阶跃，反向用反正切导数近似：

```text
前向:  s = 1[x > 0]
反向:  ds/dx ≈ (α/2) / (1 + (π·α·x/2)²)，内层截断至 ±5 防爆
```

`spike_fn(u)` 使脉冲不可微问题转为可 BPTT 的平滑梯度。这是 `H_logit` 之外的权重（`W_v, W_e` 等）的训练信号来源。

### 2.4 四阶段闭环

| 阶段    | 模块                                | 功能                                                         |
| :---- | :-------------------------------- | :--------------------------------------------------------- |
| 0 初始化 | [sinkhorn.py](ssha/sinkhorn.py)   | 对 H\_logit 做 Sinkhorn 双随机化（20 次行列归一），得到偏均匀的软超边             |
| 1 诊断  | [diagnosis.py](ssha/diagnosis.py) | 谱指标（λ₂、谱隙、ρ）、近似 Ollivier–Ricci 曲率、三阶模体丰度、脉冲统计（率/同步/熵/Fano） |
| 2 手术  | [surgery.py](ssha/surgery.py)     | prune（沉默边）/ merge（余弦 > 0.98）/ split（高活跃 × 高梯度范数）           |
| 3 精炼  | 训练循环                              | 冻结结构，只学权重                                                  |

**训练纪律**（前 30% / 中 40% / 后 30% epoch）：先只学 `W_v, W_e` 并冻结 `H_logit`；中期每 K epoch 触发一次 `structure_surgery`；后期冻结结构精修。`H` 只走手术回调，不走在线 STDP。`scripts/train.py` 已按此实现（4.5 Bug 16）。

### 2.5 结构手术判据

文件：[surgery.py](ssha/surgery.py) 的 `structure_surgery`：

- **prune**：`edge_activity = a.mean(dim=0) < prune_thr(0.02)` 判死；保留至少 `min_edges=2` 条（按活跃度 topk 兜底，且 `k = min(min_edges, M)` 防越界）
- **merge**：对 H\_base 列向量归一化后两两求余弦，> 0.98 则取均值合并；合并次数上限 `M − min_edges`，保证合并后仍保留 `min_edges` 条
- **split**（可选）：`score = ‖grad(H_logit)‖_col · a_mean > split_grad_thr` 的列复制一份并加小扰动（σ=0.01）。
  进入手术前会快照结构梯度，并让它跟随 prune/merge 的列掩码一起对齐，因此 split 在剪枝后仍然可达（4.5 Bug 9）

手术后保证 `sw.M == block.M`、`H_logit.shape[1] == M`、内部状态张量形状一致（测试硬约束）。
`SSHAModel.M` 是只读属性，实时反映 `block.M`（4.5 Bug 12 附带）。

### 2.6 退化契约（审稿人硬约束）

1. |e| ≡ 2 且 `hard=True` → 每列恰 2 个非零，等价 spiking GCN 两阶段传播
2. **H** one-hot 且 `hard=True` → 每行 ≤ 1 个非零，等价 spiking attention pooling
3. V\_th → 0（softplus 基底 −20）→ 软/硬门的 H\_eff 均值误差 **< 0.02**（修复 4.5 Bug 5 前为 0.14 量级）

### 2.7 能耗模型

文件：[energy.py](ssha/energy.py)。按 Loihi 量级建模：

```text
pJ = (n_sv + n_sve) · pJ_ac + ops · pJ_mem，  ops = Σ(‖s_v‖₁ · d · 2)
默认 pJ_ac = 1.0，pJ_mem = 0.1
```

注意该公式不为 `spikes_e`（高阶同步事件）收费，与本节公式一致，属建模取舍（见 §7）。

***

## 3. 关键接口契约

| 模块                  | 接口                                                                | 返回                                                                                    |
| :------------------ | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `SpikingWeightNode` | `forward(s_v_prev, y_e_prev, hard=False)`                         | `H_eff: [N, M]`，非负                                                                    |
| `SSHABlock`         | `forward(x, s_v_prev, y_e_prev, hard=False)`                      | `(x_new, s_v, y_e, H_eff, s_e)`                                                       |
| `SSHAModel`         | `forward(x_seq[T,N,d], hard=False)`                               | `(logits[C], aux={spike_e, spike_ve, spike_v, H_eff, struct_mass})`；`M` 为只读属性，随手术自动更新 |
| `EnergyMeter`       | `record(sv, se, sve, d)` / `report()` / `reset()`                 | `dict(spikes_v/e/ve, ops, pJ)`                                                        |
| `structure_surgery` | `(block, epoch, prune_thr, merge_cos, split_grad_thr, min_edges)` | `dict(pruned, merged, split)`                                                         |
| `motif_abundance`   | `(H, threshold=None, top_k_per_row=2)`                            | `float`；两种二值化策略互斥，`threshold` 优先                                                      |

***

## 4. Bug 修复详情

### 4.1 第一轮

实现过程中修复了 README 参考代码中的 **3 处源码 bug** 与 **1 处测试 bug**，另对训练脚本做了 **1 处适配修改**。全部修复均通过最小复现验证。

> 第二轮独立审计（见 4.5）另确认并修复 **13 处**缺陷。`block.py` 的 `de`/`de.T`
> 广播修复仍然有效且必要，但 `baselines/hgnn.py` 存在**同类未修复**问题（见 4.5 Bug 13）。

#### Bug 1：V→E 聚合的广播维度错误（block.py）

**现象**：12 个测试失败，报 `The size of tensor a (6) must match the size of tensor b (4) at non-singleton dimension 1`。

**根因**：参考代码中

```python
de = H_base.sum(dim=0, keepdim=True) + 1e-9   # 形状 [1, M]
y_e = self.phi(self.W_e((H_eff.T @ x) / de))  # [M, d] / [1, M] → 广播失败
```

`[M, d]` 与 `[1, M]` 按尾部对齐广播时要求 `d == M`，一般情形直接报错。

**修复**（[block.py:46](ssha/block.py#L46)）：`de` 转置为 `[M, 1]`；`dv` 本身已是 `[N, 1]` 无需改动。

**教训**：一度错误地把 `dv` 也转置成 `[1, N]`，导致同一位置二次报错；`keepdim=True` 下 `sum(dim=1)` 产出 `[N,1]` 恰好正确，只需修 `de`。

#### Bug 2：脉冲节点状态懒初始化（weight\_node.py）

**现象**：`test_surgery.py` 三个用例全部 `AttributeError: 'NoneType' object has no attribute 'zero_'`。

**根因**：`u/s/a/ref/rate_est` 注册为 `None`，仅在首次 `forward` 时经 `_ensure_state` 懒初始化；而手术测试在任何前向之前直接操作 `blk.sw.a.zero_()`。

**修复**（[weight\_node.py](ssha/weight_node.py)）：`__init__` 末尾直接调用 `self.reset_state()`，`_ensure_state` 保留作为安全网。

#### Bug 3：H\_base 缓存携带旧计算图（weight\_node.py）

**现象**：训练第 2 个 epoch 起报 `Trying to backward through the graph a second time`。

**根因**：`get_H_base()` 的缓存 `_H_base_cache` 是带 `grad_fn` 的 softmax 张量。`SSHAModel.forward` 每个 epoch 开头 `reset_state()` 重建膜电位等状态，但**没有清缓存**，于是新 epoch 的前向复用了上一 epoch 的图节点，反向时图已释放。

**修复**：`reset_state()` 末尾加 `self._H_base_cache = None`，状态重置即缓存失效。

#### Bug 4（最关键）：手术原地改 Parameter 形状污染 autograd 元数据（surgery.py）

**现象**：训练在 epoch 3 手术（M: 8→2）后，epoch 4 的 `loss.backward()` 必现：

```text
RuntimeError: Function SoftplusBackward0 returned an invalid gradient at
index 0 - got [32, 2] but expected shape compatible with [32, 8]
```

**根因**：PyTorch autograd 引擎为 requires-grad 叶子缓存的形状元数据与叶子张量身份绑定；`.data =` 原地替换底层存储虽更新了 Python 层 `.shape`，但引擎侧缓存仍以旧形状校验后续 backward 的梯度。**原地改变叶子 Parameter 的形状在 PyTorch 中不是安全操作。**

**修复**：`_apply_prune` / `_apply_split` 中**替换 Parameter 对象**而非原地改形状（新张量 = 新元数据）：

```python
sw.H_logit   = nn.Parameter(sw.H_logit.data[:, idx].contiguous())
sw.b         = nn.Parameter(sw.b.data[:, idx].contiguous())
sw.v_th_base = nn.Parameter(sw.v_th_base.data[:, idx].contiguous())
sw.w_post    = nn.Parameter(sw.w_post.data[idx].contiguous())
```

状态 buffer（`u/s/a/ref/rate_est/s_acc`）仍在 `@torch.no_grad()` 下切片，天然不携带 `grad_fn`，无此问题。

#### 修改 5：手术后重建优化器（train.py）

Bug 4 的修复使 Parameter 对象身份改变，原 `Adam` 持有的仍是旧参数引用（其动量状态形状也已失效）。`structure_surgery` 后必须重建优化器。

#### 修改 6：dtype 一致性测试修正（test\_shapes.py）

原测试直接将 float64 输入喂给 float32 模型，PyTorch 不允许 `float @ double` 混合精度矩阵乘。按 PyTorch 标准惯例修正为用户侧先转换模型。

***

### 4.5 第二轮修复（独立审计 + 本地硬件验证）

在 4 处原有 bug 之外，对全部模块做了独立审计（静态审查 + 最小复现 + 第二路 agent 交叉复核），
又确认并修复 **13 处**缺陷。每条都有回归测试，并已用「把 5 个模块回滚到修复前再跑测试」的方式验证：
当时 `tests/test_regressions.py` 的 17 个用例中 **14 个失败**，只有 3 个通过
（`split_is_noop_without_grad`、`surgery_keeps_state_shapes_consistent`、
`loss_sparse_on_H_base_is_constant` —— 这三个是既有正确行为的守卫，本就不该失败）。
之后又补了 4 个用例（有效阈值语义、跨配置/种子无死启动、时间平均脉冲指标、术后指标一致性），
现共 21 个用例。

#### Bug 5（最关键）：软门用了复位后的膜电位（weight\_node.py）

**现象**：`soft_gate` 取 `self.u`，而 `self.u` 已被 `self.u = self.u * (1 - s_new)`
把**恰好发放**的条目清零。于是软门与脉冲**反相关**。

**最小复现**：`SpikingWeightNode(N=1, M=2)`，`v_th=0.5`，`b=[[2.0, 0.3]]`。

| 条目    | u\_pre | s  | 旧代码门值     | 应有门值  |
| :---- | :----- | :- | :-------- | :---- |
| 0（发放） | 2.0    | 1  | **0.182** | 0.989 |
| 1（静息） | 0.3    | 0  | 0.354     | 0.354 |

**后果**：训练时压制真正发放的连接，与硬分支选到**相反**的超边；退化契约 #3 之所以仍通过，
只是因为阈值 0.5 比该缺陷引入的误差（0.1436）宽 3.5 倍。

**修复**：把复位前电位单独留存为 `u_pre`，`s` 与软门都用它。契约 #3 误差 0.1436 → **0.0157**。

#### Bug 6：`W_v` 从未参与前向（block.py）

forward hook 实测执行的 Linear 只有 `['W_e', 'gate.0', 'read_v']`；`W_v.weight.grad is None`；
扰动 `W_v` 对输出影响恰为 0.0。而训练纪律要求前 30% epoch 学 `(W_v, W_e)`。
**修复**：`x_v = W_v(x)` 接入 V→E 聚合，残差仍走原始 `x`。

#### Bug 7：`w_pre` 零初始化切断顶点脉冲→关系层的梯度（weight\_node.py）

`I = w_pre ⊙ s_v_prev` 是**乘法**，`w_pre ≡ 0` 时 `∂I/∂s_v_prev = 0`，于是
`read_v` / `lam_ref` 梯度**恰为 0**（实测 `read_v.weight += 5` 对 logits 影响为 0.0）。
**修复**：`w_pre` / `w_post` 用 `drive_init=1.0` 缩放的高斯初始化（`b` 仍为零，以保住 |e|=2 退化契约）。

#### Bug 8：关系层死启动，`hard` 模式下 `H_eff ≡ 0`（weight\_node.py / model.py）

驱动全零 ⇒ `I ≡ 0 < v_th` ⇒ 12 个 epoch 内 `sw.s` 发放率恒为 `0.00000`，
`hard=True` 时 `H_eff ≡ 0`，前向退化成 `x_new = x·(1−g·γ)`（实测每步 0.9419 倍纯衰减）。

**修复分两步**：

1. 同 Bug 7 的非零驱动初始化；
2. 把 `v_th` 改成**有效**阈值语义——原先 `v_th_base` 直接填 `0.5`，经 softplus 后实际阈值是
   `softplus(0.5) = 0.974`。顶点脉冲发放率约 10%，乘 `w_pre` 后小 N 配置普遍压不到 0.974。
   实测 `N=16` 时 8 个种子里 **4 个整条时间链零发放**；`drive_init` 调到 2.0 也只是把死启动
   从 13/40 降到 1/40，治标不治本。现改为 `v_th_base = log(expm1(v_th))`，默认有效阈值 `v_th=0.2`。

修复后：48 组 (config, seed) 组合（N=16…512，8 个种子）**零死启动**，
关系层发放率落在 \[0.024, 0.128]，`s_e`（高阶同步事件）> 0，`lam_ref` 获得非零梯度。

#### Bug 9：`split` 手术永久失效（surgery.py）

`_apply_prune` 会替换成新的 `nn.Parameter`（`.grad is None`），而 split 的守卫正是
`sw.H_logit.grad is not None`。实测：无剪枝时 `split=4`；触发一次剪枝后 `split=0`。
即使守卫通过，`g_norm`（\[M\_old]）与 `a_mean`（\[M\_new]）形状也不匹配。
**修复**：手术前快照 `grad`，并让它跟随 prune/merge 的列掩码一起对齐。

#### Bug 10：`merge` 可把 M 压到低于 `min_edges`，导致下次手术崩溃（surgery.py）

`min_edges` 原先只约束 prune。实测 4 条相同列 + `min_edges=2` ⇒ 第一次调用把 M 压到 1，
第二次调用 `edge_activity.topk(min_edges)` 抛 `RuntimeError: selected index k out of range`。
**修复**：限制合并次数为 `M - min_edges`；prune 侧用 `k = min(min_edges, sw.M)` 兜底。

#### Bug 11：`motif_abundance` 在 `H_base` 上结构性恒为 0（diagnosis.py）

`H_base` 是**行** softmax（每行和为 1），因此每行**至多一个**元素能超过 0.5；
0.5 阈值二值化得到的关联矩阵各列互不共享顶点，**模体丰度恒等于 0**（对任意 M ≥ 2）。
诊断输出里 `motif: 0.0000` 并非「没有高阶结构」，而是指标失效。
**修复**：二值化策略改为显式参数（默认按行 top-k=2 隶属；`threshold` 显式给出时优先且互斥）；
同时把 O(M³) 的 Python 三重循环换成 `trace(A³)/6` 向量化计数——M=256 时 13.33s → **0.0028s**，
300 组随机用例上数值完全一致。分母的 `+1e-9` 也去掉了（M≥3 时恒 ≥ 1，无需 epsilon，
且它会让精确比较失败）。

#### Bug 12：`diagnose_block` 读缓存的过期 `H_base`（diagnosis.py）

实测 `H_logit[:, 0] += 5` 后，缓存与真值最大差 0.9997，`λ₂` 报 `0.378330` 而真值为 `1.000000`。
**修复**：改为 `get_H_base(refresh=True)`。
附带：`SSHAModel.M` 由构造时快照改为只读属性（手术后 `model.M` 原会停在旧值）。

#### Bug 13：`baselines/hgnn.py` 广播轴错误（基线不可运行）

`L = H @ (H.T / de)` 中 `de = H.sum(dim=0, keepdim=True)` 为 `[1, M]`，与 `[M, N]` 的 `H.T`
广播要求 `N == M`。注意 **`dv`** **是** **`[N, 1]`，`L @ X / dv`** **本身正确，不应改动**；
只把 `de` 改成 `de.T` 即可。`N == M` 时旧代码不崩但**静默算错**（实测偏差 0.093）。

#### Bug 14：`baselines/spiking_gnn.py` 时间维循环把 `lin1` 作用于自身输出

`lin1` 是 `in_dim → hid_dim`，却在 T 循环里反复吃上一步的输出，`t ≥ 1` 时输入宽度已是 `hid_dim`，
除 `hid_dim == in_dim` 或 `T == 1` 外必崩。
**修复**：改为真正的 LIF 递推（输入投影一次，膜电位跨时间步积分），使 `T` 真实生效，
并把 `hard` 实现为真开关（`spike_fn` 硬脉冲 / `soft_gate` 软门）。

> 中间的过渡实现曾被写成「每步重复对同一输入投影再按 1/T 平均」，那样 `T` 会被时间均值
> 完全抵消、等于无效参数。回归测试现在显式断言 `T` 必须影响输出。

#### Bug 15：`scripts/eval.py` 喂给能耗表的顶点脉冲数是伪造的

原代码 `meter.record(torch.ones(N), aux["spike_e"], model.block.sw.s, dim)`：
`sv` 是常量全 1（不是模型真实的 `s_v`），`sve` 取的是最后一步的 `sw.s`。
同一行给出 `spikes_v: 64.0, pJ: 473.6`，换成 `zeros(64)` 就是 `0.0 / 0.0`。
**修复**：`SSHAModel.forward` 新增 `aux["spike_v"]`（T 步顶点脉冲累计）与
`aux["struct_mass"]`，`eval.py` 改用真实计数。

#### Bug 16：`scripts/train.py` 未实现文档的训练纪律

原脚本只用交叉熵、`H_logit` 从不冻结、每个 epoch 只拿**一个随机标签**训练整张图，
`losses.py` 里的 `loss_rate`/`loss_sync`/`loss_sparse`/`loss_energy` 从未被调用。
**修复**：实现 30% / 40% / 30% 三阶段纪律（冻结结构 → 手术 → 精炼）、术前快照梯度、
术后重建优化器、接入四项辅助损失、加梯度裁剪与设备自动探测。
另注意 `loss_sparse` **不能**作用在 `H_base` 上（行 softmax 使其元素均值恒为 `1/M`，梯度为零），
必须作用在 `H_eff` 上。

#### Bug 17：`spike_metrics` 只取最后一步，诊断可能全零（diagnosis.py）

`spike_metrics` 读 `sw.s`（**最后一步**的瞬时脉冲）。LIF 有不应期，最后一步可能恰好全部静默，
于是 `rate/sync/entropy/fano` 全报 0，看起来像「网络没在发放」，即使整条时间链很活跃
（实测曾出现 `trace_mean=0.21` 而 `rate=0`）。`scripts/diagnose.py` 也因此每次输出不同（未固定种子）。
**修复**：`SpikingWeightNode` 新增跨时间步累计 buffer `s_acc` 与步数 `n_steps`，
`spike_metrics` 改用 `s_acc / n_steps`（累计不可用时才退回瞬时 `s`）；
`surgery.py` 在 prune/split 时同步维护 `s_acc`；`scripts/diagnose.py` 固定随机种子。

***

## 5. 测试验证结果

### 5.1 全量测试：53/53 通过

```text
$ python -m pytest tests/ -v
======================= 53 passed, 1 warning in 0.92s ========================
```

| 测试文件                 | 用例 | 结果 | 验证点                                                      |
| :------------------- | :- | :- | :------------------------------------------------------- |
| test\_degeneracy.py  | 3  | ✅  | 三条退化契约（\|e\|≡2、one-hot、连续极限）                             |
| test\_gradients.py   | 4  | ✅  | `H_logit` 软分支梯度非零、硬分支梯度有限、替代梯度通畅、反向无 NaN/Inf             |
| test\_shapes.py      | 5  | ✅  | 各层 I/O 形状、reset 幂等、float64 一致                            |
| test\_surgery.py     | 3  | ✅  | prune 生效、M 一致、术后可前向、全活跃时 no-op                           |
| test\_energy.py      | 4  | ✅  | 计数精确、能耗随脉冲单调、确定性、reset                                   |
| test\_sanity.py      | 2  | ✅  | 模体置换检验 p 值合法、有限性                                         |
| test\_baselines.py   | 11 | ✅  | HGNN 非方阵/参考实现/静默错误守卫；SpikingGNN `T` 真实生效、`hard` 真开关、梯度有限 |
| test\_regressions.py | 21 | ✅  | 4.5 全部修复的回归（14 个用例在旧代码上失败）                               |

唯一 warning 为测试内 `float(带梯度张量)` 的提示，无语义影响。

### 5.2 端到端训练（含结构手术）

```text
$ python scripts/train.py --epochs 12 --N 64 --M 32 --dim 32 --T 8 --classes 5 \
      --batch-size 8 --steps-per-epoch 4 --eval
[env] device=cuda torch=2.10.0+cu128 threads=14
[env] gpu=NVIDIA GeForce RTX 3060 vram=12.0GiB
epoch 000 [warmup(freeze H_logit)] loss=1.8588 train_acc=0.969 test_acc=1.000 M=32
epoch 001 [warmup(freeze H_logit)] loss=1.7045 train_acc=1.000 test_acc=1.000 M=32
epoch 002 [warmup(freeze H_logit)] loss=1.4833 train_acc=1.000 test_acc=1.000 M=32
epoch 003 [surgery               ] loss=1.4770 train_acc=1.000 test_acc=1.000 M=32 surgery={'pruned': 0, 'merged': 0, 'split': 0}
epoch 004 [surgery               ] loss=1.3737 train_acc=1.000 test_acc=1.000 M=32
epoch 005 [surgery               ] loss=1.2200 train_acc=1.000 test_acc=1.000 M=32
epoch 006 [surgery               ] loss=1.1367 train_acc=1.000 test_acc=1.000 M=18 surgery={'pruned': 14, 'merged': 0, 'split': 0}
epoch 007 [surgery               ] loss=1.0505 train_acc=1.000 test_acc=1.000 M=18
epoch 008 [refine(freeze structure)] loss=1.0206 train_acc=1.000 test_acc=1.000 M=18
epoch 009 [refine(freeze structure)] loss=0.9433 train_acc=1.000 test_acc=1.000 M=18
epoch 010 [refine(freeze structure)] loss=0.8733 train_acc=1.000 test_acc=1.000 M=18
epoch 011 [refine(freeze structure)] loss=0.7921 train_acc=1.000 test_acc=1.000 M=18

=== training finished ===
epochs=12 M_final=18 params=5818 wall=10.8s
final test_acc=1.000
peak gpu mem: 25.1 MiB (reserved 30.0 MiB)
per-sample energy: pJ=2046.9839  spikes_v=27.19  spikes_e=36.53  spikes_ve=1845.76  ops=1740.3
```

三阶段纪律正确切换：epoch 0–2 冻结 `H_logit`，epoch 3–7 每 3 epoch 触发手术（32→18），
epoch 8–11 冻结结构精炼。手术剪掉的 14 条是**训练中被内稳态压到沉默**的真实死边
（epoch 3 时还全部活跃，故 `pruned: 0`；到 epoch 6 才出现沉默边），
而不是第一轮那种「关系层从不发放」造成的常数式剪枝。

> 内置合成任务是「可学习性冒烟测试」：标签定义在模型实际消费的编码量
> （`sigmoid(gain·X)` 的节点均值）上，因此 `test_acc=1.000` 只证明
> 前向/反向/手术/优化链路正确，**不构成对真实数据集的性能声明**。

### 5.3 结构诊断脚本

`scripts/diagnose.py` 现已固定随机种子（原先未固定，每次输出不同）。

```text
$ python scripts/diagnose.py
device: cuda  seed: 0
    lambda_2: 0.6664
spectral_gap: 0.6664
         rho: 1.0000
       ricci: -0.2059
       motif: 0.0839
        rate: 0.0564
        sync: 40.8398
     entropy: 0.2169
        fano: 0.1938
  trace_mean: 0.2070
     spike_v: 29
    spike_ve: 231
     spike_e: 16
```

λ₂ ≈ 0.67、ρ = 1.0 符合归一化拉普拉斯谱性质。与第一轮相比脉冲指标不再全零：
`rate`、`sync`、`entropy`、`fano`、`trace_mean`、`motif` 全部非零——即关系层真的在发放
（修复 4.5 Bug 7/8），模体指标也真的在度量（修复 4.5 Bug 11），
脉冲指标也不再受「最后一步恰好静默」影响（修复 4.5 Bug 17）。

### 5.4 能耗评估脚本

```text
$ python scripts/eval.py
device      : cuda
logits      : (8, 5)  (8 samples x 5 classes)
soft/hard max abs diff : 0.001882
energy      : {'spikes_v': 475.0, 'spikes_e': 281.0, 'spikes_ve': 1779.0, 'ops': 30400.0, 'pJ': 5294.0}
per-sample  : pJ=661.7500 spikes_v=59.38 spikes_e=35.12 spikes_ve=222.38 ops=3800.0
```

`spikes_v` 现在是模型真实的顶点脉冲数（修复 4.5 Bug 15 前是常量 `torch.ones(N)`，恒为 N）。

### 5.5 本地硬件验证（RTX 3060 12 GiB / i5-14600KF 14 核 / 32 GiB RAM）

| 配置（对应 `configs/`）      |    N |   M | dim |  T | epochs |  wall |     峰值显存 | test\_acc |
| :--------------------- | ---: | --: | --: | -: | -----: | ----: | -------: | --------: |
| 小规模冒烟                  |   64 |  32 |  32 |  8 |     12 | 10.8s |   25 MiB |     1.000 |
| `cora_spiking.yaml` 缩减 | 1024 | 256 | 128 |  8 |     12 | 17.4s |  819 MiB |     1.000 |
| `cora_spiking.yaml` 全量 | 2708 | 256 | 128 |  8 |      8 |  4.9s | 1091 MiB |     1.000 |
| `dvs_gesture.yaml`     | 1156 | 128 |  64 | 16 |      6 |  4.8s |  468 MiB |     1.000 |
| README §6 长序列预算档       |  512 | 512 |  64 | 32 |      4 |  5.2s | 1342 MiB |     1.000 |
| CPU 回退（`--device cpu`） |  256 |  64 |  64 |  8 |      6 |  2.4s |        — |     1.000 |

12 GiB 显存对文档中的全部配置都绰绰有余（最高占用 1.3 GiB）。诊断模块也已向量化：
`motif_abundance` 在 M=256 时 13.33s → 0.0028s；`spectral_metrics`（稠密 N×N 特征分解）
在 N=2708 时约 0.44s。

***

## 6. 运行方式

```bash
# 安装（亦可直接以 PYTHONPATH 运行脚本）
pip install -e .

# 全量测试 / 关键三组
pytest tests/ -v
pytest tests/test_regressions.py tests/test_degeneracy.py tests/test_surgery.py -v

# 覆盖率
pytest tests/ --cov=ssha --cov-report=term-missing

# 端到端（自动探测 CUDA；加 --device cpu 可强制回退）
python scripts/train.py --N 2708 --M 256 --dim 128 --T 8 --classes 7 --epochs 20
python scripts/train.py --N 1156 --M 128 --dim 64 --T 16 --classes 11 --epochs 80 --device cuda
python scripts/diagnose.py
python scripts/eval.py
```

> 注意：`python scripts/train.py` 在仓库根目录直接执行会报
> `ModuleNotFoundError: No module named 'ssha'`（`scripts/` 是 `sys.path[0]`，仓库根不在其中）。
> 先 `pip install -e .`，或设置 `PYTHONPATH=<repo>`。

***

## 7. 已知边界与后续工作

- `data/` 的 Cora/DVS loader 仍为占位实现（合成数据），接真实数据需 `torch_geometric` 与官方 DVS h5；
- `split` 判据现已可达（4.5 Bug 9），但 `train.py` 默认传 `split_grad_thr=None`，仍只跑 prune + merge；
  split 会按「高活跃 × 高梯度范数」复制超边，其收敛性尚未验证，正式启用前建议加多路顺序一致性检验；
- Ollivier–Ricci 曲率为粗代理（度分布矩），正式版建议接 `GraphRicciCurvature`；
- `energy.py` 的 `pJ` 公式不向 `spikes_e` 收费（与 2.7 的公式一致，属建模取舍而非代码缺陷）；
  若要把高阶同步事件计入能耗，需显式扩展该公式；
- **训练超参未做调优**：`drive_init=1.0` 与有效阈值 `v_th=0.2` 是按
  「48 组 (config, seed) 零死启动、发放率 2%–13%、`s_e` > 0」选定的，
  不同 N/M 下可能需要重标定；`w_rate`/`w_sync`/`w_sparse` 权重同样未经搜索；
- 内置合成任务是「可学习性冒烟测试」，`test_acc=1.000` 只能证明前向/反向/手术/优化链路正确，
  **不构成对真实数据集的性能声明**；
- `README.md` 内嵌的参考代码仍是修复前的版本，与当前 `ssha/` 实现不同步；
- 待证清单：表达力优于 1-WL（Proposition）、过平滑上界、泛化界（Conjecture）、手术收敛性（多路顺序一致性）。


# SSHA 技术文档：核心算法、Bug 修复与测试验证

> 结构超图重构神经模型（Structural Soft-Hypergraph with Spiking Association）
> 版本 0.2.0 · 2026-10-13

---

## 目录

1. [项目定位](#1-项目定位)
2. [核心算法原理](#2-核心算法原理)
3. [关键接口契约](#3-关键接口契约)
4. [Bug 修复详情](#4-bug-修复详情)
5. [测试验证结果](#5-测试验证结果)
6. [运行方式](#6-运行方式)
7. [已知边界与后续工作](#7-已知边界与后续工作)

---

## 1. 项目定位

把神经网络的「层 + 张量」范式换成「高阶关系结构」范式：用软超图 **H** ∈ R^{N×M} 表达节点间的高阶交互，用脉冲权重节点（Spiking Weight Node）在**关系层**（而非表征层）实现事件驱动稀疏。

三条主线：

| 主线 | 定位 | 内容 |
| :--- | :--- | :--- |
| T2（主） | 结构即算法 | SSHA + 四阶段闭环定义新前向 |
| T1（子） | 结构诊断 | 压缩已有模型，剪 20–30% 层 |
| T3（边界） | 结构反演 | 只输出后验分布，不承诺唯一解 |

---

## 2. 核心算法原理

### 2.1 脉冲权重节点（SpikingWeightNode）

关联张量 **H** 的**每个元素**是一个有膜电位、有不应期、会发放的脉冲节点。文件：[weight_node.py](ssha/weight_node.py)。

**可学参数**：

| 参数 | 形状 | 作用 |
| :--- | :--- | :--- |
| `H_logit` | [N, M] | 软关联 logits（经 softmax/temp → H_base） |
| `w_pre` | [N] | 突触前脉冲驱动权重（**非零初始化，见 4.5 Bug 7**） |
| `w_post` | [M] | 突触后超边特征驱动权重（**非零初始化**） |
| `b` | [N, M] | 逐元素偏置（零初始化） |
| `v_th_base` | [N, M] | 阈值基底（经 softplus 保证为正） |
| `lam_ref` | 标量 | 不应期抑制系数 |

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

`W_v` 此前只被构造、从未参与前向（梯度恒为 `None`），与「前 30% epoch 只学 (W_v, W_e)」
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

| 阶段 | 模块 | 功能 |
| :--- | :--- | :--- |
| 0 初始化 | [sinkhorn.py](ssha/sinkhorn.py) | 对 H_logit 做 Sinkhorn 双随机化（20 次行列归一），得到偏均匀的软超边 |
| 1 诊断 | [diagnosis.py](ssha/diagnosis.py) | 谱指标（λ₂、谱隙、ρ）、近似 Ollivier–Ricci 曲率、三阶模体丰度、脉冲统计（率/同步/熵/Fano） |
| 2 手术 | [surgery.py](ssha/surgery.py) | prune（沉默边）/ merge（余弦 > 0.98）/ split（高活跃 × 高梯度范数） |
| 3 精炼 | 训练循环 | 冻结结构，只学权重 |

**训练纪律**（前 30% / 中 40% / 后 30% epoch）：先只学 `W_v, W_e` 并冻结 `H_logit`；中期每 K epoch 触发一次 `structure_surgery`；后期冻结结构精修。`H` 只走手术回调，不走在线 STDP。`scripts/train.py` 已按此实现（4.5 Bug 16）。

### 2.5 结构手术判据

文件：[surgery.py](ssha/surgery.py) 的 `structure_surgery`：

- **prune**：`edge_activity = a.mean(dim=0) < prune_thr(0.02)` 判死；保留至少 `min_edges=2` 条（按活跃度 topk 兜底，且 `k = min(min_edges, M)` 防越界）
- **merge**：对 H_base 列向量归一化后两两求余弦，> 0.98 则取均值合并；合并次数上限 `M − min_edges`，保证合并后仍保留 `min_edges` 条
- **split**（可选）：`score = ‖grad(H_logit)‖_col · a_mean > split_grad_thr` 的列复制一份并加小扰动（σ=0.01）。
  进入手术前会快照结构梯度，并让它跟随 prune/merge 的列掩码一起对齐，因此 split 在剪枝后仍然可达（4.5 Bug 9）

手术后保证 `sw.M == block.M`、`H_logit.shape[1] == M`、内部状态张量形状一致（测试硬约束）。
`SSHAModel.M` 是只读属性，实时反映 `block.M`（4.5 Bug 12 附带）。

### 2.6 退化契约（审稿人硬约束）

1. |e| ≡ 2 且 `hard=True` → 每列恰 2 个非零，等价 spiking GCN 两阶段传播
2. **H** one-hot 且 `hard=True` → 每行 ≤ 1 个非零，等价 spiking attention pooling
3. V_th → 0（softplus 基底 −20）→ 软/硬门的 H_eff 均值误差 **< 0.02**（修复 4.5 Bug 5 前为 0.14 量级）

### 2.7 能耗模型

文件：[energy.py](ssha/energy.py)。按 Loihi 量级建模：

```text
pJ = (n_sv + n_sve) · pJ_ac + ops · pJ_mem，  ops = Σ(‖s_v‖₁ · d · 2)
默认 pJ_ac = 1.0，pJ_mem = 0.1
```

注意该公式不为 `spikes_e`（高阶同步事件）收费，与本节公式一致，属建模取舍（见 §7）。

---

## 3. 关键接口契约

| 模块 | 接口 | 返回 |
| :--- | :--- | :--- |
| `SpikingWeightNode` | `forward(s_v_prev, y_e_prev, hard=False)` | `H_eff: [N, M]`，非负 |
| `SSHABlock` | `forward(x, s_v_prev, y_e_prev, hard=False)` | `(x_new, s_v, y_e, H_eff, s_e)` |
| `SSHAModel` | `forward(x_seq[T,N,d], hard=False)` | `(logits[C], aux={spike_e, spike_ve, spike_v, H_eff, struct_mass})`；`M` 为只读属性，随手术自动更新 |
| `EnergyMeter` | `record(sv, se, sve, d)` / `report()` / `reset()` | `dict(spikes_v/e/ve, ops, pJ)` |
| `structure_surgery` | `(block, epoch, prune_thr, merge_cos, split_grad_thr, min_edges)` | `dict(pruned, merged, split)` |
| `motif_abundance` | `(H, threshold=None, top_k_per_row=2)` | `float`；两种二值化策略互斥，`threshold` 优先 |

---

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

#### Bug 2：脉冲节点状态懒初始化（weight_node.py）

**现象**：`test_surgery.py` 三个用例全部 `AttributeError: 'NoneType' object has no attribute 'zero_'`。

**根因**：`u/s/a/ref/rate_est` 注册为 `None`，仅在首次 `forward` 时经 `_ensure_state` 懒初始化；而手术测试在任何前向之前直接操作 `blk.sw.a.zero_()`。

**修复**（[weight_node.py](ssha/weight_node.py)）：`__init__` 末尾直接调用 `self.reset_state()`，`_ensure_state` 保留作为安全网。

#### Bug 3：H_base 缓存携带旧计算图（weight_node.py）

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

#### 修改 6：dtype 一致性测试修正（test_shapes.py）

原测试直接将 float64 输入喂给 float32 模型，PyTorch 不允许 `float @ double` 混合精度矩阵乘。按 PyTorch 标准惯例修正为用户侧先转换模型。

---

### 4.5 第二轮修复（独立审计 + 本地硬件验证）

在 4 处原有 bug 之外，对全部模块做了独立审计（静态审查 + 最小复现 + 第二路 agent 交叉复核），
又确认并修复 **13 处**缺陷。每条都有回归测试，并已用「把 5 个模块回滚到修复前再跑测试」的方式验证：
当时 `tests/test_regressions.py` 的 17 个用例中 **14 个失败**，只有 3 个通过
（`split_is_noop_without_grad`、`surgery_keeps_state_shapes_consistent`、
`loss_sparse_on_H_base_is_constant` —— 这三个是既有正确行为的守卫，本就不该失败）。
之后又补了 4 个用例（有效阈值语义、跨配置/种子无死启动、时间平均脉冲指标、术后指标一致性），
现共 21 个用例。

#### Bug 5（最关键）：软门用了复位后的膜电位（weight_node.py）

**现象**：`soft_gate` 取 `self.u`，而 `self.u` 已被 `self.u = self.u * (1 - s_new)`
把**恰好发放**的条目清零。于是软门与脉冲**反相关**。

**最小复现**：`SpikingWeightNode(N=1, M=2)`，`v_th=0.5`，`b=[[2.0, 0.3]]`。

| 条目 | u_pre | s | 旧代码门值 | 应有门值 |
| :--- | :--- | :--- | :--- | :--- |
| 0（发放） | 2.0 | 1 | **0.182** | 0.989 |
| 1（静息） | 0.3 | 0 | 0.354 | 0.354 |

**后果**：训练时压制真正发放的连接，与硬分支选到**相反**的超边；退化契约 #3 之所以仍通过，
只是因为阈值 0.5 比该缺陷引入的误差（0.1436）宽 3.5 倍。

**修复**：把复位前电位单独留存为 `u_pre`，`s` 与软门都用它。契约 #3 误差 0.1436 → **0.0157**。

#### Bug 6：`W_v` 从未参与前向（block.py）

forward hook 实测执行的 Linear 只有 `['W_e', 'gate.0', 'read_v']`；`W_v.weight.grad is None`；
扰动 `W_v` 对输出影响恰为 0.0。而训练纪律要求前 30% epoch 学 `(W_v, W_e)`。
**修复**：`x_v = W_v(x)` 接入 V→E 聚合，残差仍走原始 `x`。

#### Bug 7：`w_pre` 零初始化切断顶点脉冲→关系层的梯度（weight_node.py）

`I = w_pre ⊙ s_v_prev` 是**乘法**，`w_pre ≡ 0` 时 `∂I/∂s_v_prev = 0`，于是
`read_v` / `lam_ref` 梯度**恰为 0**（实测 `read_v.weight += 5` 对 logits 影响为 0.0）。
**修复**：`w_pre` / `w_post` 用 `drive_init=1.0` 缩放的高斯初始化（`b` 仍为零，以保住 |e|=2 退化契约）。

#### Bug 8：关系层死启动，`hard` 模式下 `H_eff ≡ 0`（weight_node.py / model.py）

驱动全零 ⇒ `I ≡ 0 < v_th` ⇒ 12 个 epoch 内 `sw.s` 发放率恒为 `0.00000`，
`hard=True` 时 `H_eff ≡ 0`，前向退化成 `x_new = x·(1−g·γ)`（实测每步 0.9419 倍纯衰减）。

**修复分两步**：
1. 同 Bug 7 的非零驱动初始化；
2. 把 `v_th` 改成**有效**阈值语义——原先 `v_th_base` 直接填 `0.5`，经 softplus 后实际阈值是
   `softplus(0.5) = 0.974`。顶点脉冲发放率约 10%，乘 `w_pre` 后小 N 配置普遍压不到 0.974。
   实测 `N=16` 时 8 个种子里 **4 个整条时间链零发放**；`drive_init` 调到 2.0 也只是把死启动
   从 13/40 降到 1/40，治标不治本。现改为 `v_th_base = log(expm1(v_th))`，默认有效阈值 `v_th=0.2`。

修复后：48 组 (config, seed) 组合（N=16…512，8 个种子）**零死启动**，
关系层发放率落在 [0.024, 0.128]，`s_e`（高阶同步事件）> 0，`lam_ref` 获得非零梯度。

#### Bug 9：`split` 手术永久失效（surgery.py）

`_apply_prune` 会替换成新的 `nn.Parameter`（`.grad is None`），而 split 的守卫正是
`sw.H_logit.grad is not None`。实测：无剪枝时 `split=4`；触发一次剪枝后 `split=0`。
即使守卫通过，`g_norm`（[M_old]）与 `a_mean`（[M_new]）形状也不匹配。
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
广播要求 `N == M`。注意 **`dv` 是 `[N, 1]`，`L @ X / dv` 本身正确，不应改动**；
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

---

## 5. 测试验证结果

### 5.1 全量测试：53/53 通过

```text
$ python -m pytest tests/ -v
======================= 53 passed, 1 warning in 0.92s ========================
```

| 测试文件 | 用例 | 结果 | 验证点 |
| :--- | :--- | :--- | :--- |
| test_degeneracy.py | 3 | ✅ | 三条退化契约（\|e\|≡2、one-hot、连续极限） |
| test_gradients.py | 4 | ✅ | `H_logit` 软分支梯度非零、硬分支梯度有限、替代梯度通畅、反向无 NaN/Inf |
| test_shapes.py | 5 | ✅ | 各层 I/O 形状、reset 幂等、float64 一致 |
| test_surgery.py | 3 | ✅ | prune 生效、M 一致、术后可前向、全活跃时 no-op |
| test_energy.py | 4 | ✅ | 计数精确、能耗随脉冲单调、确定性、reset |
| test_sanity.py | 2 | ✅ | 模体置换检验 p 值合法、有限性 |
| test_baselines.py | 11 | ✅ | HGNN 非方阵/参考实现/静默错误守卫；SpikingGNN `T` 真实生效、`hard` 真开关、梯度有限 |
| test_regressions.py | 21 | ✅ | 4.5 全部修复的回归（14 个用例在旧代码上失败） |

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

| 配置（对应 `configs/`） | N | M | dim | T | epochs | wall | 峰值显存 | test_acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 小规模冒烟 | 64 | 32 | 32 | 8 | 12 | 10.8s | 25 MiB | 1.000 |
| `cora_spiking.yaml` 缩减 | 1024 | 256 | 128 | 8 | 12 | 17.4s | 819 MiB | 1.000 |
| `cora_spiking.yaml` 全量 | 2708 | 256 | 128 | 8 | 8 | 4.9s | 1091 MiB | 1.000 |
| `dvs_gesture.yaml` | 1156 | 128 | 64 | 16 | 6 | 4.8s | 468 MiB | 1.000 |
| README §6 长序列预算档 | 512 | 512 | 64 | 32 | 4 | 5.2s | 1342 MiB | 1.000 |
| CPU 回退（`--device cpu`） | 256 | 64 | 64 | 8 | 6 | 2.4s | — | 1.000 |

12 GiB 显存对文档中的全部配置都绰绰有余（最高占用 1.3 GiB）。诊断模块也已向量化：
`motif_abundance` 在 M=256 时 13.33s → 0.0028s；`spectral_metrics`（稠密 N×N 特征分解）
在 N=2708 时约 0.44s。

---

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

---

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

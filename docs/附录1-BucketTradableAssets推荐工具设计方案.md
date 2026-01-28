# Bucket Tradable Assets 推荐工具设计（组合风格 Proxy + 趋势一致性筛选 + Beta 空间去同质化）

## 0. 目标与原则

**目标**：为某个 bucket（例如 GROWTH）自动检索并推荐一组可交易资产（基金）列表，使其满足：

1. **与 bucket 固定锚（anchor）proxy 的趋势一致**：保证 bucket-level 是否下注的判断与资产执行层一致；
2. **资产之间风格差异充分**：便于在 bucket 内进行因子/风格倾斜（tilt）与替换；
3. **可审计**：每只基金的入池理由、所属风格簇、替补资产清晰可解释。

**层级一致性约束**：bucket 是否下注只依赖一个固定锚 proxy $p_0$（可为组合指数）。资产筛选也必须锚定同一个 $p_0$，以避免 “bucket proxy 涨但持仓资产跌” 的结构性冲突。

---

## 1. 固定锚 Proxy：风格指数线性组合

以 GROWTH bucket 为例，选择风格指数（大/中/小盘成长）构造固定锚组合：

- $k \in \{L, M, S\}$ 分别表示大盘成长、中盘成长、小盘成长；
- 指数价格/点位序列为 $P^{(k)}_t$。

### 1.1 日收益定义（对数收益）

$$
r^{(k)}_t = \ln\frac{P^{(k)}_t}{P^{(k)}_{t-1}}.
$$

### 1.2 组合 Proxy 收益与“锚净值曲线”

定义组合（Growth）proxy 的日收益：

$$
r^{(G)}_t=\sum_{k\in\{L,M,S\}} w_k\, r^{(k)}_t,\qquad
w_k\ge 0,\ \sum_k w_k=1.
$$

为计算趋势信号，构造组合 proxy 的“净值/指数曲线”（从 1 起步）：

$$
I^{(G)}_t = I^{(G)}_{t-1}\cdot e^{r^{(G)}_t},\qquad I^{(G)}_{t_0}=1.
$$

其中 $p_0 \equiv G$ 即 bucket 的固定锚 proxy。

### 1.3 固定权重方案（可配置）

#### A) 等权（基线方案）

$$
w_L=w_M=w_S=\frac{1}{3}.
$$

#### B) 固定波动率平价（推荐默认）

先在长期历史区间 $\mathcal{H}$（例如 3–5 年日频）估计每条指数年化波动：

$$
\sigma_k=\sqrt{252}\cdot \mathrm{Std}_{t\in\mathcal{H}}(r^{(k)}_t).
$$

再取固定权重：

$$
w_k=\frac{1/\sigma_k}{\sum_j 1/\sigma_j}.
$$

> 注：波动率平价权重 **仅估计一次后固定**（不滚动更新），以保持锚点稳定与可审计性。  
> 可选加入上下限并重新归一化：$w_k \in [w_{\min}, w_{\max}]$。

---

## 2. Step 1：趋势一致性筛选与综合分 $S_{\mathrm{fit}}$

本层回答：基金 $i$ 的趋势是否与固定锚 proxy $p_0$（即 $G$）一致？

已确定使用：
- **趋势分数相关性**；
- **方向命中率**；
- 保留 **$\hat\beta>0$ 硬底线**；
- **不使用软惩罚项**（例如 TE、$|\beta-1|$ 等）。

### 2.1 数据与收益对齐

基金 $i$ 的净值序列为 $NAV^{(i)}_t$，定义日对数收益：

$$
r^{(i)}_t=\ln\frac{NAV^{(i)}_t}{NAV^{(i)}_{t-1}}.
$$

对齐基金与组合 proxy 的共同交易日集合 $\mathcal{T}$。给定评估窗口 $W\subseteq\mathcal{T}$，在 $W$ 上计算指标。

### 2.2 趋势分数定义 $T_x(t)$

对任意对象 $x$（基金 $i$ 或 proxy $G$），定义趋势分数序列：

$$
T_x(t)=\frac{MA_s(I_x,t)-MA_l(I_x,t)}{\sigma_x(t)+\varepsilon}.
$$

- $I_x(t)$ 为“价格/净值曲线”（基金用 $NAV^{(i)}_t$，proxy 用 $I^{(G)}_t$）；
- $MA_s, MA_l$ 为短/长均线（例如 20/60 或 20/120 等，与系统保持一致）；
- $\sigma_x(t)$ 为波动率估计（例如过去 $K$ 日收益标准差，可按系统统一做年化/日频计算）；
- $\varepsilon$ 为稳定项，避免除 0。

### 2.3 趋势一致性指标（窗口 $W$）

#### (1) 趋势分数相关性

$$
\rho_T(i;W)=\mathrm{Corr}_{t\in W}\bigl(T_i(t),\,T_G(t)\bigr).
$$

#### (2) 方向命中率

$$
HR(i;W)=\frac{1}{|W|}\sum_{t\in W}\mathbf{1}\Bigl(\mathrm{sign}(T_i(t))=\mathrm{sign}(T_G(t))\Bigr).
$$

### 2.4 硬底线：$\hat\beta(i;W)>\beta_{\min}$

在窗口 $W$ 上做收益回归（只用于判定同向暴露）：

$$
r^{(i)}_t=\alpha+\beta\,r^{(G)}_t+\varepsilon_t,\qquad t\in W,
$$

得到 $\hat\beta(i;W)$，要求：

$$
\hat\beta(i;W)>\beta_{\min}.
$$

其中 $\beta_{\min}$ 为最小暴露阈值，默认取 0.3（基于当前样本分布与经验校准，用于抑制跨市场或弱相关基金）。


### 2.5 窗口内综合分

将 $HR$ 映射到 $[-1,1]$ 尺度：$2HR-1$。定义窗口内综合分：

$$
S(i;W)=
\mathbf{1}\{\hat\beta(i;W)>0\}\cdot
\Bigl(
w_\rho\cdot \rho_T(i;W)
+
w_h\cdot \bigl(2\,HR(i;W)-1\bigr)
\Bigr),
\quad
w_\rho,w_h\ge 0,\ w_\rho+w_h=1.
$$

### 2.6 多窗口聚合（稳健性）

设多窗口集合 $\{W_m\}_{m=1}^M$（例如 3M/12M/36M），定义聚合得分：

$$
S_{\mathrm{fit}}(i)=\sum_{m=1}^M \lambda_m\, S(i;W_m),
\qquad
\lambda_m\ge 0,\ \sum_m\lambda_m=1.
$$

候选集筛选（示例规则，具体门槛可配置）：

- 最少有效样本数：$|W_m|\ge N_{\min}$；
- 综合分门槛：$S_{\mathrm{fit}}(i)\ge S_{\min}^{\mathrm{agg}}$；
- 或者要求在至少 $K$ 个窗口中 $S(i;W_m)\ge S_{\min}$。

输出：候选基金集合 $F$ 及每只基金的 $S_{\mathrm{fit}}(i)$ 与窗口明细（$\rho_T, HR, \hat\beta, S(i;W_m)$）。

---

## 3. Step 2：Beta 空间去同质化

本层在候选集 $F$ 上进行，回答：如何从趋势一致的候选中挑出风格差异足够的一组资产，并给出替补池？

### 3.1 因子暴露向量与不稳定估计过滤（基于 Kalman 协方差快照）

对每只候选基金 $i\in F$，数据库中存储的是 Kalman Filter 输出的因子暴露时间序列与协方差矩阵：
`code, date, MKT, SMB, HML, QMJ, const, P_bin, ...`。

本步骤用于构造“每只基金一个用于聚类的暴露向量”，并过滤掉“估计不稳定”的基金，得到稳定集合 $F'\subseteq F$。

#### (a) 聚类用 beta 的时点定义（与 Step 1 同锚）

设本次筛选/推荐的锚定日期为 $t_0$（通常取 Step 1 窗口的 end date ）。
对每只基金 $i$，取不晚于 $t_0$ 的最新一条记录：

$$
t_i^*=\max\{t\le t_0:\ \beta_i(t)\ \text{存在}\}.
$$

定义聚类用的暴露向量（需要配置fund_beta表中用于组成暴露向量的字段名，如：SMB,QMJ，以下公式以使用该配置为例）：

$$
\beta_i^{(2)}=
\bigl(\beta_{i,\mathrm{SMB}}(t_i^*),\ \beta_{i,\mathrm{QMJ}}(t_i^*)\bigr)
\in\mathbb{R}^2.
$$

> 注：虽然表中同时存储有其它因子暴露字段，但本工具的“去同质化”目标服务于 bucket 内 tilt，仅在**指定字段**对应的向量空间定义差异。

#### (b) 协方差矩阵与方差抽取

数据库中协方差矩阵 $\Sigma_i(t_i^*)$ 由 `P_bin` 提供，维度与因子顺序约定为：

$$
\Sigma_i(t_i^*)\in\mathbb{R}^{6\times 6},
\qquad
\text{order}=[\mathrm{MKT},\mathrm{SMB},\mathrm{HML},\mathrm{QMJ},\mathrm{const},\mathrm{gamma}].
$$

`P_bin` 仅存下三角，float32格式

取 SMB 与 QMJ 的对角元素作为方差：

$$
v_{i,\mathrm{SMB}}=\Sigma_{i,\mathrm{SMB,SMB}}(t_i^*),
\qquad
v_{i,\mathrm{QMJ}}=\Sigma_{i,\mathrm{QMJ,QMJ}}(t_i^*).
$$

#### (c) 不稳定度分数与过滤规则

定义不稳定度分数（越大越不稳定）：

$$
U_i=\frac{1}{2}\left(\sqrt{v_{i,\mathrm{SMB}}}+\sqrt{v_{i,\mathrm{QMJ}}}\right).
$$

过滤规则（二选一，配置项）：

- **绝对阈值**：$U_i\le U_{\max}$；
- **分位数阈值**：$U_i\le Q_q(U)$（例如 $q=0.90$ 表示保留最稳定的 90%）。

过滤后得到稳定候选集合：

$$
F'=\{i\in F:\ U_i\le \tau_U\}.
$$

> 工程约束：若 $\Sigma_i$ 解码失败、维度不匹配、或对角线存在 NaN，则该基金在 $t_0$ 下视为不可用，直接排除或进入“观察池”（由策略配置决定）。

---

### 3.2 Beta 鲁棒标准化（SMB/QMJ 二维点集）

在通过 Step 1 的趋势一致性筛选与 Kalman 协方差不稳定度过滤后，得到候选集合 $F'$。对每只基金 $i\in F'$，在锚定日期 $t_0$ 取二维因子暴露向量：

$$
\beta_i^{(2)}=
\bigl(\beta_{i,\mathrm{SMB}}(t_i^*),\ \beta_{i,\mathrm{QMJ}}(t_i^*)\bigr)\in\mathbb{R}^2,
\qquad
t_i^*=\max\{t\le t_0:\ \beta_i(t)\ \text{存在}\}.
$$

为消除 SMB/QMJ 两个维度在横截面上的尺度差异，仅做鲁棒标准化（不做单位化）。对每个维度 $j\in\{\mathrm{SMB},\mathrm{QMJ}\}$，计算 MAD 尺度：

$$
a_j=\mathrm{MAD}(\{\beta_{i,j}^{(2)}\}_{i\in F'})+\varepsilon,
\qquad
\mathrm{MAD}(x)=\mathrm{median}\bigl(|x-\mathrm{median}(x)|\bigr).
$$

构造对角缩放矩阵 $D=\mathrm{diag}(a_{\mathrm{SMB}},a_{\mathrm{QMJ}})$，并将每只基金映射为二维点：

$$
p_i=D^{-1}\beta_i^{(2)}\in\mathbb{R}^2.
$$

> 注：不进行单位化是为了保留向量模长信息（暴露强度），并避免在某个维度接近 0 时因归一化导致方向不稳定跳变。

---

### 3.3 Step 2 目标：最大化所选点集的凸包面积（风格覆盖）

在二维点集 $\{p_i\}_{i\in F'}$ 上选择 $n$ 只基金，记所选集合为 $\mathcal{S}\subseteq F'$，$|\mathcal{S}|=n$。为保证 bucket 内风格覆盖，将 Step 2 的去同质化目标定义为最大化所选点集围成的凸包面积：

$$
\max_{\mathcal{S}\subseteq F',\ |\mathcal{S}|=n}\ A\bigl(\mathrm{Hull}(\{p_i\}_{i\in\mathcal{S}})\bigr),
$$

其中 $\mathrm{Hull}(\cdot)$ 表示凸包算子，$A(\cdot)$ 表示二维多边形面积。

直觉含义：
- 若所选点过于集中（同质化），凸包面积接近 0；
- 若所选点分布在不同方向/象限（风格覆盖更广），凸包面积更大；
- 落在点云内部或近中心的点通常对凸包面积贡献较小，因此不会被优先选中。

---

### 3.4 贪心近似求解（面积增量最大）

由于精确求解上述组合优化问题代价较高，采用可解释、可审计的贪心近似算法。

定义对任意当前集合 $\mathcal{S}$ 的凸包面积为：

$$
A(\mathcal{S}) = A\bigl(\mathrm{Hull}(\{p_i\}_{i\in\mathcal{S}})\bigr).
$$

定义将候选 $x\in F'\setminus \mathcal{S}$ 加入后的面积增量：

$$
\Delta A(x\mid \mathcal{S}) = A(\mathcal{S}\cup\{x\}) - A(\mathcal{S}).
$$

贪心迭代规则为：

$$
x^*=\arg\max_{x\in F'\setminus \mathcal{S}} \Delta A(x\mid \mathcal{S}),
\qquad
\mathcal{S}\leftarrow \mathcal{S}\cup\{x^*\},
$$

重复上述步骤直至 $|\mathcal{S}|=n$。

具体贪心算法实现参考代码 `greedy_convex_hull_volume.py`

---

### 3.5 输出：去同质化的候选资产集合

贪心算法结束后得到最终集合：

$$
\mathcal{S}=\{i_1,\dots,i_n\}\subseteq F'.
$$

该集合在 $(\mathrm{SMB},\mathrm{QMJ})$ 二维风格空间中具有较大的覆盖范围，可用于后续 bucket 内的风格倾斜（tilt）与权重调整。输出同时保留每个基金的：
- 锚定日因子暴露 $\beta_i^{(2)}$ 与标准化点 $p_i$；
- Step 1 趋势一致性综合分 $S_{\mathrm{fit}}(i)$；
- Kalman 不稳定度分数 $U_i$（用于审计与回溯）。

---

## 4. 输出与审计字段（建议最小集合）

### 4.1 Bucket Proxy 说明
- 组成指数列表（例如 399372/399374/399376）
- 固定权重 $w_k$（等权或固定波动率平价）
- 若使用波动率平价：历史估计区间 $\mathcal{H}$、估计得到的 $\sigma_k$
- 组合 proxy 曲线 $I^{(G)}_t$ 与收益 $r^{(G)}_t$

### 4.2 Step 1：趋势一致性评分明细（基金级）
对每只基金 $i$：
- 各窗口 $W_m$：$\rho_T(i;W_m)$、$HR(i;W_m)$、$\hat\beta(i;W_m)$、$S(i;W_m)$、样本数 $|W_m|$
- 聚合得分：$S_{\mathrm{fit}}(i)$
- 是否入候选：pass/fail 及原因（样本不足、$\hat\beta\le 0$、得分不达标等）

### 4.3 Step 2：聚类与推荐结果（簇级）
对每个簇 $C_c$：
- 簇大小 $|C_c|$，是否满足 $m_{\min}$
- 簇中心 $\mu_c$（或其摘要）
- 代表基金 $i_c^\star$：$S_{\mathrm{fit}}(i_c^\star)$、$d_c(i_c^\star)$、$J(i_c^\star)$
- Top-K 备选池 $\mathcal{B}_c$（含每只基金的 $S_{\mathrm{fit}}$ 与 $d_c$）

---

## 5. 端到端流程摘要（可直接转为模块实现）

1. **构造固定锚 proxy**：用风格指数集合按固定权重得到 $r^{(G)}_t$ 与 $I^{(G)}_t$。  
2. **趋势一致性计算**：对每只基金 $i$ 计算 $T_i(t)$ 与 $T_G(t)$，在多个窗口上计算 $\rho_T$ 与 $HR$，并回归得到 $\hat\beta(i;W_m)$。  
3. **Step 1 打分与筛选**：按
   $$
   S(i;W_m)=\mathbf{1}\{\hat\beta>0\}\cdot\left(w_\rho\rho_T+w_h(2HR-1)\right)
   $$
   聚合为 $S_{\mathrm{fit}}(i)$，筛出候选集合 $F$。  
4. **Beta 稳定性过滤**：用不确定度指标过滤得到 $F'$。  
5. **Beta 鲁棒标准化 + Ward 聚类**：对 $\beta$ 做 MAD 缩放与单位化，Ward 聚类并施加最小簇大小约束确定 $n_{\mathrm{eff}}$。  
6. **代表与备选池**：每簇选代表
   $$
   i_c^\star=\arg\max_{i\in C_c}\left(S_{\mathrm{fit}}(i)-\eta\|\hat\beta_i-\mu_c\|_2\right),
   $$
   并输出每簇 Top-K 备选池 $\mathcal{B}_c$。  

该流程最终输出：主推荐资产集合 $\mathcal{S}$、簇级备选池 $\{\mathcal{B}_c\}$、以及完整审计明细。

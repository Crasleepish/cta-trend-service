# CTA Trend-Following Allocation System for Public Mutual Funds

**Whitepaper — Design, Rationale, and Production Implementation**

> **Core objective:** Under strict **portfolio volatility control**, build a long-horizon, low-turnover CTA-style allocation system for public mutual funds that targets **stable, smooth, and reproducible risk-adjusted returns**, rather than short-term return maximization.

---

## Table of Contents

1. Primary Goal and Practical Constraints
2. Architecture and Layering Rules
3. Asset Bucketing and Universe Philosophy
4. CTA Core — Trend Detection and Long-Only Risk Permission
5. Path Structure — Trend Quality Score and the Mapping $g(\cdot)$
6. Volatility Structure — Realized Volatility and Tradability Filtering $f(\cdot)$
7. Composite CTA Raw Weight Formula
8. Portfolio-Level Volatility Targeting and Defensive Allocation
9. Intra-Bucket Factor-Driven Adjustment (Soft Tilt)
10. Execution & Control (NAV, Holding Period, Low Turnover)
11. Parameterization and Engineering Notes (Notation & Timing Contract)

---

## 1. Primary Goal and Practical Constraints

### 1.1 Investment Objective

The system is designed to:

1. **Volatility-first**
   - Portfolio risk must stay within a predefined volatility envelope.
   - Returns are optimized **within** the volatility budget.

2. **Smooth equity curve**
   - Prefer persistent trend capture over regime-specific or tail-driven performance.

3. **Long-horizon robustness**
   - No dependence on explicit regime prediction.
   - Exposure adapts naturally as signals evolve.

4. **Production-grade deployability**
   - Signals must translate into executable portfolios.
   - Complexity must submit to interpretability and maintainability.

### 1.2 Practical Constraints (Mutual Fund Reality)

#### Execution constraints

- Only **passive public mutual funds** (index funds / ETF-like mutual funds) are tradable.
- Trades are executed at **daily NAV** (no intraday pricing, no high-frequency tactics).
- Execution is guaranteed, but **price selection is impossible**.

#### Holding and cost constraints

- Minimum holding period: **7 calendar days** after each buy.
- Early exit incurs a punitive fee (e.g., $\approx 1.5\%$).
- Therefore, the system must be **structurally low-turnover**.

#### Data constraints

- Individual funds often have limited history; complex per-asset ML is fragile.
- Modeling is allowed (and preferred) at the **factor layer** (e.g., MKT/SMB/HML/QMJ), which is more stable and longer-lived.

#### Structural constraints

- Many funds are highly correlated or track similar indices.
- The system must avoid duplicating bets on the same underlying risk factor.
- Defensive assets (RATE/CASH/GOLD) may serve as risk absorbers; capital is not required to collapse entirely into CASH.
- **Policy decision:** defensive budget is **RATE/CASH only**; **GOLD is treated as a risk bucket** (i.e., part of $\mathcal{B}_{\mathrm{risk}}$).

---

## 2. Architecture and Layering Rules

The system is organized into **strictly layered modules**, each with a clear responsibility:

1. **Data Layer**
   - Daily bucket proxy prices $P_{b,\tau}$ (computed daily; sampled weekly as $P_{b,t}$)
   - Fund NAVs
   - Daily factor returns (e.g., MKT, SMB, HML, QMJ)
   - Kalman-filtered factor betas for execution assets

2. **Signal Layer (CTA Core)**
   - Computes bucket-level risk permission and raw CTA weights

3. **Allocation Layer**
   - Converts raw CTA intent into **portfolio-level volatility-controlled** bucket weights

4. **Intra-Bucket Adjustment Layer (Factor Tilt)**
   - Softly redistributes capital among funds *within a bucket* using factor tendencies

5. **Execution & Control Layer**
   - Enforces minimum holding, low turnover, caps, and NAV-trading constraints

### 2.1 Dependency Rules

- Higher layers **may depend** on lower layers.
- Lower layers **must not depend** on higher layers.
- Cross-layer shortcuts are forbidden.

This separation keeps the CTA core interpretable and replaceable; enhancements (e.g., ML) remain optional.

---

## 3. Asset Bucketing and Universe Philosophy

The system operates on **macro-consistent buckets** rather than treating each fund as an independent asset.

### Bucket types

- **Risk buckets** (examples):
  - EQUITY: Growth / Value / Cyclical / Size
  - GOLD: inflation hedge / diversifier (**treated as a risk bucket**)

- **Defensive buckets**:
  - RATE: interest-rate sensitive defensive assets
  - CASH: residual capital and volatility absorber

Each bucket contains one or more **manually selected execution funds**. This selection is externalized from the CTA pipeline to avoid over-fitting and to respect operational constraints.

CTA logic is applied **at the bucket level**, and only later translated into fund-level weights.

---

## 4. CTA Core — Trend Detection and Long-Only Risk Permission

### 4.1 Trend Strength (Risk-Adjusted; Daily Compute, Weekly Sample)

Let $P_{b,\tau}$ be the daily close series of bucket $b$. On the weekly decision date (default: the last trading day of week $t$), sample $P_{b,t}:=P_{b,\tau(t)}$. Define the signed, risk-adjusted trend strength:

$$
T_{b,t} = \frac{MA_{\mathrm{short}}(P_{b,t}) - MA_{\mathrm{long}}(P_{b,t})}{\sigma_{b,t}},
$$

where $\sigma_{b,t}$ is **annualized** realized volatility computed from daily returns and sampled on the decision date (Section 6). The sign encodes direction:

**Notation (timing):** $MA_{\mathrm{short}}(\cdot)$ and $MA_{\mathrm{long}}(\cdot)$ are computed on the daily series and evaluated at $\tau(t)$; we write $MA(P_{b,t})$ for brevity.

- $T_{b,t} > 0$: upward drift
- $T_{b,t} < 0$: downward drift

### 4.2 Hysteresis Long-Only Gate $d^+_{b,t}$

Because NAV execution and minimum-holding constraints make signal chattering costly, the risk-on gate uses **hysteresis**.

Let $d^+_{b,t} \in \{0,1\}$ be a stateful long-only permission variable, and choose two thresholds:

$$
\theta_{\mathrm{on}} > \theta_{\mathrm{off}} > 0.
$$

Update rule:

$$
d^+_{b,t} =
\begin{cases}
1, & T_{b,t} > \theta_{\mathrm{on}},\\
0, & T_{b,t} < \theta_{\mathrm{off}},\\
d^+_{b,t-1}, & \theta_{\mathrm{off}} \le T_{b,t} \le \theta_{\mathrm{on}}.
\end{cases}
$$

Interpretation:

- $d^+_{b,t} = 1$: bucket is eligible to take **long-only directional risk**.
- $d^+_{b,t} = 0$: bucket is sidelined (signal-layer target is zero).

This gate answers:

> Does this bucket exhibit sufficiently strong upward drift to justify long-only risk, robust to noise?

### 4.3 Downward-Drift Indicator $d^-_{b,t}$ (Execution-only)

We define a downward-drift indicator:

$$
d^-_{b,t} = \mathbb{1}\left(T_{b,t} < -\theta_-\right), \qquad \theta_- > 0.
$$

**Important:**

- $d^-_{b,t}$ does **not** create negative exposure (no shorting).
- $d^-_{b,t}$ does **not** enter the signal-layer sizing formula.
- $d^-_{b,t}$ is used **only** in the execution layer for **asymmetric smoothing** (faster de-risking, slower risk-on).

Finally, the CTA gate used in sizing is:

$$
d_{b,t} = d^+_{b,t}.
$$

---

## 5. Path Structure — Trend Quality Score and the Mapping $g(\cdot)$

The CTA core is rule-based via $d_{b,t}$ and volatility scaling. Separately, we define a **continuous path-quality score** to rank or size exposures; ML is allowed only here as an optional enhancement.

### 5.1 Ex-post Path-Quality Label (Training Only, Log-Return)

Using forward horizon $N$ (weeks), define nonnegative forward runup and drawdown in log space:

$$
U^{(N)}_{b,t}=\max_{1 \le k \le N} \ln\left(\frac{P_{b,t+k}}{P_{b,t}}\right), \qquad
D^{(N)}_{b,t}=\max_{1 \le k \le N} \ln\left(\frac{P_{b,t}}{P_{b,t+k}}\right).
$$

Define the ex-post path-quality label:

$$
\mathrm{trend\_score}_{b,t} = \frac{U^{(N)}_{b,t}}{U^{(N)}_{b,t} + D^{(N)}_{b,t}} \in (0,1).
$$

Interpretation:

- near 1: forward path dominated by run-up (clean trend)
- near 0: forward path dominated by drawdown
- near 0.5: weak/noisy

### 5.2 Live Proxy $z_{b,t}$ (Optional ML)

In live trading we cannot use forward data. We may train a model to predict a proxy score using only information at time $t$:

$$
\hat{z}_{b,t} = \widehat{\mathrm{trend\_score}}_{b,t} \in [0,1].
$$

ML is **not** the hard gate; it is only a bounded sizing modifier.

### 5.3 Designing $g(\cdot)$: Production-Default Mapping (No Time Smoothing Here)

We require $g:[0,1] \to [0,1]$ to be monotone, bounded, and stable, with low sensitivity near the noisy middle.

A production-default choice is a **dead-zone power mapping**:

$$
g(x) =
\begin{cases}
0, & x \le x_0,\\
\left(\dfrac{x-x_0}{1-x_0}\right)^{\gamma}, & x > x_0,
\end{cases}
\qquad
x \in [0,1],\; x_0 \in (0.5,1),\; \gamma \ge 1.
$$

Design intent:

- $x_0$ enforces a minimum path-quality standard (avoid allocating to marginal “almost-trends”).
- $\gamma$ controls how aggressively exceptionally clean trends are rewarded.

**Central rule:** $g(\cdot)$ is intentionally **not time-smoothed**. All temporal smoothing and turnover control are handled centrally in the **Execution & Control Layer**.

### 5.4 Extensible Trend Quality Feature Suite (Design for Iteration)

This section defines an extensible, production-friendly framework to incorporate additional “trend quality” tools beyond moving averages and momentum (e.g., Donchian, ATR, Bollinger squeeze, candle direction ratio, HH–HL, price–volume confirmation), while preserving the system’s core principles: **low turnover, weekly cadence, bounded outputs, and central execution-layer smoothing**.

#### 5.4.1 Unified Interface Contract

For each bucket $b$ at week $t$, the trend-quality module exposes:

1) **Raw feature vector (unbounded):**

$$
\mathbf{x}_{b,t}=\bigl(x^{(1)}_{b,t},x^{(2)}_{b,t},\dots,x^{(m)}_{b,t}\bigr)\in\mathbb{R}^m,
$$

computed using only information available at time $t$.

2) **Quality score (bounded):**

$$
z_{b,t}=\mathcal{A}\!\left(\mathbf{x}_{b,t}\right)\in[0,1],
$$

which is consumed by the sizing mapping $g(z_{b,t})$ (Section 5.3 / Section 7).

**Interface rules (hard constraints):**

- All features are computed on the **daily** series using $P_{b,\tau}$ (daily close) and optional proxy data (e.g., volume) if defined and reliable, then sampled/recorded at the weekly decision index $t$ (default: last trading day of the week).
- Outputs must be **bounded** (final $z_{b,t}\in[0,1]$) and **not time-smoothed by design** (no extra time smoothing here). Temporal smoothing and turnover control remain centralized in the **Execution & Control** layer.
- Each feature must be **auditable**: log the raw value $x^{(j)}_{b,t}$, standardized score $s^{(j)}_{b,t}$, and its fusion weight $w_j$.

#### 5.4.2 Robust Standardization (Common Scale Across Tools)

Because tools have different units and scales, each raw feature $x^{(j)}_{b,t}$ is standardized bucket-wise using robust statistics on a reference set $\mathcal{T}$ (e.g., full history or rolling 3–5 years, updated infrequently).

Define:

$$
\tilde{x}^{(j)}_{b,t}
=
\operatorname{clip}\!\left(
\frac{x^{(j)}_{b,t}-\operatorname{Median}\!\left(x^{(j)}_{b,\cdot}\right)}
{\operatorname{IQR}\!\left(x^{(j)}_{b,\cdot}\right)+\epsilon},
\ -c,\ c
\right),
$$

where $\epsilon>0$ prevents division-by-zero and $c>0$ caps outliers.

Map standardized values to bounded scores:

$$
s^{(j)}_{b,t}=\sigma\!\left(\beta_j\,\tilde{x}^{(j)}_{b,t}\right)\in(0,1),
\qquad
\sigma(u)=\frac{1}{1+e^{-u}}.
$$

This yields comparable scores $s^{(j)}_{b,t}$ across heterogeneous tools.

#### 5.4.3 Fusion (Aggregating Multiple Tools Into One Score)

Let $\mathbf{s}_{b,t}=(s^{(1)}_{b,t},\dots,s^{(m)}_{b,t})$. We recommend a **weighted geometric mean** to reflect “joint satisfaction” while avoiding hard gates:

$$
z_{b,t}
=
\exp\!\left(
\sum_{j=1}^{m} w_j \ln\left(s^{(j)}_{b,t}\right)
\right),
\qquad
w_j\ge 0,\ \sum_{j=1}^{m} w_j=1.
$$

Properties:

- If any component score approaches 0, $z_{b,t}$ is suppressed (trend quality requires multiple supportive structures).
- Continuous and bounded; avoids binary switching.
- Weights $w_j$ encode **design priors** (initially rule-based; later can be learned, but must remain stable and auditable).

**Production safeguard:** optionally cap influence before passing to $g(\cdot)$:

$$
z_{b,t}\leftarrow \operatorname{clip}(z_{b,t}, z_{\min}, 1).
$$

#### 5.4.4 V1 Scope and Objective (Initial “From Simple” Target)

V1 keeps the trend-quality module deliberately minimal:

- **Gate:** long-only hysteresis $d^+_{b,t}$ remains the only permission mechanism (Section 4).
- **Quality:** use only **moving-average / momentum-derived information** already present in $T_{b,t}$; trend strength and stability are primarily reflected by $T_{b,t}$ and the existing mapping $g(\cdot)$.
- No additional tool-based features are required for deployment; this subsection mainly establishes the **extensible contract** for later iterations.

V1 success is not “maximum alpha,” but:

- stable weekly signals,
- low turnover under NAV execution,
- robust behavior across regimes,
- deterministic logging and auditability for later feature expansion.

#### 5.4.5 Tool Library (Mathematical Definitions as Pluggable Features)

Below are candidate feature definitions $x^{(j)}_{b,\tau}$ computed on the **daily** series for future activation, then **sampled/recorded weekly** at the decision index $t$ (default: last trading day of week $t$). All are optional and must follow the standardization and fusion rules above.

Sampling convention: for any tool $j$, define the weekly recorded value
$$
x^{(j)}_{b,t} := x^{(j)}_{b,\tau(t)}.
$$

##### (A) Donchian Channel Breakout Strength

Let $L_D$ be the lookback window:
$$
H_{b,\tau}^{(L_D)}=\max_{1\le k\le L_D} P_{b,\tau-k},
\qquad
L_{b,\tau}^{(L_D)}=\min_{1\le k\le L_D} P_{b,\tau-k}.
$$

Upward breakout strength (risk-adjusted):
$$
x^{\operatorname{don}}_{b,\tau}
=
\frac{\ln\!\left(\frac{P_{b,\tau}}{H_{b,\tau}^{(L_D)}}\right)}{\sigma_{b,\tau}}.
$$

##### (B) ATR Regime Expansion (Volatility Rising)

Daily true range (if daily high/low is available; otherwise approximate with returns):
$$
\operatorname{TR}_{b,\tau}=\max\!\left\{
P_{b,\tau}^{\operatorname{high}}-P_{b,\tau}^{\operatorname{low}},
\ \left|P_{b,\tau}^{\operatorname{high}}-P_{b,\tau-1}\right|,
\ \left|P_{b,\tau}^{\operatorname{low}}-P_{b,\tau-1}\right|
\right\}.
$$

ATR (simple moving average):
$$
\operatorname{ATR}^{(L_A)}_{b,\tau}=\frac{1}{L_A}\sum_{k=1}^{L_A}\operatorname{TR}_{b,\tau-k}.
$$

Expansion slope (log change):
$$
x^{\operatorname{atr}}_{b,\tau}
=
\Delta \ln\!\left(\operatorname{ATR}^{(L_A)}_{b,\tau}\right)
=
\ln\!\left(\frac{\operatorname{ATR}^{(L_A)}_{b,\tau}}{\operatorname{ATR}^{(L_A)}_{b,\tau-1}}\right).
$$

##### (C) Bollinger Band Squeeze $\rightarrow$ Expansion (Shape Feature)

Let $L_B$ be the Bollinger window and $k_B>0$ the width multiplier. Define:
$$
\mu_{b,\tau}=\frac{1}{L_B}\sum_{k=1}^{L_B} P_{b,\tau-k},
$$
$$
s^{(P)}_{b,\tau}=\sqrt{\frac{1}{L_B-1}\sum_{k=1}^{L_B}\left(P_{b,\tau-k}-\mu_{b,\tau}\right)^2}.
$$

Upper/lower bands:
$$
U_{b,\tau}=\mu_{b,\tau}+k_B s^{(P)}_{b,\tau},
\qquad
D_{b,\tau}=\mu_{b,\tau}-k_B s^{(P)}_{b,\tau}.
$$

Bandwidth:
$$
\operatorname{BW}_{b,\tau}=\frac{U_{b,\tau}-D_{b,\tau}}{\mu_{b,\tau}}.
$$

Shape indicator (compression then expansion):
$$
x^{\operatorname{bb}}_{b,\tau}
=
\sigma\!\left(a\left(c_b-\operatorname{BW}_{b,\tau-1}\right)\right)
\cdot
\sigma\!\left(b\left(\Delta \operatorname{BW}_{b,\tau}-x_b\right)\right),
$$
where $\Delta \operatorname{BW}_{b,\tau}=\operatorname{BW}_{b,\tau}-\operatorname{BW}_{b,\tau-1}$ and $a,b,c_b,x_b$ are bucket-specific calibration constants.

##### (D) Candle Direction Consistency Ratio

Let $K_C$ be a window length:
$$
x^{\operatorname{dir}}_{b,\tau}
=
\frac{1}{K_C}\sum_{k=1}^{K_C}\mathbb{1}\!\left(r_{b,\tau-k}>0\right)\in[0,1],
$$
where
$$
r_{b,\tau}=\ln\left(\frac{P_{b,\tau}}{P_{b,\tau-1}}\right)
$$
is the daily log return.

##### (E) HH–HL Structure (Higher High + Higher Low)

Using two consecutive blocks of length $L_H$:
$$
H^{(1)}_{b,\tau}=\max_{1\le k\le L_H} P_{b,\tau-k},
\qquad
H^{(2)}_{b,\tau}=\max_{L_H< k\le 2L_H} P_{b,\tau-k},
$$
$$
L^{(1)}_{b,\tau}=\min_{1\le k\le L_H} P_{b,\tau-k},
\qquad
L^{(2)}_{b,\tau}=\min_{L_H< k\le 2L_H} P_{b,\tau-k}.
$$

Binary structure score:
$$
x^{\operatorname{hhhl}}_{b,\tau}
=
\mathbb{1}\!\left(H^{(1)}_{b,\tau}>H^{(2)}_{b,\tau}\right)
\cdot
\mathbb{1}\!\left(L^{(1)}_{b,\tau}>L^{(2)}_{b,\tau}\right)\in\{0,1\}.
$$

(Optionally replace indicators with normalized differences for a continuous feature.)

##### (F) Price–Volume Confirmation (Conditional, Proxy-Dependent)

**Note:** Mutual fund NAV has no meaningful traded volume; this feature is defined only if a reliable **proxy volume** series $V_{b,\tau}$ exists for the bucket (e.g., representative ETF / index proxy).

Relative volume:
$$
x^{\operatorname{vol}}_{b,\tau}=\ln\!\left(\frac{V_{b,\tau}}{\operatorname{MA}_{L_V}(V_{b,\tau})}\right).
$$

A conditional confirmation can be defined only under a breakout regime (e.g., Donchian breakout). Exact logic is **version-dependent** and should be introduced only after proxy data-quality validation.

#### 5.4.6 Iteration Roadmap (Feature Activation Plan)

- **V1 (current):** only MA/momentum-derived information; implement this section’s interface, standardization, and fusion scaffolding with minimal active features.
- **V2–V5:** *to be determined* (feature activation order, calibration policy, and ablation criteria will be specified after V1 deployment stability is validated).

---

## 6. Volatility Structure — Realized Volatility and Tradability Filtering $f(\cdot)$

### 6.1 Daily Returns and Rolling Standard Deviation (Annualized), Sampled Weekly

Define daily log return:

$$
r_{b,\tau}=\ln\left(\frac{P_{b,\tau}}{P_{b,\tau-1}}\right).
$$

Let the rolling mean over the past $K$ trading days be:

$$
\bar r_{b,\tau}=\frac{1}{K}\sum_{k=1}^{K} r_{b,\tau-k}.
$$

We define realized volatility from **daily returns** as the **sample standard deviation** (mean-adjusted, scaled by $K-1$):

$$
\sigma^{\mathrm{daily}}_{b,\tau}=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(r_{b,\tau-k}-\bar r_{b,\tau}\right)^2}.
$$

Annualization:

$$
\sigma_{b,\tau}=\sigma^{\mathrm{daily}}_{b,\tau}\sqrt{252}.
$$

Sampling note: on the weekly decision date $t$ (default: last trading day of week $t$), take $\sigma_{b,t}:=\sigma_{b,\tau(t)}$.

### 6.2 Effective Volatility Floor $\sigma^{\mathrm{eff}}$ (to Stabilize $1/\sigma$)

To prevent $1/\sigma$ from over-amplifying exposure under unusually low-volatility regimes:

$$
\sigma^{\mathrm{eff}}_{b,t}=\max\left(\sigma_{b,t},\ \sigma_{\min,b}\right).
$$

$\sigma_{\min,b}$ is set per bucket using a robust historical rule (e.g., $\max\{\mathrm{Quantile}_{q_{\min}}(\sigma_{b,\cdot}),\ c_{\min}\cdot\mathrm{Median}(\sigma_{b,\cdot})\}$).

### 6.3 Tradability Filter $f(\cdot)$ (Logistic)

We suppress **untradeably high** volatility regimes smoothly:

$$
f(\sigma_{b,t}) = \frac{1}{1+\exp\left(\frac{\sigma_{b,t}-\sigma_{\max,b}}{\kappa_{\sigma,b}}\right)},
\qquad
\kappa_{\sigma,b}>0.
$$

**Interpretation.** The logistic form is a *soft feasibility filter*:

- $\sigma_{\max,b}$ is the *center* of the cutoff: $f(\sigma_{\max,b})=0.5$.
- $\kappa_{\sigma,b}$ is the *transition bandwidth*: smaller values yield a sharper cutoff; larger values yield a gentler decay.

#### 6.3.1 Selecting $\sigma_{\max,b}$ (Bucket-specific, robust)

We choose $\sigma_{\max,b}$ from the historical distribution of $\sigma_{b,t}$ so that it represents a volatility regime where CTA exposure should begin to be meaningfully suppressed.

Recommended robust choices (pick one policy and keep it consistent across buckets):

1. **Quantile rule (recommended default):**

$$
\sigma_{\max,b} = \mathrm{Quantile}_{q_{\max}}\left(\{\sigma_{b,t}\}_{t\in\mathcal{T}}\right),
\qquad q_{\max}\in[0.80,0.95].
$$

- Intuition: treat the top $(1-q_{\max})$ tail as progressively “hard-to-trade”.
- Engineering: stable, scale-free, and easy to explain.

2. **Robust location–scale rule:**

$$
\sigma_{\max,b} = \mathrm{Median}(\sigma_{b,\cdot}) + c_{\max}\cdot \mathrm{IQR}(\sigma_{b,\cdot}),
\qquad c_{\max}\in[0.5,2.0].
$$

- Intuition: suppress regimes materially above the typical volatility level while remaining robust to outliers.

**Estimation window.** Use a long and stable window $\mathcal{T}$ (e.g., full history if reliable, or a rolling 3–5 year window) and update infrequently (e.g., monthly or quarterly) to avoid parameter drift.

#### 6.3.2 Selecting $\kappa_{\sigma,b}$ (Bandwidth / slope calibration)

$\kappa_{\sigma,b}$ should be chosen so that $f(\sigma)$ transitions smoothly across the “high-vol” region without creating a new chattering mechanism.

A practical calibration uses two anchor points:

- Define $\sigma_{\max,b}$ as above.
- Define an upper anchor $\sigma_{\mathrm{hi},b}$ (e.g., $\sigma_{\mathrm{hi},b}=\mathrm{Quantile}_{q_{\mathrm{hi}}}(\sigma_{b,\cdot})$ with $q_{\mathrm{hi}}\in[0.90,0.99]$).
- Choose a small target weight $\varepsilon \in (0,0.2)$ such that $f(\sigma_{\mathrm{hi},b})=\varepsilon$.

Solving the logistic gives:

$$
\kappa_{\sigma,b} = \frac{\sigma_{\mathrm{hi},b}-\sigma_{\max,b}}{\ln\left(\frac{1}{\varepsilon}-1\right)}.
$$

Typical defaults:

- $q_{\max}=0.85$ to $0.90$
- $q_{\mathrm{hi}}=0.95$ to $0.98$
- $\varepsilon=0.10$ (so at $\sigma_{\mathrm{hi}}$ the filter reduces exposure to about 10% of normal)

**Sanity checks (recommended):**

- Ensure monotonicity and boundedness are preserved by construction.
- Verify that $f(\sigma)$ is close to 1 for the bulk of observations (e.g., below the median or below $q_{\max}$).
- Confirm that $f(\sigma)$ does not become the dominant driver of exposure under normal regimes (otherwise you are effectively trading a volatility strategy).

---

## 7. Composite CTA Raw Weight Formula

At the bucket level, the raw CTA intent is the product of three structures:

$$
\tilde{w}_{b,t} =
\underbrace{d_{b,t}}_{\text{Momentum (permission)}}
\cdot
\underbrace{f(\sigma_{b,t})}_{\text{Volatility (tradability)}}
\cdot
\underbrace{g(z_{b,t})}_{\text{Path quality (cleanliness)}}
\cdot
\frac{1}{\sigma^{\mathrm{eff}}_{b,t}}.
$$

Where:

- $d_{b,t} = d^+_{b,t}$ is the long-only hysteresis gate.
- $z_{b,t}\in[0,1]$ is the path-quality score available at time $t$:
  - offline analysis: $z_{b,t} = \mathrm{trend\_score}_{b,t}$
  - live (optional): $z_{b,t} = \hat{z}_{b,t}$

**Important:** $d^-_{b,t}$ does **not** enter this formula; it is reserved for execution-layer asymmetric smoothing.

---

## 8. Portfolio-Level Volatility Targeting and Defensive Allocation

This layer converts raw CTA intent into a portfolio that respects the volatility budget and a stable defensive policy.

### 8.1 Normalize Across Risk Buckets (with “All-Zero” Fallback)

Let $\mathcal{B}_{\mathrm{risk}}$ denote risk buckets (equity buckets + GOLD).

Define:

$$
S_t = \sum_{j\in\mathcal{B}_{\mathrm{risk}}} \tilde{w}_{j,t}.
$$

If $S_t>0$, normalize raw weights:

$$
w^{(0)}_{b,t} = \frac{\tilde{w}_{b,t}}{S_t}, \qquad b\in\mathcal{B}_{\mathrm{risk}}.
$$

**Degenerate case:** if $S_t=0$ (all risk buckets gated off), then

$$
w^{\mathrm{risk}}_{b,t}=0,\qquad \forall b\in\mathcal{B}_{\mathrm{risk}},
$$

and the whole portfolio becomes defensive (RATE/CASH) via Sections 8.4–8.5.

### 8.2 Ex-ante Volatility Implied by CTA Intent (Risk-Sleeve Only)

Let $\Sigma_t$ be the covariance matrix of **risk-bucket** returns (daily, annualized; shrinkage recommended), defined on $\mathcal{B}_{\mathrm{risk}}$ only. The implied ex-ante volatility of the **risk sleeve** is:

$$
\hat{\sigma}^{\mathrm{risk}}_{p,t} = \sqrt{\left(w^{(0)}_t\right)^\top \Sigma_t\, w^{(0)}_t}.
$$

**Clarification (policy):** This volatility quantity and the subsequent scaling in Section 8.3 apply **only** to the **risk-bucket sub-portfolio** (the “risk sleeve”). Defensive buckets (RATE/CASH) are handled separately via Sections 8.4–8.5.

### 8.3 Global Volatility Scaling (Cap / Upper Bound, Risk-Sleeve Only)

Let $\sigma_{\mathrm{target}}$ denote the volatility budget. Apply global scaling to the **risk sleeve**:

$$
\alpha_t = \min\left(1,\; \frac{\sigma_{\mathrm{target}}}{\hat{\sigma}^{\mathrm{risk}}_{p,t}}\right),
\qquad
w^{\mathrm{risk}}_{b,t} = \alpha_t\, w^{(0)}_{b,t},\qquad b\in\mathcal{B}_{\mathrm{risk}}.
$$

**Clarification:** This acts as a **cap** on the **risk sleeve**; the full portfolio may run below $\sigma_{\mathrm{target}}$ depending on signal sparsity and defensive allocation. We do not force volatility to “hit” the target.

**Important (explicit scope):** Since RATE/CASH are not included in $\hat{\sigma}^{\mathrm{risk}}_{p,t}$, this cap is defined as:  
> “Portfolio volatility control is enforced as a cap on the risk-bucket sub-portfolio (risk sleeve), while defensive allocation is policy-driven (RATE/CASH).”

### 8.4 Defensive Budget

Defensive budget is the residual:

$$
w_{\mathrm{def},t} = 1 - \sum_{b\in\mathcal{B}_{\mathrm{risk}}} w^{\mathrm{risk}}_{b,t}.
$$

### 8.5 Defensive Allocation Policy: RATE vs CASH (RATE Trend-Based)

Defensive capital is allocated between RATE and CASH deterministically.

Define RATE’s medium-horizon trend strength:

Let $P_{\mathrm{RATE},\tau}$ be the daily close series for RATE. On the weekly decision date (default: the last trading day of week $t$), sample $P_{\mathrm{RATE},t}:=P_{\mathrm{RATE},\tau(t)}$ and set $\sigma_{\mathrm{RATE},t}:=\sigma_{\mathrm{RATE},\tau(t)}$ where $\sigma_{\mathrm{RATE},\tau}$ is the annualized realized volatility computed from daily returns (Section 6.1).
 
$$
T_{\mathrm{RATE},t}=\frac{MA_{\mathrm{short}}(P_{\mathrm{RATE},t})-MA_{\mathrm{long}}(P_{\mathrm{RATE},t})}{\sigma_{\mathrm{RATE},t}}.
$$

Map to bounded preference:

$$
r^{\mathrm{RATE}}_t
=\frac{1}{1+\exp\left(-k\left(T_{\mathrm{RATE},t}-\theta_{\mathrm{RATE}}\right)\right)},\qquad k>0.
$$

Then:

$$
w_{\mathrm{RATE},t} = w_{\mathrm{def},t}\, r^{\mathrm{RATE}}_t,\qquad
w_{\mathrm{CASH},t} = w_{\mathrm{def},t} - w_{\mathrm{RATE},t}.
$$

Parameter defaults (per prior decisions):

- $k\approx 2$ (or bandwidth-calibrated, e.g., 1.8–2.8)
- $\theta_{\mathrm{RATE}} = \mathrm{Median}(T_{\mathrm{RATE}})$ (full-sample or rolling 3-year median)

---

## 9. Intra-Bucket Factor-Driven Adjustment (Soft Tilt)

**Purpose:** Conditionally tilt capital among execution assets *within a bucket* using slow-moving factor tendencies, without introducing new directional decisions.

This layer does **not** create independent alpha; it improves **capital efficiency** given existing bucket-level risk.

**Low-turnover guardrails (decision):**

- This layer updates **weekly**.
- It remains subject to the execution-layer **dead-band** (Section 10.5).
- Tilt strength is set to $\epsilon=0.5$ initially and may be tuned later.

### 9.1 Design Rules

1. Bucket-level CTA signals are authoritative.
2. No cross-sectional selection (no dynamic inclusion or exclusion).
3. Adjustments are continuous and bounded.
4. Weekly recompute; trading occurs only if the resulting weight change passes the dead-band.

### 9.2 Factor Tendency Vector (Cumulative Over $H$ Trading Days), Sampled Weekly

Let $\mathbf{F}_{\tau}$ be **daily** factor returns (e.g., SMB, QMJ). Define medium-horizon cumulative factor returns over $H$ **trading days**:

$$
\mathbf{R}^{(H)}_{\tau} = \sum_{k=1}^{H} \mathbf{F}_{\tau-k}.
$$

Sampling note: on the weekly decision date $t$ (default: last trading day of week $t$), sample the cumulative factor return and downstream tendency vector to obtain $\mathbf{R}^{(H)}_t$ and $\mathbf{s}_t$.

To avoid scale dominance across factor dimensions, apply **dimension-wise robust standardization** (per factor component $j$):

$$
\tilde{R}^{(H)}_{j,t}
=
\operatorname{clip}\!\left(
\frac{R^{(H)}_{j,t}-\operatorname{Median}\!\left(R^{(H)}_{j,\cdot}\right)}
{\operatorname{IQR}\!\left(R^{(H)}_{j,\cdot}\right)+\epsilon_R},
\ -c_R,\ c_R
\right).
$$
where $\epsilon_R>0$ prevents division-by-zero and $c_R>0$ is the clipping bound.

Then map to a bounded tendency vector:

$$
\mathbf{s}_t = \tanh\!\left(\gamma_{\mathrm{tilt}}\, \tilde{\mathbf{R}}^{(H)}_t\right)\in[-1,1]^d.
$$

(Any optional smoothing of $\mathbf{s}_t$ is not required; avoid double-counting with execution-layer EWMA.)

### 9.3 Execution Asset Exposure Vectors (Diagonal Scale Calibration)

For each execution asset $i$ inside bucket $b$, define its factor exposure vector via Kalman-filtered betas:

$$
\mathbf{g}_{i,t} = \left(\beta_{i,1,t},\beta_{i,2,t},\dots,\beta_{i,d,t}\right).
$$

To prevent any single factor dimension from dominating due to scale, introduce a **diagonal scale calibration**:

$$
\mathbf{D}=\operatorname{diag}(a_1,\dots,a_d),
\qquad
a_j = \operatorname{IQR}\!\left(g_{i,j,\cdot}\right) \ \text{(pooled across assets $i$ in the same bucket, over a reference window)}.
$$

Define calibrated exposures:

$$
\tilde{\mathbf{g}}_{i,t} = \mathbf{D}^{-1}\mathbf{g}_{i,t}.
$$

This is a low-complexity alternative to full covariance whitening, while addressing cross-factor scale mismatch.

### 9.4 Alignment via Cosine Similarity and Soft Tilt

Compute **cosine alignment** (bounded in $[-1,1]$):

$$
u_{i,t}
=
\frac{\mathbf{s}_t^\top \tilde{\mathbf{g}}_{i,t}}
{\left\|\mathbf{s}_t\right\|_2\left\|\tilde{\mathbf{g}}_{i,t}\right\|_2+\epsilon_u}.
$$
where $\epsilon_u>0$ is a small constant for numerical stability.

Convert into bounded multipliers:

$$
\mu_{i,t} = 1 + \epsilon\,\tanh(u_{i,t}),\qquad \epsilon=0.5.
$$

Bucket-normalized intra-weights:

$$
\pi_{i\mid b,t} = \frac{\mu_{i,t}}{\sum_{j\in\mathcal{I}(b)} \mu_{j,t}}.
$$

Final fund-level target weights:

$$
w^*_{i,t} = w_{b,t}\,\pi_{i\mid b,t}.
$$

---

## 10. Execution & Control (NAV, Holding Period, Low Turnover)

This layer turns finalized target weights into **tradeable executed weights** under mutual-fund constraints.

### 10.1 Target Weights vs Executed Weights

Let $w^*_{i,t}$ be finalized post-normalization target weights after:

1. bucket-level signal construction and volatility targeting, and
2. intra-bucket adjustment.

Let $w^{\mathrm{exec}}_{i,t}$ be executed weights after turnover and constraint controls.

**Central rule:** Execution-layer EWMA smoothing is applied **only** to $w^*_{i,t}$ (post-normalization), ensuring a single source of truth for temporal smoothing and turnover control.

### 10.2 Minimum Holding Period Constraint

Maintain last-buy timestamp $\tau_i$ for each asset $i$. Let $H_{\mathrm{hold}}$ be the minimum holding period (calendar days, e.g., $H_{\mathrm{hold}}=7$). Define a sell-eligibility mask $m_{i,t}\in\{0,1\}$:

- $m_{i,t}=1$: selling is allowed at time $t$
- $m_{i,t}=0$: selling is disallowed (avoid penalty)

### 10.3 Asymmetric EWMA Smoothing (Risk-off Faster Than Risk-on)

Use bucket-dependent asymmetric EWMA.

Let $b(i)$ be the bucket of asset $i$. Define:

$$
\alpha_{b,t}=
\begin{cases}
\alpha_{\mathrm{off}}, & d^-_{b,t}=1,\\
\alpha_{\mathrm{on}}, & d^-_{b,t}=0,
\end{cases}
\qquad 0<\alpha_{\mathrm{on}}<\alpha_{\mathrm{off}}<1.
$$

Then:

$$
\tilde{w}^{\mathrm{exec}}_{i,t} =
(1-\alpha_{b(i),t})\,w^{\mathrm{exec}}_{i,t-1}
+ \alpha_{b(i),t}\,w^*_{i,t}.
$$

**Trigger semantics (decision):** $d^-_{b,t}$ only changes the **adjustment speed** (typically faster de-risking). It does **not** modify signal-layer sizing and does not imply shorting.

### 10.4 Constraint-Aware Adjustment Under Holding Restrictions (Priority Rules)

When $m_{i,t}=0$, selling is infeasible. The system must remain feasible and deterministic. Priority rules:

1. **Feasibility first:** do not violate holding constraints.
2. **No forced sells:** assets with $m_{i,t}=0$ are locked; the system adjusts only the sellable subset.
3. **If budget is insufficient:** suppress new buys first (prefer holding CASH/RATE) rather than causing turnover.

One deterministic approach:

- Lock non-sellable positions:

$$
w^{\mathrm{exec}}_{i,t} = w^{\mathrm{exec}}_{i,t-1}, \qquad \forall i\ \text{with } m_{i,t}=0.
$$

- Update sellable positions toward $\tilde{w}^{\mathrm{exec}}_{i,t}$ and renormalize the sellable subset to satisfy the budget constraint and nonnegativity.

(Any equivalent projection scheme is acceptable if deterministic and auditable.)

### 10.5 Rebalance Dead-Band (Trade Trigger)

To avoid micro-trades under NAV execution, apply a dead-band:

$$
\text{trade if } \left|w^*_{i,t} - w^{\mathrm{exec}}_{i,t-1}\right| > \delta_w.
$$

This dead-band also bounds the turnover impact of the factor tilt in Section 9.

### 10.6 Caps and Concentration Controls

To avoid single-asset or single-bucket concentration:

- Per-asset cap:

$$
w^{\mathrm{exec}}_{i,t} \le w^{\max}_i.
$$

- Optional per-bucket cap:

$$
\sum_{i\in\mathcal{I}(b)} w^{\mathrm{exec}}_{i,t} \le W^{\max}_b.
$$

Excess weight is redistributed deterministically into defensive buckets (RATE/CASH) or via proportional scaling.

### 10.7 Robustness Summary

The execution layer ensures that:

- Signal intent (permission, sizing, volatility targeting) remains intact.
- Turnover is controlled via EWMA and the dead-band.
- Minimum holding restrictions are respected without infeasible trades.
- Downward drift is used only to accelerate de-risking via asymmetric EWMA, without shorting and without adding extra signal-layer sizing terms.

---

## 11. Parameterization and Engineering Notes (Notation & Timing Contract)

### 11.1 Timing Contract (Daily Compute, Weekly Sampling, T+1 NAV Execution)

We distinguish:

- $t$: weekly decision index (sampling point)
- $\tau$: daily index (NAV execution)

**Bucket price definition:** $P_{b,\tau}$ is the daily close series; the sampled weekly price is $P_{b,t}:=P_{b,\tau(t)}$, i.e., the last trading day close of week $t$.

**Signal timing:** signals are computed on the daily series and sampled after week $t$ closes.

**Execution timing:** signals computed at week $t$ are executed at the **next trading day NAV** (the chosen “option 2” alignment).

### 11.2 Key Parameters

- Trend MAs: $MA_{\mathrm{short}}, MA_{\mathrm{long}}$ (daily; sampled weekly)
- Hysteresis thresholds: $\theta_{\mathrm{on}}, \theta_{\mathrm{off}}$
- Downward threshold: $\theta_-$
- Volatility window: $K$ (trading days; sampled weekly)
- Volatility bounds: $\sigma_{\min,b}$ and $\sigma_{\max,b}$; logistic slope $\kappa_{\sigma,b}$
- Path mapping: $x_0, \gamma$
- Portfolio volatility budget: $\sigma_{\mathrm{target}}$ (cap behavior)
- Execution smoothing: $\alpha_{\mathrm{on}}, \alpha_{\mathrm{off}}$
- Dead-band: $\delta_w$
- Defensive policy: $k, \theta_{\mathrm{RATE}}$
- Intra-bucket tilt: $H, \gamma_{\mathrm{tilt}}, \epsilon$
- Minimum holding: $H_{\mathrm{hold}}$ (calendar days; default 7)
- Tilt standardization: $\epsilon_R$, $c_R$
- Cosine stability: $\epsilon_u$

### 11.3 Determinism and Auditability

Every transformation from data $\to$ signal $\to$ target $\to$ executed weights must be:

- deterministic,
- reproducible,
- logged (inputs, outputs, parameters, and constraint masks),
- robust to missing or weak information (graceful degradation).

---

## Appendix: Design Philosophy Summary

1. **Regime-agnostic** rather than regime-predictive
2. **Path-dependent** rather than return-forecasting
3. **Engineering-first**, prioritizing stability over cleverness
4. **Modular**, enabling upgrades without architectural breakage
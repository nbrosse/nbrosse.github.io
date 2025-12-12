# Flow Matching Collapse: A Statistical Physics Perspective

This document provides a rigorous analysis of the collapse phenomenon in Flow Matching using tools from statistical physics, specifically the Random Energy Model (REM) and Large Deviation Principles (LDP).

---

## 1. Gibbs Measure and Energy Formulation

### 1.1 The Marginal FM Weights as a Gibbs Measure

The marginal velocity field in Flow Matching is given by:
$$
u_t(x) = \sum_{i=1}^n \lambda_i(x,t) \frac{x^{(i)} - x}{1-t}
$$
where the weights $\lambda_i(x,t)$ take the form:
$$
\lambda_i(x,t) = \frac{\exp\left(-\frac{\|x - tx^{(i)}\|^2}{2(1-t)^2}\right)}{\sum_{j=1}^n \exp\left(-\frac{\|x - tx^{(j)}\|^2}{2(1-t)^2}\right)}.
$$

This is precisely a **Boltzmann distribution** at inverse temperature $\beta = 1$ over "states" $i \in \{1, \dots, n\}$ with energy:
$$
E_i(x,t) = \frac{\|x - tx^{(i)}\|^2}{2(1-t)^2}.
$$

The denominator defines the **partition function**:
$$
Z(x,t) = \sum_{j=1}^n e^{-E_j(x,t)}.
$$

### 1.2 Assumptions on the Data

We consider a **Gaussian proxy** for the training data:
$$
x^{(i)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2 I_d), \quad i = 1, \dots, n.
$$

This approximation captures the essential high-dimensional geometry while enabling analytical tractability. For CIFAR-10 after standard normalization, $\sigma^2 \approx 0.24$ provides a reasonable proxy.

### 1.3 The "Planted" (Teacher) Trajectory

Consider a trajectory starting from noise and targeting a specific training image:
$$
x_t = (1-t)x_0 + tx_1, \quad x_0 \sim \mathcal{N}(0, I_d), \quad x_1 = x^{(1)}.
$$

We call $i = 1$ the **special (planted) index**. For this index:
$$
x_t - tx^{(1)} = (1-t)x_0
$$
which gives the **planted energy**:
$$
E_1(x_t, t) = \frac{\|(1-t)x_0\|^2}{2(1-t)^2} = \frac{\|x_0\|^2}{2}.
$$

By the law of large numbers for chi-squared random variables:
$$
\frac{E_1}{d} \xrightarrow{d \to \infty} \frac{1}{2} \quad \text{almost surely.}
$$

Hence the planted contribution to the partition function is:
$$
Z_1 = e^{-E_1} \approx e^{-d/2}.
$$

### 1.4 The "Bulk" (REM-like) Energies

For $j \neq 1$, we have:
$$
x_t - tx^{(j)} = (1-t)x_0 + t(x_1 - x^{(j)}).
$$

Under the Gaussian proxy, $x_1 - x^{(j)} \sim \mathcal{N}(0, 2\sigma^2 I_d)$ (independent of $x_0$). Therefore:
$$
x_t - tx^{(j)} \sim \mathcal{N}\left(0, s^2(t) I_d\right), \quad s^2(t) := (1-t)^2 + 2\sigma^2 t^2.
$$

The bulk energies become:
$$
E_j(x_t, t) = \frac{\|x_t - tx^{(j)}\|^2}{2(1-t)^2} \approx \frac{s^2(t)}{2(1-t)^2} \cdot \chi_d^2.
$$

Define the **time-dependent energy scale**:
$$
c(t) := \frac{s^2(t)}{2(1-t)^2} = \frac{1}{2}\left(1 + \frac{2\sigma^2 t^2}{(1-t)^2}\right).
$$

**Key observation:** The bulk energies are approximately i.i.d. across $j \geq 2$, each extensive (order $d$). This is the classic setup where **Random Energy Model (REM)** tools apply.

---

## 2. Collapsed vs Uncollapsed States (Glass vs Liquid)

### 2.1 Definition of Collapse

The central question is whether the partition function $Z(x_t, t)$ is dominated by:
- **One special term** (the planted state), or
- **Exponentially many comparable terms** (the bulk).

This leads to two phases:

| Phase | Statistical Physics | Flow Matching | Behavior |
|-------|---------------------|---------------|----------|
| **Collapsed** | Glass / Condensed | Memorization | $\lambda_1 \approx 1$ |
| **Uncollapsed** | Liquid | Generalization | Mass spread over many $j$'s |

### 2.2 Glass Phase (Collapsed / Memorization)

In the collapsed phase, $Z \approx Z_1 = e^{-E_1}$, meaning $\lambda_1 \approx 1$.

The marginal velocity field then reduces to:
$$
u_t(x_t) \approx \frac{x^{(1)} - x_t}{1-t} = \frac{x_1 - x_t}{1-t} = x_1 - x_0.
$$

This is precisely the **conditional velocity field**. The marginal flow aligns perfectly with the teacher trajectory, leading to memorization of training data.

### 2.3 Liquid Phase (Uncollapsed / Generalization)

In the liquid phase, $Z \approx Z_{\text{bulk}} = \sum_{j=2}^n e^{-E_j}$, with mass distributed across many states.

The velocity field is a genuine mixture of directions toward different training points, enabling interpolation and generalization.

### 2.4 The Phase Transition

The transition between phases depends on:
- **Energy advantage** of the planted state (order $d$)
- **Entropy** from having $n$ competitors (order $\log n$)

The competition is controlled by the **dimensionless ratio**:
$$
\alpha := \frac{\log n}{d}.
$$

For typical datasets (e.g., CIFAR-10 with $d = 3072$, $n = 50{,}000$):
$$
\alpha = \frac{\log(50{,}000)}{3072} \approx 0.0035 \ll 1.
$$

This tiny $\alpha$ means the entropy budget is negligible compared to extensive energy gaps, placing the system deep in the collapsed regime for most values of $t$.

---

## 3. Simple Heuristic Analysis

### 3.1 Typical Energy Gap

Compare a typical bulk energy to the planted energy. The **typical energy gap** is:
$$
E_j - E_1 \approx \frac{d}{2}\left(\frac{s^2(t)}{(1-t)^2} - 1\right) = \frac{d}{2} \cdot \frac{2\sigma^2 t^2}{(1-t)^2} = \frac{d \sigma^2 t^2}{(1-t)^2}.
$$

Therefore, the log-weight ratio is:
$$
\log \frac{\lambda_j}{\lambda_1} \approx -(E_j - E_1) \approx -\frac{d \sigma^2 t^2}{(1-t)^2}.
$$

This shows **exponential-in-$d$ suppression** of bulk weights relative to the planted weight.

### 3.2 Crude Annealed Entropy Estimate

A naive estimate of the bulk partition function uses the **annealed approximation**:
$$
Z_{\text{bulk}} \approx (n-1) \cdot \mathbb{E}[e^{-E_j}].
$$

However, this is often wrong because $e^{-E_j}$ has exponentially broad fluctuations. The sum is dominated by **rare low-energy states**, not typical ones. This is precisely where REM physics becomes essential.

### 3.3 Energy vs Entropy Competition

The competition can be summarized as:
- **Planted contribution:** $Z_1 \sim e^{-d/2}$
- **Bulk contribution (naive):** $Z_{\text{bulk}} \sim n \cdot e^{-d \cdot c(t)} = e^{\alpha d - d \cdot c(t)}$

For the planted state to dominate, we need the energy advantage to exceed the entropy:
$$
\frac{d \sigma^2 t^2}{(1-t)^2} \gg \log n.
$$

Rearranging, collapse holds whenever:
$$
\frac{t}{1-t} \gg \sqrt{\frac{\log n}{d \sigma^2}}.
$$

For CIFAR-like numbers, the right-hand side is tiny ($\approx 0.12$), so collapse can occur at small $t$.

### 3.4 Simple Collapse Time Estimate

Setting the two sides equal gives a heuristic collapse time:
$$
t_C^{\text{heuristic}} \approx \frac{\sqrt{\alpha/\sigma^2}}{1 + \sqrt{\alpha/\sigma^2}}.
$$

For CIFAR-10 ($\alpha \approx 0.0035$, $\sigma^2 \approx 0.24$):
$$
t_C^{\text{heuristic}} \approx 0.11.
$$

**Caveat:** This estimate compares typical bulk energy to planted energy. A rigorous analysis must consider the **best** competitor among $n-1$ states, not a typical one. This requires REM/LDP tools.

---

## 4. Rigorous REM/LDP Derivation

### 4.1 Large Deviation Principles: Foundations

#### Why LDP?

The Law of Large Numbers tells us *where* random variables concentrate, but not *how unlikely* deviations are. Large Deviation Theory provides **exponential decay rates** for rare events.

#### Definition

A sequence of random variables $X_d$ satisfies a **Large Deviation Principle (LDP)** with rate function $I(x)$ if:
$$
\mathbb{P}(X_d \approx x) \asymp e^{-d \, I(x)}
$$
where:
- $d$ is the large parameter (dimension)
- $I(x) \geq 0$ with $I(x) = 0$ at the typical value
- Larger $I(x)$ means exponentially rarer events

#### Cramér's Theorem

For i.i.d. random variables $X_1, \dots, X_d$ with finite moment generating function, the empirical average $S_d = \frac{1}{d}\sum_{k=1}^d X_k$ satisfies an LDP:
$$
\mathbb{P}(S_d \approx x) \asymp e^{-d \, I(x)}
$$
where the **rate function** is the Legendre transform of the log-MGF:
$$
I(x) = \sup_{\lambda \in \mathbb{R}} \left( \lambda x - \log \mathbb{E}[e^{\lambda X_1}] \right).
$$

### 4.2 LDP for Chi-Squared (Energy Density)

#### Deriving the Rate Function

Let $Z_k \sim \mathcal{N}(0,1)$ i.i.d., so $\chi_d^2 = \sum_{k=1}^d Z_k^2$. Define:
$$
Y_d := \frac{\chi_d^2}{d} = \frac{1}{d}\sum_{k=1}^d Z_k^2.
$$

For $X = Z^2$, the log-MGF is:
$$
\log \mathbb{E}[e^{\lambda Z^2}] = -\frac{1}{2}\log(1 - 2\lambda), \quad \lambda < \frac{1}{2}.
$$

Applying Cramér's theorem and computing the Legendre transform:
$$
I_\chi(y) = \sup_{\lambda < 1/2}\left(\lambda y + \frac{1}{2}\log(1 - 2\lambda)\right) = \frac{1}{2}(y - 1 - \log y), \quad y > 0.
$$

**Key properties:**
- Minimum at $y = 1$ with $I_\chi(1) = 0$ (typical value)
- Convex and non-negative
- Deviations are exponentially unlikely in $d$

#### Energy Density LDP

For bulk energies with $E = c(t) \cdot \chi_d^2$, the energy density $e = E/d = c \cdot Y_d$ satisfies:
$$
\mathbb{P}(e \approx \varepsilon) \asymp e^{-d \, I(\varepsilon)}
$$
where:
$$
\boxed{I(\varepsilon) = \frac{1}{2}\left(\frac{\varepsilon}{c} - 1 - \log\frac{\varepsilon}{c}\right)}
$$

This is the **rigorous backbone** for all REM arguments.

### 4.3 Bulk Free Energy via Laplace Principle

#### Complexity (Entropy of States)

With $n = e^{\alpha d}$ bulk energies, the expected **number of states** with energy density near $\varepsilon$ is:
$$
\#(\varepsilon) \approx n \cdot \mathbb{P}(e \approx \varepsilon) \approx \exp\{d(\alpha - I(\varepsilon))\}.
$$

The quantity:
$$
\Sigma(\varepsilon) := \alpha - I(\varepsilon)
$$
is the **complexity** (entropy density of states at energy $\varepsilon$).

#### Laplace Principle

The bulk partition function is approximately:
$$
Z_{\text{bulk}} = \sum_{j=2}^n e^{-E_j} \approx \int d\varepsilon \, \exp\{d(\Sigma(\varepsilon) - \varepsilon)\}.
$$

By the Laplace principle (large-$d$ saddle point):
$$
\boxed{\Phi(1) := \lim_{d \to \infty} \frac{1}{d}\log Z_{\text{bulk}} = \sup_{\varepsilon > 0}\left[\alpha - I(\varepsilon) - \varepsilon\right]}
$$

### 4.4 Glass vs Liquid Phase Transition

#### The Variational Problem

Define $F(\varepsilon) = \alpha - I(\varepsilon) - \varepsilon$. The maximizer determines the phase.

#### Liquid Phase (Interior Solution)

If the maximizer is interior, it satisfies $F'(\varepsilon) = 0$, i.e., $I'(\varepsilon) = -1$.

Computing:
$$
I'(\varepsilon) = \frac{1}{2}\left(\frac{1}{c} - \frac{1}{\varepsilon}\right) = -1
$$
yields the **liquid saddle point**:
$$
\boxed{\varepsilon_* = \frac{c}{1 + 2c}}
$$

This solution is valid only if there are exponentially many states there: $\Sigma(\varepsilon_*) = \alpha - I(\varepsilon_*) \geq 0$.

#### Glass Phase (Boundary Solution)

If $\alpha - I(\varepsilon_*) < 0$, there are too few states at the liquid saddle. The supremum hits the boundary where complexity vanishes:
$$
I(\varepsilon_g) = \alpha
$$

The partition sum is then dominated by the **lowest-energy extremes**: this is the glass/condensed phase.

#### Freezing Threshold

The critical entropy level separating phases is:
$$
\boxed{\alpha_c(c) := I(\varepsilon_*) = \frac{1}{2}\left(\log(1 + 2c) - \frac{2c}{1 + 2c}\right)}
$$

**Phase diagram:**
- **Liquid phase:** $\alpha \geq \alpha_c(c)$, maximizer at $\varepsilon_*$, free energy $\Phi(1) = \alpha - I(\varepsilon_*) - \varepsilon_*$
- **Glass phase:** $\alpha < \alpha_c(c)$, maximizer at $\varepsilon_g$ with $I(\varepsilon_g) = \alpha$, free energy $\Phi(1) = -\varepsilon_g$

### 4.5 Collapse Criterion from Free Energy

#### Full Partition Function

The complete partition function is:
$$
Z = e^{-E_1} + Z_{\text{bulk}}, \quad \lambda_1 = \frac{e^{-E_1}}{Z}.
$$

Since $E_1/d \to 1/2$, we have:
$$
\frac{1}{d}\log\frac{Z_{\text{bulk}}}{e^{-E_1}} \to \Phi(1) + \frac{1}{2}.
$$

#### Collapse Condition

$$
\boxed{\lambda_1 \to 1 \iff \Phi(1) < -\frac{1}{2}}
$$

More precisely, for large $d$:
$$
\lambda_1 \approx \frac{1}{1 + \exp\{d(\Phi(1) + 1/2)\}}.
$$

#### Glass Phase Collapse Criterion

In the glass phase, $\Phi(1) = -\varepsilon_g$, so collapse requires:
$$
\boxed{-\varepsilon_g < -\frac{1}{2} \iff \varepsilon_g > \frac{1}{2}}
$$

where $\varepsilon_g$ is defined implicitly by $I(\varepsilon_g) = \alpha$.

**Interpretation:** Collapse occurs when even the *best* spurious competitor (with energy density $\varepsilon_g$) has higher energy than the planted state (energy density $1/2$).

#### Liquid Phase Collapse Criterion

In the liquid phase, collapse requires:
$$
\alpha - I(\varepsilon_*) - \varepsilon_* < -\frac{1}{2}.
$$

This gives an explicit inequality in $c$ and $\alpha$.

### 4.6 Explicit Collapse Time

#### Collapse Time Equation

The collapse time $t_C$ is defined by $\varepsilon_g(t_C) = 1/2$. Setting $\varepsilon_g = 1/2$ in $I(\varepsilon_g) = \alpha$:
$$
\alpha = I(1/2) = \frac{1}{2}\left(\frac{1/2}{c} - 1 - \log\frac{1/2}{c}\right) = \frac{1}{2}\left(\frac{1}{2c} - 1 + \log(2c)\right).
$$

This gives the **REM-rigorous collapse boundary**:
$$
\boxed{\alpha = \frac{1}{2}\left(\log(2c(t_C)) + \frac{1}{2c(t_C)} - 1\right)}
$$

with $c(t) = \frac{1}{2}\left(1 + \frac{2\sigma^2 t^2}{(1-t)^2}\right)$.

#### CIFAR-10 Proxy Calculation

For CIFAR-10: $d = 3072$, $n = 50{,}000$, $\sigma^2 = 0.24$.

Compute:
$$
\alpha = \frac{\log(50{,}000)}{3072} \approx 0.003522.
$$

Solving the collapse equation yields:
$$
\boxed{t_C \approx 0.341}
$$

with corresponding $c(t_C) \approx 0.5644$.

**Interpretation:** Under the i.i.d. Gaussian proxy + REM idealization, for $t \lesssim 0.34$, the planted datapoint is no longer guaranteed to dominate because among $n$ training points, one can typically find at least one spurious point whose energy matches or beats the planted energy. For $t \gtrsim 0.34$, collapse occurs and the planted state dominates.

#### Why CIFAR is Always in the Glass Phase

For CIFAR-like $\alpha \ll 1$, we have $\alpha \ll \alpha_c(c(t))$ for all $t \in [0,1)$. The system is **always in the glass phase**, meaning:
1. The bulk partition function is dominated by a few extreme low-energy states
2. The free-energy criterion reduces to the minimum-energy criterion
3. The "going one level deeper" analysis confirms the simpler extreme-value argument

### 4.7 Summary of Key Equations

#### Bulk Energy LDP
$$
\mathbb{P}\left(\frac{E}{d} \approx \varepsilon\right) \asymp e^{-d \, I(\varepsilon)}, \quad I(\varepsilon) = \frac{1}{2}\left(\frac{\varepsilon}{c} - 1 - \log\frac{\varepsilon}{c}\right)
$$

#### Bulk Quenched Free Energy
$$
\Phi(1) = \sup_{\varepsilon > 0}\left[\alpha - I(\varepsilon) - \varepsilon\right], \quad \alpha = \frac{\log n}{d}
$$

#### Glass vs Liquid Threshold
$$
\alpha_c(c) = \frac{1}{2}\left(\log(1 + 2c) - \frac{2c}{1 + 2c}\right)
$$

#### Collapse Criterion
$$
\lambda_1 \to 1 \iff \Phi(1) < -\frac{1}{2}
$$

In the glass phase ($\alpha < \alpha_c$):
$$
\lambda_1 \to 1 \iff \varepsilon_g > \frac{1}{2}, \quad I(\varepsilon_g) = \alpha
$$

#### Collapse Time (Glass Phase)
$$
\alpha = \frac{1}{2}\left(\log(2c(t_C)) + \frac{1}{2c(t_C)} - 1\right)
$$

---

## Appendix: Extension to Non-Isotropic Covariance

### Motivation

The isotropic Gaussian proxy $x^{(i)} \sim \mathcal{N}(0, \sigma^2 I_d)$ is convenient but unrealistic. Real image datasets have structured covariance with varying eigenvalues (some directions have high variance, others low).

**Key insight:** The REM/LDP framework extends naturally to non-isotropic covariance. The structure of the theory remains identical — only the **rate function** $I(\varepsilon)$ changes.

### Setup: General Covariance Model

Assume training images follow:
$$
x^{(i)} \sim \mathcal{N}(0, C), \quad i = 1, \dots, n
$$
where $C$ is a $d \times d$ positive definite covariance matrix with eigendecomposition:
$$
C = U \text{diag}(\mu_1, \dots, \mu_d) U^\top.
$$

### Deriving the Effective Covariance

Recall the bulk energy for $j \geq 2$:
$$
E_j = \frac{\|x_t - tx^{(j)}\|^2}{2(1-t)^2}, \quad x_t = (1-t)x_0 + tx_1.
$$

**Step 1: Decompose the displacement**
$$
x_t - tx^{(j)} = (1-t)x_0 + t(x_1 - x^{(j)}).
$$

**Step 2: Identify the distributions**
- $x_0 \sim \mathcal{N}(0, I_d)$ (noise)
- $x_1 - x^{(j)} \sim \mathcal{N}(0, 2C)$ (difference of two i.i.d. samples from $\mathcal{N}(0, C)$)

**Step 3: Compute the covariance of $x_t - tx^{(j)}$**

Since $x_0$ and $(x_1 - x^{(j)})$ are independent:
$$
\text{Cov}(x_t - tx^{(j)}) = (1-t)^2 I_d + 2t^2 C =: \Sigma_t.
$$

**Step 4: Define the effective covariance for the energy**

The energy involves $\|x_t - tx^{(j)}\|^2 / (2(1-t)^2)$. Let $g \sim \mathcal{N}(0, I_d)$ be a standard Gaussian vector. Then:
$$
x_t - tx^{(j)} \stackrel{d}{=} \Sigma_t^{1/2} g
$$
and the energy becomes:
$$
E_j = \frac{g^\top \Sigma_t g}{2(1-t)^2} = \frac{1}{2} g^\top C_t g
$$
where the **effective covariance** is:
$$
C_t := \frac{\Sigma_t}{(1-t)^2} = I_d + \frac{2t^2}{(1-t)^2} C.
$$

### Eigenvalues of the Effective Covariance

If $C$ has eigenvalues $\{\mu_k\}_{k=1}^d$, then $C_t$ has eigenvalues:
$$
\lambda_k^{(t)} = 1 + \frac{2t^2}{(1-t)^2} \mu_k, \quad k = 1, \dots, d.
$$

The bulk energy density is:
$$
\frac{E_j}{d} = \frac{1}{2d} \sum_{k=1}^d \lambda_k^{(t)} Z_k^2
$$
where $Z_k \sim \mathcal{N}(0,1)$ are i.i.d. (the components of $g$ in the eigenbasis of $C_t$).

**Recovery of isotropic case:** When $C = \sigma^2 I_d$, all $\mu_k = \sigma^2$, so:
$$
\lambda_k^{(t)} = 1 + \frac{2\sigma^2 t^2}{(1-t)^2} = 2c(t)
$$
which matches our earlier formula (the factor of 2 accounts for the $1/2$ in $E_j = \frac{1}{2}g^\top C_t g$).

### LDP for Weighted Chi-Squared: Gärtner-Ellis Theorem

The energy density is a **weighted sum of chi-squared variables**:
$$
\frac{E_j}{d} = \frac{1}{2d} \sum_{k=1}^d \lambda_k^{(t)} Z_k^2.
$$

To find its LDP, we use the **Gärtner-Ellis theorem**, which generalizes Cramér's theorem to non-i.i.d. settings.

**Step 1: Compute the scaled cumulant generating function**

For a single term $\lambda Z^2$ with $Z \sim \mathcal{N}(0,1)$:
$$
\mathbb{E}[e^{\theta \lambda Z^2}] = \frac{1}{\sqrt{1 - 2\theta\lambda}}, \quad \theta < \frac{1}{2\lambda}.
$$
Thus:
$$
\log \mathbb{E}[e^{\theta \lambda Z^2}] = -\frac{1}{2}\log(1 - 2\theta\lambda).
$$

**Step 2: Sum over all eigenvalues**

The scaled log-MGF for the energy density $e = E_j/d$ is:
$$
\Lambda_t(\theta) := \lim_{d \to \infty} \frac{1}{d} \log \mathbb{E}\left[e^{d\theta \cdot \frac{1}{2d}\sum_k \lambda_k^{(t)} Z_k^2}\right] = \lim_{d \to \infty} \frac{1}{d} \sum_{k=1}^d \left(-\frac{1}{2}\log\left(1 - \theta\lambda_k^{(t)}\right)\right).
$$

This limit exists if the **empirical spectral distribution** of $C_t$ converges:
$$
\Lambda_t(\theta) = \int \left(-\frac{1}{2}\log(1 - \theta\lambda)\right) d\nu_t(\lambda)
$$
where $\nu_t$ is the limiting spectral measure of $\{lambda_k^{(t)}\}$.

**Step 3: Legendre transform gives the rate function**

By the Gärtner-Ellis theorem:
$$
\boxed{I_t(\varepsilon) = \sup_{\theta < 1/\lambda_{\max}^{(t)}} \left(\theta\varepsilon - \Lambda_t(\theta)\right)}
$$

This is the **drop-in replacement** for the isotropic rate function.

### Verifying the Isotropic Case

For $C = \sigma^2 I_d$, all eigenvalues are equal: $\lambda_k^{(t)} = \lambda^{(t)} = 1 + \frac{2\sigma^2 t^2}{(1-t)^2}$.

Then:
$$
\Lambda_t(\theta) = -\frac{1}{2}\log(1 - \theta\lambda^{(t)}).
$$

The Legendre transform:
$$
I_t(\varepsilon) = \sup_\theta \left(\theta\varepsilon + \frac{1}{2}\log(1 - \theta\lambda^{(t)})\right).
$$

Setting the derivative to zero:
$$
\varepsilon - \frac{\lambda^{(t)}/2}{1 - \theta\lambda^{(t)}} = 0 \quad \Rightarrow \quad \theta^* = \frac{1}{\lambda^{(t)}} - \frac{1}{2\varepsilon}.
$$

Substituting back and simplifying:
$$
I_t(\varepsilon) = \frac{1}{2}\left(\frac{2\varepsilon}{\lambda^{(t)}} - 1 - \log\frac{2\varepsilon}{\lambda^{(t)}}\right).
$$

With $c(t) = \lambda^{(t)}/2$, this becomes:
$$
I_t(\varepsilon) = \frac{1}{2}\left(\frac{\varepsilon}{c(t)} - 1 - \log\frac{\varepsilon}{c(t)}\right)
$$
which matches our earlier isotropic formula exactly.

### The Full Pipeline with General Covariance

The REM framework carries through unchanged:

1. **Bulk free energy:**
$$
\Phi(1,t) = \sup_{\varepsilon > 0}\left[\alpha - I_t(\varepsilon) - \varepsilon\right]
$$

2. **Collapse criterion:**
$$
\lambda_1 \to 1 \iff \Phi(1,t) < -\frac{1}{2}
$$

3. **Planted weight approximation:**
$$
\lambda_1(t) \approx \frac{1}{1 + \exp\{d(\Phi(1,t) + 1/2)\}}
$$

Only $I_t(\varepsilon)$ changes — the rest of the theory is identical.

### Practical Implementation

**To compute $t_C$ and $\lambda_1(t)$ for real data:**

1. **Estimate the covariance spectrum:** Compute the eigenvalues $\{\mu_k\}$ of the sample covariance of your training data.

2. **Compute the effective spectrum at each $t$:**
$$
\lambda_k^{(t)} = 1 + \frac{2t^2}{(1-t)^2} \mu_k
$$

3. **Evaluate the scaled log-MGF numerically:**
$$
\Lambda_t(\theta) = \frac{1}{d}\sum_{k=1}^d \left(-\frac{1}{2}\log(1 - \theta\lambda_k^{(t)})\right)
$$

4. **Compute the rate function via numerical Legendre transform:**
$$
I_t(\varepsilon) = \sup_{\theta} \left(\theta\varepsilon - \Lambda_t(\theta)\right)
$$

5. **Solve the variational problem:**
$$
\Phi(1,t) = \sup_\varepsilon\left[\alpha - I_t(\varepsilon) - \varepsilon\right]
$$

6. **Find $t_C$:** Solve $\Phi(1,t_C) = -1/2$.

### Spiked Covariance Model (Analytical Tractability)

For interpretability, consider a **spiked covariance** model:
$$
C = v I_d + (\Lambda - v) uu^\top
$$
where:
- $v > 0$ is the bulk variance
- $\Lambda > v$ is the spike strength (top eigenvalue)
- $u \in \mathbb{R}^d$ is the principal direction ($\|u\| = 1$)

This model has:
- One eigenvalue $\Lambda$ (the spike)
- $(d-1)$ eigenvalues equal to $v$ (the bulk)

The effective spectrum becomes:
- One eigenvalue: $\lambda_{\text{spike}}^{(t)} = 1 + \frac{2t^2}{(1-t)^2}\Lambda$
- $(d-1)$ eigenvalues: $\lambda_{\text{bulk}}^{(t)} = 1 + \frac{2t^2}{(1-t)^2}v$

As $d \to \infty$, the spike contributes negligibly to $\Lambda_t(\theta)$, and we recover the isotropic formula with $\sigma^2 = v$. However, the spike can significantly affect the **planted energy** if the planted trajectory aligns with the principal direction — this connects to the statistical physics of "principal component recovery".

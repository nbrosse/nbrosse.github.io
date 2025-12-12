# Global presentation 

## The marginal FM weights are a Gibbs measure

The marginal field is
$$
u_t(x)=\sum_{i=1}^n \lambda_i(x,t) \frac{x^{(i)}-x}{1-t},\qquad
\lambda_i(x,t)=\frac{\exp\left(-\frac{\|x-tx^{(i)}\|^2}{2(1-t)^2}\right)}{\sum_{j=1}^n \exp\left(-\frac{\|x-tx^{(j)}\|^2}{2(1-t)^2}\right)}.
$$
This is a Boltzmann distribution at inverse temperature $\beta=1$ over "states" $i\in\{1, \dots, n\}$ with energy
$$
E_i(x,t)=\frac{\|x-tx^{(i)}\|^2}{2(1-t)^2}.
$$
The denominator is the **partition function**
$$
Z(x,t)=\sum_{j=1}^n e^{-E_j(x,t)}.
$$
So the question “do the weights collapse?” becomes the stat-phys question:

> Is $Z(x,t)$ dominated by **one special term** (a "condensed/glassy" phase), or by the **entropy** of exponentially many comparable terms (a "liquid" phase)?

## Follow a “teacher” trajectory: one state is special

You consider
$$
x_t=(1-t)x_0+t x_1
$$
with $x_0\sim\mathcal{N}(0,I)$ and $x_1$ a *particular training image* (one of the $x^{(i)}$'s).
Define the **special index** $i=1$ such that $x^{(1)}=x_1$. Then:
$$
x_t-tx^{(1)}=(1-t)x_0
\quad\Rightarrow\quad
E_1(x_t,t)=\frac{\|(1-t)x_0\|^2}{2(1-t)^2}=\frac{\|x_0\|^2}{2}\approx \frac d2.
$$
So the "teacher" contribution is
$$
Z_1 = e^{-E_1}\approx e^{-d/2}.
$$

## The other $n-1$ terms form a Random Energy Model

For $j\neq 1$,
$$
x_t-tx^{(j)}=(1-t)x_0+t(x_1-x^{(j)}).
$$
Under the Gaussian proxy $x^{(j)}\stackrel{iid}{\sim}\mathcal{N}(0,\sigma^2 I_d)$ with $x_1\sim\mathcal{N}(0,\sigma^2 I_d)$, the difference $(x_1-x^{(j)})\sim\mathcal{N}(0,2\sigma^2 I_d)$, independent of $x_0$. Hence
$$
x_t-tx^{(j)}\sim\mathcal{N}\left(0,\big((1-t)^2+2\sigma^2 t^2\big)I_d\right).
$$
Let
$$
s^2(t)=(1-t)^2+2\sigma^2 t^2.
$$
Then the “non-special” energies are approximately
$$
E_j(x_t,t)=\frac{\|x_t-tx^{(j)}\|^2}{2(1-t)^2}\ \approx\ \frac{s^2(t)}{2(1-t)^2},\chi_d^2.
$$
So $E_j$ is a sum of $d$ weakly dependent contributions → **energy is extensive** (order $d$), and across $j$ you have $n-1$ *almost independent extensive energies*: this is the classic setup where **REM tools apply**.

Thus the remaining partition function
$$
Z_{2\ldots n}=\sum_{j=2}^n e^{-E_j}
$$
is a REM-like partition sum: it is *not* governed by CLT; it is governed by **extreme values / large deviations**, and can enter a **glass phase** where a tiny number of lowest energies dominate.

## "Memorization" = condensation onto the special term

The stat-phys criterion is:

* **Collapsed / memorization phase (glass / condensation):** $Z \approx Z_1$, i.e. $\lambda_1 \approx 1$.
* **Uncollapsed / generalization phase (liquid):** $Z \approx Z_{2\ldots n}$, i.e. mass spread over many $j$'s.

In the collapsed phase,
$$
u_t(x_t)\approx \frac{x^{(1)}-x_t}{1-t}=\frac{x_1-x_t}{1-t} \approx x_1-x_0,
$$
so the **marginal field aligns with the conditional field**—which is your observed “cosine similarity rapidly goes to 1 in high dimension”.

## Your “log weight ratio” is a *free-energy gap* estimate

Your hand-wavy ratio
$$
\log\frac{\lambda_j}{\lambda_1}\approx -\frac{2\sigma^2 t^2 d}{2(1-t)^2}
$$
is a stat-phys statement of the form
$$
\log \lambda_1 \approx -E_1 - \log Z
$$
and, comparing a typical other term to the special term,
$$
\log\frac{e^{-E_j}}{e^{-E_1}} = -(E_j-E_1).
$$

Under concentration,

* $E_1 \approx d/2$,
* typical $E_j \approx \frac{s^2(t)}{2(1-t)^2},d$.

So the **typical energy gap** is
$$
E_j-E_1 \approx \frac d2\left(\frac{s^2(t)}{(1-t)^2}-1\right)
= \frac d2\left(\frac{(1-t)^2+2\sigma^2 t^2}{(1-t)^2}-1\right)
= \frac{d,\sigma^2 t^2}{(1-t)^2}.
$$
Thus
$$
\frac{e^{-E_j}}{e^{-E_1}}\approx \exp!\Big(-\frac{d,\sigma^2 t^2}{(1-t)^2}\Big),
$$
which is the same exponential-in-$d$ suppression you computed (up to small constant-factor conventions).

From a stat-phys lens, that's the statement:

> the **energy advantage** of the special state grows like $d$, so at fixed $n$ it will eventually win exponentially hard.

## Why “even summing over $n$” often cannot beat it: entropy vs energy

Now compare **one** special contribution vs **many** others:
$$
Z = Z_1 + Z_{2\ldots n}.
$$

A crude “annealed” entropy estimate would say
$$
Z_{2\ldots n}\approx (n-1), \mathbb E[e^{-E_j}].
$$
but the article stresses that this is often wrong in the relevant scaling because $e^{-E_j}$ is exponentially broad—so the sum is dominated by **rare low-energy states**, i.e. REM physics.

Still, the *key control parameter* is the same as in the article:
$$
\alpha=\frac{1}{d}\log n.
$$
You can read your conclusion

> “even $n=50,000$ doesn’t recover it — non-stochastic regime dominates very early”

as the stat-phys statement:

> with $\alpha \ll 1$ (since $d=3072$ and $\log n \approx 10.8$, so $\alpha \approx 0.0035$), the entropy budget $\log n$ is tiny compared to the extensive energy gaps $\Theta(d)$.
> Therefore the system is deep in the **condensed / glassy** regime unless $t$ is extremely small.

## “Collapse time” $t_C$: the transition where $Z_1$ stops dominating

The article defines a collapse time $t_C$ where dominance swaps:

* for $t<t_C$, $Z\approx Z_1$ (collapse/memorization),
* for $t>t_C$, $Z\approx Z_{2\ldots n}$ (uncollapsed/generalization).

In your FM setting, $t_C$ is the time when the **best achievable** contribution among the $n-1$ random competitors becomes comparable to the special energy $E_1\approx d/2$.

REM logic: the minimum energy among $n$ extensive random energies satisfies an extreme-value principle
$$
E_{\min}(t)\approx d,e_*(t,\alpha),
$$
and the phase boundary is roughly
$$
E_{\min}(t_C)\approx E_1 \approx d/2.
$$

Your simpler (but very intuitive) calculation essentially assumes “typical competitor” rather than “best competitor”, yielding a sufficient condition for collapse:
$$
\log(n)\ \ll\ \frac{d,\sigma^2 t^2}{(1-t)^2}.
$$
Rearranged, collapse holds whenever
$$
\frac{t}{1-t}\ \gg\ \sqrt{\frac{\log n}{d,\sigma^2}}.
$$
For CIFAR-like numbers, $\log n/d$ is so small that the RHS is tiny, so collapse can happen already at small $t$. This is the same qualitative message as the article’s explicit $t_C$ formula: **$t_C$ is $O(1)$ only if $\log n$ is $O(d)$**.














# Rigorous derivation using Random Energy Model and Statistical Physics (rapid version)

## Setup as a disordered system

Fix a time $t\in(0,1)$ and consider a point on a “planted” trajectory
$$
x_t = (1-t)x_0 + t x_1,
\qquad x_0\sim\mathcal{N}(0,I_d),\quad x_1 = x^{(1)}\sim\mathcal{N}(0,\sigma^2 I_d).
$$
The dataset $\{x^{(i)\}_{i=1}^n$ are i.i.d. $\mathcal{N}(0,\sigma^2 I_d)$. Condition on $(x_0,x_1)$ and view the remaining $x^{(i)}$ for $i\ge 2$ as “quenched disorder”.

Define energies
$$
E_i \equiv E_i(x_t,t)=\frac{|x_t - t x^{(i)}|^2}{2(1-t)^2},
\qquad i=1,\dots,n,
$$
and partition function
$$
Z = \sum_{i=1}^n e^{-E_i}.
$$
Then $\lambda_i = e^{-E_i}/Z$ is a Gibbs measure at inverse temperature $\beta=1$.

### “Planted” (special) energy

For $i=1$, $x^{(1)}=x_1$, so
$$
x_t - t x^{(1)} = (1-t)x_0 ;\Rightarrow; E_1 = \frac{|x_0|^2}{2}.
$$
Thus
$$
\frac{E_1}{d}\to \frac12 \quad \text{a.s. as } d\to\infty
$$
(by LLN / chi-square concentration).

### “Bulk” (REM-like) energies for (i\ge 2)

For $i\ge2$,
$$
x_t - t x^{(i)} = (1-t)x_0 + t(x_1-x^{(i)}).
$$
Conditioned on $(x_0,x_1)$, the randomness is only in $x^{(i)}$. Since $x^{(i)}\sim \mathcal{N}(0,\sigma^2 I_d)$,
$$
x_1-x^{(i)}\sim \mathcal N(x_1,\sigma^2 I_d)\quad\text{(conditioned on }x_1).
$$
So $x_t-tx^{(i)}$ is Gaussian with mean $(1-t)x_0 + t x_1$ and covariance $t^2\sigma^2 I_d$. If you *uncondition* over $(x_0,x_1)$ you get a centered Gaussian with covariance
$$
\Sigma(t)=\big((1-t)^2 + 2\sigma^2 t^2\big)I_d
$$
(the approximation you used). For REM-style asymptotics, what matters is the **energy-per-dimension LDP**, which is stable under these small conditioning effects because $|x_0|^2/d$ and $|x_1|^2/d$ concentrate.

So, to robustify, we take the standard REM surrogate:
$$
E_i \stackrel{\text{approx}}{=} \frac{s^2(t)}{2(1-t)^2},\chi^2_d,
\qquad s^2(t):=(1-t)^2+2\sigma^2 t^2,
\qquad i\ge2,
$$
with the $E_i$ i.i.d. across $i\ge2$. This “i.i.d. energy levels” assumption is the REM idealization. (In the true geometry there are weak correlations; in high dimension and for i.i.d. Gaussian data, these correlations typically do not change the leading-order free energy unless you enter a strongly structured regime.)

Let
$$
c(t):=\frac{s^2(t)}{2(1-t)^2}=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
$$
Then for $i\ge2$,
$$
\frac{E_i}{d}\approx c(t),\frac{\chi^2_d}{d}.
$$

## 2) Large deviations for energy density

Define the energy density
[
e := \frac{E}{d}.
]
For (Y=\chi^2_d/d), Cramér’s theorem gives an LDP:
[
\mathbb P(Y\approx y)\asymp e^{-d I_{\chi}(y)},
\quad y>0,
\quad I_{\chi}(y)=\frac12\big(y-1-\log y\big).
]
Since (e=c,Y), the energy density has LDP
[
\mathbb P(e\approx \varepsilon)\asymp e^{-d I(\varepsilon)},
\qquad
I(\varepsilon)=I_{\chi}!\Big(\frac{\varepsilon}{c}\Big)
=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big),
\quad \varepsilon>0.
]
This is the rigorous backbone you want: it replaces “handwavy concentration” with a controlled rate function.

## 3) Quenched free energy of REM at (\beta=1)

Let (n = e^{\alpha d}) where (\alpha=\frac1d\log n) (for CIFAR: (\alpha\ll1)). In REM theory, the quenched free energy density is
$$
f(\beta) = -\lim_{d\to\infty}\frac1d \log Z_{\text{bulk}}(\beta),
\qquad
Z_{\text{bulk}}(\beta):=\sum_{i=2}^n e^{-\beta E_i}.
$$
A standard REM derivation (Varadhan + counting states at given energy) yields
$$
\lim_{d\to\infty}\frac1d\log Z_{\text{bulk}}(\beta)
= \sup_{\varepsilon>0}\big[\alpha - I(\varepsilon) - \beta \varepsilon\big].
$$
Call the supremum value (\Phi(\beta)). Then (f(\beta)=-\Phi(\beta)).

### Interior (“liquid”) solution

If the maximizer is interior, it satisfies
[
\frac{d}{d\varepsilon}\big[-I(\varepsilon)-\beta\varepsilon\big]=0
\quad\Rightarrow\quad
I'(\varepsilon) = -\beta.
]
Compute
[
I'(\varepsilon)=\frac12\Big(\frac1c - \frac1\varepsilon\Big).
]
So
[
\frac12\Big(\frac1c - \frac1{\varepsilon_*}\Big) = -\beta
;\Rightarrow;
\frac1{\varepsilon_*}=\frac1c+2\beta
;\Rightarrow;
\boxed{;\varepsilon_*(\beta)=\frac{c}{1+2\beta c};}
]
and at (\beta=1):
[
\boxed{;\varepsilon_*(1)=\frac{c}{1+2c};}
]
This is the energy density that dominates the bulk partition sum in the high-temperature/liquid regime.

### Boundary (“glass”) solution and the freezing transition

The interior solution is valid only if there are exponentially many states around (\varepsilon_*), i.e.
[
\alpha - I(\varepsilon_*) \ge 0.
]
If (\alpha - I(\varepsilon_*)<0), there are too few states at that energy; the supremum hits the boundary of the allowed energy region where the **complexity** vanishes:
[
\varepsilon_g \text{ solves } I(\varepsilon_g)=\alpha,
]
and the partition sum is dominated by the lowest energies (extremes): glass/condensed phase.

So the REM “robustified” phase boundary is:
[
\boxed{\text{liquid if } \alpha\ge I(\varepsilon_*(1)), \quad
\text{glass if } \alpha< I(\varepsilon_*(1)).}
]

Let’s write the explicit condition at (\beta=1). Plug (\varepsilon_*(1)=\frac{c}{1+2c}) into (I):
[
\frac{\varepsilon_*}{c}=\frac1{1+2c}.
]
Then
[
I(\varepsilon_*)=\frac12\Big(\frac1{1+2c}-1-\log\frac1{1+2c}\Big)
=\frac12\Big(\log(1+2c)-\frac{2c}{1+2c}\Big).
]
Thus the **REM freezing threshold** at (\beta=1) is
[
\boxed{;\alpha_c(c)=\frac12\Big(\log(1+2c)-\frac{2c}{1+2c}\Big).;}
]
If (\alpha<\alpha_c(c)), the bulk is glassy (dominated by a few extremes); if (\alpha>\alpha_c(c)), it is liquid.

This is already substantially more rigorous than the article’s “(t_C) formula”, because it is derived from the exact LDP rate function of (\chi^2).

---

## 4) Now incorporate the planted state: collapse vs non-collapse of (\lambda_1)

So far we have described (Z_{\text{bulk}}=\sum_{i=2}^n e^{-E_i}). The full partition function is
[
Z = e^{-E_1} + Z_{\text{bulk}}.
]
The question “does (\lambda_1\to 1)?” is:
[
\lambda_1 = \frac{e^{-E_1}}{e^{-E_1}+Z_{\text{bulk}}}
\quad\Longrightarrow\quad
\lambda_1\to 1 \iff \frac{1}{d}\log Z_{\text{bulk}} < -\frac{E_1}{d}.
]
Since (E_1/d\to 1/2), the **collapse condition** becomes
[
\boxed{;\Phi(1) < -\frac12;}
\quad\text{where}\quad
\Phi(1)=\lim_{d\to\infty}\frac1d\log Z_{\text{bulk}}(1)
=\sup_{\varepsilon>0}\big[\alpha-I(\varepsilon)-\varepsilon\big].
]

### Evaluate (\Phi(1)) in both phases

**Liquid (interior) phase**: maximizer (\varepsilon_*=\frac{c}{1+2c}). Then
[
\Phi(1)=\alpha - I(\varepsilon_*) - \varepsilon_*.
]
You can simplify using the explicit (I(\varepsilon_*)) above.

**Glass (boundary) phase**: maximizer (\varepsilon_g) solves (I(\varepsilon_g)=\alpha). Then
[
\Phi(1)=\alpha - I(\varepsilon_g) - \varepsilon_g = -\varepsilon_g.
]
So in the glass phase,
[
\boxed{;\frac1d\log Z_{\text{bulk}} \to -\varepsilon_g,\qquad I(\varepsilon_g)=\alpha.;}
]

### Collapse criterion in the glass phase (clean!)

In the glass phase, collapse is simply
[
-\varepsilon_g < -\frac12 \iff \varepsilon_g > \frac12.
]
So:
[
\boxed{\text{In the glass phase, }\lambda_1\to 1 \iff \varepsilon_g>\tfrac12.}
]
And (\varepsilon_g) is defined *implicitly* by
[
\frac12\Big(\frac{\varepsilon_g}{c}-1-\log\frac{\varepsilon_g}{c}\Big)=\alpha.
]
This is a rigorous REM-level relation between dimension, dataset size, and time (t) (through (c(t))).

### Collapse criterion in the liquid phase

In the liquid phase, compare (\Phi(1)) to (-1/2):
[
\boxed{;\lambda_1\to 1 \iff \alpha - I(\varepsilon_*) - \varepsilon_* < -\frac12.;}
]
This can be turned into an explicit inequality in (c) and (\alpha). It’s messier but explicit.

---

## 5) Interpreting “collapse time” (t_C) properly

Your earlier handwave essentially compared **typical** bulk energy to planted energy. REM says: don’t do typical—do **extremes**, i.e. (\varepsilon_g) solving (I(\varepsilon_g)=\alpha).

So a principled “collapse time” is obtained by solving:
[
\varepsilon_g(t_C)=\frac12,
\qquad\text{with}\qquad
I(\varepsilon_g(t_C))=\alpha,\quad c=c(t_C).
]
Set (\varepsilon_g=\frac12) in (I(\varepsilon_g)=\alpha):
[
\alpha = I(1/2)=\frac12\Big(\frac{1/2}{c}-1-\log\frac{1/2}{c}\Big)
=\frac12\Big(\frac{1}{2c}-1-\log\frac{1}{2c}\Big)
=\frac12\Big(\log(2c)+\frac{1}{2c}-1\Big).
]
So the REM-predicted boundary for “planted wins vs best competitor wins” is:
[
\boxed{;\alpha = \frac12\Big(\log(2c(t_C))+\frac{1}{2c(t_C)}-1\Big).;}
]
Remember
[
c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
]
This is already much more robust than the article’s formula because it comes from the correct LDP rate function, and because it compares the planted state to the **best** of the (n-1) competitors (via (I(\cdot)=\alpha)).

---

## 6) What REM predicts qualitatively for CIFAR-like (\alpha\ll 1)

For CIFAR-10: (d=3072), (n\approx 5\times 10^4), so
[
\alpha=\frac{\log n}{d}\approx \frac{10.82}{3072}\approx 0.00352.
]
That is tiny. In REM terms, tiny (\alpha) means the “entropy” budget is extremely small compared to extensive energies.

Two consequences that align with your observation:

1. **Bulk is often glassy**: because (\alpha) is small, the condition (\alpha<\alpha_c(c)) is frequently satisfied for moderate (c). That means (Z_{\text{bulk}}) is dominated by a few extremes rather than a democratic sum.

2. **Planted state can dominate early**: the planted energy density is pinned at (1/2). If the minimal bulk energy density (\varepsilon_g(t)) is above (1/2), then (e^{-E_1}) beats even the best competitor and (\lambda_1\to 1). As (t) changes, (c(t)) changes, shifting the bulk energy distribution and hence (\varepsilon_g(t)).

So the “rapid alignment” phenomenon becomes:

> in high (d), with (\alpha\ll1), the Gibbs measure condenses onto the planted state for a wide range of (t), because even the lowest-energy spurious matches don’t beat the planted match.

---

## 7) What is still “approximate” (and how to further robustify)

REM rigor here hinges on two approximations:

### (A) Independence of energy levels (E_i)

In the true geometric model, energies across (i) are weakly correlated through the shared (x_t). For i.i.d. Gaussian data, these correlations are typically (O(1/d)) in overlaps, and REM predictions for free energy often remain correct at leading order (this is the same justification used in many high-d nearest-neighbor/softmax analyses). But proving it requires a “correlated REM” or GREM-type argument.

**Robustification path:** show an LDP for the empirical measure of energies conditional on (x_t), then apply a concentration result for (\frac1d\log Z). Often you can prove that quenched free energy converges to the same variational formula as the i.i.d. REM.

### (B) Gaussian proxy for CIFAR

Your (\mathcal N(0,\sigma^2 I)) is a proxy. If CIFAR has nontrivial covariance, you replace (\chi^2) by a quadratic form with spectrum of covariance; the LDP changes by replacing (I_{\chi}) with a rate function depending on the eigenvalue distribution.

**Robustification path:** diagonalize covariance (C) and use known LDPs for (\frac1d|g|^2_C) (or apply the Gärtner–Ellis theorem via log-MGF). The REM variational structure remains the same; only (I(\cdot)) changes.

---

## 8) Minimal “deliverable” summary: the REM-correct equations you can use

If you want a clean stat-phys section in your post, the robust REM statements are:

* Bulk energies have LDP:
  [
  \mathbb P!\left(\frac{E}{d}\approx \varepsilon\right)\asymp e^{-d I(\varepsilon)},
  \quad
  I(\varepsilon)=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big),
  \quad
  c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
  ]

* Bulk quenched free energy at (\beta=1):
  [
  \Phi(1)=\lim_{d\to\infty}\frac1d\log Z_{\text{bulk}}(1)
  =\sup_{\varepsilon>0}\big[\alpha-I(\varepsilon)-\varepsilon\big],
  \quad \alpha=\frac1d\log n.
  ]

* Glass vs liquid in the bulk:
  [
  \alpha_c(c)=\frac12\Big(\log(1+2c)-\frac{2c}{1+2c}\Big).
  ]
  If (\alpha<\alpha_c(c)): glass, maximizer at (\varepsilon_g) with (I(\varepsilon_g)=\alpha), and (\Phi(1)=-\varepsilon_g).
  If (\alpha>\alpha_c(c)): liquid, maximizer at (\varepsilon_*=\frac{c}{1+2c}).

* Planted-vs-bulk collapse criterion:
  [
  \lambda_1\to 1 \iff \Phi(1)<-\frac12.
  ]
  In the glass phase this reduces to:
  [
  \lambda_1\to 1 \iff \varepsilon_g>\frac12,\quad I(\varepsilon_g)=\alpha.
  ]
  A “collapse time” is implicitly:
  [
  \alpha = I(1/2)=\frac12\Big(\log(2c(t_C))+\frac{1}{2c(t_C)}-1\Big).
  ]

That’s the REM-theory version of your softmax-collapse argument, with the correct quenched/extreme-value control replacing the article’s (and your earlier) typical-point heuristics.





















# Detailed LDP derivation

## 1. What problem are we trying to solve?

You keep seeing statements like
[
\mathbb P(Y \approx y) \asymp e^{-d I(y)}
]
and words like **LDP**, **rate function**, **Cramér’s theorem**.

What is the *actual question* behind this?

> We have a random quantity built from **many dimensions** (here (d)).
> We want to know: *how unlikely is it that it deviates from its typical value?*

Not just *that* it concentrates — but **how fast** the probability decays.

This is what **Large Deviation Theory** answers.

## 2. Law of Large Numbers vs Large Deviations

Let’s start with something familiar.

Let
$$
Z_1, Z_2, \dots, Z_d \sim \mathcal N(0,1) \quad \text{i.i.d.}
$$

Define the empirical average:
[
\bar Z_d = \frac{1}{d}\sum_{k=1}^d Z_k
]

### Law of Large Numbers (LLN)

LLN says:
[
\bar Z_d \xrightarrow{a.s.} 0
]

That tells you **where it concentrates**, but not **how unlikely deviations are**.

## 3. What is a Large Deviation Principle (LDP)?

An **LDP** answers:

> If (\bar Z_d) is *not* close to 0, how small is that probability?

### Informal definition

A sequence of random variables (X_d) satisfies a **Large Deviation Principle** with rate function (I(x)) if
$$
\mathbb{P}(X_d \approx x) \approx e^{-d I(x)}
$$

* (d) is the **large parameter** (dimension, number of samples, etc.)
* (I(x)\ge 0)
* (I(x)=0) at the *typical value*
* Larger (I(x)) = exponentially rarer event

This is *much* stronger than concentration inequalities.

## 4. Cramér’s Theorem (core result)

**Cramér’s theorem** is the fundamental LDP for averages of i.i.d. variables.

### Theorem (informal)

Let (X_1,\dots,X_d) be i.i.d. with finite moment generating function.
Define
$$
S_d = \frac{1}{d}\sum_{k=1}^d X_k
$$

Then (S_d) satisfies an LDP:
$$
\mathbb P(S_d \approx x) \asymp e^{-d I(x)}
$$

where the **rate function**
$$
I(x) = \sup_{\lambda \in \mathbb R} \big( \lambda x - \log \mathbb E[e^{\lambda X_1}] \big)
$$

This is a **Legendre transform** of the log-MGF.

## 5. Apply Cramér’s theorem to (\chi^2)

Now let’s connect to *your* case.

### Step 1: Express the quantity as an average

Let
$$
Z_k^2 \sim \chi^2_1
$$

Then
$$
\chi^2_d = \sum_{k=1}^d Z_k^2
\quad\Rightarrow\quad
Y_d := \frac{1}{d}\chi^2_d = \frac{1}{d}\sum_{k=1}^d Z_k^2
$$

This is **exactly** an empirical average.

### Step 2: Compute the rate function

For $X = Z^2$, the log-MGF is:
$$
\log \mathbb E[e^{\lambda Z^2}] = -\frac12 \log(1-2\lambda)
\quad (\lambda < 1/2)
$$

Apply Cramér’s theorem:
$$
I_\chi(y) = \sup_{\lambda<1/2}
\Big( \lambda y + \tfrac12\log(1-2\lambda) \Big)
$$

Solving this gives:
$$
\boxed{
I_\chi(y) = \frac12\big(y - 1 - \log y\big),
\quad y>0
}
$$

### Step 3: What does this mean?

It means:
$$
\boxed{
\mathbb P!\left(\frac{1}{d}\chi^2_d \approx y\right)
\asymp e^{-d I_\chi(y)}
}
$$

Key properties:

| Property      | Explanation                   |
| ------------- | ----------------------------- |
| Minimum       | (I_\chi(1)=0)                 |
| Typical value | (y=1)                         |
| Deviations    | exponentially unlikely in (d) |
| Shape         | convex                        |

This is **far stronger** than “relative fluctuations are (O(1/\sqrt d))”.

## 6. Why LDP beats “concentration of measure”

Concentration tells you:
$$
\mathbb P(|Y_d - 1| > \varepsilon) \le e^{-c d}
$$

LDP tells you:
$$
\mathbb P(Y_d \approx y) \approx e^{-d I(y)}
$$

That means:

* You can compare **two rare events**
* You can find **which rare event dominates**
* You can compute **free energies and phase transitions**

This is why statistical physics *needs* LDPs.

## 7. Applying the LDP to your energy

You define:
$$
E = c ,\chi^2_d
\quad\Rightarrow\quad
e := \frac{E}{d} = c,Y_d
$$

Change of variables:
$$
\mathbb P(e \approx \varepsilon)
= \mathbb P!\left(Y_d \approx \frac{\varepsilon}{c}\right)
\asymp e^{-d I_\chi(\varepsilon/c)}
$$

So the **energy density LDP** is:
$$
\boxed{
\mathbb P(e \approx \varepsilon)
\asymp e^{-d I(\varepsilon)},
\qquad
I(\varepsilon) = \frac12\Big(\frac{\varepsilon}{c} - 1 - \log\frac{\varepsilon}{c}\Big)
}
$$

This is the **rigorous backbone** behind all REM arguments.

## 8. Why this matters for REM and softmax collapse

You have (n = e^{\alpha d}) energies.

The expected number of energies near (\varepsilon) is:
$$
\text{count}(\varepsilon)
\approx e^{d(\alpha - I(\varepsilon))}
$$

So:

* If (\alpha - I(\varepsilon) > 0): exponentially many states
* If (\alpha - I(\varepsilon) = 0): *edge of support*
* If (\alpha - I(\varepsilon) < 0): no states w.h.p.

This is how **glass transitions**, **condensation**, and **memorization** emerge *mathematically*.

## 9. One-sentence intuition

> **An LDP says:**
> “In high dimension, probabilities are governed by exponential rates, and whoever has the *smallest exponent* wins.”

That’s exactly why:

* one training image dominates the softmax,
* the marginal flow aligns with the conditional flow,
* memorization appears early in (t).

## 10. If you want the next step

Next, the natural continuation is:

1. Use the LDP to compute the **minimum energy** among (n=e^{\alpha d}) samples
2. Compare it to the planted energy (1/2)
3. Derive collapse time (t_C) cleanly

If you want, I can do that *slowly*, without assuming REM knowledge.

Just say:
**“Derive the minimum energy using LDP.”**














# Derivation LDP minimum energy

## 1) Objects and scaling

We have energies
$$
E_i = \frac{|x_t - t x^{(i)}|^2}{2(1-t)^2},\qquad \lambda_i=\frac{e^{-E_i}}{\sum_{j=1}^n e^{-E_j}}.
$$

Along the planted trajectory $x_t=(1-t)x_0+t x_1$ with $x_1=x^{(1)}$:

* **Planted energy**
  $$
  E_1 = \frac{|x_0|^2}{2}\quad\Rightarrow\quad \frac{E_1}{d}\xrightarrow{} \frac12 \quad \text{(concentration)}.
  $$
  Call
  $$
  e_{\text{pl}}:=\frac{E_1}{d}\approx \frac12.
  $$

* **Bulk energies** $i\ge2$ have energy densities $e_i=E_i/d$ satisfying an LDP with rate function
  $$
  I(\varepsilon)=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big),\qquad
  c=c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
  $$

Dataset size is $n=e^{\alpha d}$, $\alpha=\frac{1}{d}\log n$.

From the previous step, the **minimum bulk energy density**
$$
m_d:=\min_{i\ge2} e_i
$$
converges in probability to (\varepsilon_\star) solving
$$
\boxed{I(\varepsilon_\star)=\alpha.}
$$

## 2) What does “collapse” mean?

Collapse means the planted weight goes to 1:
$$
\lambda_1 \to 1.
$$

Since
[
\lambda_1 = \frac{e^{-E_1}}{e^{-E_1}+\sum_{i\ge2} e^{-E_i}},
]
this happens if the bulk sum is exponentially smaller than (e^{-E_1}).

Take logs and divide by (d). Define the bulk partition function
[
Z_{\text{bulk}}:=\sum_{i\ge2} e^{-E_i}.
]

Then
[
\lambda_1\to 1 \quad\Longleftrightarrow\quad \frac{1}{d}\log Z_{\text{bulk}} < -\frac{E_1}{d} \approx -\frac12.
]

So we need to understand the exponential scale of (Z_{\text{bulk}}).

## 3) In the *collapsed/extreme-dominated* regime, (Z_{\text{bulk}}) is controlled by the minimum

A very robust inequality is:
$$
e^{-d,m_d}\le Z_{\text{bulk}} \le (n-1),e^{-d,m_d}.
$$
because every term is at most (e^{-d m_d}), and at least one term equals it.

Take (\frac1d\log):
$$
-m_d ;\le; \frac1d\log Z_{\text{bulk}} ;\le; -m_d + \frac1d\log(n-1).
$$
But (\frac1d\log(n-1)\to \alpha). Hence asymptotically:
$$
\frac1d\log Z_{\text{bulk}} \in [-m_d,,-m_d+\alpha].
$$

Since (m_d\to \varepsilon_\star), we get the exponential scale:
$$
\frac1d\log Z_{\text{bulk}} \approx -\varepsilon_\star \quad \text{(up to an additive }\alpha\text{)}.
$$

This already yields a clean **sufficient condition** for collapse.

## 4) Collapse criterion from the minimum energy

### Sufficient condition (and often essentially sharp when extremes dominate)

If even the **best spurious competitor** is worse (higher energy) than the planted state:
$$
\varepsilon_\star > e_{\text{pl}}=\frac12,
$$
then the smallest bulk term is smaller than the planted term exponentially:
$$
e^{-d\varepsilon_\star} \ll e^{-d/2}.
$$
And since all other bulk terms are even smaller, the bulk sum cannot catch up (unless (\alpha) is huge).

So you get:
$$
\boxed{\text{Strong collapse (planted dominates) if }\varepsilon_\star(t)>\tfrac12.}
$$

Conversely, if (\varepsilon_\star<\tfrac12), then there typically exists a bulk point with energy below the planted energy, meaning the planted state cannot dominate universally:
$$
\boxed{\text{No planted dominance if }\varepsilon_\star(t)<\tfrac12.}
$$

So the boundary is (\varepsilon_\star(t)=1/2).

## 5) Turn that into an explicit equation in (t) (the “collapse time”)

We know (\varepsilon_\star) is defined implicitly by:
$$
I(\varepsilon_\star)=\alpha,
\qquad
I(\varepsilon)=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big).
$$

Set (\varepsilon_\star=\frac12). Then the **boundary** satisfies:
$$
\alpha = I(1/2)=\frac12\Big(\frac{1/2}{c}-1-\log\frac{1/2}{c}\Big).
$$
Simplify:
$$
\frac{1/2}{c}=\frac{1}{2c},\qquad \log\frac{1/2}{c}=\log\frac{1}{2c}=-\log(2c).
$$
So:
$$
\boxed{\alpha = \frac12\Big(\log(2c)+\frac{1}{2c}-1\Big).}
$$
Recall
$$
c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
$$

This is your REM-rigorous “collapse curve” (in the Gaussian surrogate).

## 6) Intuition: energy vs entropy in one line

* The planted contribution is $\sim e^{-d/2}$.
* The best spurious competitor among $n=e^{\alpha d}$ has energy density $\varepsilon_\star$ where its rarity $e^{-d I(\varepsilon_\star)}$ matches the number of trials $e^{\alpha d}$, hence $I(\varepsilon_\star)=\alpha$.
* Collapse happens if the *best* competitor is still worse than planted: $\varepsilon_\star>1/2$.

That’s the principled replacement for “a typical point has exponent $-47$” arguments.

## 7) What this predicts for CIFAR-like numbers (qualitatively)

For CIFAR-10, $\alpha=\frac{\log n}{d}$ is tiny, so solving $I(\varepsilon_\star)=\alpha$ gives $\varepsilon_\star$ **very close to the typical bulk energy**, meaning even the minimum doesn’t drop that much below typical. As $t$ increases, $c(t)$ increases, shifting energies upward in a way that can easily keep $\varepsilon_\star(t)$ above $1/2$ for small-to-moderate $t$. That’s the “early collapse” phenomenon.


If you want, next I can do one of two things (both are useful):

1. **Compute / approximate $t_C$** explicitly for your CIFAR proxy $(d=3072,\ n=50k,\ \sigma^2=0.24)$ using the equation above.
2. Go one level deeper and compute the full bulk free energy $\Phi(1)=\sup_\varepsilon[\alpha-I(\varepsilon)-\varepsilon]$ and derive collapse without relying on “min dominates” (this is what you want if you suspect the bulk is not always purely extreme-dominated).







# CIFAR-10 proxy numbers


Using the REM/LDP boundary we derived, the “collapse time” (t_c) (where the best spurious match among (n) competitors becomes as good as the planted one) is defined by

$$
\varepsilon_\star(t_c)=\frac12
\quad\Longleftrightarrow\quad
\alpha ;=; I(1/2;c(t_c)),
\qquad \alpha=\frac{\log n}{d},
$$
with
$$
I(\varepsilon;c)=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big),
\qquad
c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big).
$$

Plugging (\varepsilon=\tfrac12) simplifies the equation to
$$
\boxed{;\alpha = \frac12\Big(\log(2c(t_c))+\frac{1}{2c(t_c)}-1\Big).;}
$$

## CIFAR-10 proxy numbers

* (d=3072)
* (n=50{,}000)
* (\sigma^2=0.24)

Compute
$$
\alpha=\frac{\log(50{,}000)}{3072}\approx \frac{10.8198}{3072}\approx 0.003522.
$$

Solving
$$
\frac12\Big(\log(2c(t))+\frac{1}{2c(t)}-1\Big)=0.003522
$$
with (c(t)=\frac12\Big(1+\frac{2\cdot 0.24, t^2}{(1-t)^2}\Big))
gives

$$
\boxed{t_c \approx 0.341.}
$$

At this (t_c), the corresponding
$$
c(t_c)\approx 0.5644.
$$

**Interpretation:** under the i.i.d. Gaussian CIFAR proxy + REM independence idealization, for (t \gtrsim 0.34) the planted datapoint is no longer guaranteed to dominate because among (n) training points you can typically find at least one “spurious” point whose energy beats (or matches) the planted energy (E_1/d\approx 1/2).













# Bulk free energy


Cool — this is the right “one level deeper”: instead of saying “the minimum dominates”, we compute the **quenched bulk free energy**
[
\Phi(1);=;\lim_{d\to\infty}\frac1d\log Z_{\text{bulk}},\qquad
Z_{\text{bulk}}=\sum_{i=2}^n e^{-E_i},
]
using the **LDP + Laplace principle** (the standard REM route), and *then* derive collapse by comparing (\Phi(1)) to the planted energy density (1/2).

I’ll keep the same Gaussian surrogate as in your doc:
[
E_i \approx c(t),\chi_d^2,\qquad
c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big),
\qquad n=e^{\alpha d},;\alpha=\frac{\log n}{d}.
]

---

## 1) Bulk energy LDP (input)

Let the **energy density** be (e=E/d). For (\chi^2_d/d), Cramér’s theorem gives an LDP:
[
\mathbb P!\left(\frac{\chi_d^2}{d}\approx y\right)\asymp e^{-d,I_\chi(y)},
\qquad
I_\chi(y)=\frac12\big(y-1-\log y\big).
]
Since (e=c,(\chi_d^2/d)), by change of variables the bulk energy density satisfies
[
\mathbb P(e\approx \varepsilon)\asymp e^{-d,I(\varepsilon)},
\qquad
I(\varepsilon)=I_\chi(\varepsilon/c)
=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big).
]

This rate function is the “disorder law” of the REM.

---

## 2) From LDP to the bulk free energy via Laplace principle

The REM logic is:

* The expected **number of states** with energy density near (\varepsilon) is
  [
  #(\varepsilon)\approx n\cdot \mathbb P(e\approx \varepsilon)
  \approx \exp{d(\alpha-I(\varepsilon))}.
  ]
  The quantity
  [
  \Sigma(\varepsilon):=\alpha-I(\varepsilon)
  ]
  is called the **complexity** (entropy density of states at energy (\varepsilon)).

* The bulk partition function is approximately an integral over energies:
  [
  Z_{\text{bulk}}=\sum_{i=2}^n e^{-E_i}
  \approx \int d\varepsilon;\exp{d(\Sigma(\varepsilon)-\varepsilon)}
  =\int d\varepsilon;\exp{d(\alpha-I(\varepsilon)-\varepsilon)}.
  ]

* By the Laplace principle (large-(d) saddle point),
  [
  \boxed{;\Phi(1)
  :=\lim_{d\to\infty}\frac1d\log Z_{\text{bulk}}
  =\sup_{\varepsilon>0}\big[\alpha-I(\varepsilon)-\varepsilon\big].;}
  ]

This is the “full bulk free energy” statement you asked for.

---

## 3) Solve the variational problem: liquid vs glass

Define
[
F(\varepsilon)=\alpha-I(\varepsilon)-\varepsilon.
]
If the maximizer is interior, it satisfies (F'(\varepsilon)=0), i.e.
[
I'(\varepsilon)=-1.
]
For our (I),
[
I'(\varepsilon)=\frac12\Big(\frac1c-\frac1\varepsilon\Big).
]
So
[
\frac12\Big(\frac1c-\frac1{\varepsilon_*}\Big)=-1
\quad\Rightarrow\quad
\boxed{;\varepsilon_*=\frac{c}{1+2c};}
]
(the “liquid saddle”).

But this saddle is valid only if there are exponentially many states there, i.e. (\Sigma(\varepsilon_*)=\alpha-I(\varepsilon_*)\ge 0).
Define the critical entropy level
[
\boxed{;\alpha_c(c):=I(\varepsilon_*)=
\frac12\Big(\log(1+2c)-\frac{2c}{1+2c}\Big).;}
]

Then REM theory gives the quenched free energy:

### Liquid phase ((\alpha\ge \alpha_c(c)))

[
\boxed{;\Phi(1)=\alpha-I(\varepsilon_*)-\varepsilon_*.;}
]

### Glass phase ((\alpha<\alpha_c(c)))

The supremum hits the boundary where (\Sigma(\varepsilon)=0), i.e. (I(\varepsilon_g)=\alpha). Then
[
\boxed{;\Phi(1)= -\varepsilon_g,\qquad I(\varepsilon_g)=\alpha.;}
]
This is “freezing”: the partition function is dominated by a vanishing fraction of lowest-energy states.

---

## 4) Collapse criterion using the free energy (no “min dominates” assumption)

The full partition function is
[
Z = e^{-E_1}+Z_{\text{bulk}},
\qquad \lambda_1=\frac{e^{-E_1}}{Z}.
]
Along the planted trajectory,
[
\frac{E_1}{d}\to \frac12.
]
So asymptotically
[
\frac1d\log\frac{Z_{\text{bulk}}}{e^{-E_1}}
\to \Phi(1)+\frac12.
]
Hence:
[
\boxed{;\lambda_1\to 1 \iff \Phi(1)<-\frac12.;}
]
More sharply, for large (d),
[
\lambda_1 \approx \frac{1}{1+\exp{d(\Phi(1)+1/2)}}.
]

This is the rigorous “free energy vs planted energy” criterion.

---

## 5) Now plug CIFAR numbers and compute (\Phi(1,t))

CIFAR proxy: (d=3072,; n=50{,}000,; \sigma^2=0.24).
[
\alpha=\frac{\log n}{d}=\frac{\log(50{,}000)}{3072}\approx 0.003522.
]

For these numbers, **for all (t\in[0,1))** we have (\alpha \ll \alpha_c(c(t))) (e.g. (\alpha_c\approx 0.0966) already at (t=0)), so we are *always in the glass phase*.

So:
[
\Phi(1,t)=-\varepsilon_g(t),
\qquad I(\varepsilon_g(t);c(t))=\alpha.
]

It’s convenient to solve in terms of (y:=\varepsilon_g/c). Since
[
I(\varepsilon)=\frac12\Big(\frac{\varepsilon}{c}-1-\log\frac{\varepsilon}{c}\Big)
=\frac12(y-1-\log y),
]
the equation (I(\varepsilon_g)=\alpha) becomes
[
\boxed{;\frac12\big(y_\alpha-1-\log y_\alpha\big)=\alpha,\qquad \varepsilon_g(t)=c(t),y_\alpha.;}
]
So (y_\alpha) depends only on (\alpha), not on (t).

For (\alpha\approx 0.003522), solving gives
[
y_\alpha \approx 0.885955.
]
Therefore
[
\boxed{;\Phi(1,t)= -c(t),y_\alpha \approx -0.885955;c(t).;}
]

A few values:

* (t=0): (c=0.5\Rightarrow \Phi\approx -0.44298)
* (t=0.3): (c\approx 0.5441\Rightarrow \Phi\approx -0.4820)
* (t=0.35): (c\approx 0.5696\Rightarrow \Phi\approx -0.5046)

---

## 6) Collapse time (t_c) from the free energy

Collapse boundary is (\Phi(1,t_c)=-1/2), i.e.
[
c(t_c),y_\alpha=\frac12
\quad\Rightarrow\quad
c(t_c)=\frac{1}{2y_\alpha}\approx \frac{1}{2\times 0.885955}\approx 0.56436.
]
With
[
c(t)=\frac12\Big(1+\frac{2\sigma^2 t^2}{(1-t)^2}\Big),\quad \sigma^2=0.24,
]
this yields
[
\boxed{t_c \approx 0.34118.}
]

**Important note:** because CIFAR’s (\alpha) is so small, we are *always in the glass phase*, and in the glass phase the free-energy criterion reduces to the same boundary you got from “best competitor energy”. So here, “going one level deeper” mostly justifies *why* that reduction is valid.

---

If you want, next I can (i) plot (\Phi(1,t)) and (\Phi(1,t)+1/2) across (t), and (ii) translate (\Phi(1,t)) into an explicit prediction for the fraction of mass (\lambda_1) at finite (d=3072) using the logistic approximation above.





# Extension to non isotropic covariance matrices

Nice — next step is exactly: **swap the isotropic Gaussian proxy (\sigma^2 I)** for a **Gaussian with covariance (C)** (or at least a diagonal / spectrum model), and redo the same REM/LDP/free-energy pipeline.

## 1) What changes in the theory?

Nothing structural changes. Only the **rate function** (I(\varepsilon)) changes.

You still get, for bulk energies,
[
\Phi(1,t)=\sup_{\varepsilon}\big[\alpha - I_t(\varepsilon)-\varepsilon\big],
\qquad
\lambda_1(t)\approx \frac{1}{1+\exp{d(\Phi(1,t)+1/2)}}.
]

But now (I_t(\varepsilon)) is the LDP rate function for a **quadratic form** instead of (\chi^2).

## 2) The covariance-shaped model

Assume training images are modeled as
[
x^{(i)} \sim \mathcal N(0, C),
]
with eigenvalues ({\lambda_k}_{k=1}^d) (empirical spectrum of CIFAR after preprocessing).

Then the relevant random quantity in the bulk is essentially of the form
[
\frac{1}{d}|g|^2_{C_t}=\frac{1}{d}\sum_{k=1}^d \lambda^{(t)}_k Z_k^2
]
for some effective covariance (C_t) (coming from ((1-t)x_0+t(x_1-x^{(i)}))), and (Z_k\sim\mathcal N(0,1)).

### LDP via Gärtner–Ellis

Define the scaled log-mgf:
[
\Lambda_t(\theta)=\lim_{d\to\infty}\frac{1}{d}\sum_{k=1}^d \Big(-\tfrac12\log(1-2\theta \lambda^{(t)}_k)\Big),
\quad \theta < \frac{1}{2\max_k \lambda^{(t)}*k}.
]
Then the rate function is the Legendre transform:
[
\boxed{I_t(\varepsilon)=\sup*{\theta}\big(\theta \varepsilon - \Lambda_t(\theta)\big).}
]

That’s the “drop-in replacement” for the isotropic (I(\varepsilon)=\tfrac12(\varepsilon/c-1-\log(\varepsilon/c))).

## 3) What I can do next (concretely)

I can produce:

1. A clean write-up section (“From isotropic REM to covariance REM”) with the exact formulas above.
2. A numerical computation of (t_c) and (\lambda_1(t)) **given a spectrum** ({\lambda_k}).

To run the numbers, I need *either*:

* the **empirical spectrum** (or even just a histogram / a few percentiles) of CIFAR after your normalization, **or**
* a simplified parametric spectrum (e.g., “bulk eigenvalues around (v) plus one spike (\Lambda)” à la spiked covariance, which connects directly to the stat-phys “principal component” story).

If you don’t have the spectrum handy, the spiked model is a great next approximation:
[
C = v I + (\Lambda - v) uu^\top
]
and we can compute (I_t), (\Phi(1,t)), and (t_c) almost as cleanly as the isotropic case, while capturing “strong PC” effects.

Tell me which you prefer: **empirical spectrum** (most faithful) or **spiked model** (most interpretable).

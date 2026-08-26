"""Boundary layer at the edge of an interval.

q has density rho(x) = (2/3)(1+x) on [0,1]; the active stratum is the endpoint x=0,
with inward direction +1.  This is the case d=m=1, c=1, k=0 of the general theorem.
Everything is computed exactly (quadrature) and compared to the tangent-cone
prediction, whose one-dimensional cone is the half-line [0,infty).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "cm",
})

rho = lambda x: (2.0 / 3.0) * (1.0 + x)

def _int(f, s, y):
    return quad(lambda x: rho(x) * f(x, y, s) * norm.pdf((y - x) / s) / s, 0.0, 1.0,
                limit=400)[0]

p    = lambda y, s: _int(lambda x, y, s: 1.0, s, y)
dp   = lambda y, s: _int(lambda x, y, s: (x - y) / s**2, s, y)
d2p  = lambda y, s: _int(lambda x, y, s: (x - y) ** 2 / s**4 - 1.0 / s**2, s, y)

def exact(y, s):
    p0, p1, p2 = p(y, s), dp(y, s), d2p(y, s)
    sc = p1 / p0
    return p0, sc, p2 / p0 - sc**2

# tangent-cone (half-line) profile
lam   = lambda a: norm.pdf(a) / norm.cdf(a)
lamp  = lambda a: -lam(a) * (a + lam(a))
lampp = lambda a: -lamp(a) * (a + lam(a)) - lam(a) * (1.0 + lamp(a))

C0 = lambda a: rho(0.0) * norm.cdf(a)
C1 = lambda a: rho(0.0) * (a * norm.cdf(a) + norm.pdf(a))   # rho'(0)=2/3, rho(0)=2/3
L1 = lambda a: a + lam(a)                                    # = C1/C0

a = np.linspace(-2.5, 3.0, 400)
sigmas = [0.20, 0.05]
colors = ["#c05a2a", "#2a5fa8"]

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
axes[0].plot(a, C0(a), "k-", lw=1.6, label=r"leading: $\rho(x)\,\Phi_{\mathrm{N}}(a)$")
axes[1].plot(a, lam(a), "k-", lw=1.6, label=r"leading: $\lambda(a)$")
axes[2].plot(a, lamp(a), "k-", lw=1.6, label=r"leading: $\lambda'(a)$")

for s, c in zip(sigmas, colors):
    ex = np.array([exact(s * ai, s) for ai in a])
    axes[0].plot(a, ex[:, 0], color=c, lw=1.3, label=rf"exact, $\sigma={s}$")
    axes[1].plot(a, s * ex[:, 1], color=c, lw=1.3, label=rf"exact, $\sigma={s}$")
    axes[2].plot(a, s**2 * ex[:, 2], color=c, lw=1.3, label=rf"exact, $\sigma={s}$")
    axes[0].plot(a, C0(a) + s * C1(a), color=c, ls=":", lw=1.5)
    axes[1].plot(a, lam(a) + s * (1.0 + lamp(a)), color=c, ls=":", lw=1.5)
    axes[2].plot(a, lamp(a) + s * lampp(a), color=c, ls=":", lw=1.5)

axes[0].plot([], [], color="0.45", ls=":", lw=1.5, label="two-term")

titles = [r"density $p_\sigma(\sigma a)$",
          r"rescaled score $\sigma\, s_\sigma$",
          r"rescaled log-Hessian $\sigma^2 H_\sigma$"]
for ax, t in zip(axes, titles):
    ax.set_xlabel("$a$"); ax.set_title(t, fontsize=9.5)
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "figures" / "mills-boundary-layer.png",
            bbox_inches="tight")

# --- convergence check: leading term O(sigma), two-term O(sigma^2) --------------
ac = np.linspace(-2.0, 3.0, 60)
print(f"{'sigma':>7} | {'p: 1-term':>10} {'p: 2-term':>10} | "
      f"{'s: 1-term':>10} {'s: 2-term':>10} | {'H: 1-term':>10} {'H: 2-term':>10}")
prev = None
for s in [0.1, 0.05, 0.025, 0.0125]:
    ex = np.array([exact(s * ai, s) for ai in ac])
    e = [np.max(np.abs(ex[:, 0] - C0(ac))),
         np.max(np.abs(ex[:, 0] - (C0(ac) + s * C1(ac)))),
         np.max(np.abs(s * ex[:, 1] - lam(ac))),
         np.max(np.abs(s * ex[:, 1] - (lam(ac) + s * (1.0 + lamp(ac))))),
         np.max(np.abs(s**2 * ex[:, 2] - lamp(ac))),
         np.max(np.abs(s**2 * ex[:, 2] - (lamp(ac) + s * lampp(ac))))]
    print(f"{s:>7} | " + " ".join(f"{v:10.2e}" for v in e[:2]) + " | "
          + " ".join(f"{v:10.2e}" for v in e[2:4]) + " | "
          + " ".join(f"{v:10.2e}" for v in e[4:]))
    if prev is not None:
        print(f"{'  ratio':>7} | " + " ".join(f"{prev[i]/e[i]:10.2f}" for i in range(6)))
    prev = e

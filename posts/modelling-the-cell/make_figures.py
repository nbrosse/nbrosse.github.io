"""Figures for the post "Modelling the cell".

Numbers come from the study in https://github.com/nbrosse/modelling-the-cell
(report/results.md; the re-split gaps are copied from study/outputs/*/repeated_splits.csv
into data/). Run from this folder:  python make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"

# Categorical palette (validated: blue, orange, aqua) and neutral inks.
PCA, STRING, PRIOR = "#2a78d6", "#eb6834", "#1baf7a"
BASE, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"
RP, OTHER = "#e34948", "#a3a29c"

plt.rcParams.update({
    "font.size": 14,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def hits_bars(name, n_hits, rows, chance, xmax):
    """Horizontal bars of hits found among the n_hits strongest test perturbations."""
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    notes = [r[3] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, values, height=0.62, color=colors)
    ax.axvline(chance, color=MUTED, ls=":", lw=1.5)
    ax.text(chance, len(rows) - 0.35, " chance", color=MUTED, fontsize=12, va="bottom")
    for yi, v, note in zip(y, values, notes):
        label = f"{v:g} / {n_hits}"
        if note:
            label += f"   {note}"
        ax.text(max(v, chance) + 0.12, yi, label, va="center", fontsize=13, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, int(xmax) + 1))
    ax.set_xlabel(f"Hits found among the {n_hits} strongest test perturbations")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    save(fig, name)


# A. K562 test, k = 5 % (14 hits); results.md, seed 0 table and exploratory section.
hits_bars(
    "k562-hits.png", 14,
    [
        ("ridge-PCA", 5, PCA, "← chosen by hit recall"),
        ("knn-PCA", 5, PCA, ""),
        ("ridge-STRING", 1, STRING, ""),
        ("knn-STRING", 1, STRING, "← chosen by all 5 usual metrics"),
        ("target expression\n(no model)", 5, PRIOR, ""),
    ],
    chance=0.7, xmax=9,
)

# B. K562 test without ribosomal-protein targets (265 candidates, 14 hits);
# study/outputs/exploratory.json, without_ribosomal_targets["5%"].
hits_bars(
    "k562-no-rp.png", 14,
    [
        ("ridge-PCA", 2, PCA, ""),
        ("knn-PCA", 0, PCA, ""),
        ("ridge-STRING", 0, STRING, ""),
        ("knn-STRING", 0, STRING, ""),
        ("target expression\n(no model)", 2, PRIOR, ""),
    ],
    chance=0.053 * 14, xmax=9,
)

# C. Share of perturbations passing the >= 20 cells per fold filter; report rp-filter figure.
fig, ax = plt.subplots(figsize=(8, 4.6))
lines = ["K562", "RPE1"]
rp = [(62, 78), (6, 78)]
other = [(1224, 1979), (790, 2315)]
x = np.arange(2)
w = 0.36
for off, data, color, lab in [(-w / 2, rp, RP, "ribosomal-protein targets"),
                              (w / 2, other, OTHER, "other targets")]:
    pct = [100 * a / b for a, b in data]
    ax.bar(x + off, pct, width=w - 0.02, color=color, label=lab)
    for xi, p, (a, b) in zip(x + off, pct, data):
        ax.text(xi, p + 2, f"{a}/{b}", ha="center", fontsize=13, color=INK)
ax.set_xticks(x, lines)
ax.set_ylim(0, 100)
ax.set_ylabel("Pass the cell-count filter (%)")
ax.legend(frameon=False, loc="upper right", fontsize=13)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
ax.set_axisbelow(True)
save(fig, "rp-filter.png")

# D. Hits gained by choosing with hit recall instead of Pearson on expression, 50 splits.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (line, n_hits) in zip(axes, [("k562", 14), ("rpe1", 8)]):
    gap = np.loadtxt(HERE / "data" / f"resplits-{line}.csv", delimiter=",", skiprows=1)[:, 1]
    g = gap * n_hits
    hits = (np.sign(g) * np.ceil(np.abs(g) - 1e-9)).astype(int)  # away from zero: any gain or loss stays visible
    vals = np.arange(-3, 6)
    counts = [(hits == v).sum() for v in vals]
    colors = [BASE if v == 0 else MUTED for v in vals]
    ax.bar(vals, counts, width=0.8, color=colors)
    for v, c in zip(vals, counts):
        if c:
            ax.text(v, c + 0.4, str(c), ha="center", fontsize=12, color=INK)
    ax.set_xticks(vals)
    ax.set_title(f"{line.upper()}  ({n_hits} test hits)", fontsize=14, loc="left")
    ax.set_xlabel("Hits gained by choosing with hit recall")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Number of splits (of 50)")
fig.tight_layout()
save(fig, "resplits.png")

print("written:", sorted(p.name for p in OUT.glob("*.png")))

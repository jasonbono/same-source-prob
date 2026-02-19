"""
Figure 2 in the paper — Toy Set A exact enumeration histograms.

PURPOSE:
  Validates Theorems 1 and 2 by *exact* enumeration (not sampling) of all
  feasible contingency tables C for three small examples sharing o=(4,4,4)
  but with increasing entity-degree heterogeneity:
    A1: ℓ=(4,4,4)  σ²=0   — uniform entities
    A2: ℓ=(6,3,3)  σ²=2   — mild heterogeneity
    A3: ℓ=(10,1,1) σ²=18  — one dominant emitter

  Each panel shows:
    - Blue bars: exact discrete distribution of θ(C) = P(S|collision),
      weighted by config-model probability  w(C) ∝ 1/∏ c_ia!
    - Green dashed line: ensemble mean computed from the enumerated distribution
    - Red dotted line: the σ²-only closed-form estimate (Theorem 1, Algorithm 1),
      computed using only (o, n, σ²) with no knowledge of ℓ
    - Grey band: ±1 sd from Theorem 2

  The key visual: the green and red lines coincide, proving the variance-only
  estimator equals the ensemble mean.

OUTPUT: fig_toyA_hist.pdf (3-panel, publication-ready vector PDF)
USED IN: \\includegraphics in main.tex Figure 2
"""

import math
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Contingency-table enumeration
# ---------------------------------------------------------------------------

def enumerate_tables(ell, o):
    """
    Yield every n×m non-negative integer matrix with row sums ell, col sums o.
    Feasible only for small n, m, T — used for toy validation, not at scale.
    """
    n, m = len(ell), len(o)
    table = [[0] * m for _ in range(n)]
    row_rem = list(ell)
    col_rem = list(o)
    yield from _fill(table, row_rem, col_rem, n, m, 0, 0)


def _fill(table, row_rem, col_rem, n, m, i, j):
    """Recursive cell-by-cell fill. Prunes via remaining row/col budgets."""
    if i == n:
        yield [row[:] for row in table]
        return

    ni, nj = (i, j + 1) if j + 1 < m else (i + 1, 0)

    if j == m - 1:
        val = row_rem[i]
        if val < 0 or val > col_rem[j]:
            return
        table[i][j] = val
        row_rem[i] -= val
        col_rem[j] -= val
        yield from _fill(table, row_rem, col_rem, n, m, ni, nj)
        row_rem[i] += val
        col_rem[j] += val
        table[i][j] = 0
        return

    if i == n - 1:
        val = col_rem[j]
        if val < 0 or val > row_rem[i]:
            return
        table[i][j] = val
        row_rem[i] -= val
        col_rem[j] -= val
        yield from _fill(table, row_rem, col_rem, n, m, ni, nj)
        row_rem[i] += val
        col_rem[j] += val
        table[i][j] = 0
        return

    lo = max(0, row_rem[i] - sum(col_rem[j + 1:]))
    hi = min(row_rem[i], col_rem[j])
    for val in range(lo, hi + 1):
        table[i][j] = val
        row_rem[i] -= val
        col_rem[j] -= val
        yield from _fill(table, row_rem, col_rem, n, m, ni, nj)
        row_rem[i] += val
        col_rem[j] += val
        table[i][j] = 0


# ---------------------------------------------------------------------------
# Config-model weight and theta
# ---------------------------------------------------------------------------

def log_weight(C):
    """
    Log of unnormalised config-model weight: -Σ log(c_ia!).
    Under uniform stub matching, the number of matchings that produce table C
    is proportional to 1/∏ c_ia!, so tables with more spread-out entries are
    more probable.
    """
    return -sum(math.lgamma(c + 1) for row in C for c in row)


def theta(C, D):
    """θ(C) = N(C)/D where N(C) = Σ c_ia² (collision numerator)."""
    N = sum(c * c for row in C for c in row)
    return N / D


# ---------------------------------------------------------------------------
# Analytic formulas (Theorems 1 & 2)
# ---------------------------------------------------------------------------

def falling(x, k):
    """Falling factorial x_(k) = x(x-1)...(x-k+1)."""
    r = 1
    for j in range(k):
        r *= (x - j)
    return r


def analytic_mean(ell, o):
    """
    Ensemble mean E[θ(C) | ℓ, o] using full knowledge of ℓ.
    Formula: (T + M₂·p₂) / D  where M₂ = Σ ℓ_i(ℓ_i-1).
    This is what the enumerated distribution's weighted mean should match.
    """
    T = sum(o)
    D = sum(a * a for a in o)
    M2 = sum(li * (li - 1) for li in ell)
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


def variance_only_estimate(o, n, sigma2):
    """
    Theorem 1 closed-form estimator — the paper's main contribution.
    Uses only (o, n, σ²): no knowledge of the individual ℓ_i values.
    Computes α = 1/n + nσ²/T², then M₂ = T²α - T, then (T + M₂·p₂)/D.
    Should exactly equal analytic_mean() for the corresponding ℓ — that's the theorem.
    """
    T = sum(o)
    D = sum(a * a for a in o)
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


def analytic_sd(ell, o):
    """
    Theorem 2: analytic standard deviation of θ(C) across the config-model
    ensemble. Depends on M₃ = Σ ℓ_i(ℓ_i-1)(ℓ_i-2) in addition to M₂,
    so it requires more than just σ² — it needs the full ℓ (or a skewness
    assumption).
    """
    T = sum(o)
    D = sum(a * a for a in o)
    M2 = sum(li * (li - 1) for li in ell)
    M3 = sum(li * (li - 1) * (li - 2) for li in ell)

    p2 = sum(falling(a, 2) for a in o) / falling(T, 2)
    p3 = sum(falling(a, 3) for a in o) / falling(T, 3)

    S2 = sum(falling(a, 2) for a in o)
    S4 = sum(falling(a, 4) for a in o)
    S22 = sum(falling(a, 2) ** 2 for a in o)
    p22 = (S4 + S2 * S2 - S22) / falling(T, 4)

    EX2 = 2 * M2 * p2 + M2 * (M2 - 2) * p22 + 4 * M3 * (p3 - p22)
    EX = M2 * p2
    var_X = EX2 - EX ** 2
    var_theta = var_X / (D * D)
    return math.sqrt(max(var_theta, 0.0))


# ---------------------------------------------------------------------------
# Build discrete distribution for one example
# ---------------------------------------------------------------------------

def build_distribution(ell, o):
    """
    Enumerate all feasible tables, weight by config-model probability,
    and aggregate into a discrete distribution over θ values.
    Returns {θ_value: probability} dict.
    """
    D = sum(a * a for a in o)
    log_ws = []
    thetas = []
    for C in enumerate_tables(ell, o):
        log_ws.append(log_weight(C))
        thetas.append(theta(C, D))

    max_lw = max(log_ws)
    ws = [math.exp(lw - max_lw) for lw in log_ws]
    total = sum(ws)
    ws = [w / total for w in ws]

    dist = defaultdict(float)
    for t, w in zip(thetas, ws):
        dist[round(t, 6)] += w
    return dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

EXAMPLES = [
    # (label, o, ell, sigma2)
    ("A1", (4, 4, 4), (4, 4, 4), 0),
    ("A2", (4, 4, 4), (6, 3, 3), 2),
    ("A3", (4, 4, 4), (10, 1, 1), 18),
]

COLORS = {
    "bar": "#4878CF",
    "ensemble_mean": "#2CA02C",
    "estimator": "#E24A33",
    "sd_band": "#888888",
}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

    for ax, (label, o, ell, sigma2) in zip(axes, EXAMPLES):
        dist = build_distribution(list(ell), list(o))
        xs = sorted(dist.keys())
        ps = [dist[x] for x in xs]
        n = len(ell)

        mu = analytic_mean(list(ell), list(o))
        est = variance_only_estimate(list(o), n, sigma2)
        sd = analytic_sd(list(ell), list(o))

        emp_mean = sum(x * p for x, p in zip(xs, ps))
        print(f"{label}: ensemble mean={mu:.6f}  σ²-only estimate={est:.6f}  "
              f"emp mean={emp_mean:.6f}  sd={sd:.6f}")

        bar_width = 0.018 if len(xs) > 3 else 0.012
        ymax = max(ps)

        ax.bar(xs, ps, width=bar_width, color=COLORS["bar"], edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.85)

        ax.axvspan(mu - sd, mu + sd, color=COLORS["sd_band"], alpha=0.12,
                   zorder=1, label=rf"$\pm 1\,$ sd  ({sd:.4f})")

        ax.axvline(mu, color=COLORS["ensemble_mean"], linewidth=2.0,
                   linestyle="--", zorder=5,
                   label=f"Ensemble mean = {mu:.4f}")

        ax.axvline(est, color=COLORS["estimator"], linewidth=2.0,
                   linestyle=":", zorder=6,
                   label=rf"$\sigma^2$-only estimate = {est:.4f}")

        ax.set_xlabel(r"$\theta(C) = P(S \mid \mathcal{C})$", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Probability", fontsize=11)

        ax.set_title(
            r"$\ell$" + f" = {ell},  "
            rf"$\sigma^2 = {sigma2}$",
            fontsize=10.5, pad=8,
        )

        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.88)
        ax.tick_params(labelsize=9)
        ax.set_xlim(min(xs) - 0.04, max(xs) + 0.04)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        r"Toy Set A  ($n = 3,\; m = 3,\; o = (4,4,4)$)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig_toyA_hist.pdf", bbox_inches="tight", dpi=300)
    print("\nSaved fig_toyA_hist.pdf")


if __name__ == "__main__":
    main()

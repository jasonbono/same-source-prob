"""
Figure 3 in the paper — Toy Set B exact enumeration histogram.

PURPOSE:
  Same method as fig_toyA_histograms.py but for a single example with
  skewed value marginals: o=(8,2,2) with uniform entities ℓ=(4,4,4), σ²=0.
  Demonstrates that the estimator works when value labels are unevenly
  distributed (one dominant value).

  Shows the same overlays as the Toy A panels:
    - Blue bars: exact discrete distribution of θ(C)
    - Green dashed: ensemble mean
    - Red dotted: σ²-only estimate (Theorem 1) — coincides with ensemble mean
    - Grey band: ±1 sd (Theorem 2)

OUTPUT: fig_toyB_hist.pdf (single-panel vector PDF)
USED IN: \\includegraphics in main.tex Figure 3

NOTE: The enumeration and formula code is duplicated from fig_toyA_histograms.py
rather than shared, so each script is self-contained and runnable independently.
"""

import math
from collections import defaultdict

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Contingency-table enumeration (same logic as fig_toyA)
# ---------------------------------------------------------------------------

def enumerate_tables(ell, o):
    n, m = len(ell), len(o)
    table = [[0] * m for _ in range(n)]
    row_rem = list(ell)
    col_rem = list(o)
    yield from _fill(table, row_rem, col_rem, n, m, 0, 0)


def _fill(table, row_rem, col_rem, n, m, i, j):
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


def log_weight(C):
    return -sum(math.lgamma(c + 1) for row in C for c in row)


def theta(C, D):
    N = sum(c * c for row in C for c in row)
    return N / D


def falling(x, k):
    r = 1
    for j in range(k):
        r *= (x - j)
    return r


def analytic_mean(ell, o):
    T = sum(o)
    D = sum(a * a for a in o)
    M2 = sum(li * (li - 1) for li in ell)
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


def variance_only_estimate(o, n, sigma2):
    T = sum(o)
    D = sum(a * a for a in o)
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


def analytic_sd(ell, o):
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
    var_theta = (EX2 - EX ** 2) / (D * D)
    return math.sqrt(max(var_theta, 0.0))


def build_distribution(ell, o):
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


def main():
    o = [8, 2, 2]
    ell = [4, 4, 4]
    sigma2 = 0
    n = len(ell)

    dist = build_distribution(ell, o)
    xs = sorted(dist.keys())
    ps = [dist[x] for x in xs]

    mu = analytic_mean(ell, o)
    est = variance_only_estimate(o, n, sigma2)
    sd = analytic_sd(ell, o)

    emp_mean = sum(x * p for x, p in zip(xs, ps))
    print(f"B: ensemble mean={mu:.6f}  σ²-only estimate={est:.6f}  "
          f"emp mean={emp_mean:.6f}  sd={sd:.6f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    bar_width = 0.018
    ax.bar(xs, ps, width=bar_width, color="#4878CF", edgecolor="white",
           linewidth=0.5, zorder=3, alpha=0.85)

    ax.axvspan(mu - sd, mu + sd, color="#888888", alpha=0.12,
               zorder=1, label=rf"$\pm 1\,$ sd  ({sd:.4f})")

    ax.axvline(mu, color="#2CA02C", linewidth=2.0, linestyle="--", zorder=5,
               label=f"Ensemble mean = {mu:.4f}")

    ax.axvline(est, color="#E24A33", linewidth=2.0, linestyle=":", zorder=6,
               label=rf"$\sigma^2$-only estimate = {est:.4f}")

    ax.set_xlabel(r"$\theta(C) = P(S \mid \mathcal{C})$", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(
        r"Example B  ($n = 3,\; m = 3,\; o = (8,2,2)$)",
        fontsize=11, pad=8,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.88)
    ax.tick_params(labelsize=9)
    ax.set_xlim(min(xs) - 0.04, max(xs) + 0.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig("fig_toyB_hist.pdf", bbox_inches="tight", dpi=300)
    print("\nSaved fig_toyB_hist.pdf")


if __name__ == "__main__":
    main()

"""
ℓ-shape invariance figure — same σ², different degree-sequence shapes.

PURPOSE:
  Demonstrates the core claim of Theorem 1: the ensemble mean of θ(C)
  depends on the entity-degree sequence ℓ *only through σ²*.  Three
  hand-constructed ℓ vectors all have σ²=36 but very different shapes
  (and therefore different M₃ = Σ ℓ_i(ℓ_i-1)(ℓ_i-2)):

    "Spread":   5×16 + 5×4   — mass distributed across many high entities
    "Moderate":  2×22 + 8×7   — moderate concentration in two entities
    "Peaked":   1×28 + 9×8   — single dominant emitter

  The figure overlays MC histograms of θ(C) for all three shapes.
  All three distributions centre on the same Theorem 1 estimate (black
  dashed line), even though the ℓ vectors differ.  The spread of each
  histogram depends on M₃ (Theorem 2), which differs across shapes,
  though the effect is modest at this scale.

PARAMETERS: n=10, m=10, T=100, o=(10,...,10), σ²=36, B=5000 MC samples.

OUTPUT: fig_ell_shapes.pdf (single-panel vector PDF)
NOT YET in main.tex — generated for review.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# Theorem 1 estimator
# ---------------------------------------------------------------------------

def theta_estimate(o, n, sigma2):
    """Theorem 1 closed-form estimator from (o, n, σ²) alone."""
    T = sum(o)
    D = sum(a * a for a in o)
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


# ---------------------------------------------------------------------------
# Stub-matching MC  (same method as fig_sigma_sweep.py)
# ---------------------------------------------------------------------------

def stub_match_theta(ell, o, rng):
    """One MC sample of θ(C) via uniform stub matching."""
    T = sum(ell)
    n, m = len(ell), len(o)
    entity_stubs = np.repeat(np.arange(n), ell)
    value_stubs = np.repeat(np.arange(m), o)
    rng.shuffle(value_stubs)
    counts = np.zeros((n, m), dtype=np.int64)
    for idx in range(T):
        counts[entity_stubs[idx], value_stubs[idx]] += 1
    D = sum(a * a for a in o)
    N = int(np.sum(counts * counts))
    return N / D


def mc_samples(ell, o, B, rng):
    return np.array([stub_match_theta(ell, o, rng) for _ in range(B)])


# ---------------------------------------------------------------------------
# Degree-sequence statistics
# ---------------------------------------------------------------------------

def compute_sigma2(ell):
    """Entity-degree variance σ² = (1/n) Σ (ℓ_i - μ)²."""
    n = len(ell)
    mu = sum(ell) / n
    return sum((li - mu) ** 2 for li in ell) / n


def compute_M3(ell):
    """Third falling-factorial moment M₃ = Σ ℓ_i(ℓ_i-1)(ℓ_i-2). Appears in Theorem 2."""
    return sum(li * (li - 1) * (li - 2) for li in ell)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n, T, m = 10, 100, 10
    B = 5000

    o = [T // m] * m

    # Three ℓ vectors, all with σ² = 36, but different concentration patterns.
    # μ = 10, n = 10, T = 100.
    #
    # "Spread":   5 entities at 16, 5 at 4  → mass spread across many high entities
    # "Moderate":  2 entities at 22, 8 at 7  → moderate concentration
    # "Peaked":   1 entity at 28, 9 at 8    → single dominant emitter
    shapes = [
        ("Spread (5×16 + 5×4)",    [16]*5 + [4]*5),
        ("Moderate (2×22 + 8×7)",  [22]*2 + [7]*8),
        ("Peaked (1×28 + 9×8)",    [28]*1 + [8]*9),
    ]

    for desc, ell in shapes:
        s2 = compute_sigma2(ell)
        m3 = compute_M3(ell)
        print(f"  {desc}: σ²={s2:.1f}, M₃={m3}, sum={sum(ell)}")

    est = theta_estimate(o, n, compute_sigma2(shapes[0][1]))
    print(f"\nThm 1 estimate: {est:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#4878CF", "#E24A33", "#2CA02C"]

    for idx, (desc, ell) in enumerate(shapes):
        samples = mc_samples(ell, o, B, RNG)
        mc_mean = np.mean(samples)
        mc_std = np.std(samples)
        m3 = compute_M3(ell)

        ax.hist(samples, bins=50, density=True, alpha=0.35,
                color=colors[idx], edgecolor="white", linewidth=0.3,
                label=f"{desc}\n"
                      rf"  $M_3 = {m3}$,  MC sd = {mc_std:.4f}")

        print(f"  {desc}: MC mean={mc_mean:.6f}, MC sd={mc_std:.4f}")

    ax.axvline(est, color="black", linewidth=2.0, linestyle="--",
               label=f"Thm 1 estimate = {est:.4f}", zorder=10)

    ax.set_xlabel(r"$\theta(C) = P(S \mid \mathcal{C})$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        r"$n = 10,\; T = 100,\; \sigma^2 = 36$:  "
        r"same mean, different $\ell$ shapes",
        fontsize=11, pad=8,
    )
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.88)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig("fig_ell_shapes.pdf", bbox_inches="tight", dpi=300)
    print("\nSaved fig_ell_shapes.pdf")


if __name__ == "__main__":
    main()

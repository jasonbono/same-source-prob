"""
End-to-end demonstration: ground-truth graphs → simulated measurements →
estimated marginals → Theorem 1 estimator → comparison with truth.

Addresses the paper's TODO:
  (i)   Sample ground-truth graphs from the configuration model.
  (ii)  Simulate measurements (draw edges, observe value endpoints).
  (iii) Estimate value marginals from finite data.
  (iv)  Compare the true posterior to our mean estimate and uncertainty bands.

OUTPUT:
  fig_demo_convergence.pdf — 3 panels (low / medium / high σ²).
    Shows convergence of the Theorem 1 estimator as the number of observed
    measurement draws increases.  At large N_obs the estimated-marginals
    curve collapses onto the oracle (true-o) estimate; the gray band shows
    the irreducible ensemble uncertainty across graphs.

  fig_demo_empirical.pdf — 1 panel.
    Validates the measurement model directly: the empirical same-source
    rate among collisions from simulated draw pairs converges to the exact
    θ(C*) computed from the known graph.

PARAMETERS:
  n = 20 entities, m = 15 values, T = 200 total edges, uniform o.
  Three σ² levels spanning the feasible range.
"""

import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Theorem 1 closed-form estimator
# ---------------------------------------------------------------------------

def theta_estimate(o, n, sigma2):
    """
    Theorem 1 estimator: E[θ(C) | o, σ²].
    Takes value degrees o (or scaled estimates), entity count n, assumed σ².
    """
    o = np.asarray(o, dtype=float)
    T = o.sum()
    D = (o * o).sum()
    if D == 0 or T <= 1:
        return np.nan
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


# ---------------------------------------------------------------------------
# Exact θ(C) from a known graph
# ---------------------------------------------------------------------------

def theta_exact(C):
    """θ(C) = Σ c²_{ia} / Σ o²_a for a known incidence matrix C."""
    o = C.sum(axis=0)
    D = int((o * o).sum())
    N = int((C * C).sum())
    return N / D if D > 0 else 0.0


# ---------------------------------------------------------------------------
# Analytic ensemble sd (Theorem 2)
# ---------------------------------------------------------------------------

def falling(x, k):
    r = 1
    for j in range(k):
        r *= (x - j)
    return r


def analytic_sd(ell, o):
    """
    Theorem 2: sd of θ(C) across the configuration-model ensemble.
    Requires the full entity-degree sequence ℓ (not just σ²), because the
    variance formula depends on M₃ = Σ ℓ_i(ℓ_i-1)(ℓ_i-2).
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
    var_theta = (EX2 - EX ** 2) / (D * D)
    return math.sqrt(max(var_theta, 0.0))


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def make_uniform_o(m, T):
    """Value degrees as equal as possible: each o_a ≈ T/m."""
    base = T // m
    o = [base] * m
    for i in range(T - sum(o)):
        o[i] += 1
    return o


def two_level_degrees(n, T, sigma2_target):
    """
    Build a two-level ℓ vector (k entities at d_high, rest at d_low)
    matching sigma2_target as closely as integer constraints allow.
    Reused from fig_sigma_sweep.py.
    """
    mu = T / n
    best_ell, best_err = None, float("inf")
    for k in range(1, n):
        nk = n - k
        gap_sq = sigma2_target * n * n / (k * nk)
        if gap_sq < 0:
            continue
        gap = math.sqrt(gap_sq)
        d_low_exact = mu - k * gap / n
        for d_low in (math.floor(d_low_exact), math.ceil(d_low_exact)):
            if d_low < 1:
                continue
            d_high_needed = T - nk * d_low
            if d_high_needed % k != 0:
                continue
            d_high = d_high_needed // k
            if d_high < 1:
                continue
            ell = [d_high] * k + [d_low] * nk
            actual_s2 = sum((li - mu) ** 2 for li in ell) / n
            err = abs(actual_s2 - sigma2_target)
            if err < best_err:
                best_err = err
                best_ell = ell
    if best_ell is None:
        best_ell = [T // n] * n
        leftover = T - sum(best_ell)
        for i in range(leftover):
            best_ell[i] += 1
    return best_ell


def sample_config_model(ell, o, rng):
    """Sample one graph C from the bipartite configuration model via stub matching."""
    T = sum(ell)
    n, m = len(ell), len(o)
    entity_stubs = np.repeat(np.arange(n), ell)
    value_stubs = np.repeat(np.arange(m), o)
    rng.shuffle(value_stubs)
    C = np.zeros((n, m), dtype=np.int64)
    np.add.at(C, (entity_stubs, value_stubs), 1)
    return C


# ---------------------------------------------------------------------------
# Measurement simulation
# ---------------------------------------------------------------------------

def simulate_draws(C, N_obs, rng):
    """
    Simulate N_obs single measurement draws from graph C.
    Each draw samples an edge uniformly at random and returns the value
    endpoint.  (The entity endpoint is latent and unobserved.)
    """
    o = C.sum(axis=0).astype(float)
    p = o / o.sum()
    return rng.choice(len(o), size=N_obs, p=p)


def estimate_o_hat(values, m, T):
    """
    From N_obs observed value labels, estimate the value-degree vector ô
    by scaling empirical frequencies to sum to T (assumed known).
    """
    counts = np.bincount(values, minlength=m).astype(float)
    return counts * (T / counts.sum())


def simulate_pairs(C, N_pairs, rng):
    """
    Simulate N_pairs independent pairs of measurement draws from graph C.
    Each draw in the pair samples an edge uniformly at random.
    Returns (collision_mask, same_source_mask) boolean arrays.
    """
    n, m = C.shape
    T = int(C.sum())

    # Build flat arrays: edge index → (entity, value)
    entity_idx = np.repeat(np.arange(n), C.sum(axis=1).astype(int))
    value_idx = np.empty(T, dtype=np.int64)
    pos = 0
    for i in range(n):
        for a in range(m):
            cnt = int(C[i, a])
            value_idx[pos:pos + cnt] = a
            pos += cnt

    idx1 = rng.integers(T, size=N_pairs)
    idx2 = rng.integers(T, size=N_pairs)

    collision = value_idx[idx1] == value_idx[idx2]
    same_source = entity_idx[idx1] == entity_idx[idx2]
    return collision, same_source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n, m, T = 20, 15, 200
    o = make_uniform_o(m, T)

    # Three σ² levels: low (uniform entities), moderate, high heterogeneity.
    sigma2_targets = [0, 200, 1200]

    N_obs_grid = np.unique(
        np.logspace(np.log10(30), np.log10(20000), 22).astype(int)
    )
    R = 300          # measurement-simulation repetitions per N_obs
    K_extra = 8      # additional C* samples to show graph-to-graph spread

    # ---- Build ground-truth scenarios ----
    print("Building scenarios...")
    scenarios = []
    for s2_target in sigma2_targets:
        ell = two_level_degrees(n, T, s2_target)
        actual_s2 = sum((li - T / n) ** 2 for li in ell) / n

        C_star = sample_config_model(ell, o, RNG)
        extra_thetas = [
            theta_exact(sample_config_model(ell, o, RNG))
            for _ in range(K_extra)
        ]

        sc = {
            "ell": ell,
            "sigma2": actual_s2,
            "C_star": C_star,
            "theta_true": theta_exact(C_star),
            "theta_oracle": theta_estimate(o, n, actual_s2),
            "sd_ensemble": analytic_sd(ell, o),
            "extra_thetas": extra_thetas,
        }
        scenarios.append(sc)

        print(f"  ℓ shape: {dict(sorted(Counter(ell).items()))},  "
              f"σ²={actual_s2:.1f},  θ*={sc['theta_true']:.4f},  "
              f"oracle={sc['theta_oracle']:.4f},  sd={sc['sd_ensemble']:.4f}")

    # ==================================================================
    # Figure 1: Convergence of estimator with finite-sample marginals
    # ==================================================================
    print("\nComputing convergence curves...")
    fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, sc in zip(axes, scenarios):
        C_star = sc["C_star"]
        oracle = sc["theta_oracle"]
        sd = sc["sd_ensemble"]

        medians, lo_band, hi_band = [], [], []
        for j, N_obs in enumerate(N_obs_grid):
            ests = np.empty(R)
            for r in range(R):
                vals = simulate_draws(C_star, int(N_obs), RNG)
                o_hat = estimate_o_hat(vals, m, T)
                ests[r] = theta_estimate(o_hat, n, sc["sigma2"])
            medians.append(np.median(ests))
            lo_band.append(np.percentile(ests, 10))
            hi_band.append(np.percentile(ests, 90))
            if j % 7 == 0:
                print(f"    σ²={sc['sigma2']:.0f}  N_obs={N_obs:>6d}: "
                      f"median θ̂={medians[-1]:.4f}")

        # Ensemble uncertainty band (±1 sd from Theorem 2)
        ax.axhspan(oracle - sd, oracle + sd, color="#CCCCCC", alpha=0.45,
                   zorder=1, label=rf"Ensemble $\pm 1$ sd")

        # Additional sampled C* (thin lines showing graph-to-graph spread)
        for k, et in enumerate(sc["extra_thetas"]):
            kw = {"label": r"Other $C$ samples"} if k == 0 else {}
            ax.axhline(et, color="#E24A33", linewidth=0.6, alpha=0.40,
                       zorder=2, **kw)

        # True θ(C*)
        ax.axhline(sc["theta_true"], color="#E24A33", linewidth=1.8, zorder=4,
                   label=rf"True $\theta(C^*)$ = {sc['theta_true']:.3f}")

        # Oracle: Theorem 1 with true o
        ax.axhline(oracle, color="#2CA02C", linewidth=1.5, linestyle="--",
                   zorder=5,
                   label=rf"Thm 1 (true $o$) = {oracle:.3f}")

        # Estimated-marginals convergence curve
        ax.fill_between(N_obs_grid, lo_band, hi_band,
                        color="#4878CF", alpha=0.22, zorder=3)
        ax.plot(N_obs_grid, medians, color="#4878CF", linewidth=2, zorder=6,
                label=r"Thm 1 (est. $\hat{o}$)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$N_{\rm obs}$ (measurement draws)", fontsize=10)
        # Format ℓ shape as "k₁×d₁ + k₂×d₂"
        counts = Counter(sc["ell"])
        ell_desc = " + ".join(
            f"{cnt}×{deg}" for deg, cnt in sorted(counts.items(), reverse=True)
        )
        ax.set_title(
            rf"$\sigma^2 = {sc['sigma2']:.0f}$"
            rf"$\;\;(\ell$: {ell_desc}$)$",
            fontsize=9.5, pad=8,
        )
        ax.legend(fontsize=6.5, loc="best", framealpha=0.92)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\hat{\theta} \approx P(S \mid \mathcal{C})$",
                       fontsize=11)

    fig1.suptitle(
        r"Estimator convergence with finite-sample marginals"
        rf"  ($n = {n},\; m = {m},\; T = {T}$)",
        fontsize=11.5, y=1.02,
    )
    fig1.tight_layout()
    fig1.savefig("fig_demo_convergence.pdf", bbox_inches="tight", dpi=300)
    print("\nSaved fig_demo_convergence.pdf")

    # ==================================================================
    # Figure 2: Empirical θ from simulated measurement pairs
    #
    # This validates the measurement model itself: draw many pairs of
    # edges from C*, count collisions, count same-source collisions,
    # and verify that the ratio converges to the exact θ(C*).
    # ==================================================================
    print("\nSimulating draw pairs for empirical θ...")
    sc = scenarios[1]     # use the moderate-σ² scenario
    C_star = sc["C_star"]
    N_pairs_total = 200_000

    collision, same_source = simulate_pairs(C_star, N_pairs_total, RNG)

    checkpoints = np.unique(
        np.logspace(np.log10(200), np.log10(N_pairs_total), 80).astype(int)
    )
    running_theta = []
    for cp in checkpoints:
        n_coll = collision[:cp].sum()
        if n_coll > 0:
            running_theta.append(
                (collision[:cp] & same_source[:cp]).sum() / n_coll
            )
        else:
            running_theta.append(np.nan)

    total_coll = collision.sum()
    print(f"  Total pairs: {N_pairs_total:,},  collisions: {total_coll:,}  "
          f"({100 * total_coll / N_pairs_total:.1f}%)")
    print(f"  Final empirical θ: {running_theta[-1]:.4f}  "
          f"vs true θ(C*): {sc['theta_true']:.4f}")

    fig2, ax = plt.subplots(figsize=(7, 4.2))

    ax.axhline(sc["theta_true"], color="#E24A33", linewidth=1.5,
               label=rf"True $\theta(C^*)$ = {sc['theta_true']:.4f}")
    ax.axhline(sc["theta_oracle"], color="#2CA02C", linewidth=1.5,
               linestyle="--",
               label=rf"Thm 1 (true $o$) = {sc['theta_oracle']:.4f}")

    ax.plot(checkpoints, running_theta, color="#4878CF", linewidth=1.3,
            label=r"Empirical $\hat{\theta}$ from pairs")

    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated draw pairs", fontsize=11)
    ax.set_ylabel(r"$\theta = P(S \mid \mathcal{C})$", fontsize=11)
    ax.set_title(
        r"Empirical collision-based $\theta$  "
        rf"($\sigma^2 = {sc['sigma2']:.0f}$,  "
        rf"$n = {n}$,  $T = {T}$)",
        fontsize=10.5, pad=8,
    )
    ax.legend(fontsize=8.5, framealpha=0.92)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig2.tight_layout()
    fig2.savefig("fig_demo_empirical.pdf", bbox_inches="tight", dpi=300)
    print("Saved fig_demo_empirical.pdf")


if __name__ == "__main__":
    main()

"""
σ² sensitivity sweep — Theorem 1 estimator at scale.

PURPOSE:
  Shows the Theorem 1 closed-form estimate as a function of σ² (entity-degree
  variance) for a larger problem (n=20, m=15, T=200) where exact enumeration
  is infeasible. Instead, Monte Carlo stub matching provides the ground truth.

  Two side-by-side panels, one per value-marginal vector:
    - Left:  uniform o (each value ~13–14 edges)
    - Right: Zipf-skewed o (one value has 60 edges, long tail)

  Each panel shows:
    - Solid line: Theorem 1 closed-form estimate (pure formula, no sampling)
    - Shaded band: 5th–95th percentile of θ(C) from B=500 MC stub-matching
      samples at each σ² grid point
    - Dashed lines: Theorem 1 estimate ± 1 analytic sd (Theorem 2)

  The solid line sitting in the centre of the band demonstrates the mean
  estimator works at scale. The dashed lines validate the analytic variance
  formula (Theorem 2) against the MC spread.

METHOD:
  For each target σ², a "two-level" ℓ vector is constructed deterministically:
  k entities at d_high, (n-k) at d_low, with integer values chosen to hit σ²
  as closely as possible. Both the Theorem 1 curve and the MC band are plotted
  at the *actual* achieved σ² (not the target) to avoid alignment artefacts
  from integer rounding.

OUTPUT: fig_sigma_sweep.pdf (2-panel vector PDF)
NOT YET in main.tex — generated for review.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Theorem 1 closed-form estimator (no access to ℓ, only o, n, σ²)
# ---------------------------------------------------------------------------

def theta_estimate(o, n, sigma2):
    T = sum(o)
    D = sum(a * a for a in o)
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


# ---------------------------------------------------------------------------
# Theorem 2 analytic sd (requires full ℓ for M₃)
# ---------------------------------------------------------------------------

def falling(x, k):
    r = 1
    for j in range(k):
        r *= (x - j)
    return r


def analytic_sd(ell, o):
    """Theorem 2: ensemble sd of θ(C) across the config-model ensemble."""
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
# Feasible σ² range
# ---------------------------------------------------------------------------

def sigma2_bounds(n, T):
    """
    Feasible range of entity-degree variance given n entities and T total edges.
    Min: degrees as equal as possible (T = qn+r gives r entities at q+1, rest at q).
    Max: one dominant emitter with degree T-n+1, all others at 1.
    """
    q, r = divmod(T, n)
    s2_min = r * (n - r) / (n * n)
    s2_max = (n - 1) * (T - n) ** 2 / (n * n)
    return s2_min, s2_max


# ---------------------------------------------------------------------------
# Two-level degree constructor: target a specific σ²
# ---------------------------------------------------------------------------

def two_level_degrees(n, T, sigma2_target):
    """
    Build ℓ with k entries at d_high and (n-k) at d_low,
    choosing the integer solution closest to σ²_target.
    """
    mu = T / n
    best_ell = None
    best_err = float("inf")

    for k in range(1, n):
        nk = n - k
        # With k high and nk low: d_high*k + d_low*nk = T
        # variance = k*(d_high-mu)^2 + nk*(d_low-mu)^2) / n = sigma2
        # Solve: d_high - d_low = sqrt(sigma2 * n / (k * nk / n))
        #   = sqrt(sigma2 * n^2 / (k * nk))
        denom = k * nk
        if denom == 0:
            continue
        gap_sq = sigma2_target * n * n / denom
        if gap_sq < 0:
            continue
        gap = math.sqrt(gap_sq)

        d_low_exact = mu - k * gap / n
        d_high_exact = d_low_exact + gap

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
            actual_mu = T / n
            actual_s2 = sum((li - actual_mu) ** 2 for li in ell) / n
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


# ---------------------------------------------------------------------------
# Stub-matching Monte Carlo
# ---------------------------------------------------------------------------

def stub_match_theta(ell, o, rng):
    """
    One Monte Carlo sample of θ(C) via the bipartite configuration model.
    Creates T entity-side stubs and T value-side stubs, shuffles the value
    stubs uniformly at random, pairs them off, and reads the resulting
    contingency table C. Returns θ(C) = N(C)/D = (Σ c_ia²) / (Σ o_a²).
    """
    T = sum(ell)
    n, m = len(ell), len(o)

    entity_stubs = np.repeat(np.arange(n), ell)
    value_stubs = np.repeat(np.arange(m), o)
    rng.shuffle(value_stubs)

    D = sum(a * a for a in o)
    counts = np.zeros((n, m), dtype=np.int64)
    for idx in range(T):
        counts[entity_stubs[idx], value_stubs[idx]] += 1
    N = int(np.sum(counts * counts))
    return N / D


def mc_band(ell, o, B, rng):
    samples = [stub_match_theta(ell, o, rng) for _ in range(B)]
    return np.array(samples)


# ---------------------------------------------------------------------------
# Value-marginal vectors
# ---------------------------------------------------------------------------

def make_uniform_o(m, T):
    """Value degrees as equal as possible: each o_a ≈ T/m."""
    base = T // m
    o = [base] * m
    for i in range(T - sum(o)):
        o[i] += 1
    return o


def make_zipf_o(m, T):
    """Value degrees proportional to 1/a (Zipf), rounded to integers summing to T."""
    raw = [1.0 / (a + 1) for a in range(m)]
    total = sum(raw)
    o = [max(1, int(round(r / total * T))) for r in raw]
    diff = T - sum(o)
    i = 0
    while diff > 0:
        o[i % m] += 1
        diff -= 1
        i += 1
    while diff < 0:
        for j in range(m - 1, -1, -1):
            if o[j] > 1:
                o[j] -= 1
                diff += 1
                if diff == 0:
                    break
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n, m, T = 20, 15, 200
    B = 500
    n_grid = 40

    o_uniform = make_uniform_o(m, T)
    o_zipf = make_zipf_o(m, T)

    print(f"o_uniform: {o_uniform}  sum={sum(o_uniform)}")
    print(f"o_zipf:    {o_zipf}  sum={sum(o_zipf)}")

    s2_min, s2_max = sigma2_bounds(n, T)
    print(f"σ² range: [{s2_min:.2f}, {s2_max:.2f}]")

    grid = np.linspace(s2_min, s2_max, n_grid)

    results = {}
    for label, o in [("uniform", o_uniform), ("Zipf", o_zipf)]:
        sigma2_actual = []
        est_curve = []
        sd_curve = []
        mc_mean_curve = []
        lo_band = []
        hi_band = []

        seen_s2 = set()
        for idx, s2 in enumerate(grid):
            ell = two_level_degrees(n, T, s2)
            actual_s2 = sum((li - T / n) ** 2 for li in ell) / n
            key = round(actual_s2, 4)
            if key in seen_s2:
                continue
            seen_s2.add(key)

            est = theta_estimate(o, n, actual_s2)
            sd = analytic_sd(ell, o)
            est_curve.append(est)
            sd_curve.append(sd)
            sigma2_actual.append(actual_s2)

            samples = mc_band(ell, o, B, RNG)
            mc_mean_curve.append(np.mean(samples))
            lo_band.append(np.percentile(samples, 5))
            hi_band.append(np.percentile(samples, 95))

            if idx % 10 == 0:
                print(f"  {label} target σ²={s2:.1f}  actual σ²={actual_s2:.1f}: "
                      f"est={est:.4f}  MC mean={mc_mean_curve[-1]:.4f}")

        results[label] = {
            "s2": np.array(sigma2_actual),
            "est": np.array(est_curve),
            "sd": np.array(sd_curve),
            "mc_mean": mc_mean_curve,
            "lo": lo_band,
            "hi": hi_band,
        }

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    color_line = "#4878CF"
    color_band = "#4878CF"
    color_sd = "#E24A33"
    color_prior = "#888888"

    panels = [
        ("uniform", o_uniform, axes[0]),
        ("Zipf", o_zipf, axes[1]),
    ]

    for label, o, ax in panels:
        r = results[label]
        s2 = r["s2"]

        alpha_curve = np.array([1.0 / n + n * s / (T * T) for s in s2])

        ax.fill_between(s2, r["lo"], r["hi"], color=color_band, alpha=0.18,
                        label="MC 5th–95th pctl")
        ax.plot(s2, r["est"], color=color_line, linewidth=2.0,
                label="Thm 1 estimate")
        ax.plot(s2, r["est"] + r["sd"], color=color_sd, linewidth=1.2,
                linestyle="--", label=r"Thm 1 $\pm\,1$ sd (Thm 2)")
        ax.plot(s2, r["est"] - r["sd"], color=color_sd, linewidth=1.2,
                linestyle="--")
        ax.plot(s2, alpha_curve, color=color_prior, linewidth=1.5,
                linestyle=":", label=r"Prior $\alpha = P(S)$")

        ax.set_xlabel(r"Entity-degree variance $\sigma^2$", fontsize=11)
        ax.set_title(
            rf"$o$ {label}  ($n = {n},\; m = {m},\; T = {T}$)",
            fontsize=10.5, pad=8,
        )
        ax.legend(fontsize=8, loc="upper left", framealpha=0.88)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\theta = P(S \mid \mathcal{C})$", fontsize=11)

    fig.tight_layout()
    fig.savefig("fig_sigma_sweep.pdf", bbox_inches="tight", dpi=300)
    print("\nSaved fig_sigma_sweep.pdf")


if __name__ == "__main__":
    main()

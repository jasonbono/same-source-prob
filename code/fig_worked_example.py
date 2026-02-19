"""
Worked example: quasi-identifier collisions in de-identified event logs.

Demonstrates how an analyst would apply the Theorem 1 estimator to a
realistic scenario where entity identifiers are unavailable.

Scenario
--------
  n = 100 entities (e.g. patients, devices, or users)
  T = 5,000 total event-log rows
  Entity activity volumes follow a heavy-tailed (lognormal) distribution:
    a few power users generate many events, most generate few.
  Two quasi-identifier keys of different granularity:
    Coarse (m =  30): e.g. {age bracket, sex}
    Fine   (m = 300): e.g. {birth year, sex, ZIP3}

For each key a sigma-squared sensitivity sweep shows theta-hat as a function
of the assumed entity heterogeneity.  The true theta(C*) from the ground-truth
graph and the true sigma-squared are marked, showing that the estimator
recovers the correct value at the correct heterogeneity level.

OUTPUT: fig_worked_example.pdf (2-panel figure)
"""

import math

import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Core estimator (Theorem 1)
# ---------------------------------------------------------------------------

def theta_estimate(o, n, sigma2):
    """Theorem 1 closed-form estimator: E[theta(C) | o, sigma^2]."""
    o = np.asarray(o, dtype=float)
    T = o.sum()
    D = (o * o).sum()
    if D == 0 or T <= 1:
        return np.nan
    alpha = 1.0 / n + n * sigma2 / (T * T)
    M2 = T * T * alpha - T
    p2 = (D - T) / (T * (T - 1))
    return (T + M2 * p2) / D


def theta_exact(C):
    """Exact theta(C) = sum c_ia^2 / sum o_a^2 for a known incidence matrix."""
    o = C.sum(axis=0)
    D = float((o * o).sum())
    N = float((C * C).sum())
    return N / D if D > 0 else 0.0


def sample_config_model(ell, o, rng):
    """Sample one graph C from the bipartite configuration model."""
    n, m = len(ell), len(o)
    entity_stubs = np.repeat(np.arange(n), ell)
    value_stubs = np.repeat(np.arange(m), o)
    rng.shuffle(value_stubs)
    C = np.zeros((n, m), dtype=np.int64)
    np.add.at(C, (entity_stubs, value_stubs), 1)
    return C


def sigma2_bounds(n, T):
    """Feasible range of entity-degree variance."""
    _, r = divmod(T, n)
    s2_min = r * (n - r) / (n * n)
    s2_max = (n - 1) * ((T - n) ** 2) / (n ** 2)
    return s2_min, s2_max


# ---------------------------------------------------------------------------
# Degree generators
# ---------------------------------------------------------------------------

def make_lognormal_degrees(n, T, log_sigma, rng):
    """
    Integer entity degrees from a discretised lognormal distribution.
    Guarantees ell_i >= 1 and sum ell_i = T.
    """
    mu = T / n
    log_mu = math.log(mu) - log_sigma ** 2 / 2
    raw = rng.lognormal(log_mu, log_sigma, size=n)
    raw = np.maximum(raw, 1.0)
    raw = raw / raw.sum() * T
    ell = np.maximum(np.round(raw).astype(int), 1)
    diff = T - ell.sum()
    if diff > 0:
        for i in rng.choice(n, size=abs(diff), replace=True):
            ell[i] += 1
    elif diff < 0:
        for _ in range(-diff):
            cand = np.where(ell > 1)[0]
            if len(cand) == 0:
                break
            ell[cand[np.argmax(ell[cand])]] -= 1
    return ell


def make_zipf_values(m, T, exponent=1.0):
    """Value degrees with Zipf-like distribution, o_a >= 1, sum o_a = T."""
    raw = np.array([1.0 / (a + 1) ** exponent for a in range(m)])
    raw = raw / raw.sum() * T
    o = np.maximum(np.round(raw).astype(int), 1)
    diff = T - o.sum()
    if diff > 0:
        for i in range(diff):
            o[i % m] += 1
    elif diff < 0:
        for i in range(m - 1, -1, -1):
            while o[i] > 1 and diff < 0:
                o[i] -= 1
                diff += 1
    return o


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n = 100
    T = 5000

    ell = make_lognormal_degrees(n, T, log_sigma=1.5, rng=RNG)
    mu = T / n
    true_sigma2 = float(np.sum((ell.astype(float) - mu) ** 2) / n)

    print(f"Scenario: n = {n} entities,  T = {T} events")
    print(f"  Entity degrees: min={ell.min()}, median={int(np.median(ell))}, "
          f"max={ell.max()}, mean={ell.mean():.1f}, sigma^2={true_sigma2:.1f}")

    configs = [
        {"label": "Coarse key", "m": 30,  "zipf_exp": 0.8},
        {"label": "Fine key",   "m": 300, "zipf_exp": 0.5},
    ]

    s2_min, s2_max = sigma2_bounds(n, T)
    sweep_hi = min(true_sigma2 * 4, s2_max)
    sigma2_grid = np.linspace(s2_min, sweep_hi, 120)

    K_extra = 8

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, cfg in zip(axes, configs):
        m = cfg["m"]
        o = make_zipf_values(m, T, exponent=cfg["zipf_exp"])

        C_star = sample_config_model(ell, o, RNG)
        theta_true = theta_exact(C_star)
        theta_hat_at_true = theta_estimate(o, n, true_sigma2)
        theta_hat_at_zero = theta_estimate(o, n, s2_min)

        extra_thetas = [
            theta_exact(sample_config_model(ell, o, RNG))
            for _ in range(K_extra)
        ]

        o_arr = np.asarray(o, dtype=float)
        D = float((o_arr ** 2).sum())
        p_coll = D / (T * T)

        print(f"\n  {cfg['label']} (m={m}):")
        print(f"    Value counts: min={min(o)}, max={max(o)}")
        print(f"    Collision rate P(coll) = {p_coll:.4f}")
        print(f"    True theta(C*)         = {theta_true:.4f}")
        print(f"    theta-hat(true sigma^2)= {theta_hat_at_true:.4f}")
        print(f"    theta-hat(sigma^2=min) = {theta_hat_at_zero:.4f}")
        print(f"    Extra C* thetas: "
              + ", ".join(f"{et:.4f}" for et in extra_thetas))

        curve = [theta_estimate(o, n, s2) for s2 in sigma2_grid]
        alpha_curve = [1.0 / n + n * s2 / (T * T) for s2 in sigma2_grid]

        # Ensemble spread (additional C* samples)
        for k, et in enumerate(extra_thetas):
            kw = {"label": r"Other $C$ samples"} if k == 0 else {}
            ax.axhline(et, color="#E24A33", linewidth=0.5, alpha=0.35,
                       zorder=2, **kw)

        # True theta(C*)
        ax.axhline(theta_true, color="#E24A33", linewidth=1.5, zorder=3,
                   label=rf"True $\theta(C^*)$ = {theta_true:.3f}")

        # True sigma^2
        ax.axvline(true_sigma2, color="#AAAAAA", linewidth=1.0, linestyle=":",
                   zorder=2, label=rf"True $\sigma^2$ = {true_sigma2:.0f}")

        # Theorem 1 sensitivity curve
        ax.plot(sigma2_grid, curve, color="#4878CF", linewidth=2.0, zorder=5,
                label=r"$\hat{\theta}(\sigma^2)$ (Thm 1)")

        # Prior baseline
        ax.plot(sigma2_grid, alpha_curve, color="#888888", linewidth=1.5,
                linestyle=":", zorder=4, label=r"Prior $\alpha = P(S)$")

        # Intersection dot
        ax.plot(true_sigma2, theta_hat_at_true, "o", color="#4878CF",
                markersize=7, zorder=6, markeredgecolor="white",
                markeredgewidth=1.2)

        ax.set_xlabel(r"Assumed entity-degree variance $\sigma^2$",
                      fontsize=10)
        ax.set_title(rf"{cfg['label']}  ($m = {m}$)",
                     fontsize=10.5, pad=8)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.90)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        r"$\hat{\theta} \approx P(S \mid \mathcal{C})$", fontsize=11
    )

    fig.suptitle(
        r"Worked example: $\hat{\theta}$ sensitivity to $\sigma^2$"
        rf"  ($n = {n},\; T = {T:,}$)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig_worked_example.pdf", bbox_inches="tight", dpi=300)
    print(f"\nSaved fig_worked_example.pdf")


if __name__ == "__main__":
    main()

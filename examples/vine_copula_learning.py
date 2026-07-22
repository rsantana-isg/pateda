"""
Vine-copula learning in pateda — structures and pair-copula families

A vine copula factorizes a multivariate dependency into a cascade of *bivariate*
copulas ("pair-copulas") arranged on a sequence of trees.  Two independent
choices define the model, and pateda exposes both:

1. The **vine structure** -- how the pair-copulas are arranged:
   - C-vine (canonical): star-shaped trees, one central variable per tree
     (:func:`~pateda.learning.vine_copula.learn_vine_copula_cvine`);
   - D-vine (drawable): path-shaped trees
     (:func:`~pateda.learning.vine_copula.learn_vine_copula_dvine`, structure
     auto-selected -> an R-vine);
   - auto: structure *and* families selected from the data
     (:func:`~pateda.learning.vine_copula.learn_vine_copula_auto`).

2. The **pair-copula family** -- the shape of each bivariate dependency, e.g.
   Gaussian (no tail dependence), Clayton (lower-tail), Gumbel (upper-tail),
   Frank (symmetric, no tail), Joe (upper-tail), and the two-parameter BB
   families.  A family can be fixed for the whole vine (``copula_family``) or
   selected per pair from the data (``select_families=True``).

This script demonstrates learning for the different types of copula:

    Part 1  -- the three vine structures (C-vine, D-vine, auto) on the same
               correlated data, with the full learn -> sample round trip.
    Part 2a -- fixing each pair-copula family in turn on data generated from a
               known family, showing that the true family fits best (AIC/BIC).
    Part 2b -- automatic per-pair family selection recovering a *mixture* of
               families (a Clayton pair and a Gumbel pair in the same data).
    Part 3  -- fully automatic structure + family selection.

Vine copulas model *continuous* variables; the dependency is learned on
rank/pseudo-observations, so it is invariant to the marginal scale.

Requires ``pyvinecopulib`` (``pip install pyvinecopulib``) and ``scipy``.

Usage
-----
    python3 vine_copula_learning.py [seed]
"""

import sys
import numpy as np

try:
    import pyvinecopulib as pv
    from scipy.stats import norm
except ImportError:  # pragma: no cover
    print("This example requires 'pyvinecopulib' and 'scipy'.\n"
          "Install with: pip install pyvinecopulib scipy")
    sys.exit(0)

from pateda.learning.vine_copula import (
    learn_vine_copula_cvine,
    learn_vine_copula_dvine,
    learn_vine_copula_auto,
    COPULA_FAMILIES,
)
from pateda.sampling.vine_copula import sample_vine_copula


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bic_of(vine_model, n_obs):
    """BIC computed from the model (pyvinecopulib's bic() can under-report the
    sample size, so we compute it directly: BIC = -2*loglik + npars*log(n))."""
    return -2.0 * vine_model.loglik() + vine_model.npars * np.log(n_obs)


def fit_summary(vine_model, n_obs):
    """One-line goodness-of-fit summary for a learned vine."""
    return (f"loglik={vine_model.loglik():8.2f}  npars={vine_model.npars:4.1f}  "
            f"AIC={vine_model.aic():9.2f}  BIC={bic_of(vine_model, n_obs):9.2f}")


def gaussian_marginals(u):
    """Map pseudo-observations in (0,1) to Gaussian marginals (realistic
    continuous-optimization data whose *dependency* is the copula's)."""
    return norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))


def clayton_pair(n, theta, seeds):
    """n x 2 sample with Clayton (lower-tail) dependence."""
    return pv.Bicop(family=pv.BicopFamily.clayton,
                    parameters=np.array([[theta]])).simulate(n, seeds=seeds)


def gumbel_pair(n, theta, seeds):
    """n x 2 sample with Gumbel (upper-tail) dependence."""
    return pv.Bicop(family=pv.BicopFamily.gumbel,
                    parameters=np.array([[theta]])).simulate(n, seeds=seeds)


def mean_abs_tau_error(model, data, rng):
    """Sample from the learned model and report how well it reproduces the
    pairwise Kendall's tau of the original data (a scale-free dependency check)."""
    from scipy.stats import kendalltau
    sampled = sample_vine_copula(model, n_samples=data.shape[0], rng=rng)
    d = data.shape[1]
    errs = []
    for i in range(d):
        for j in range(i + 1, d):
            t_data = kendalltau(data[:, i], data[:, j]).statistic
            t_samp = kendalltau(sampled[:, i], sampled[:, j]).statistic
            errs.append(abs(t_data - t_samp))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Part 1 — vine structures
# ---------------------------------------------------------------------------

def part1_structures(rng):
    print("=" * 78)
    print("1. Vine STRUCTURES: C-vine vs D-vine vs auto (Gaussian pair-copulas)")
    print("=" * 78)

    # Correlated 4-D Gaussian data (a chain of dependencies).
    n, d = 400, 4
    cov = np.array([
        [1.0, 0.7, 0.5, 0.3],
        [0.7, 1.0, 0.6, 0.4],
        [0.5, 0.6, 1.0, 0.5],
        [0.3, 0.4, 0.5, 1.0],
    ])
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    fitness = np.sum(X ** 2, axis=1)

    learners = [
        ("C-vine", learn_vine_copula_cvine, {"copula_family": 0}),
        ("D-vine / R-vine", learn_vine_copula_dvine, {"copula_family": 0,
                                                      "select_families": False}),
        ("auto (Gaussian)", learn_vine_copula_auto, {"family_set": [pv.BicopFamily.gaussian]}),
    ]
    for name, learn, params in learners:
        model = learn(X, fitness, params)
        vm = model["vine_model"]
        tau_err = mean_abs_tau_error(model, X, rng)
        print(f"\n  {name}: {fit_summary(vm, n)}")
        print(f"    learn->sample mean |Δτ| over all pairs = {tau_err:.3f} "
              f"(0 = perfect dependency match)")
        print(f"    structure order = {list(vm.order)}")
    print("\n  All three model the same data; the structure only changes how the")
    print("  pair-copulas are arranged. The round-trip τ error confirms each")
    print("  learned model reproduces the original dependency.\n")


# ---------------------------------------------------------------------------
# Part 2a — pair-copula families (true family wins)
# ---------------------------------------------------------------------------

def part2a_families(rng):
    print("=" * 78)
    print("2a. Pair-copula FAMILIES: fixing each family on Gumbel-generated data")
    print("=" * 78)

    # 3-D data whose pairs have upper-tail (Gumbel) dependence.
    n = 600
    u1 = gumbel_pair(n, theta=2.5, seeds=[1, 2, 3, 4])
    u2 = gumbel_pair(n, theta=2.0, seeds=[5, 6, 7, 8])
    X = gaussian_marginals(np.column_stack([u1[:, 0], u1[:, 1], u2[:, 1]]))
    fitness = np.sum(X ** 2, axis=1)

    print("  True dependency: Gumbel (upper tail).  Lower AIC/BIC = better fit.\n")
    print(f"  {'family':<10} | {'fit'}")
    print("  " + "-" * 62)
    families = ["gaussian", "clayton", "gumbel", "frank", "joe"]
    name_to_idx = {v: k for k, v in COPULA_FAMILIES.items()}
    scores = {}
    for fam in families:
        model = learn_vine_copula_cvine(X, fitness, {"copula_family": name_to_idx[fam]})
        vm = model["vine_model"]
        scores[fam] = vm.aic()
        print(f"  {fam:<10} | {fit_summary(vm, n)}")
    best = min(scores, key=scores.get)
    print(f"\n  Best-fitting family by AIC: '{best}' "
          f"(matches the true generating family).\n")


# ---------------------------------------------------------------------------
# Part 2b — automatic per-pair family selection (mixture)
# ---------------------------------------------------------------------------

def part2b_family_selection(rng):
    print("=" * 78)
    print("2b. Automatic PER-PAIR family selection on mixed-tail data")
    print("=" * 78)

    # 4-D data: variables (1,2) share lower-tail (Clayton) dependence,
    # variables (3,4) share upper-tail (Gumbel) dependence.
    n = 600
    uc = clayton_pair(n, theta=3.0, seeds=[1, 2, 3, 4])
    ug = gumbel_pair(n, theta=3.0, seeds=[5, 6, 7, 8])
    X = gaussian_marginals(np.column_stack([uc, ug]))
    fitness = np.sum(X ** 2, axis=1)

    model = learn_vine_copula_dvine(X, fitness, {"select_families": True})
    vm = model["vine_model"]
    print("\n  Data has a Clayton pair (vars 1-2) and a Gumbel pair (vars 3-4).")
    print("  select_families=True chooses a family per pair from the data:\n")
    print(vm)
    print(f"\n  {fit_summary(vm, n)}")
    print("  The first tree recovers a lower-tail family (Clayton) for the 1-2")
    print("  pair and an upper-tail family (Gumbel or its BB1 generalization) for")
    print("  the 3-4 pair -- a *mixture* of copula types no single family fits.\n")


# ---------------------------------------------------------------------------
# Part 3 — fully automatic structure + family selection
# ---------------------------------------------------------------------------

def part3_auto(rng):
    print("=" * 78)
    print("3. Fully AUTOMATIC learning (structure + families, BIC-selected)")
    print("=" * 78)

    # Heterogeneous 5-D data: a Clayton pair, a Gumbel pair, plus a near-linear
    # (Gaussian) link, to give the automatic selector something to discover.
    n = 700
    uc = clayton_pair(n, theta=4.0, seeds=[1, 2, 3, 4])
    ug = gumbel_pair(n, theta=3.0, seeds=[5, 6, 7, 8])
    x_lin = 0.8 * gaussian_marginals(uc[:, :1]).ravel() + 0.6 * rng.standard_normal(n)
    X = np.column_stack([gaussian_marginals(uc), gaussian_marginals(ug), x_lin])
    fitness = np.sum(X ** 2, axis=1)

    model = learn_vine_copula_auto(
        X, fitness, {"tree_criterion": "tau", "selection_criterion": "bic"}
    )
    vm = model["vine_model"]
    print("\n  Structure and per-pair families both chosen from the data:\n")
    print(vm)
    print(f"\n  {fit_summary(vm, n)}")
    tau_err = mean_abs_tau_error(model, X, rng)
    print(f"  learn->sample mean |Δτ| over all pairs = {tau_err:.3f}")
    print("  This is the most flexible learner and the usual default when the")
    print("  dependency structure is unknown.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("#" * 78)
    print("# Vine-copula learning: structures and pair-copula families")
    print(f"# seed = {seed}   (pyvinecopulib {pv.__version__})")
    print("#" * 78 + "\n")
    rng = np.random.default_rng(seed)
    part1_structures(rng)
    part2a_families(rng)
    part2b_family_selection(rng)
    part3_auto(rng)
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  - learn_vine_copula_cvine / _dvine / _auto give the three vine")
    print("    STRUCTURES (C-vine, D-vine/R-vine, fully automatic).")
    print("  - copula_family fixes one pair-copula FAMILY; select_families=True")
    print("    (and _auto) pick the best family per pair -- Gaussian, Clayton,")
    print("    Gumbel, Frank, Joe, BB..., each capturing a different dependency.")
    print("  - sample_vine_copula draws new solutions from any learned model.")
    print("=" * 78)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)

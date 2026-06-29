"""
Compare discrete *neural-network* EDAs (pateda_nn) on real-world benchmarks using
a plain **binary** representation (every variable has cardinality 2).

This is the binary-representation counterpart of
``compare_mixed_cardinality_nn_edas_rw.py``.  Instead of grouping the binary
variables into mixed-cardinality super-variables, each benchmark is optimised
directly in its native binary space, so the NN-EDAs operate on {0,1} vectors of
length ``n_binary``.  This lets the comparison use the rich family of
*binary-specific* models in ``pateda_nn`` (including all five binary Denoising
Diffusion variants), with several variants per model family that differ in their
relevant hyper-parameters / learning strategies.

Benchmarks (binary, reduced sizes)
----------------------------------
  * Ising -> SG_64_1   (64 binary spins)
  * UBQP  -> bqp50      (50 binary variables)
  * SAT   -> uf50-01    (50 binary variables)   <- SAT instance with >= 50 vars

Model families and variants
---------------------------
  * VAE       : basic, large latent, low-KL, Extended (fitness predictor),
                Regression-VAE, Moment-matching VAE, Conditional VAE
  * GAN       : basic, WGAN-GP, repulsion
  * DBD       : Current->Selected (CS), Current->Closest (CD)
  * Backdrive : standard, Huber loss, ranking loss, perturb-best init
  * Dendiff   : Gumbel-Softmax, corruption, straight-through (STE), deterministic

Usage
-----
    # defaults: pop_size=200, n_gen=15, n_runs=2
    python scripts/compare_binary_nn_edas_rw.py

    # choose the run budget on the command line
    python scripts/compare_binary_nn_edas_rw.py --pop-size 300 --n-gen 30 --n-runs 5

    # fast smoke preset (pop_size=60, n_gen=4, n_runs=1); explicit flags still win
    python scripts/compare_binary_nn_edas_rw.py --quick
    python scripts/compare_binary_nn_edas_rw.py --help
"""

import argparse
import sys
import time
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from pateda.functions.discrete.ising import (
    eval_ising,
    load_ising_benchmark_instance,
)
from pateda.functions.discrete.sat import (
    evaluate_sat,
    load_sat_benchmark_instance,
)
from pateda.functions.discrete.ubqp import (
    evaluate_ubqp,
    load_ubqp_benchmark_instance,
)

# --- Binary NN-EDA learning functions ---------------------------------------
from pateda_nn.learning.discrete_vae import (
    learn_binary_vae,
    learn_binary_cvae,
    learn_binary_regvae,
    learn_binary_momvae,
)
from pateda_nn.learning.discrete_gan import (
    learn_binary_gan,
    learn_binary_gan_wgan_gp,
    learn_binary_gan_repulsion,
)
from pateda_nn.learning.discrete_dbd import (
    learn_binary_dbd_cs,
    learn_binary_dbd_cd,
)
from pateda_nn.learning.discrete_backdrive import learn_binary_backdrive
from pateda_nn.learning.discrete_backdrive_huber import learn_binary_backdrive_huber
from pateda_nn.learning.discrete_backdrive_ranking import learn_binary_backdrive_ranking
from pateda_nn.learning.discrete_dendiff_gumbel import learn_discrete_dendiff_gumbel
from pateda_nn.learning.discrete_dendiff_corruption import learn_discrete_dendiff_corruption
from pateda_nn.learning.discrete_dendiff_ste import learn_discrete_dendiff_ste
from pateda_nn.learning.discrete_dendiff_deterministic import learn_discrete_dendiff_deterministic

# --- Binary NN-EDA sampling functions ---------------------------------------
from pateda_nn.sampling.discrete_neural import (
    sample_binary_vae,
    sample_binary_cvae,
    sample_binary_regvae,
    sample_binary_momvae,
    sample_binary_gan,
    sample_binary_backdrive,
)
from pateda_nn.sampling.discrete_dbd import (
    sample_binary_dbd_cs,
    sample_binary_dbd_cd,
)
from pateda_nn.sampling.discrete_dendiff import (
    sample_discrete_dendiff_gumbel,
    sample_discrete_dendiff_corruption,
    sample_discrete_dendiff_ste,
    sample_discrete_dendiff_deterministic,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

#   (problem_type, instance_name)  -- all evaluated in native binary space
PROBLEMS = [
    ("Ising", "SG_64_1"),
    ("UBQP",  "bqp50"),
    ("SAT",   "uf50-01"),   # SAT instance with >= 50 variables
]

# Reduced budgets (NN-EDAs retrain a network every generation).  These are the
# defaults; override them on the command line (see ``parse_cli``).
POP_SIZE  = 200
N_GEN     = 15
SEL_RATIO = 0.30
N_RUNS    = 2
SEEDS     = list(range(1, N_RUNS + 1))
QUICK     = False

# Preset applied by --quick (a fast smoke run that still exercises every family).
QUICK_PRESET = dict(pop_size=60, n_gen=4, n_runs=1)


def parse_cli(argv):
    """Parse command-line overrides for the run budget and apply them.

    Recognised options (all optional)::

        --pop-size N    population size          (default: 200)
        --n-gen N       number of generations    (default: 15)
        --n-runs N      number of runs / seeds   (default: 2; seeds = 1..N)
        --sel-ratio F   truncation selection ratio (default: 0.30)
        --quick         fast smoke preset (pop=60, gen=4, runs=1); any explicit
                        flag given alongside --quick overrides the preset value

    The parsed values are written back to the module-level globals used by the
    EDA driver.
    """
    global POP_SIZE, N_GEN, N_RUNS, SEL_RATIO, SEEDS, QUICK

    parser = argparse.ArgumentParser(
        description="Compare binary neural-network EDAs (pateda_nn) on "
                    "real-world benchmarks.",
    )
    parser.add_argument("--pop-size", type=int, default=None,
                        help=f"population size (default: {POP_SIZE})")
    parser.add_argument("--n-gen", type=int, default=None,
                        help=f"number of generations (default: {N_GEN})")
    parser.add_argument("--n-runs", type=int, default=None,
                        help=f"number of runs / seeds, seeds = 1..n_runs (default: {N_RUNS})")
    parser.add_argument("--sel-ratio", type=float, default=None,
                        help=f"truncation selection ratio (default: {SEL_RATIO})")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke preset (pop=60, gen=4, runs=1); "
                             "explicit flags still override it")

    argv = list(argv)
    if argv and argv[0].lower() == "quick":   # backwards-compatible positional
        argv[0] = "--quick"
    args = parser.parse_args(argv)

    if args.quick:
        QUICK = True
        POP_SIZE = QUICK_PRESET["pop_size"]
        N_GEN = QUICK_PRESET["n_gen"]
        N_RUNS = QUICK_PRESET["n_runs"]
    if args.pop_size is not None:
        POP_SIZE = args.pop_size
    if args.n_gen is not None:
        N_GEN = args.n_gen
    if args.n_runs is not None:
        N_RUNS = args.n_runs
    if args.sel_ratio is not None:
        SEL_RATIO = args.sel_ratio
    SEEDS = list(range(1, N_RUNS + 1))
    return args


# ---------------------------------------------------------------------------
# Problem loading (native binary space)
# ---------------------------------------------------------------------------

def _single_objective(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type: str, instance_name: str):
    """Load a benchmark instance and return a binary fitness function.

    Returns:
        fitness_func : callable(1-D binary array) -> scalar fitness (maximised).
        n_vars       : number of binary variables.
        optimal      : known optimum (or "Unknown").
    """
    problem_key = problem_type.upper()

    if problem_key == "SAT":
        sat_instance, optimal = load_sat_benchmark_instance(instance_name)
        n_vars = sat_instance.n_vars

        def fitness_func(solution):
            return _single_objective(evaluate_sat(np.asarray(solution), sat_instance))

    elif problem_key == "ISING":
        n_vars, lattice, inter, optimal = load_ising_benchmark_instance(instance_name)

        def fitness_func(solution):
            return -eval_ising(np.asarray(solution), lattice, inter)

    elif problem_key == "UBQP":
        ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
        n_vars = ubqp_instance.n_vars

        def fitness_func(solution):
            return _single_objective(evaluate_ubqp(np.asarray(solution), ubqp_instance))

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return fitness_func, n_vars, optimal


# ---------------------------------------------------------------------------
# Binary NN-EDA variant registry
#
# Each variant exposes two closures hiding the family-specific calling
# conventions behind a uniform interface:
#   learn(selected, sel_fit, current, current_fit)             -> model
#   sample(model, n_samples, selected, current, current_fit)   -> population
# ---------------------------------------------------------------------------

class NNEDAVariant:
    def __init__(self, name: str, family: str, learn: Callable, sample: Callable):
        self.name = name
        self.family = family
        self.learn = learn
        self.sample = sample


def _simple_variant(name, family, learn_fn, sample_fn, learn_params, sample_params):
    """VAE / GAN / Dendiff: learn(pop, fit, params); sample(model, n, params)."""
    return NNEDAVariant(
        name, family,
        lambda sel, sf, cur, cf, lp=learn_params, fn=learn_fn: fn(sel, sf, lp),
        lambda m, n, sel, cur, cf, sp=sample_params, fn=sample_fn: fn(m, n, sp),
    )


def _dbd_cs_variant(name, learn_params, sample_params):
    """DbD-CS learns current->selected (paired) and refines the selected pop."""
    def _learn(sel, sf, cur, cf, lp=learn_params):
        n = len(sel)
        idx = np.random.choice(len(cur), size=n, replace=len(cur) < n)
        return learn_binary_dbd_cs(cur[idx], sel, lp)

    return NNEDAVariant(
        name, "DBD",
        _learn,
        lambda m, n, sel, cur, cf, sp=sample_params: sample_binary_dbd_cs(m, n, sel, sp),
    )


def _dbd_cd_variant(name, learn_params, sample_params):
    """DbD-CD learns current->selected and refines from the current pop."""
    def _learn(sel, sf, cur, cf, lp=learn_params):
        n = len(sel)
        idx = np.random.choice(len(cur), size=n, replace=len(cur) < n)
        return learn_binary_dbd_cd(cur[idx], sel, lp)

    return NNEDAVariant(
        name, "DBD",
        _learn,
        lambda m, n, sel, cur, cf, sp=sample_params: sample_binary_dbd_cd(m, n, cur, sp),
    )


def _backdrive_variant(name, learn_fn, learn_params, sample_params):
    def _sample(m, n, sel, cur, cf, sp=sample_params):
        sp = dict(sp)
        if sp.get("init_method") in ("perturb_best", "perturb_selected"):
            sp["current_population"] = cur
            sp["current_fitness"] = cf
        return sample_binary_backdrive(m, n, sp)

    return NNEDAVariant(
        name, "Backdrive",
        lambda sel, sf, cur, cf, lp=learn_params, fn=learn_fn: fn(sel, sf, lp),
        _sample,
    )


def build_variants() -> List[NNEDAVariant]:
    """Construct the list of binary NN-EDA variants to compare."""
    epochs = 8 if QUICK else 30
    gan_epochs = 8 if QUICK else 40

    variants: List[NNEDAVariant] = []

    # ---- VAE family --------------------------------------------------------
    base_vae = dict(epochs=epochs, learning_rate=1e-3,
                    hidden_dims_enc=[64, 32], hidden_dims_dec=[32, 64])
    variants += [
        _simple_variant("VAE", "VAE", learn_binary_vae, sample_binary_vae,
                        dict(base_vae, latent_dim=8, beta_end=1.0), dict(temperature=0.5)),
        _simple_variant("VAE-bigZ", "VAE", learn_binary_vae, sample_binary_vae,
                        dict(base_vae, latent_dim=20, beta_end=1.0), dict(temperature=0.5)),
        _simple_variant("VAE-lowBeta", "VAE", learn_binary_vae, sample_binary_vae,
                        dict(base_vae, latent_dim=8, beta_end=0.3), dict(temperature=0.5)),
        _simple_variant("E-VAE", "VAE", learn_binary_vae, sample_binary_vae,
                        dict(base_vae, latent_dim=8, use_extended=True, fitness_weight=0.2),
                        dict(temperature=0.5)),
        _simple_variant("Reg-VAE", "VAE", learn_binary_regvae, sample_binary_regvae,
                        dict(base_vae, latent_dim=8), dict(temperature=0.5)),
        _simple_variant("Mom-VAE", "VAE", learn_binary_momvae, sample_binary_momvae,
                        dict(base_vae, latent_dim=8, moment_weight=0.2), dict(temperature=0.5)),
        _simple_variant("C-VAE", "VAE", learn_binary_cvae, sample_binary_cvae,
                        dict(base_vae, latent_dim=8), dict(temperature=0.5)),
    ]

    # ---- GAN family --------------------------------------------------------
    base_gan = dict(epochs=gan_epochs, hidden_dims_g=[64, 64], hidden_dims_d=[64, 64],
                    learning_rate_g=2e-4, learning_rate_d=2e-4)
    variants += [
        _simple_variant("GAN", "GAN", learn_binary_gan, sample_binary_gan,
                        dict(base_gan, latent_dim=16), dict()),
        _simple_variant("GAN-WGAN-GP", "GAN", learn_binary_gan_wgan_gp, sample_binary_gan,
                        dict(base_gan, latent_dim=16), dict()),
        _simple_variant("GAN-repulse", "GAN", learn_binary_gan_repulsion, sample_binary_gan,
                        dict(base_gan, latent_dim=16), dict()),
    ]

    # ---- DBD family --------------------------------------------------------
    variants += [
        _dbd_cs_variant("DBD-CS", dict(epochs=epochs), dict()),
        _dbd_cd_variant("DBD-CD", dict(epochs=epochs), dict()),
    ]

    # ---- Backdrive family --------------------------------------------------
    base_bd = dict(epochs=epochs, learning_rate=1e-3)
    bd_sample = dict(n_iterations=50, learning_rate=0.05, temperature=2.0)
    variants += [
        _backdrive_variant("Backdrive", learn_binary_backdrive, base_bd,
                           dict(bd_sample, init_method="random")),
        _backdrive_variant("Backdrive-huber", learn_binary_backdrive_huber, base_bd,
                           dict(bd_sample, init_method="random")),
        _backdrive_variant("Backdrive-rank", learn_binary_backdrive_ranking, base_bd,
                           dict(bd_sample, init_method="random")),
        _backdrive_variant("Backdrive-pBest", learn_binary_backdrive, base_bd,
                           dict(bd_sample, init_method="perturb_best", init_noise=0.1)),
    ]

    # ---- Dendiff family (binary; five denoising-diffusion variants) --------
    base_dd = dict(epochs=epochs, n_timesteps=50, hidden_dims=[64, 32])
    variants += [
        _simple_variant("Dendiff-gumbel", "Dendiff",
                        learn_discrete_dendiff_gumbel, sample_discrete_dendiff_gumbel,
                        dict(base_dd), dict(temperature=0.5)),
        _simple_variant("Dendiff-corrupt", "Dendiff",
                        learn_discrete_dendiff_corruption, sample_discrete_dendiff_corruption,
                        dict(base_dd), dict(temperature=0.5)),
        _simple_variant("Dendiff-ste", "Dendiff",
                        learn_discrete_dendiff_ste, sample_discrete_dendiff_ste,
                        dict(base_dd), dict(temperature=0.5)),
        _simple_variant("Dendiff-determ", "Dendiff",
                        learn_discrete_dendiff_deterministic, sample_discrete_dendiff_deterministic,
                        dict(base_dd), dict(temperature=0.3, deterministic=True)),
    ]

    return variants


# ---------------------------------------------------------------------------
# Generic binary NN-EDA driver
# ---------------------------------------------------------------------------

def _random_binary_population(pop_size: int, n_vars: int,
                              rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, n_vars)).astype(int)


def _evaluate_population(population: np.ndarray, fitness_func: Callable) -> np.ndarray:
    return np.array([float(fitness_func(ind)) for ind in population], dtype=float)


def _binarize(population: np.ndarray) -> np.ndarray:
    return (np.asarray(population) > 0.5).astype(int)


def run_single_seed(variant: NNEDAVariant, n_vars: int, fitness_func: Callable,
                    seed: int) -> Tuple[float, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    population = _random_binary_population(POP_SIZE, n_vars, rng)
    fitness = _evaluate_population(population, fitness_func)

    best_idx = int(np.argmax(fitness))
    best_fit = float(fitness[best_idx])
    best_sol = population[best_idx].copy()

    n_sel = max(2, int(POP_SIZE * SEL_RATIO))
    t0 = time.time()

    for _ in range(N_GEN):
        sel_idx = np.argsort(fitness)[-n_sel:]
        selected = population[sel_idx]
        sel_fit = fitness[sel_idx]

        model = variant.learn(selected, sel_fit, population, fitness)
        new_pop = _binarize(variant.sample(model, POP_SIZE, selected, population, fitness))
        new_fit = _evaluate_population(new_pop, fitness_func)

        # Elitism: carry the global best into the new population.
        worst = int(np.argmin(new_fit))
        if best_fit > float(new_fit[worst]):
            new_pop[worst] = best_sol
            new_fit[worst] = best_fit

        population, fitness = new_pop, new_fit

        gen_best = int(np.argmax(fitness))
        if float(fitness[gen_best]) > best_fit:
            best_fit = float(fitness[gen_best])
            best_sol = population[gen_best].copy()

    return best_fit, time.time() - t0


def run_all_seeds(variant: NNEDAVariant, n_vars: int,
                  fitness_func: Callable) -> Tuple[List[float], List[float]]:
    bests, times = [], []
    for seed in SEEDS:
        best, elapsed = run_single_seed(variant, n_vars, fitness_func, seed)
        bests.append(best)
        times.append(elapsed)
    return bests, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    variants = build_variants()

    print(f"\n{'=' * 96}")
    print("Binary-representation NN-EDA comparison (pateda_nn)")
    print(f"  variants={len(variants)}  pop_size={POP_SIZE}  n_gen={N_GEN}  "
          f"selection_ratio={SEL_RATIO}  n_runs={N_RUNS}  seeds={SEEDS}"
          + ("   [QUICK MODE]" if QUICK else ""))
    print(f"  representation: binary (cardinality 2 for every variable)")
    print(f"  families: VAE, GAN, DBD, Backdrive, Dendiff")
    print(f"{'=' * 96}")

    name_w = max(len(v.name) for v in variants) + 1

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_vars, optimal = load_problem(problem_type, instance_name)

        print(f"\n{'=' * 96}")
        print(f"Problem: {problem_type}  Instance: {instance_name}  "
              f"(n_vars={n_vars}, representation=binary, optimal={optimal})")
        print(f"{'=' * 96}")

        problem_tag = f"{problem_type} {instance_name}"
        for variant in variants:
            try:
                bests, times = run_all_seeds(variant, n_vars, fitness_func)
                mean_best = float(np.mean(bests))
                mean_time = float(np.mean(times))
                bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                print(f"{variant.name:<{name_w}} {variant.family:<10} {problem_tag:<16}: "
                      f"{bests_str}  mean={mean_best:.4f}  time={mean_time:.2f}s")
            except Exception as exc:
                print(f"{variant.name:<{name_w}} {variant.family:<10} {problem_tag:<16}: "
                      f"ERROR -- {exc}")
                traceback.print_exc()

    print()


if __name__ == "__main__":
    parse_cli(sys.argv[1:])
    main()

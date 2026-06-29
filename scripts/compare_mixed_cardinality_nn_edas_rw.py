"""
Compare discrete *neural-network* EDAs (pateda_nn) on real-world benchmarks with
a mixed-cardinality super-variable encoding.

This is the neural-network counterpart of
``compare_mixed_cardinality_edas_rw.py``.  Instead of the classical pateda EDAs
(UMDA, EBNA, ...) it evaluates the deep-generative EDAs implemented in
``packages/pateda_nn`` -- VAE, GAN, DBD, Backdrive and (categorical) Denoising
Diffusion -- under the *mixed-cardinality* assumption: every super-variable may
have a different cardinality.

Super-variable encoding
-----------------------
The binary variables of each benchmark are randomly partitioned into
non-overlapping groups of size uniformly drawn from {1,...,MAX_GROUP_SIZE}.
Group i becomes a single super-variable y_i with cardinality 2^|group_i|, so the
cardinalities are mixed (e.g. {2, 4, 8, 16}).  Each wrapped fitness function
decodes a super-variable solution back to binary (LSB-first) and evaluates the
original binary objective.  A *separate* grouping is built per problem because
the reduced instances differ in size.

Reduced instances (see task: 25 or 64 variables, depending on availability)
---------------------------------------------------------------------------
  * Ising -> SG_64_1   (64 binary spins)            -> matches the "64" target
  * UBQP  -> bqp50      (50 binary variables)        -> smallest packaged UBQP
  * SAT   -> uf20-01    (20 binary variables)        -> closest to the "25" target

Mixed-cardinality support
-------------------------
All NN-EDAs used here accept a *per-variable* cardinality vector:
  * VAE / GAN / DBD  -> one-hot encoding with per-variable category offsets
  * Backdrive        -> per-variable embedding layers
  * Dendiff          -> the categorical denoising-diffusion model added in
                        ``pateda_nn.learning.categorical_dendiff`` (the binary
                        dendiff variants only support cardinality 2).

Many *variants of the same model family* are included, differing in their
relevant hyper-parameters / learning strategies (latent size, KL weight beta,
Gumbel temperature, GAN learning rates, diffusion schedule / steps, Backdrive
initialisation method, ...).

Usage
-----
    # defaults: pop_size=200, n_gen=15, n_runs=2
    python scripts/compare_mixed_cardinality_nn_edas_rw.py

    # choose the run budget on the command line
    python scripts/compare_mixed_cardinality_nn_edas_rw.py --pop-size 300 --n-gen 30 --n-runs 5
    python scripts/compare_mixed_cardinality_nn_edas_rw.py --pop-size 100 --n-gen 10

    # fast smoke preset (pop_size=60, n_gen=4, n_runs=1); explicit flags still win
    python scripts/compare_mixed_cardinality_nn_edas_rw.py --quick
    python scripts/compare_mixed_cardinality_nn_edas_rw.py quick          # legacy form
    python scripts/compare_mixed_cardinality_nn_edas_rw.py --quick --n-runs 3

    python scripts/compare_mixed_cardinality_nn_edas_rw.py --help         # full option list
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

# --- NN-EDA learning / sampling functions (all mixed-cardinality capable) -----
from pateda_nn.learning.discrete_vae import learn_categorical_vae
from pateda_nn.sampling.discrete_neural import (
    sample_categorical_vae,
    sample_categorical_gan,
    sample_discrete_backdrive,
)
from pateda_nn.learning.discrete_gan import learn_categorical_gan
from pateda_nn.learning.discrete_dbd import learn_categorical_dbd
from pateda_nn.sampling.discrete_dbd import sample_categorical_dbd
from pateda_nn.learning.discrete_backdrive import learn_discrete_backdrive
from pateda_nn.learning.categorical_dendiff import learn_categorical_dendiff
from pateda_nn.sampling.categorical_dendiff import sample_categorical_dendiff


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

#   (problem_type, instance_name, expected_n_binary)
PROBLEMS = [
    ("Ising", "SG_64_1", 64),
    ("UBQP",  "bqp50",   50),
    ("SAT",   "uf20-01", 20),
]

# Reduced budgets: NN-EDAs retrain a network every generation, so they are far
# more expensive than the classical EDAs.  These module-level values are the
# defaults; they can be overridden on the command line (see ``parse_cli``).
POP_SIZE  = 200
N_GEN     = 15
SEL_RATIO = 0.30
N_RUNS    = 2
SEEDS     = list(range(1, N_RUNS + 1))
QUICK     = False   # set by --quick; also reduces per-network training epochs

# Super-variable encoding.  MAX_GROUP_SIZE = 4 -> cardinalities in {2, 4, 8, 16}
# (a genuine mix) while keeping the one-hot dimension manageable for training.
MAX_GROUP_SIZE = 4
GROUPING_SEED  = 42
UBQP_THRESHOLD_RATIO = 0.5  # kept for parity with the classical script

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

    The legacy bare ``quick`` positional (``... .py quick``) is still accepted.
    The parsed values are written back to the module-level globals used by the
    EDA driver.
    """
    global POP_SIZE, N_GEN, N_RUNS, SEL_RATIO, SEEDS, QUICK

    parser = argparse.ArgumentParser(
        description="Compare mixed-cardinality neural-network EDAs (pateda_nn) "
                    "on real-world benchmarks.",
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

    # Start from the --quick preset (if requested), then let explicit flags win.
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
# Super-variable partitioning / decoding
# ---------------------------------------------------------------------------

def make_grouping(n_binary_vars: int, max_group_size: int, seed: int) -> List[np.ndarray]:
    """Randomly partition binary variable indices into non-overlapping groups."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_binary_vars)
    groups, i = [], 0
    while i < n_binary_vars:
        remaining = n_binary_vars - i
        size = int(rng.integers(1, min(max_group_size, remaining) + 1))
        groups.append(perm[i:i + size])
        i += size
    return groups


def decode_super_to_binary(super_solution: np.ndarray, groups: List[np.ndarray],
                           n_binary_vars: int) -> np.ndarray:
    """Decode one super-variable individual to binary (LSB-first)."""
    binary = np.zeros(n_binary_vars, dtype=int)
    for sv_idx, group in enumerate(groups):
        val = int(super_solution[sv_idx])
        for bit_pos, bin_idx in enumerate(group):
            binary[int(bin_idx)] = (val >> bit_pos) & 1
    return binary


def _single_objective(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type: str, instance_name: str, groups: List[np.ndarray]):
    """Load an instance and wrap its fitness for super-variable inputs."""
    n_sv = len(groups)
    cardinality_vec = np.array([2 ** len(g) for g in groups], dtype=int)
    problem_key = problem_type.upper()

    if problem_key == "SAT":
        sat_instance, optimal = load_sat_benchmark_instance(instance_name)
        n_binary = sat_instance.n_vars

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return _single_objective(evaluate_sat(binary, sat_instance))

    elif problem_key == "ISING":
        n_binary, lattice, inter, optimal = load_ising_benchmark_instance(instance_name)

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return -eval_ising(binary, lattice, inter)

    elif problem_key == "UBQP":
        ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
        n_binary = ubqp_instance.n_vars

        def fitness_func(solution):
            binary = decode_super_to_binary(np.asarray(solution), groups, n_binary)
            return _single_objective(evaluate_ubqp(binary, ubqp_instance))

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return fitness_func, n_sv, cardinality_vec, n_binary, optimal


# ---------------------------------------------------------------------------
# NN-EDA variant registry
#
# Each variant is a NNEDAVariant with two closures that hide the family-specific
# calling conventions behind a uniform interface:
#   learn(selected, sel_fit, cardinality, current_pop, current_fit) -> model
#   sample(model, n_samples, current_pop, current_fit)             -> population
# ---------------------------------------------------------------------------

class NNEDAVariant:
    def __init__(self, name: str, family: str, learn: Callable, sample: Callable):
        self.name = name
        self.family = family
        self.learn = learn
        self.sample = sample


def _vae_variant(name, learn_params, sample_params):
    return NNEDAVariant(
        name, "VAE",
        lambda sel, f, card, cur, cf, lp=learn_params: learn_categorical_vae(sel, f, card, lp),
        lambda m, n, cur, cf, sp=sample_params: sample_categorical_vae(m, n, sp),
    )


def _gan_variant(name, learn_params, sample_params):
    return NNEDAVariant(
        name, "GAN",
        lambda sel, f, card, cur, cf, lp=learn_params: learn_categorical_gan(sel, f, card, lp),
        lambda m, n, cur, cf, sp=sample_params: sample_categorical_gan(m, n, sp),
    )


def _dbd_variant(name, learn_params, sample_params):
    # DBD learns the transition source -> target by *pairing* the two
    # populations index-by-index, so they must have equal length.  We pair the
    # elite (selected) targets with an equal-sized random sample of the current
    # population as the source.
    def _learn(sel, f, card, cur, cf, lp=learn_params):
        n = len(sel)
        idx = np.random.choice(len(cur), size=n, replace=len(cur) < n)
        source = cur[idx]
        return learn_categorical_dbd(source, sel, card, lp)

    return NNEDAVariant(
        name, "DBD",
        _learn,
        lambda m, n, cur, cf, sp=sample_params: sample_categorical_dbd(m, n, sp),
    )


def _backdrive_variant(name, learn_params, sample_params):
    def _sample(m, n, cur, cf, sp=sample_params):
        sp = dict(sp)
        # Perturb-based initialisations need a reference population.
        if sp.get("init_method") in ("perturb_best", "perturb_selected"):
            sp["current_population"] = cur
            sp["current_fitness"] = cf
        return sample_discrete_backdrive(m, n, sp)

    return NNEDAVariant(
        name, "Backdrive",
        lambda sel, f, card, cur, cf, lp=learn_params: learn_discrete_backdrive(sel, f, card, lp),
        _sample,
    )


def _dendiff_variant(name, learn_params, sample_params):
    return NNEDAVariant(
        name, "Dendiff",
        lambda sel, f, card, cur, cf, lp=learn_params: learn_categorical_dendiff(sel, f, card, lp),
        lambda m, n, cur, cf, sp=sample_params: sample_categorical_dendiff(m, n, sp),
    )


def build_variants() -> List[NNEDAVariant]:
    """Construct the list of NN-EDA variants to compare."""
    # Modest training budgets so the whole sweep finishes in reasonable time.
    epochs = 8 if QUICK else 30
    gan_epochs = 8 if QUICK else 40

    variants: List[NNEDAVariant] = []

    # ---- VAE family: vary latent size, KL weight (beta), Gumbel temperature ----
    base_vae = dict(epochs=epochs, learning_rate=1e-3,
                    hidden_dims_enc=[64, 32], hidden_dims_dec=[32, 64])
    variants += [
        _vae_variant("VAE",          dict(base_vae, latent_dim=8,  beta=1.0),
                     dict(temperature=0.5)),
        _vae_variant("VAE-bigZ",     dict(base_vae, latent_dim=16, beta=1.0),
                     dict(temperature=0.5)),
        _vae_variant("VAE-lowBeta",  dict(base_vae, latent_dim=8,  beta=0.2),
                     dict(temperature=0.5)),
        _vae_variant("VAE-highBeta", dict(base_vae, latent_dim=8,  beta=2.0),
                     dict(temperature=0.5)),
        _vae_variant("VAE-coldT",    dict(base_vae, latent_dim=8,  beta=1.0),
                     dict(temperature=0.2)),
    ]

    # ---- GAN family: vary latent size and generator/discriminator balance ----
    base_gan = dict(epochs=gan_epochs, hidden_dims_g=[64, 64], hidden_dims_d=[64, 64])
    variants += [
        _gan_variant("GAN",        dict(base_gan, latent_dim=16,
                                        learning_rate_g=2e-4, learning_rate_d=2e-4),
                     dict(temperature=0.5)),
        _gan_variant("GAN-bigZ",   dict(base_gan, latent_dim=32,
                                        learning_rate_g=2e-4, learning_rate_d=2e-4),
                     dict(temperature=0.5)),
        _gan_variant("GAN-fastG",  dict(base_gan, latent_dim=16,
                                        learning_rate_g=5e-4, learning_rate_d=2e-4,
                                        k_discriminator=2),
                     dict(temperature=0.5)),
    ]

    # ---- DBD family: vary diffusion blending samples and denoising steps ----
    variants += [
        _dbd_variant("DBD",        dict(epochs=epochs, num_alpha_samples=10),
                     dict(n_steps=10, temperature=1.0)),
        _dbd_variant("DBD-fine",   dict(epochs=epochs, num_alpha_samples=20),
                     dict(n_steps=25, temperature=0.7)),
    ]

    # ---- Backdrive family: vary the gradient-search initialisation strategy ----
    base_bd = dict(epochs=epochs, learning_rate=1e-3, embedding_dim=8)
    bd_sample = dict(n_iterations=50, learning_rate=0.05, temperature=2.0)
    variants += [
        _backdrive_variant("Backdrive-rand",    base_bd,
                           dict(bd_sample, init_method="random")),
        _backdrive_variant("Backdrive-pBest",   base_bd,
                           dict(bd_sample, init_method="perturb_best", init_noise=0.1)),
        _backdrive_variant("Backdrive-pSel",    base_bd,
                           dict(bd_sample, init_method="perturb_selected", init_noise=0.1)),
        _backdrive_variant("Backdrive-gumbel",  base_bd,
                           dict(bd_sample, init_method="random", use_gumbel_noise=True)),
    ]

    # ---- Categorical Dendiff family: vary schedule and number of steps ----
    base_dd = dict(epochs=epochs, time_emb_dim=32, hidden_dims=[64, 32])
    variants += [
        _dendiff_variant("Dendiff",        dict(base_dd, n_timesteps=50,
                                                beta_schedule="linear"),
                         dict(temperature=0.5)),
        _dendiff_variant("Dendiff-cosine", dict(base_dd, n_timesteps=50,
                                                beta_schedule="cosine"),
                         dict(temperature=0.5)),
        _dendiff_variant("Dendiff-fewT",   dict(base_dd, n_timesteps=20,
                                                beta_schedule="linear"),
                         dict(temperature=0.3, deterministic=True)),
    ]

    return variants


# ---------------------------------------------------------------------------
# Generic NN-EDA driver
# ---------------------------------------------------------------------------

def _random_categorical_population(pop_size: int, cardinality: np.ndarray,
                                   rng: np.random.Generator) -> np.ndarray:
    pop = np.empty((pop_size, len(cardinality)), dtype=int)
    for i, card in enumerate(cardinality):
        pop[:, i] = rng.integers(0, int(card), size=pop_size)
    return pop


def _evaluate_population(population: np.ndarray, fitness_func: Callable) -> np.ndarray:
    return np.array([float(fitness_func(ind)) for ind in population], dtype=float)


def _clip_to_cardinality(population: np.ndarray, cardinality: np.ndarray) -> np.ndarray:
    population = np.asarray(population).astype(int)
    upper = (np.asarray(cardinality, dtype=int) - 1)[None, :]
    return np.clip(population, 0, upper)


def run_single_seed(variant: NNEDAVariant, n_vars: int, cardinality: np.ndarray,
                    fitness_func: Callable, seed: int) -> Tuple[float, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    population = _random_categorical_population(POP_SIZE, cardinality, rng)
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

        model = variant.learn(selected, sel_fit, cardinality, population, fitness)
        new_pop = variant.sample(model, POP_SIZE, population, fitness)
        new_pop = _clip_to_cardinality(new_pop, cardinality)
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


def run_all_seeds(variant: NNEDAVariant, n_vars: int, cardinality: np.ndarray,
                  fitness_func: Callable) -> Tuple[List[float], List[float]]:
    bests, times = [], []
    for seed in SEEDS:
        best, elapsed = run_single_seed(variant, n_vars, cardinality, fitness_func, seed)
        bests.append(best)
        times.append(elapsed)
    return bests, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    variants = build_variants()

    print(f"\n{'=' * 96}")
    print("Mixed-cardinality NN-EDA comparison (pateda_nn)")
    print(f"  variants={len(variants)}  pop_size={POP_SIZE}  n_gen={N_GEN}  "
          f"selection_ratio={SEL_RATIO}  n_runs={N_RUNS}  seeds={SEEDS}"
          + ("   [QUICK MODE]" if QUICK else ""))
    print(f"  families: VAE, GAN, DBD, Backdrive, Dendiff (categorical)")
    print(f"{'=' * 96}")

    name_w = max(len(v.name) for v in variants) + 1

    for problem_type, instance_name, _expected_nb in PROBLEMS:
        # Per-problem grouping (instances differ in size).
        n_binary_probe = _expected_nb
        groups = make_grouping(n_binary_probe, MAX_GROUP_SIZE,
                               GROUPING_SEED + hash(problem_type) % 1000)
        fitness_func, n_sv, card_vec, n_binary, optimal = load_problem(
            problem_type, instance_name, groups
        )

        card_counts: Dict[int, int] = {}
        for c in card_vec:
            card_counts[int(c)] = card_counts.get(int(c), 0) + 1

        print(f"\n{'=' * 96}")
        print(f"Problem: {problem_type}  Instance: {instance_name}  "
              f"(n_binary={n_binary}, n_supervars={n_sv}, optimal={optimal})")
        print("  cardinalities: "
              + "  ".join(f"card={c}x{cnt}" for c, cnt in sorted(card_counts.items())))
        print(f"{'=' * 96}")

        problem_tag = f"{problem_type} {instance_name}"
        for variant in variants:
            try:
                bests, times = run_all_seeds(variant, n_sv, card_vec, fitness_func)
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

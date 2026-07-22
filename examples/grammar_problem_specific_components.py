"""
Using the components the generic pipeline grammar does not cover

The generic grammar of :mod:`pateda.pipelines.grammar`
(see ``eda_pipeline_grammar_demo.py``) reaches ~77% of pateda's discrete-EDA
implementations.  The uncovered ones are not missing by oversight: they need a
*problem-specific argument* that a problem-agnostic grammar cannot invent.  This
script shows, for each such component, (a) where its argument comes from and
(b) a working use, and finally (c) how a *problem-aware* grammar injects those
arguments so the components become coverable.

The problem-specific arguments group as follows:

    interaction matrix   -> Tree-EDA_r, MN-FDA_r, MN-FDAg_r
    unitation n_ones     -> CFDA / CUMDA samplers, constrained seeding & repair
    template mask        -> partial (conditional) sampling
    target optimum       -> MaxGenerationsOrOptimum
    bias / seed pop      -> BiasInit, SeedThisPop
    binary representation-> bit-flip / frequency-balance mutation
    (hyper-parameter)    -> RTRReplacement, mixture models

Usage
-----
    python3 grammar_problem_specific_components.py [seed]
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit, BiasInit, SeedThisPop, SeedingUnitationConstraint
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement, RTRReplacement
from pateda.stop_conditions import MaxGenerations, MaxGenerationsOrOptimum
from pateda.repairing import UnitationRepairing
from pateda.learning import (
    LearnUMDA, LearnTreeModelR, LearnMNFDAR, LearnMNFDAGR, LearnCUMDA,
    LearnMixtureTrees,
)
from pateda.sampling import (
    SampleFDA, SamplePartialFDA, SampleCUMDA, SampleMixtureTrees,
)
from pateda.learning.interaction_learning import (
    find_matrix_interactions_additive_decomposable,
)
from pateda.mutation import RandomResetMutation, bit_flip_mutation, FrequencyBalanceMutation

from structure_problems import make_trap


# ---------------------------------------------------------------------------
# 1. Interaction matrix  ->  Tree-EDA_r, MN-FDA_r, MN-FDAg_r
# ---------------------------------------------------------------------------

def interaction_matrix_components(rng):
    print("=" * 78)
    print("1. INTERACTION MATRIX  (Tree-EDA_r, MN-FDA_r, MN-FDAg_r)")
    print("=" * 78)
    # The argument is the problem's variable-interaction graph.  For an additive
    # function it is block-diagonal; here a concatenated trap-4 gives the blocks.
    fitness, _, optimum, label = make_trap(n_blocks=4, k=4)
    n = 16
    blocks = [list(range(b, b + 4)) for b in range(0, n, 4)]
    IM = find_matrix_interactions_additive_decomposable(blocks, n)
    print(f"  {label}: interaction matrix has "
          f"{int((np.triu(IM, 1) > 0).sum())} edges (the trap blocks)\n")

    learners = {
        "Tree-EDA_r": lambda: LearnTreeModelR(IM),
        "MN-FDA_r": lambda: LearnMNFDAR(IM),
        "MN-FDAg_r": lambda: LearnMNFDAGR(IM),
    }
    card = np.full(n, 2)
    for name, mk in learners.items():
        comp = EDAComponents(
            seeding=RandomInit(), learning=mk(), sampling=SampleFDA(n_samples=200),
            selection=TruncationSelection(ratio=0.3), replacement=ElitistReplacement(),
            stop_condition=MaxGenerations(max_gen=20))
        eda = EDA(pop_size=200, n_vars=n, fitness_func=fitness, cardinality=card,
                  components=comp, random_seed=int(rng.integers(1e6)))
        stats, _ = eda.run(verbose=False)
        print(f"  {name:12s} best={stats.best_fitness_overall:5.1f}/{optimum:.0f}   "
              f"(MI computed only for the {int((np.triu(IM,1)>0).sum())} allowed pairs)")
    print("\n  Only pairs with IM[i,j]=1 may become dependencies, restricting the")
    print("  model to the known problem structure.\n")


# ---------------------------------------------------------------------------
# 2. Unitation constraint (num_ones)  ->  CUMDA/CFDA sampler + constrained seeding/repair
# ---------------------------------------------------------------------------

def unitation_components(rng):
    print("=" * 78)
    print("2. UNITATION CONSTRAINT num_ones  (CUMDA/CFDA sampler, seeding, repair)")
    print("=" * 78)
    # Problem: every solution must have exactly num_ones ones; maximize matches
    # to a target that also has num_ones ones.
    n, num_ones = 16, 8
    target = np.zeros(n, dtype=int); target[:num_ones] = 1; rng.shuffle(target)

    def fitness(x):
        return float(np.sum(np.asarray(x) == target))

    print(f"  n={n}, num_ones={num_ones}; optimum = {n} (exact match).\n")
    comp = EDAComponents(
        # Constrained seeding: every initial solution already has num_ones ones.
        seeding=SeedingUnitationConstraint(),
        seeding_params={"num_ones": num_ones},
        learning=LearnCUMDA(),
        sampling=SampleCUMDA(n_samples=200, n_ones=num_ones),   # constrained sampler
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        # Constrained repair keeps offspring on the num_ones surface.
        repairing=UnitationRepairing(min_ones=num_ones, max_ones=num_ones),
        stop_condition=MaxGenerations(max_gen=30))
    eda = EDA(pop_size=200, n_vars=n, fitness_func=fitness, cardinality=np.full(n, 2),
              components=comp, random_seed=int(rng.integers(1e6)))
    stats, _ = eda.run(verbose=False)
    ones = int(stats.best_individual.sum())
    print(f"  CUMDA (constrained) best={stats.best_fitness_overall:.0f}/{n}, "
          f"ones in best solution = {ones} (== num_ones: {ones == num_ones})")
    print("  The num_ones argument feeds the sampler, the seeding and the repair,")
    print("  keeping the whole pipeline on the feasible (fixed-unitation) subspace.\n")


# ---------------------------------------------------------------------------
# 3. Template mask  ->  partial (conditional) sampling
# ---------------------------------------------------------------------------

def partial_sampling(rng):
    print("=" * 78)
    print("3. TEMPLATE MASK  (SamplePartialFDA: conditional / partial sampling)")
    print("=" * 78)
    n, N = 12, 300
    # Learn a model biased towards ones.
    data = (rng.random((N, n)) < 0.75).astype(int)
    model = LearnUMDA(alpha=1.0).learn(0, n, np.full(n, 2), data, data.sum(1).astype(float))

    # The problem-specific argument is a *template*: fixed positions (values) and
    # free positions (NaN) to be filled from the model conditioned on the fixed.
    template = np.full((200, n), np.nan)
    template[:, :4] = 0                          # clamp the first 4 variables to 0
    samples = SamplePartialFDA(n_samples=200).sample(
        n, model, np.full(n, 2), aux_pop=template, rng=rng)
    print(f"  Fixed prefix (first 4 = 0) preserved: "
          f"{bool(np.all(samples[:, :4] == 0))}")
    print(f"  Free tail mean (model-driven, ~0.75): {samples[:, 4:].mean():.2f}")
    print("  Partial sampling completes a template from the model -- used for")
    print("  local moves, injecting a-priori knowledge, or seeding sub-solutions.\n")


# ---------------------------------------------------------------------------
# 4. Target optimum  ->  MaxGenerationsOrOptimum (early stop)
# ---------------------------------------------------------------------------

def optimum_stop(rng):
    print("=" * 78)
    print("4. TARGET OPTIMUM  (MaxGenerationsOrOptimum: stop when reached)")
    print("=" * 78)
    n = 16

    def onemax(x):
        return float(np.sum(x))

    comp = EDAComponents(
        seeding=RandomInit(), learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=200), selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        # The known optimum (n) lets the run stop as soon as it is found.
        stop_condition=MaxGenerationsOrOptimum(max_gen=100, optimal_fitness=float(n)))
    eda = EDA(pop_size=200, n_vars=n, fitness_func=onemax, cardinality=np.full(n, 2),
              components=comp, random_seed=int(rng.integers(1e6)))
    stats, _ = eda.run(verbose=False)
    print(f"  OneMax optimum={n} found at generation {stats.generation_found}, "
          f"stopped early (max_gen was 100).\n")


# ---------------------------------------------------------------------------
# 5. Bias / seed population, mixture model, RTR replacement, binary mutation
# ---------------------------------------------------------------------------

def other_specific(rng):
    print("=" * 78)
    print("5. OTHER problem/representation-specific components")
    print("=" * 78)
    n = 12
    card = np.full(n, 2)

    # BiasInit: seed with a per-variable bias 'p' (a-priori knowledge).
    biased = BiasInit().seed(n, 100, card, rng=rng, p=0.8)
    print(f"  BiasInit(p=0.8): mean ones in seed = {biased.mean():.2f} (biased high)")

    # SeedThisPop: start from a specific, user-supplied population.
    given = np.tile(np.arange(n) % 2, (10, 1))
    seeded = SeedThisPop().seed(n, 10, card, initial_population=given)
    print(f"  SeedThisPop: seeded from a given population, shape {seeded.shape}")

    # Binary mutation (representation-specific: needs binary variables).
    pop = rng.integers(0, 2, size=(50, n))
    flipped = bit_flip_mutation(n, card, pop, {"mutation_prob": 0.1})
    print(f"  bit_flip_mutation: {int((flipped != pop).sum())} bits flipped "
          f"(binary-only operator)")
    fb = FrequencyBalanceMutation(alpha=0.3).mutate(n, card, pop.copy())
    print(f"  FrequencyBalanceMutation: {int((fb != pop).sum())} changes "
          f"(binary frequency balancing)")

    # Mixture-of-trees model with its dedicated sampler + RTR niching replacement.
    def dec(x):  # simple deceptive-ish to exercise niching
        x = np.asarray(x); t = 0.0
        for b in range(0, n, 4):
            u = int(x[b:b + 4].sum()); t += 4.0 if u == 4 else float(3 - u)
        return t
    comp = EDAComponents(
        seeding=RandomInit(), learning=LearnMixtureTrees(n_components=2),
        sampling=SampleMixtureTrees(n_samples=200),
        selection=TruncationSelection(ratio=0.3),
        replacement=RTRReplacement(window_size=15),      # niching (hyper-parameter)
        stop_condition=MaxGenerations(max_gen=15))
    eda = EDA(pop_size=200, n_vars=n, fitness_func=dec, cardinality=card,
              components=comp, random_seed=int(rng.integers(1e6)))
    stats, _ = eda.run(verbose=False)
    print(f"  Mixture-of-Trees + RTR replacement: best={stats.best_fitness_overall:.1f}\n")


# ---------------------------------------------------------------------------
# 6. Injecting the arguments into a *problem-aware* grammar
# ---------------------------------------------------------------------------

def problem_aware_grammar(rng):
    print("=" * 78)
    print("6. A PROBLEM-AWARE GRAMMAR injects these arguments")
    print("=" * 78)
    print("  The generic grammar cannot cover the above because it has no problem")
    print("  context.  Given a context (interaction matrix, num_ones, optimum...),")
    print("  extra grammar terminals bound to that context become available:\n")

    from pateda.pipelines.grammar import Terminal

    # A problem context supplied by the caller / benchmark loader.
    n = 16
    blocks = [list(range(b, b + 4)) for b in range(0, n, 4)]
    context = {
        "interaction_matrix": find_matrix_interactions_additive_decomposable(blocks, n),
        "num_ones": 8,
        "optimum": float(n),
    }

    def problem_aware_terminals(ctx):
        """Return grammar terminals whose factories are bound to the problem ctx."""
        IM = ctx["interaction_matrix"]
        no = ctx["num_ones"]
        opt = ctx["optimum"]
        return {
            "LearnTreeModelR": Terminal("LearnTreeModelR", "learner", "learner",
                                        build=lambda c, IM=IM: LearnTreeModelR(IM)),
            "LearnMNFDAR": Terminal("LearnMNFDAR", "learner", "learner",
                                    build=lambda c, IM=IM: LearnMNFDAR(IM)),
            "SampleCUMDA": Terminal("SampleCUMDA", "sampler", "sampler",
                                    build=lambda c, no=no: SampleCUMDA(c["pop_size"], n_ones=no)),
            "UnitationSeeding": Terminal("UnitationSeeding", "seeding", "seeding",
                                         build=lambda c: SeedingUnitationConstraint()),
            "MaxGenOrOpt": Terminal("MaxGenOrOpt", "stop", "stop",
                                    build=lambda c, opt=opt: MaxGenerationsOrOptimum(
                                        max_gen=c["n_gen"], optimal_fitness=opt)),
        }

    extra = problem_aware_terminals(context)
    print("  Problem-aware terminals now emittable by the grammar:")
    for name, t in extra.items():
        print(f"    + {name:18s} (role={t.role})")
    # Show one builds and produces a component bound to the context.
    lm = extra["LearnTreeModelR"].build({"pop_size": 100, "n_gen": 10})
    print(f"\n  e.g. grammar emits 'LearnTreeModelR' -> {type(lm).__name__} bound to the")
    print(f"  {int((np.triu(context['interaction_matrix'],1)>0).sum())}-edge interaction matrix of this instance.")
    print("\n  So the uncovered components are covered once the grammar is given the")
    print("  problem context to bind their arguments -- the same mechanism the")
    print("  meta-optimizer will use per benchmark instance.\n")


def main(seed=42):
    print("#" * 78)
    print("# Grammar-uncovered components and their problem-specific arguments")
    print(f"# seed = {seed}")
    print("#" * 78 + "\n")
    rng = np.random.default_rng(seed)
    interaction_matrix_components(rng)
    unitation_components(rng)
    partial_sampling(rng)
    optimum_stop(rng)
    other_specific(rng)
    problem_aware_grammar(rng)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)

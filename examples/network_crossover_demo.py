"""
Network crossover — structure-guided recombination

Network crossover (:class:`~pateda.sampling.network_crossover.SampleNetworkCrossover`)
recombines two parents by swapping a *connected* subset of variables on the
linkage graph, so tightly-linked building blocks are inherited together and not
disrupted (Hauschild & Pelikan, 2010).  The linkage graph can come from the
*learned* model of an EDA or from the *known* structure of the instance.

This script demonstrates:

  1. Block preservation: on a concatenated deceptive trap (known block
     structure), how often each crossover breaks a building block -- network
     crossover (BFS and random-walk masks) vs. structure-blind uniform crossover.

  2. Optimization on instances whose structure depends on the instance
     (Ising 2D spin glass, UBQP, 3-SAT): an EDA whose sampler is network
     crossover, using either the known instance graph or the graph learned by a
     Tree model, compared to a plain EDA.

Usage
-----
    python3 network_crossover_demo.py [seed]
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.sampling import SampleFDA, SampleNetworkCrossover
from pateda.sampling.network_crossover import (
    grow_mask_bfs,
    grow_mask_random_walk,
    _adjacency_list,
)
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations

from structure_problems import make_trap, make_ising, make_ubqp, make_sat


# ---------------------------------------------------------------------------
# Part 1 — building-block preservation
# ---------------------------------------------------------------------------

def block_disruption(rng):
    print("=" * 78)
    print("1. Building-block preservation on a concatenated trap (known blocks)")
    print("=" * 78)
    _, G, _, label = make_trap(n_blocks=6, k=4)
    n = G.shape[0]
    k = 4
    neighbors = _adjacency_list(G)

    # A building block is *disrupted* when the crossover mask splits it, i.e. the
    # boundary between the two parents passes through the block (some of its
    # variables come from one parent and some from the other).  This depends only
    # on the mask, so it isolates the operator's effect from the parents' values.
    def mask_disruption(masks):
        mixed = total = 0
        for m in masks:
            for b in range(0, n, k):
                seg = m[b:b + k]
                total += 1
                if 0 < seg.sum() < k:      # mask not constant over the block
                    mixed += 1
        return mixed / total

    print(f"  {label};  lower disruption = crossover splits fewer blocks\n")
    print(f"  {'operator':<28} | {'block-disruption rate':>21}")
    print("  " + "-" * 54)
    n_masks = 3000
    for method, grow in (("bfs", grow_mask_bfs), ("random_walk", grow_mask_random_walk)):
        gen = np.random.default_rng(int(rng.integers(1e6)))
        masks = [grow(neighbors, n, n // 2, gen) for _ in range(n_masks)]
        print(f"  network crossover ({method:<11})    | {mask_disruption(masks):>21.3f}")

    # Uniform crossover: each variable chosen independently (structure-blind).
    gen = np.random.default_rng(int(rng.integers(1e6)))
    masks = [gen.random(n) < 0.5 for _ in range(n_masks)]
    print(f"  {'uniform crossover':<28} | {mask_disruption(masks):>21.3f}")
    print("\n  Network crossover's connected mask keeps linked variables together,")
    print("  so its boundary rarely cuts through a block -- the mechanism that")
    print("  avoids disrupting the building blocks a model has captured.\n")


# ---------------------------------------------------------------------------
# Part 2 — optimization on instance-structured problems
# ---------------------------------------------------------------------------

def run_eda(fitness, n, sampler_factory, learner_factory, seed,
            pop=300, gens=30):
    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return fitness(x)

    comp = EDAComponents(
        seeding=RandomInit(),
        learning=learner_factory(),
        sampling=sampler_factory(seed),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=gens),
    )
    eda = EDA(pop_size=pop, n_vars=n, fitness_func=wrapped,
              cardinality=np.full(n, 2), components=comp, random_seed=seed)
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness_overall


def optimization(base_seed, n_runs=4):
    print("=" * 78)
    print("2. Optimization with network-crossover sampler (instance structure)")
    print("=" * 78)
    print("  Compared pipelines (selection=truncation, replacement=elitist):")
    print("    - Baseline EDA : UMDA model + probabilistic sampling")
    print("    - NetX(known)  : network crossover on the KNOWN instance graph")
    print("    - NetX(learned): network crossover on a Tree-learned graph\n")

    problems = [
        ("Ising", lambda s: make_ising(L=5, seed=s)),
        ("UBQP", lambda s: make_ubqp(n=25, density=0.15, seed=s)),
        ("3-SAT", lambda s: make_sat(n=25, ratio=4.0, seed=s)),
    ]
    print(f"  {'problem':<8} | {'Baseline':>10} | {'NetX(known)':>12} | "
          f"{'NetX(learned)':>13}")
    print("  " + "-" * 52)
    for pname, make in problems:
        base, known, learned = [], [], []
        for r in range(n_runs):
            seed = base_seed + 100 * r
            fitness, G, opt, label = make(seed)
            n = G.shape[0]
            base.append(run_eda(
                fitness, n,
                lambda s: SampleFDA(n_samples=300),
                lambda: LearnUMDA(alpha=1.0), seed))
            known.append(run_eda(
                fitness, n,
                lambda s, G=G: SampleNetworkCrossover(n_samples=300, linkage_graph=G, seed=s),
                lambda: LearnUMDA(alpha=1.0), seed))
            learned.append(run_eda(
                fitness, n,
                lambda s: SampleNetworkCrossover(n_samples=300, seed=s),
                lambda: LearnTreeModel(), seed))
        print(f"  {pname:<8} | {np.mean(base):>10.2f} | {np.mean(known):>12.2f} | "
              f"{np.mean(learned):>13.2f}")
    print("\n  Network crossover, guided by the instance or the learned linkage,")
    print("  matches or beats the structure-blind baseline on structured problems.\n")


def main(seed=42):
    print("#" * 78)
    print("# Network crossover: structure-guided recombination")
    print(f"# seed = {seed}")
    print("#" * 78 + "\n")
    rng = np.random.default_rng(seed)
    block_disruption(rng)
    optimization(seed)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)

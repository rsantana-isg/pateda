"""
Generate multi-objective NK landscape (MNK) instances.

Usage (positional args, seed first)::

    python generate_mnk_instances.py SEED N_VARS K N_OBJECTIVES [OUTDIR]

``K`` may be a single integer (homogeneous objectives) or a ``+``-separated list
for heterogeneous objectives, e.g. ``2+4+3`` (in which case ``N_OBJECTIVES`` is
inferred from the list).  Each objective is an independent random-neighbourhood
NK landscape over the same ``N_VARS`` binary variables; all objectives are
maximised and lie in ``[0, 1)``.  Saves an ``.npz`` instance.

Reload with::

    from pateda.functions.discrete_binary.multiobjective.mnk_landscape import MNKLandscape, create_mnk_objective_function
    inst = MNKLandscape.load(path)
    objective = create_mnk_objective_function(inst)
"""

import sys
import numpy as np

from pateda.functions.discrete_binary.multiobjective.mnk_landscape import generate_mnk, create_mnk_objective_function
from pateda.functions.instance_generators import instances_dir


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
    n_vars = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    k_arg = sys.argv[3] if len(sys.argv) > 3 else "2"
    n_obj = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    if "+" in k_arg:
        k = [int(v) for v in k_arg.split("+")]
        n_obj = len(k)
        k_tag = k_arg
    else:
        k = int(k_arg)
        k_tag = str(k)

    outdir = sys.argv[5] if len(sys.argv) > 5 else str(instances_dir("mnk"))

    inst = generate_mnk(n_vars=n_vars, k=k, n_objectives=n_obj, seed=seed)

    fname = f"mnk_n_{n_vars}_k_{k_tag}_m_{inst.n_objectives}_seed_{seed}.npz"
    path = f"{outdir.rstrip('/')}/{fname}"
    inst.save(path)

    print(f"Seed:              {seed}")
    print(f"Variables:         {n_vars}")
    print(f"K per objective:   {inst.ks}")
    print(f"# objectives:      {inst.n_objectives}")
    print(f"Heterogeneous:     {len(set(inst.ks)) > 1}")
    print(f"Saved instance:    {path}")

    objective = create_mnk_objective_function(inst)
    F = objective(np.random.default_rng(seed).integers(0, 2, size=(5, n_vars)))
    print(f"Sample objective values (maximise, in [0,1)):\n{np.round(F, 4)}")


if __name__ == "__main__":
    main()

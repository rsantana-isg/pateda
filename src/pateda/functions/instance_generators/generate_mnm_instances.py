"""
Generate mNM (truncated-Walsh) multi-objective instances.

Usage (positional args, seed first)::

    python generate_mnm_instances.py SEED N_VARS MAX_ORDER SIGMA [ORDER1 ORDER2] [OUTDIR]

Builds a bi-objective mNM instance: a base Walsh model of ``MAX_ORDER`` with
coefficient-decay ``SIGMA``, whose two objectives use maximum interaction orders
``ORDER1`` and ``ORDER2`` (default ``MAX_ORDER, MAX_ORDER``) and sign transforms
``[+1, -1]``.  Saves an ``.npz`` instance and prints its configuration.

Reload with::

    from pateda.functions.discrete_binary.multiobjective.mnm import MNMInstance, create_mnm_objective_function
    inst = MNMInstance.load(path)
    objective = create_mnm_objective_function(inst)
"""

import sys
import numpy as np

from pateda.functions.discrete_binary.multiobjective.mnm import generate_mnm, create_mnm_objective_function
from pateda.functions.instance_generators import instances_dir


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
    n_vars = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    max_order = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    if len(sys.argv) > 6:
        orders = [int(sys.argv[5]), int(sys.argv[6])]
        outdir_arg = 7
    else:
        orders = [max_order, max_order]
        outdir_arg = 5
    outdir = sys.argv[outdir_arg] if len(sys.argv) > outdir_arg else str(instances_dir("mnm"))

    inst = generate_mnm(n_vars=n_vars, max_order=max_order, sigma=sigma,
                        objective_orders=orders, seed=seed)

    fname = f"mnm_n_{n_vars}_M_{max_order}_sigma_{sigma:g}_o_{orders[0]}-{orders[1]}_seed_{seed}.npz"
    path = f"{outdir.rstrip('/')}/{fname}"
    inst.save(path)

    print(f"Seed:                 {seed}")
    print(f"Variables:            {n_vars}")
    print(f"Base model order M:   {max_order}")
    print(f"Sigma (decay):        {sigma}")
    print(f"Objective orders:     {orders}")
    print(f"Objective signs:      {inst.objective_signs}")
    print(f"# model components:   {len(inst.model.components)}")
    print(f"Saved instance:       {path}")

    # quick sanity: evaluate a small random population
    objective = create_mnm_objective_function(inst)
    F = objective(np.random.default_rng(seed).integers(0, 2, size=(5, n_vars)))
    print(f"Sample objective values (maximise):\n{np.round(F, 4)}")


if __name__ == "__main__":
    main()

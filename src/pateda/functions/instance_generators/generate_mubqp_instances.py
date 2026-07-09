"""
Generate multi-objective UBQP (mUBQP) instances.

Usage (positional args, seed first)::

    python generate_mubqp_instances.py SEED METHOD N_VARS [ARG1 ARG2 ...] [OUTDIR]

METHOD selects the construction:

* ``standard   RHO [DENSITY]``  -- Liefooghe generator: density + objective
  correlation ``RHO`` (negative = conflicting = harder).
* ``artificial ITYPE``          -- one of the five hand-designed structures
  (ITYPE in 1..5).
* ``hard       KIND``           -- hard instance built from order-5 building
  blocks; KIND is ``tiled`` (block-separable) or ``heavy`` (random overlapping
  placement).  Uses :func:`select_hard_chunk_pairs`.

All objectives are maximised.  Instances are written in the ``.dat`` format used
by the reference MATLAB code.

Reload with::

    from pateda.functions.discrete_binary.multiobjective.mubqp import MUBQPInstance, create_mubqp_objective_function
    inst = MUBQPInstance.load(path)
    objective = create_mubqp_objective_function(inst)
"""

import sys
import numpy as np

from pateda.functions.discrete_binary.multiobjective.mubqp import (
    generate_mubqp, create_artificial_mubqp, create_mubqp_objective_function,
    select_hard_chunk_pairs, create_mubqp_from_chunk, create_heavy_mubqp_from_chunks,
)
from pateda.functions.instance_generators import instances_dir


def build(method, n_vars, args, seed):
    """Return (instance, filename_tag) for the requested construction."""
    if method == "standard":
        rho = float(args[0]) if args else -0.5
        density = float(args[1]) if len(args) > 1 else 0.4
        inst = generate_mubqp(n_vars, n_objectives=2, density=density, rho=rho, seed=seed)
        return inst, f"standard_rho_{rho:g}_d_{density:g}"
    if method == "artificial":
        itype = int(args[0]) if args else 3
        inst = create_artificial_mubqp(n_vars, itype, seed=seed)
        return inst, f"artificial_type_{itype}"
    if method == "hard":
        kind = args[0] if args else "heavy"
        hard = select_hard_chunk_pairs(max_pairs=20, n_candidates=4000, min_pareto=3, seed=seed)
        if not hard:
            raise RuntimeError("no hard chunk pairs found; increase n_candidates")
        pairs = [(w1, w2) for (w1, w2, _) in hard]
        if kind == "tiled":
            w1, w2, _ = hard[0]
            inst = create_mubqp_from_chunk(w1, w2, n_vars=n_vars, k=5, seed=seed)
        elif kind == "heavy":
            inst = create_heavy_mubqp_from_chunks(pairs, n_vars=n_vars, k=5,
                                                  n_chunks=n_vars, seed=seed)
        else:
            raise ValueError("hard KIND must be 'tiled' or 'heavy'")
        return inst, f"hard_{kind}"
    raise ValueError(f"unknown METHOD '{method}' (standard|artificial|hard)")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
    method = sys.argv[2] if len(sys.argv) > 2 else "hard"
    n_vars = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    rest = sys.argv[4:]
    # last argument is an output directory if it is not numeric/keyword-ish path
    outdir = str(instances_dir("mubqp"))
    if rest and ("/" in rest[-1] or rest[-1].startswith(".")):
        outdir = rest[-1]
        rest = rest[:-1]

    inst, tag = build(method, n_vars, rest, seed)

    fname = f"mubqp_{tag}_n_{n_vars}_seed_{seed}.dat"
    path = f"{outdir.rstrip('/')}/{fname}"
    inst.save(path)

    print(f"Seed:            {seed}")
    print(f"Method:          {method} {' '.join(rest)}")
    print(f"Variables:       {n_vars}")
    print(f"# objectives:    {inst.n_objectives}")
    print(f"Edges obj1/obj2: {len(inst.edges(0))} / {len(inst.edges(1))}")
    print(f"Saved instance:  {path}")

    objective = create_mubqp_objective_function(inst)
    F = objective(np.random.default_rng(seed).integers(0, 2, size=(5, n_vars)))
    print(f"Sample objective values (maximise):\n{np.round(F, 2)}")


if __name__ == "__main__":
    main()

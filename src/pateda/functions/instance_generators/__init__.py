"""Instance generators for multi-objective benchmark problems.

Command-line scripts that build and save instances of the three multi-objective
models implemented in
:mod:`pateda.functions.discrete_binary.multiobjective`:

* ``generate_mnm_instances``   -- mNM (truncated-Walsh / Markov-network) model.
* ``generate_mnk_instances``   -- multi-objective NK landscapes (MNK).
* ``generate_mubqp_instances`` -- multi-objective UBQP, incl. hard instances.

Each script follows pateda's cluster conventions: positional arguments with the
seed first, self-describing output filenames, and a printed configuration.
Generated files are written under ``functions/MultiObjective_Instances/`` and
can be re-loaded with the ``load`` methods / ``get_function`` helpers of the
corresponding model classes to obtain objective functions for testing pateda
algorithms.
"""

from pathlib import Path


def instances_dir(model: str = "") -> Path:
    """Return the packaged multi-objective instances directory.

    Args:
        model: optional sub-directory (``"mnm"``, ``"mnk"`` or ``"mubqp"``).
    """
    base = Path(__file__).resolve().parent.parent / "MultiObjective_Instances"
    return base / model if model else base

"""
Quasiparticle braid approximation of quantum gates.

Topological quantum computing realises a single-qubit gate as a *braid*: an
ordered product of elementary braid-generator matrices. Approximating a target
gate ``T`` therefore becomes a combinatorial problem -- find a sequence of ``n``
generators whose matrix product ``B`` is as close as possible to ``T`` (and,
secondarily, as short as possible).

Representation (following Santana et al., SOCO braid paper, based on the
fitness of McDonald & Katzgraber, 2013):

    A solution ``x = (x_1, ..., x_n)`` is a fixed-length vector where each
    ``x_i`` in ``{0, ..., 2g-1}`` selects one of ``2g`` elementary matrices
    (``g`` generators plus their ``g`` inverses).  With ``g`` generators
    ``sigma_1, ..., sigma_g`` ordered first, ``x_i = j`` with ``j < g`` picks
    ``sigma_{j+1}`` and ``x_i = j`` with ``j >= g`` picks the inverse
    ``sigma^{-1}_{(j-g)+1}``.  For the Fibonacci-anyon set (``g = 2``) the
    cardinality is 4: ``0->sigma_1, 1->sigma_2, 2->sigma_1^{-1}, 3->sigma_2^{-1}``.
    Example: ``sigma_1 sigma_1 sigma_2 sigma_2^{-1} sigma_1^{-1}`` -> ``(0,0,1,3,2)``.

The braid error is the Frobenius distance between the braid matrix and the
target (Eq. ERROR of the paper)::

    epsilon = |B - T|,   |M| = sqrt(sum_ij |M_ij|^2)

and the (maximised) fitness combines error and length with a trade-off
``lambda`` (Eq. FITNESS)::

    f(x) = (1 - lambda) / (1 + epsilon) + lambda / l

where ``l`` is the braid length (or the *effective* length ``elen`` after
cancelling adjacent inverse pairs).

References:
    R. Santana, Z. Zhu, ... "Solving the quasiparticle braid problem with
    estimation of distribution algorithms" (SOCOBraidPaper4).
    R. B. McDonald and H. G. Katzgraber, "Genetic braid optimization",
    Phys. Rev. B 87, 054414 (2013).
    M. Burrello, H. Xu, G. Mussardo, X. Wan, "Topological quantum hashing with
    the icosahedral group", arXiv:0903.1497.
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

# Golden-ratio conjugate used by the Fibonacci-anyon generators.
TAU = (np.sqrt(5.0) - 1.0) / 2.0
# Golden ratio used by the icosahedral-group axis/angle definitions.
TAU_ICO = (np.sqrt(5.0) + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# Elementary matrices
# --------------------------------------------------------------------------- #

def fibonacci_anyon_generators() -> List[np.ndarray]:
    """Return the 4 Fibonacci-anyon elementary matrices.

    Order matches the paper representation:
    ``[sigma_1, sigma_2, sigma_1^{-1}, sigma_2^{-1}]``.  Each is a ``2x2``
    complex SU(2) matrix; the inverses are the conjugate transposes.
    """
    sigma1 = np.array([
        [np.exp(-1j * 7.0 * np.pi / 10.0), 0.0],
        [0.0, -np.exp(-1j * 3.0 * np.pi / 10.0)],
    ], dtype=complex)
    a = -TAU * np.exp(-1j * np.pi / 10.0)
    b = -1j * np.sqrt(TAU)
    sigma2 = np.array([
        [a, b],
        [-np.conj(b), np.conj(a)],
    ], dtype=complex)
    return [sigma1, sigma2, sigma1.conj().T, sigma2.conj().T]


def su2_from_axis_angle(x: float, y: float, z: float, phi: float) -> np.ndarray:
    """SU(2) matrix for a rotation of angle ``phi`` about axis ``(x, y, z)``.

    Port of ``UMatrix::set_axis_angle`` from the Burrello et al. reference code.
    """
    norm = np.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        # Zero axis only occurs together with phi = 0 (the identity element).
        a = complex(np.cos(phi / 2.0), 0.0)
        b = 0j
    else:
        s = np.sin(phi / 2.0)
        a = complex(np.cos(phi / 2.0), -z / norm * s)
        b = complex(-y / norm * s, -x / norm * s)
    return np.array([[a, b], [-np.conj(b), np.conj(a)]], dtype=complex)


# Axis/angle definitions of the 60 icosahedral rotations, transcribed from
# ``icosahedral_group.cc`` (Burrello et al.).  ``t`` is the golden ratio and
# ``p1, p2, p3`` are the three rotation angles used per symmetry class.
def _icosahedral_axis_angles():
    t = TAU_ICO
    p1 = 0.4 * np.pi
    p2 = np.pi
    p3 = 2.0 * np.pi / 3.0
    s = 1.0 + 2.0 * t  # shorthand appearing in the 3-fold axes
    specs = [
        (0.0, 0.0, 1.0, 0.0),
    ]
    # 5-fold axes, angles p1 and 2*p1
    fivefold = [
        (0.0, 1.0, t), (0.0, 1.0, -t), (0.0, -1.0, t), (0.0, -1.0, -t),
        (t, 0.0, 1.0), (-t, 0.0, 1.0), (t, 0.0, -1.0), (-t, 0.0, -1.0),
        (1.0, t, 0.0), (1.0, -t, 0.0), (-1.0, t, 0.0), (-1.0, -t, 0.0),
    ]
    for ax in fivefold:
        specs.append((*ax, p1))
    for ax in fivefold:
        specs.append((*ax, 2.0 * p1))
    # 2-fold axes, angle p2 = pi
    twofold = [
        (0.0, 0.0, 1.0),
        (1.0, 1.0 / t, t), (-1.0, 1.0 / t, t), (1.0, -1.0 / t, t), (-1.0, -1.0 / t, t),
        (1.0, 0.0, 0.0),
        (t, 1.0, 1.0 / t), (t, -1.0, 1.0 / t), (t, 1.0, -1.0 / t), (t, -1.0, -1.0 / t),
        (0.0, 1.0, 0.0),
        (1.0 / t, t, 1.0), (-1.0 / t, t, 1.0), (1.0 / t, t, -1.0), (-1.0 / t, t, -1.0),
    ]
    for ax in twofold:
        specs.append((*ax, p2))
    # 3-fold axes, angles p3 and 2*p3
    threefold = [
        (t, 0.0, s), (-t, 0.0, s), (s, t, 0.0), (s, -t, 0.0),
        (0.0, s, t), (0.0, s, -t),
        (s, s, s), (s, -s, s), (-s, s, s), (-s, -s, s),
    ]
    for ax in threefold:
        specs.append((*ax, p3))
    for ax in threefold:
        specs.append((*ax, 2.0 * p3))
    return specs


def icosahedral_group() -> List[np.ndarray]:
    """Return the 60 SU(2) matrices representing the icosahedral rotation group.

    These are the target gates of the icosahedral braid benchmark.
    """
    return [su2_from_axis_angle(x, y, z, phi) for (x, y, z, phi) in _icosahedral_axis_angles()]


# --------------------------------------------------------------------------- #
# Braid evaluation
# --------------------------------------------------------------------------- #

def default_inverse_index(n_generators: int) -> np.ndarray:
    """Inverse map for a generator set of ``g`` generators + ``g`` inverses.

    Returns an array ``inv`` of length ``2g`` where ``inv[j]`` is the index of
    the elementary matrix that inverts matrix ``j``.  Pairs are ``(j, j+g)``.
    """
    g = n_generators // 2
    inv = np.empty(n_generators, dtype=int)
    for j in range(g):
        inv[j] = j + g
        inv[j + g] = j
    return inv


def braid_matrix(x: Sequence[int], generators: Sequence[np.ndarray]) -> np.ndarray:
    """Matrix product ``B = generators[x_0] @ generators[x_1] @ ...``."""
    x = np.asarray(x).ravel()
    B = np.array(generators[int(x[0])], dtype=complex)
    for i in range(1, len(x)):
        B = B @ generators[int(x[i])]
    return B


def braid_error(x: Sequence[int], generators: Sequence[np.ndarray],
                target: np.ndarray, phase_invariant: bool = False) -> float:
    """Frobenius distance ``epsilon = |B - T|`` between braid and target.

    If ``phase_invariant`` is True the global-phase-agnostic distance
    ``min(|B - T|, |B + T|)`` is used (matching the reference C++ code, which
    quotients out the unobservable overall sign).
    """
    B = braid_matrix(x, generators)
    T = np.asarray(target, dtype=complex)
    d = np.linalg.norm(B - T)
    if phase_invariant:
        d = min(d, np.linalg.norm(B + T))
    return float(d)


def effective_length(x: Sequence[int], inverse_index: Optional[np.ndarray]) -> int:
    """Effective braid length ``elen`` after cancelling adjacent inverse pairs.

    Uses free-reduction on a stack: a generator immediately followed by its
    inverse is removed, repeatedly.  If ``inverse_index`` is None (a generator
    set with no inverses), the length equals ``len(x)``.
    """
    x = np.asarray(x).ravel()
    if inverse_index is None:
        return int(len(x))
    stack: List[int] = []
    for v in x:
        v = int(v)
        if stack and inverse_index[stack[-1]] == v:
            stack.pop()
        else:
            stack.append(v)
    return len(stack)


def braid_fitness(x: Sequence[int], generators: Sequence[np.ndarray], target: np.ndarray,
                  lam: float = 0.0, inverse_index: Optional[np.ndarray] = None,
                  use_effective_length: bool = False, phase_invariant: bool = False) -> float:
    """McDonald-Katzgraber fitness (to be maximised).

    ``f(x) = (1 - lam) / (1 + epsilon) + lam / l``.  With ``lam = 0`` this is
    ``1 / (1 + epsilon)``, so maximising it minimises the approximation error.
    """
    eps = braid_error(x, generators, target, phase_invariant=phase_invariant)
    if lam == 0.0:
        return 1.0 / (1.0 + eps)
    if use_effective_length:
        l = effective_length(x, inverse_index)
    else:
        l = len(np.asarray(x).ravel())
    l = max(1, l)
    return (1.0 - lam) / (1.0 + eps) + lam / l


class BraidProblem:
    """A braid approximation instance: elementary matrices + a target gate.

    Attributes
    ----------
    generators : list of 2x2 complex ndarray
        The elementary matrices selectable at each position.
    target : 2x2 complex ndarray
        The gate to approximate.
    n_matrices : int
        Braid length (number of variables ``n``).
    cardinality : int
        Number of selectable matrices (``= len(generators)``).
    inverse_index : ndarray or None
        Index of the inverse of each generator (for effective length); None if
        the generator set is not closed under inversion.
    """

    def __init__(self, generators: Sequence[np.ndarray], target: np.ndarray,
                 n_matrices: int, inverse_index: Optional[np.ndarray] = None,
                 name: str = ""):
        self.generators = [np.asarray(g, dtype=complex) for g in generators]
        self.target = np.asarray(target, dtype=complex)
        self.n_matrices = int(n_matrices)
        self.cardinality = len(self.generators)
        self.inverse_index = inverse_index
        self.name = name

    def matrix(self, x: Sequence[int]) -> np.ndarray:
        return braid_matrix(x, self.generators)

    def error(self, x: Sequence[int], phase_invariant: bool = False) -> float:
        return braid_error(x, self.generators, self.target, phase_invariant=phase_invariant)

    def effective_length(self, x: Sequence[int]) -> int:
        return effective_length(x, self.inverse_index)

    def fitness(self, x: Sequence[int], lam: float = 0.0,
                use_effective_length: bool = False, phase_invariant: bool = False) -> float:
        return braid_fitness(x, self.generators, self.target, lam=lam,
                             inverse_index=self.inverse_index,
                             use_effective_length=use_effective_length,
                             phase_invariant=phase_invariant)


def make_fibonacci_braid_problem(target: np.ndarray, n_matrices: int,
                                 name: str = "") -> BraidProblem:
    """Build a :class:`BraidProblem` on the Fibonacci-anyon generator set."""
    gens = fibonacci_anyon_generators()
    return BraidProblem(gens, target, n_matrices,
                        inverse_index=default_inverse_index(len(gens)), name=name)


# --------------------------------------------------------------------------- #
# Packaged instances
# --------------------------------------------------------------------------- #

def _default_braid_instances_dir() -> Path:
    """Return the packaged directory containing braid benchmark instances."""
    return Path(__file__).resolve().parent.parent.parent / "Braid_Instances"


def save_braid_instances(instances_dir: Optional[str] = None) -> Path:
    """Write the packaged braid instances (generators + icosahedral targets).

    Produces two ``.npz`` files: ``fibonacci_anyon_generators.npz`` (a stacked
    ``(4, 2, 2)`` complex array) and ``icosahedral_group.npz`` (a stacked
    ``(60, 2, 2)`` complex array).  The data is deterministic, so this simply
    materialises what the code generates for inspection / reuse.
    """
    d = Path(instances_dir) if instances_dir is not None else _default_braid_instances_dir()
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "fibonacci_anyon_generators.npz",
             generators=np.array(fibonacci_anyon_generators()),
             order=np.array(["sigma_1", "sigma_2", "sigma_1_inv", "sigma_2_inv"]))
    np.savez(d / "icosahedral_group.npz",
             targets=np.array(icosahedral_group()),
             axis_angles=np.array(_icosahedral_axis_angles()))
    return d


def load_icosahedral_targets(instances_dir: Optional[str] = None) -> np.ndarray:
    """Load the 60 icosahedral target matrices as a ``(60, 2, 2)`` array.

    Reads the packaged ``icosahedral_group.npz`` if present, otherwise generates
    the group on the fly (the definition is deterministic).
    """
    d = Path(instances_dir) if instances_dir is not None else _default_braid_instances_dir()
    f = d / "icosahedral_group.npz"
    if f.exists():
        return np.load(f)["targets"]
    return np.array(icosahedral_group())


def load_anyon_generators(instances_dir: Optional[str] = None) -> np.ndarray:
    """Load the 4 Fibonacci-anyon generator matrices as a ``(4, 2, 2)`` array."""
    d = Path(instances_dir) if instances_dir is not None else _default_braid_instances_dir()
    f = d / "fibonacci_anyon_generators.npz"
    if f.exists():
        return np.load(f)["generators"]
    return np.array(fibonacci_anyon_generators())


def make_icosahedral_benchmark_problem(target_index: int, n_matrices: int,
                                       instances_dir: Optional[str] = None) -> BraidProblem:
    """Build the braid problem approximating icosahedral element ``target_index``.

    ``target_index`` in ``[0, 59]`` selects one of the 60 icosahedral gates as
    the target; the elementary set is the 4 Fibonacci-anyon generators.
    """
    if not 0 <= target_index < 60:
        raise ValueError(f"target_index must be in [0, 59], got {target_index}")
    targets = load_icosahedral_targets(instances_dir)
    gens = load_anyon_generators(instances_dir)
    return BraidProblem([gens[i] for i in range(len(gens))], targets[target_index],
                        n_matrices, inverse_index=default_inverse_index(len(gens)),
                        name=f"icosahedral[{target_index}]")


def create_braid_objective_function(problem: BraidProblem, lam: float = 0.0,
                                    use_effective_length: bool = False,
                                    phase_invariant: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Create a single-objective braid fitness function for EDAs.

    The returned callable maximises :func:`braid_fitness`; with ``lam = 0`` this
    is ``1 / (1 + error)``, so the optimum minimises the approximation error.

    Accepts either a single individual (1-D array -> scalar) or a whole
    population (2-D array -> 1-D array of fitness values).
    """
    def objective(population: np.ndarray) -> np.ndarray:
        population = np.asarray(population)
        if population.ndim == 1:
            return problem.fitness(population, lam=lam,
                                   use_effective_length=use_effective_length,
                                   phase_invariant=phase_invariant)
        fitness = np.empty(population.shape[0])
        for i in range(population.shape[0]):
            fitness[i] = problem.fitness(population[i], lam=lam,
                                         use_effective_length=use_effective_length,
                                         phase_invariant=phase_invariant)
        return fitness

    return objective

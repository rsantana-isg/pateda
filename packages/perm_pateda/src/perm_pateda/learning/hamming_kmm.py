from typing import Dict, Any
import numpy as np
from scipy.special import comb
from perm_pateda.distances import hamming_distance, compute_derangements


def _compute_theta_from_expected_distance(expected_dist: float, n: int) -> float:
    """
    Encuentra θ tal que E[K] bajo Hamming MM = expected_dist,
    usando búsqueda binaria sobre θ.
    """
    # S(n,k) = C(n,k) * D(k), número de permutaciones a distancia k
    derangements = compute_derangements(n)
    s_nk = np.array([comb(n, k, exact=True) * derangements[k] for k in range(n + 1)])

    def compute_expected_dist(theta):
        log_weights = np.log(s_nk + 1e-300) - theta * np.arange(n + 1)
        log_weights[1] = -np.inf  # k=1 imposible en Hamming
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()
        return np.dot(weights, np.arange(n + 1))

    # Búsqueda binaria
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if compute_expected_dist(mid) > expected_dist:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class LearnHammingKMM:
    """
    Kernels of Mallows Model bajo distancia Hamming.

    El 'modelo' aprendido es:
      - Las propias soluciones seleccionadas como centros
      - θ calculado a partir de E[K] que decrece exponencialmente
    """

    def __init__(
        self,
        expected_dist_start: float = None,   # E[K]_0, por defecto n/2
        expected_dist_end: float = 0.25,     # E[K]_tmax
        gamma: float = 5.14,                 # velocidad de decaimiento
        n_gen: int = 50,                     # total de generaciones
    ):
        self.expected_dist_start = expected_dist_start
        self.expected_dist_end = expected_dist_end
        self.gamma = gamma
        self.n_gen = n_gen

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:

        e0 = self.expected_dist_start if self.expected_dist_start is not None else n_vars / 2
        e_end = self.expected_dist_end

        # Progreso exponencial: delta(p) = (exp(-gamma*p) - 1) / (exp(-gamma) - 1)
        p = generation / max(self.n_gen, 1)
        delta = (np.exp(-self.gamma * p) - 1) / (np.exp(-self.gamma) - 1)
        expected_dist = e_end + delta * (e0 - e_end)
        expected_dist = max(expected_dist, e_end)

        theta = _compute_theta_from_expected_distance(expected_dist, n_vars)

        return {
            "centers": population.copy(),   # todas las seleccionadas son centros
            "theta": theta,
            "expected_dist": expected_dist,
            "n_vars": n_vars,
        }

    def learn(self, generation, n_vars, cardinality, population, fitness, **kwargs):
        return self.__call__(generation, n_vars, cardinality, population, fitness, **kwargs)
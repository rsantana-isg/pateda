# sampling/hamming_kmm.py

import numpy as np
from scipy.special import comb
from typing import Dict, Any
from perm_pateda.distances import hamming_distance, compute_derangements


def _sample_at_hamming_distance(center: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construye una permutación a distancia Hamming exactamente k de center.
    Elige k posiciones y las desarregla (ninguna queda en su sitio original).
    """
    n = len(center)
    result = center.copy()

    # Elegir k posiciones a permutar
    positions = rng.choice(n, size=k, replace=False)
    items = result[positions].copy()

    # Desarreglar los items: ninguno puede quedar en su posición original
    # Usamos Fisher-Yates con rechazo hasta tener un desarreglo válido
    for _ in range(1000):
        rng.shuffle(items)
        if all(items[i] != result[positions[i]] for i in range(k)):
            break

    result[positions] = items
    return result


class SampleHammingKMM:
    """
    Muestreador para Kernels of Mallows Model bajo distancia Hamming.

    Algoritmo (Algorithm 2 del paper):
      1. Elegir un centro σ_0 del conjunto uniformemente al azar
      2. Calcular p(k) ∝ S(n,k) * e^(-θk), k ∈ {0} ∪ [n] \ {1}
      3. Muestrear k según esas probabilidades
      4. Construir σ a distancia Hamming k de σ_0
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:

        centers = model["centers"]
        theta = model["theta"]
        n = n_vars

        # Calcular S(n,k) = C(n,k) * D(k)
        derangements = compute_derangements(n)
        s_nk = np.array([
            int(comb(n, k, exact=True)) * int(derangements[k])
            for k in range(n + 1)
        ], dtype=float)

        # p(k) ∝ S(n,k) * e^(-θk), k=1 imposible en Hamming
        log_weights = np.log(s_nk + 1e-300) - theta * np.arange(n + 1)
        log_weights[1] = -np.inf
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

        new_pop = np.empty((sample_size, n), dtype=int)

        for i in range(sample_size):
            # Paso 1: elegir centro aleatorio
            center = centers[rng.integers(len(centers))]

            # Paso 2-3: muestrear distancia k
            k = int(rng.choice(n + 1, p=weights))

            # Paso 4: construir permutación a distancia k
            if k == 0:
                new_pop[i] = center.copy()
            else:
                new_pop[i] = _sample_at_hamming_distance(center, k, rng)

        return new_pop
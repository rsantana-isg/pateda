"""
Sampling module for Mixture of Plackett-Luce Models.
Generates new rankings (permutations) based on a learned mixture model.
Uses the Gumbel-max trick for efficient parallel sampling of Plackett-Luce distributions.
"""
import numpy as np
from typing import Dict, Any


class SamplePlackettLuceMixture:
    """
    Sample permutations from a learned Mixture of Plackett-Luce models.
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
        **kwargs,
    ) -> np.ndarray:
        """Sample method to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
        )

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate new permutations from the learned mixture model.

        Args:
            n_vars:      Permutation length
            model:       Dictionary returned by LearnPlackettLuceMixture
            cardinality: Not used, kept para compatibilidad con interfaz
            population:  Not used, kept para compatibilidad con interfaz
            fitness:     Not used, kept para compatibilidad con interfaz
            sample_size: Number of permutations to generate
            rng:         Random generator passed by _PermEDA
        Returns:
            samples: (sample_size, n_vars) array of generated permutations
        """
        beta = model["mixing_weights"]
        weights_per_comp = model["weights_per_component"]
        K = model["n_components"]

        samples = np.zeros((sample_size, n_vars), dtype=int)

        # Elegir componente para cada muestra
        Z = rng.choice(K, size=sample_size, p=beta)

        # Gumbel-max trick
        for i in range(sample_size):
            k = Z[i]
            w = weights_per_comp[k]

            log_w = np.log(np.maximum(w, 1e-12))

            u = rng.uniform(0, 1, size=n_vars)
            gumbel_noise = -np.log(-np.log(u + 1e-12))

            scores = log_w + gumbel_noise
            samples[i] = np.argsort(scores)[::-1]

        return samples
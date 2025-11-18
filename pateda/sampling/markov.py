"""
Sampling from k-order Markov Chain models

Samples from k-order Markov chain EDAs by sequentially sampling each variable
conditioned on the k previous variables.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, FactorizedModel
from pateda.sampling.fda import SampleFDA


class SampleMarkovChain(SamplingMethod):
    """
    Sample population from a k-order Markov Chain model

    The k-order Markov model samples variables sequentially:
    1. Sample first k+1 variables from their joint distribution
    2. For each subsequent variable, sample conditioned on k previous variables

    This is implemented as a specialized case of FDA sampling, since Markov chains
    are a specific factorization structure.
    """

    def __init__(self, n_samples: int):
        """
        Initialize Markov chain sampling

        Args:
            n_samples: Number of individuals to sample
        """
        self.n_samples = n_samples
        self._fda_sampler = SampleFDA(n_samples)

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from Markov Chain model

        Args:
            n_vars: Number of variables
            model: FactorizedModel with Markov chain structure
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used for basic Markov sampling)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples

        Returns:
            Sampled population (n_samples, n_vars)

        Note:
            The sampling proceeds sequentially:
            - First, sample the initial k+1 variables from their joint distribution
            - Then, for each remaining variable, sample from p(x_i | x_{i-k}, ..., x_{i-1})
        """
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Validate it's a Markov chain structure
        if not self._is_markov_structure(model):
            raise ValueError("Model does not have valid Markov chain structure")

        # Delegate to FDA sampler (Markov chains are a special case of FDA)
        return self._fda_sampler.sample(
            n_vars, model, cardinality, aux_pop, aux_fitness, **params
        )

    def _is_markov_structure(self, model: FactorizedModel) -> bool:
        """
        Check if model has valid Markov chain structure

        Args:
            model: Model to validate

        Returns:
            True if structure matches Markov chain pattern
        """
        cliques = model.structure
        metadata = model.metadata

        # Check metadata indicates Markov chain
        if metadata.get("model_type") == "Markov Chain":
            return True

        # Additional structural validation could be added here
        # For now, we trust the model type metadata

        return False


class SampleMarkovChainForward(SamplingMethod):
    """
    Forward sampling from k-order Markov Chain (explicit implementation)

    This is an explicit implementation that directly samples from the Markov chain
    without delegating to FDA. Useful for understanding the Markov sampling process.
    """

    def __init__(self, n_samples: int, k: int):
        """
        Initialize forward Markov sampling

        Args:
            n_samples: Number of individuals to sample
            k: Order of the Markov chain
        """
        self.n_samples = n_samples
        self.k = k

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample using forward sampling algorithm

        Args:
            n_vars: Number of variables
            model: FactorizedModel with Markov chain structure
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters

        Returns:
            Sampled population (n_samples, n_vars)
        """
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)

        # Get model parameters
        tables = model.parameters

        # Initialize population
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample first k+1 variables from joint distribution
        joint_table = tables[0]  # First table is joint over first k+1 variables

        # Flatten joint table to 1D for sampling
        joint_probs_flat = joint_table.flatten()

        # Sample configurations for all individuals
        for i in range(n_samples):
            # Sample configuration index
            config_idx = np.random.choice(len(joint_probs_flat), p=joint_probs_flat)

            # Convert configuration index to variable values
            init_cards = [int(cardinality[j]) for j in range(self.k + 1)]
            config_values = self._index_to_values(config_idx, init_cards)

            # Assign to population
            new_pop[i, :self.k + 1] = config_values

        # Sample remaining variables conditionally
        for var in range(self.k + 1, n_vars):
            cpd = tables[var - self.k]  # CPD for this variable

            for i in range(n_samples):
                # Get parent configuration (k previous variables)
                parent_values = new_pop[i, var - self.k:var]
                parent_cards = [int(cardinality[j]) for j in range(var - self.k, var)]

                # Convert parent values to configuration index
                parent_config_idx = self._values_to_index(parent_values, parent_cards)

                # Get conditional probabilities for this parent configuration
                cond_probs = cpd[parent_config_idx, :]

                # Sample variable value
                var_value = np.random.choice(len(cond_probs), p=cond_probs)
                new_pop[i, var] = var_value

        return new_pop

    def _index_to_values(self, index: int, cards: list) -> np.ndarray:
        """Convert configuration index to variable values (mixed-radix)"""
        values = []
        for card in cards:
            values.append(index % card)
            index //= card
        return np.array(values, dtype=int)

    def _values_to_index(self, values: np.ndarray, cards: list) -> int:
        """Convert variable values to configuration index (mixed-radix)"""
        index = 0
        mult = 1
        for val, card in zip(values, cards):
            index += int(val) * mult
            mult *= card
        return index

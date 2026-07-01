"""
Frequency Balance Mutation Component for Multi-value EDA Framework

Implements MutationMethod interface for frequency-based balance mutation
for variables with cardinality > 2, that can be integrated into the EDA
execution pipeline.
"""

from typing import Any
import numpy as np

from pateda.core.components import MutationMethod
from pateda.mutation.frequency_balance_multivalue import (
    frequency_balance_multivalue_mutation,
)


class FrequencyBalanceMultivalueMutation(MutationMethod):
    """
    Frequency balance mutation component for multi-value EDA framework (c > 2)

    This component ensures no variable has an excessive frequency of any
    single value in the population by applying multi-value frequency balance
    mutation.
    """

    def __init__(self, alpha: float = 0.0):
        """
        Initialize multi-value frequency balance mutation

        Args:
            alpha: Maximum allowed frequency threshold (default 0.0, no mutation)
                   If alpha=0, no mutation is applied
        """
        self.alpha = alpha

    def mutate(
        self,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Apply multi-value frequency balance mutation to population

        Args:
            n_vars: Number of variables
            cardinality: Variable cardinalities (one entry per variable)
            population: Population to mutate
            **params: Additional parameters (not used, alpha is set in constructor)

        Returns:
            Mutated population
        """
        mutation_params = {'alpha': self.alpha}

        return frequency_balance_multivalue_mutation(
            n_vars,
            cardinality,
            population,
            mutation_params,
        )

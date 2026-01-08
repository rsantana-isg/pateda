"""
Frequency Balance Mutation Component for EDA Framework

Implements MutationMethod interface for frequency-based balance mutation
that can be integrated into the EDA execution pipeline.
"""

from typing import Any
import numpy as np

from pateda.core.components import MutationMethod
from pateda.mutation.frequency_balance import frequency_balance_mutation


class FrequencyBalanceMutation(MutationMethod):
    """
    Frequency balance mutation component for EDA framework
    
    This component ensures no variable has an excessive frequency of
    ones or zeros in the population by applying frequency balance mutation.
    """
    
    def __init__(self, alpha: float = 0.0):
        """
        Initialize frequency balance mutation
        
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
        Apply frequency balance mutation to population
        
        Args:
            n_vars: Number of variables
            cardinality: Variable cardinalities or ranges
            population: Population to mutate
            **params: Additional parameters (alpha can override instance alpha)
        
        Returns:
            Mutated population
        """
        # Get alpha from params or use instance alpha
        alpha = params.get('alpha', self.alpha)
        
        # Create mutation params dict
        mutation_params = {'alpha': alpha}
        
        # Apply frequency balance mutation
        return frequency_balance_mutation(
            n_vars,
            cardinality,
            population,
            mutation_params
        )

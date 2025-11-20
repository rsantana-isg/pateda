"""
Unitation constraint repairing method wrapper

This module provides a RepairingMethod wrapper for the unitation repairing
function, allowing it to be used in the EDA component framework.
"""

from typing import Any
import numpy as np

from pateda.core.components import RepairingMethod
from pateda.repairing.unitation import unitation_repairing


class UnitationRepairing(RepairingMethod):
    """
    Repairing method for maintaining unitation constraint (fixed number of ones).

    This method wraps the unitation_repairing function to enforce that all
    solutions in a binary population have a number of ones within a specified range.

    For the special case of fixing the number of ones to exactly k, set both
    min_ones and max_ones to k.

    Attributes:
        min_ones: Minimum number of ones allowed
        max_ones: Maximum number of ones allowed

    Examples:
        >>> # Create repairing method for exactly 5 ones
        >>> repairing = UnitationRepairing(min_ones=5, max_ones=5)
        >>> # Or use the factory method
        >>> repairing = UnitationRepairing.exact_k_ones(k=5)
    """

    def __init__(self, min_ones: int = None, max_ones: int = None):
        """
        Initialize unitation repairing method.

        Args:
            min_ones: Minimum number of ones (required)
            max_ones: Maximum number of ones (required)

        Raises:
            ValueError: If min_ones or max_ones is not provided
        """
        if min_ones is None or max_ones is None:
            raise ValueError("Both min_ones and max_ones must be specified")

        if min_ones > max_ones:
            raise ValueError(f"min_ones ({min_ones}) must be <= max_ones ({max_ones})")

        self.min_ones = min_ones
        self.max_ones = max_ones

    def repair(
        self,
        population: np.ndarray,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Repair population to satisfy unitation constraint.

        Args:
            population: Population to repair (pop_size, n_vars)
            cardinality: Variable cardinalities (not used for binary)
            **params: Additional parameters
                min_ones: Override minimum number of ones
                max_ones: Override maximum number of ones

        Returns:
            Repaired population where each solution has number of ones
            in [min_ones, max_ones]
        """
        # Allow overriding via params
        min_ones = params.get('min_ones', self.min_ones)
        max_ones = params.get('max_ones', self.max_ones)

        range_values = (min_ones, max_ones)

        # Apply unitation repairing
        repaired_pop = unitation_repairing(population, range_values)

        return repaired_pop

    @classmethod
    def exact_k_ones(cls, k: int) -> "UnitationRepairing":
        """
        Factory method to create repairing for exactly k ones.

        Args:
            k: Exact number of ones required

        Returns:
            UnitationRepairing instance configured for exactly k ones

        Examples:
            >>> repairing = UnitationRepairing.exact_k_ones(k=10)
            >>> # All repaired solutions will have exactly 10 ones
        """
        return cls(min_ones=k, max_ones=k)

    def __repr__(self) -> str:
        if self.min_ones == self.max_ones:
            return f"UnitationRepairing(exactly {self.min_ones} ones)"
        else:
            return f"UnitationRepairing({self.min_ones} to {self.max_ones} ones)"

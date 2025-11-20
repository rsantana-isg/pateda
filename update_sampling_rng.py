#!/usr/bin/env python3
"""
Script to update all sampling files with rng parameter support
"""

import os
import re

def update_file(filepath, updates):
    """Apply a list of (old_string, new_string) updates to a file"""
    with open(filepath, 'r') as f:
        content = f.read()

    for old, new in updates:
        if old not in content:
            print(f"Warning: Could not find pattern in {filepath}")
            print(f"Pattern: {old[:100]}...")
            continue
        content = content.replace(old, new)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Updated {filepath}")

# Define updates for each file
base_dir = "/home/user/pateda/pateda/sampling"

# cumda.py - uses stochastic_universal_sampling
cumda_updates = [
    (
        """    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:""",
        """    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:"""
    ),
    (
        """        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables""",
        """        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables"""
    ),
    (
        "            selected_indices = stochastic_universal_sampling(n_ones, cum_probs)",
        "            selected_indices = stochastic_universal_sampling(n_ones, cum_probs, rng)"
    ),
    (
        "            n_ones_this = np.random.randint(min_ones, max_ones + 1)",
        "            n_ones_this = rng.integers(min_ones, max_ones + 1)"
    ),
    (
        "                selected_indices = stochastic_universal_sampling(n_ones_this, cum_probs)",
        "                selected_indices = stochastic_universal_sampling(n_ones_this, cum_probs, rng)"
    )
]

# cfda.py - delegates to fda
cfda_updates = [
    (
        """    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:""",
        """    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:"""
    ),
    (
        """        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables""",
        """        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables"""
    ),
    (
        """        unconstrained_pop = self.fda_sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            n_samples=n_samples,
        )""",
        """        unconstrained_pop = self.fda_sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            rng=rng,
            n_samples=n_samples,
        )"""
    )
]

# Apply updates
update_file(os.path.join(base_dir, "cumda.py"), cumda_updates)
update_file(os.path.join(base_dir, "cfda.py"), cfda_updates)

print("\nCompleted!")

"""
Inference methods for probabilistic graphical models.

This module provides inference capabilities for:
- MAP (Maximum A Posteriori) computation
- k-MAP (k most probable configurations)
- Belief propagation
- Junction tree inference
- KMPC (K Most Probable Configurations) via Nilsson's partition scheme
"""

from pateda.inference.map_inference import (
    compute_map,
    compute_k_map,
    compute_map_decimation,
    MAPInference
)

from pateda.inference.kmpc import (
    KMPCResult,
    KMPCUnivariate,
    KMPCTree,
    KMPCJunctionTree,
    KMPCBayesianNetwork,
    compute_kmpc,
)

from pateda.inference.max_product_mpc import (
    max_product_mpc,
    MPCIntractable,
)

__all__ = [
    'compute_map',
    'compute_k_map',
    'compute_map_decimation',
    'MAPInference',
    # exact MPC via junction-tree max-product
    'max_product_mpc',
    'MPCIntractable',
    # KMPC
    'KMPCResult',
    'KMPCUnivariate',
    'KMPCTree',
    'KMPCJunctionTree',
    'KMPCBayesianNetwork',
    'compute_kmpc',
]

"""
Inference methods for probabilistic graphical models.

This module provides inference capabilities for:
- MAP (Maximum A Posteriori) computation
- k-MAP (k most probable configurations)
- Belief propagation
- Junction tree inference
"""

from pateda.inference.map_inference import (
    compute_map,
    compute_k_map,
    compute_map_decimation,
    MAPInference
)

__all__ = [
    'compute_map',
    'compute_k_map',
    'compute_map_decimation',
    'MAPInference'
]

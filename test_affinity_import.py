"""Quick test to verify affinity module imports correctly"""

import sys
import numpy as np

# Test basic imports
try:
    from pateda.learning.affinity import (
        LearnAffinityFactorization,
        LearnAffinityFactorizationElim,
    )
    print("✓ Successfully imported affinity learning classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test sklearn dependency
try:
    from sklearn.cluster import AffinityPropagation
    print("✓ scikit-learn AffinityPropagation is available")
except ImportError as e:
    print(f"✗ sklearn import error: {e}")
    sys.exit(1)

# Test initialization
try:
    learner1 = LearnAffinityFactorization(max_clique_size=5)
    learner2 = LearnAffinityFactorizationElim(max_clique_size=5)
    print("✓ Successfully initialized both affinity learning classes")
except Exception as e:
    print(f"✗ Initialization error: {e}")
    sys.exit(1)

# Test mutual information matrix computation
try:
    # Create simple test data
    n_vars = 6
    n_samples = 50
    population = np.random.randint(0, 2, size=(n_samples, n_vars))
    cardinality = np.full(n_vars, 2)

    # Import marginal prob function
    from pateda.learning.utils.marginal_prob import find_marginal_prob

    univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality)

    mi_matrix = learner1._compute_mutual_information_matrix(
        population, n_vars, cardinality, univ_prob, biv_prob
    )

    print(f"✓ Computed MI matrix shape: {mi_matrix.shape}")
    print(f"  MI matrix range: [{mi_matrix.min():.4f}, {mi_matrix.max():.4f}]")

except Exception as e:
    print(f"✗ MI computation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test affinity clustering
try:
    labels, converged = learner1._affinity_clustering(mi_matrix)
    n_clusters = len(np.unique(labels))
    print(f"✓ Affinity clustering successful")
    print(f"  Found {n_clusters} clusters, converged: {converged}")
except Exception as e:
    print(f"✗ Clustering error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! Affinity-based factorization is ready to use.")
print("=" * 60)

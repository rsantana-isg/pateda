"""Tests for joint-table size constraints in discrete learning algorithms."""

import numpy as np

from pateda.learning.boa import LearnBOA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.fda import LearnFDA
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfda_r import LearnMNFDAR
from pateda.learning.utils.table_size import joint_table_size


def _all_structure_table_sizes(structure: np.ndarray, cardinality: np.ndarray):
    sizes = []
    for row in structure:
        n_overlap = int(row[0])
        n_new = int(row[1])
        vars_in_row = row[2 : 2 + n_overlap + n_new].astype(int)
        sizes.append(joint_table_size(cardinality, vars_in_row))
    return sizes


def test_ebna_limits_parent_sets_by_table_size():
    n_vars = 6
    n_samples = 128
    cardinality = np.array([64] * n_vars)

    population = np.zeros((n_samples, n_vars), dtype=int)
    base = np.random.randint(0, 64, size=n_samples)
    for i in range(n_vars):
        population[:, i] = base
    fitness = np.sum(population, axis=1).astype(float)

    model = LearnEBNA(max_parents=3).learn(0, n_vars, cardinality, population, fitness)
    for var in range(n_vars):
        assert len(model.parameters[var]["parents"]) == 0


def test_boa_limits_parent_sets_by_table_size():
    n_vars = 6
    n_samples = 128
    cardinality = np.array([64] * n_vars)

    population = np.zeros((n_samples, n_vars), dtype=int)
    base = np.random.randint(0, 64, size=n_samples)
    for i in range(n_vars):
        population[:, i] = base
    fitness = np.sum(population, axis=1).astype(float)

    model = LearnBOA(max_parents=3).learn(0, n_vars, cardinality, population, fitness)
    for var in range(n_vars):
        assert len(model.parameters[var]["parents"]) == 0


def test_fda_splits_oversized_cliques():
    n_vars = 4
    n_samples = 128
    cardinality = np.array([64] * n_vars)
    population = np.random.randint(0, 64, size=(n_samples, n_vars))
    fitness = np.sum(population, axis=1).astype(float)

    # One oversized clique with all variables: 64^4 > n_samples
    cliques = np.array([[0, 4, 0, 1, 2, 3]])
    model = LearnFDA(cliques=cliques).learn(0, n_vars, cardinality, population, fitness)

    for table_size in _all_structure_table_sizes(model.structure, cardinality):
        assert table_size <= n_samples


def test_mnfda_limits_clique_table_size():
    n_vars = 6
    n_samples = 128
    cardinality = np.array([64] * n_vars)
    population = np.random.randint(0, 64, size=(n_samples, n_vars))
    population[:, 1] = population[:, 0]
    population[:, 2] = population[:, 0]
    fitness = np.sum(population, axis=1).astype(float)

    model = LearnMNFDA(max_clique_size=3, return_factorized=True).learn(
        0, n_vars, cardinality, population, fitness
    )

    for table_size in _all_structure_table_sizes(model.structure, cardinality):
        assert table_size <= n_samples


def test_mnfdar_limits_clique_table_size():
    n_vars = 6
    n_samples = 128
    cardinality = np.array([64] * n_vars)
    population = np.random.randint(0, 64, size=(n_samples, n_vars))
    population[:, 1] = population[:, 0]
    population[:, 2] = population[:, 0]
    fitness = np.sum(population, axis=1).astype(float)
    interaction_matrix = np.ones((n_vars, n_vars), dtype=int)

    model = LearnMNFDAR(
        interaction_matrix=interaction_matrix,
        max_clique_size=3,
        return_factorized=True,
    ).learn(0, n_vars, cardinality, population, fitness)

    for table_size in _all_structure_table_sizes(model.structure, cardinality):
        assert table_size <= n_samples

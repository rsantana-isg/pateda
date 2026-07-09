"""
Tree-Max function for optimization benchmarking

Based on Section 3.2 of "Using Optimal Dependency-Trees for Combinatorial Optimization"
(Baluja & Davies, 1997).
The objective is to find a bit-string that maximizes the probability under a randomly 
generated tree-structured Bayesian network.
"""

import numpy as np


class TreeMaxInstance:
    """
    Represents an instance of the Tree-Max problem, consisting of a tree structure
    and conditional probability tables for each node.
    """
    def __init__(self, parents: np.ndarray, root: int, q_root: float, q_cond: np.ndarray):
        """
        Args:
            parents: Array of parent indices. parents[i] is the parent of node i, with parents[root] = -1.
            root: Index of the root node.
            q_root: Marginal probability P(X_root = 1).
            q_cond: Array of shape (n_vars, 2) where q_cond[i, val] is P(X_i = 1 | X_parent = val).
        """
        self.parents = parents
        self.root = root
        self.q_root = q_root
        self.q_cond = q_cond
        self.n_vars = len(parents)


def generate_random_tree_max(n_vars: int, dist_type: str = 'uniform') -> TreeMaxInstance:
    """
    Generate a random Tree-Max instance.

    Args:
        n_vars: Number of variables.
        dist_type: Distribution for random probabilities. Options:
            - 'uniform': Uniform in [0, 1]
            - 'extreme': Uniform in [0, 0.2] or [0.8, 1.0] (50/50 probability)
            - 'middle': Uniform in [0.4, 0.6]

    Returns:
        A TreeMaxInstance object.
    """
    # 1. Generate a random tree structure using a random permutation
    perm = np.random.permutation(n_vars)
    root = perm[0]
    parents = np.zeros(n_vars, dtype=int) - 1

    for i in range(1, n_vars):
        node = perm[i]
        # Choose a parent from the already added nodes
        parent_idx = np.random.randint(0, i)
        parents[node] = perm[parent_idx]

    # Helper to generate random probabilities based on distribution type
    def sample_probs(size):
        if dist_type == 'uniform':
            return np.random.uniform(0.0, 1.0, size)
        elif dist_type == 'extreme':
            low = np.random.uniform(0.0, 0.2, size)
            high = np.random.uniform(0.8, 1.0, size)
            choices = np.random.binomial(1, 0.5, size)
            return low * (1 - choices) + high * choices
        elif dist_type == 'middle':
            return np.random.uniform(0.4, 0.6, size)
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}. Choose 'uniform', 'extreme', or 'middle'.")

    q_root = float(sample_probs(1)[0])
    q_cond = np.zeros((n_vars, 2))

    # Generate P(X_i = 1 | X_parent = 0) and P(X_i = 1 | X_parent = 1)
    q_cond[:, 0] = sample_probs(n_vars)
    q_cond[:, 1] = sample_probs(n_vars)

    # Root node has no parent conditional probability, but we fill it with q_root for consistency
    q_cond[root, :] = q_root

    return TreeMaxInstance(parents, root, q_root, q_cond)


def eval_tree_max(x: np.ndarray, instance: TreeMaxInstance, log_prob: bool = True) -> float:
    """
    Evaluate the joint probability of a binary vector under the tree-max network.

    Args:
        x: Binary vector representing a solution.
        instance: The TreeMaxInstance to evaluate against.
        log_prob: If True, returns log-probability (default, more stable).
                  If False, returns raw probability.

    Returns:
        Probability (or log-probability) value.
    """
    if x.ndim == 2:
        x = x.flatten()

    if len(x) != instance.n_vars:
        raise ValueError(f"Solution length ({len(x)}) does not match instance variable count ({instance.n_vars})")

    # Evaluate the root node
    root_val = int(x[instance.root])
    p_root = instance.q_root if root_val == 1 else 1.0 - instance.q_root

    if log_prob:
        val = np.log(p_root + 1e-15)
        for i in range(instance.n_vars):
            if i == instance.root:
                continue
            parent_val = int(x[instance.parents[i]])
            prob = instance.q_cond[i, parent_val]
            node_val = int(x[i])
            p_node = prob if node_val == 1 else 1.0 - prob
            val += np.log(p_node + 1e-15)
    else:
        val = p_root
        for i in range(instance.n_vars):
            if i == instance.root:
                continue
            parent_val = int(x[instance.parents[i]])
            prob = instance.q_cond[i, parent_val]
            node_val = int(x[i])
            p_node = prob if node_val == 1 else 1.0 - prob
            val *= p_node

    return float(val)


def create_tree_max_objective_function(instance: TreeMaxInstance, log_prob: bool = True):
    """
    Create a Tree-Max objective function for use with EDAs

    Args:
        instance: The TreeMaxInstance to evaluate against.
        log_prob: If True, evaluates log-probabilities (recommended for stability).

    Returns:
        Objective function that takes a population array and returns fitness values.
    """
    def objective(population: np.ndarray) -> np.ndarray:
        """
        Evaluate Tree-Max function for a population

        Args:
            population: 2D array of shape (pop_size, n_vars) or 1D array of shape (n_vars,)

        Returns:
            1D array of fitness values of shape (pop_size,) or shape (1,)
        """
        if population.ndim == 1:
            return np.array([eval_tree_max(population, instance, log_prob)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = eval_tree_max(population[i], instance, log_prob)

        return fitness

    return objective

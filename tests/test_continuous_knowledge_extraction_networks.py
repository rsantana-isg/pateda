"""
Tests for the continuous-EDA knowledge-extraction modules:
``gaussian_networks`` (interaction networks from Gaussian covariance) and
``vine_analysis`` (vine-copula structure/parameter analysis).
"""
import numpy as np
import pytest

from pateda.knowledge_extraction import gaussian_networks as gnet


# ---------------------------------------------------------------------------
# Gaussian networks
# ---------------------------------------------------------------------------

def _block_covariance():
    """Covariance whose strong correlations are (0,1) and (2,3); 4,5 independent."""
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal((n, 6))
    x = np.empty((n, 6))
    x[:, 0] = z[:, 0]
    x[:, 1] = 0.9 * z[:, 0] + 0.1 * z[:, 1]
    x[:, 2] = z[:, 2]
    x[:, 3] = 0.9 * z[:, 2] + 0.1 * z[:, 3]
    x[:, 4] = z[:, 4]
    x[:, 5] = z[:, 5]
    return np.cov(x, rowvar=False)


def test_extract_gaussian_parameters_from_model():
    from pateda.core.models import GaussianModel
    cov = _block_covariance()
    gm = GaussianModel(structure=None, parameters={"mean": np.zeros(6), "cov": cov})
    mean, c = gnet.extract_gaussian_parameters(gm)
    assert mean.shape == (6,)
    assert c.shape == (6, 6)
    np.testing.assert_allclose(c, cov)


def test_precision_and_partial_correlation():
    cov = _block_covariance()
    prec = gnet.covariance_to_precision(cov)
    pc = gnet.partial_correlation_matrix(prec)
    assert prec.shape == (6, 6)
    np.testing.assert_allclose(np.diag(pc), 1.0, atol=1e-8)
    # Strong direct dependencies should have large |partial correlation|.
    assert abs(pc[0, 1]) > 0.5
    assert abs(pc[2, 3]) > 0.5
    # Independent variables: near-zero partial correlation.
    assert abs(pc[4, 5]) < 0.2


def test_gaussian_interaction_network_recovers_blocks():
    cov = _block_covariance()
    for method in ("partial_correlation", "glasso", "correlation"):
        net = gnet.gaussian_interaction_network(cov, method=method, threshold=0.2)
        adj = net["adjacency"]
        assert adj.shape == (6, 6)
        assert np.array_equal(adj, adj.T)          # undirected
        assert adj[0, 1] == 1 and adj[2, 3] == 1   # strong pairs present
        assert adj[4, 5] == 0                       # independent pair absent
        assert net["directed"] is False


def test_orientation_produces_directed_graph():
    cov = _block_covariance()
    net = gnet.gaussian_interaction_network(cov, method="partial_correlation", threshold=0.2)
    directed = gnet.orient_edges_likelihood_score(net["adjacency"], cov)
    # Each undirected edge becomes exactly one arc.
    n_undirected = int(net["adjacency"].sum() // 2)
    assert int(directed.sum()) == n_undirected
    assert not np.array_equal(directed, directed.T)


def test_compare_and_combine_networks():
    a = np.zeros((6, 6), dtype=int); a[0, 1] = a[1, 0] = a[2, 3] = a[3, 2] = 1
    b = np.zeros((6, 6), dtype=int); b[0, 1] = b[1, 0] = b[4, 5] = b[5, 4] = 1
    cmp = gnet.compare_networks(a, b)
    assert cmp["common_edges"] == 1            # only (0,1) shared
    assert cmp["only_in_a"] == 1 and cmp["only_in_b"] == 1
    assert cmp["jaccard"] == pytest.approx(1 / 3)
    union = gnet.combine_networks(a, b, mode="union")
    inter = gnet.combine_networks(a, b, mode="agreement")
    assert int(union.sum() // 2) == 3
    assert int(inter.sum() // 2) == 1


def test_gaussian_network_evolution():
    from pateda.core.models import GaussianModel
    cov = _block_covariance()
    models = [GaussianModel(structure=None,
                            parameters={"mean": np.zeros(6), "cov": cov}) for _ in range(4)]
    evo = gnet.gaussian_network_evolution(models, method="partial_correlation", threshold=0.2)
    assert evo["n_generations"] == 4
    assert len(evo["adjacencies"]) == 4
    assert evo["n_edges"].shape == (4,)


# ---------------------------------------------------------------------------
# Vine analysis (skipped when pyvinecopulib is unavailable)
# ---------------------------------------------------------------------------

pv = pytest.importorskip("pyvinecopulib", reason="pyvinecopulib not installed")


def _fit_vine():
    from scipy.stats import rankdata
    rng = np.random.default_rng(7)
    n = 600
    z = rng.standard_normal((n, 6))
    x = np.empty((n, 6))
    x[:, 0] = z[:, 0]; x[:, 1] = 0.85 * z[:, 0] + 0.15 * z[:, 1]
    x[:, 2] = z[:, 2]; x[:, 3] = 0.85 * z[:, 2] + 0.15 * z[:, 3]
    x[:, 4] = z[:, 4]; x[:, 5] = 0.7 * z[:, 4] + 0.3 * z[:, 5]
    u = np.column_stack([rankdata(x[:, j]) / (n + 1) for j in range(6)])
    fam = [pv.BicopFamily.gaussian, pv.BicopFamily.clayton, pv.BicopFamily.frank,
           pv.BicopFamily.indep, pv.BicopFamily.gumbel]
    vc = pv.Vinecop.from_data(data=u, controls=pv.FitControlsVinecop(family_set=fam))
    return {"vine_model": vc, "type": "vine_copula_auto"}


def test_vine_structure_decoding():
    from pateda.knowledge_extraction import vine_analysis as va
    model = _fit_vine()
    s = va.vine_structure(model)
    assert s["n_vars"] == 6
    assert len(s["edges"]) == 6 * 5 // 2          # n(n-1)/2 pair-copulas
    first_tree = [e for e in s["edges"] if e["tree"] == 1]
    assert len(first_tree) == 5                     # n-1 edges in T1


def test_vine_first_tree_network_recovers_strong_pairs():
    from pateda.knowledge_extraction import vine_analysis as va
    model = _fit_vine()
    net = va.first_tree_network(model, tau_threshold=0.2)
    adj = net["adjacency"]
    assert np.array_equal(adj, adj.T)
    # The three strong pairs (0,1),(2,3),(4,5) should be first-tree edges.
    assert adj[0, 1] == 1 and adj[2, 3] == 1 and adj[4, 5] == 1


def test_vine_family_and_tau_statistics():
    from pateda.knowledge_extraction import vine_analysis as va
    model = _fit_vine()
    comp = va.family_composition(model)
    assert comp["total_pair_copulas"] == 15
    assert comp["n_non_independence"] >= 3
    taus = va.tau_by_tree(model)
    # Dependence is strongest in the first tree.
    assert taus["mean_abs_tau_by_tree"][1] >= taus["mean_abs_tau_by_tree"].get(2, 0.0)
    assert va.effective_truncation(model) >= 1


def test_vine_evolution():
    from pateda.knowledge_extraction import vine_analysis as va
    models = [_fit_vine() for _ in range(3)]
    evo = va.vine_evolution(models)
    assert evo["n_generations"] == 3
    assert len(evo["first_tree_adjacencies"]) == 3
    assert evo["series"]["first_tree_edges"].shape == (3,)

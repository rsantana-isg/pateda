"""
Reorganization script: creates pateda and pateda_nn packages from the
root-level pateda source tree.

Run from the packages/ directory:
    python reorganize.py
"""

import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent  # pateda repo root
PACKAGES = Path(__file__).parent     # packages/

PATEDA_SRC = PACKAGES / "pateda" / "src" / "pateda"
PATEDA_NN_SRC = PACKAGES / "pateda_nn" / "src" / "pateda_nn"

# ── NN learning modules (go to pateda_nn) ────────────────────────────────────
NN_LEARNING = {
    "backdrive", "dae", "dbd", "dendiff", "dendiff_relu",
    "discrete_backdrive", "discrete_backdrive_descriptors",
    "discrete_backdrive_huber", "discrete_backdrive_ranking",
    "discrete_backdrive_weighted_mse", "discrete_dbd",
    "discrete_dendiff_corruption", "discrete_dendiff_corruption_enhanced",
    "discrete_dendiff_deterministic", "discrete_dendiff_deterministic_enhanced",
    "discrete_dendiff_gumbel", "discrete_dendiff_gumbel_enhanced",
    "discrete_dendiff_hard_concrete", "discrete_dendiff_hard_concrete_enhanced",
    "discrete_dendiff_ste", "discrete_dendiff_ste_enhanced",
    "discrete_dendiff_utils", "discrete_gan", "discrete_vae",
    "gan", "nn_eda", "nn_utils", "rbm", "vae",
}

# ── NN sampling modules (go to pateda_nn) ────────────────────────────────────
NN_SAMPLING = {
    "backdrive", "dae", "dbd", "dendiff", "dendiff_relu",
    "discrete_dbd", "discrete_dendiff", "discrete_neural",
    "gan", "nn_eda", "rbm", "vae",
}


def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, transform=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    if transform:
        text = transform(text)
    dst.write_text(text, encoding="utf-8")


# ── Import rewriting helpers ──────────────────────────────────────────────────

def _nn_module_names(subpackage: str) -> set:
    return NN_LEARNING if subpackage == "learning" else NN_SAMPLING


def rewrite_for_pateda(text: str) -> str:
    """Remove sys.path hacks; otherwise no-op (imports already use pateda.*)."""
    # remove sys.path.insert lines
    text = re.sub(r'^\s*sys\.path\.insert\([^\n]*\)\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*import sys\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*import os\n', '', text, flags=re.MULTILINE)
    return text


def rewrite_for_pateda_nn(text: str) -> str:
    """
    Rewrite imports so that:
      - References to NN learning/sampling modules use pateda_nn.*
      - References to classical pateda modules stay as pateda.*
      - Old sys.path hacks become proper pateda_nn imports
    """
    # Fix sys.path hacks first (sampling files)
    # Pattern: sys.path.insert + from learning.X import Y
    #       -> from pateda_nn.learning.X import Y
    text = re.sub(
        r'import sys\nimport os\nsys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(__file__\)\)\)\nfrom (learning|sampling)\.(\w+) import',
        r'from pateda_nn.\1.\2 import',
        text,
    )
    # Also handle cases where sys.path / os are imported earlier
    text = re.sub(r'^\s*sys\.path\.insert\([^\n]*\)\n', '', text, flags=re.MULTILINE)

    # Rewrite: from pateda.learning.<nn_module> -> from pateda_nn.learning.<nn_module>
    nn_learn_pattern = '|'.join(re.escape(m) for m in sorted(NN_LEARNING, key=len, reverse=True))
    text = re.sub(
        rf'from pateda\.learning\.({nn_learn_pattern})\b',
        r'from pateda_nn.learning.\1',
        text,
    )

    # Rewrite: from pateda.sampling.<nn_module> -> from pateda_nn.sampling.<nn_module>
    nn_sample_pattern = '|'.join(re.escape(m) for m in sorted(NN_SAMPLING, key=len, reverse=True))
    text = re.sub(
        rf'from pateda\.sampling\.({nn_sample_pattern})\b',
        r'from pateda_nn.sampling.\1',
        text,
    )

    # Fix remaining bare module imports (from learning.X / from sampling.X)
    text = re.sub(
        rf'from learning\.({nn_learn_pattern})\b',
        r'from pateda_nn.learning.\1',
        text,
    )
    text = re.sub(
        rf'from sampling\.({nn_sample_pattern})\b',
        r'from pateda_nn.sampling.\1',
        text,
    )

    return text


# ── Copy helpers ──────────────────────────────────────────────────────────────

def copy_dir_filtered(src_dir: Path, dst_dir: Path, exclude_stems: set = None,
                      only_stems: set = None, transform=None):
    """Copy .py files from src_dir to dst_dir with optional filtering."""
    for f in sorted(src_dir.glob("*.py")):
        stem = f.stem
        if exclude_stems and stem in exclude_stems:
            continue
        if only_stems and stem not in only_stems and stem != "__init__":
            continue
        copy_file(f, dst_dir / f.name, transform)


def copy_dir_all(src_dir: Path, dst_dir: Path, transform=None):
    """Copy all .py files recursively."""
    for f in sorted(src_dir.rglob("*.py")):
        rel = f.relative_to(src_dir)
        copy_file(f, dst_dir / rel, transform)


def touch_init(path: Path):
    init = path / "__init__.py"
    if not init.exists():
        init.write_text('"""Package init"""\n')


# ── Build pateda package ──────────────────────────────────────────────────────

def build_pateda():
    print("Building pateda package...")
    make_dirs(PATEDA_SRC)

    T = rewrite_for_pateda  # transform (mostly no-op)

    # core
    copy_dir_all(ROOT / "core", PATEDA_SRC / "core", T)

    # learning – exclude NN modules
    dst_learn = PATEDA_SRC / "learning"
    src_learn = ROOT / "learning"
    copy_dir_filtered(src_learn, dst_learn, exclude_stems=NN_LEARNING, transform=T)
    # learning/utils
    utils_src = src_learn / "utils"
    if utils_src.exists():
        copy_dir_all(utils_src, dst_learn / "utils", T)
    # rewrite learning/__init__.py: strip NN try/except blocks
    _write_pateda_learning_init(dst_learn)

    # sampling – exclude NN modules
    dst_sample = PATEDA_SRC / "sampling"
    src_sample = ROOT / "sampling"
    copy_dir_filtered(src_sample, dst_sample, exclude_stems=NN_SAMPLING, transform=T)
    _write_pateda_sampling_init(dst_sample)

    # all other pure-Python modules
    for mod in ["selection", "seeding", "mutation", "replacement", "repairing",
                "stop_conditions", "local_optimization", "crossover",
                "inference", "statistics", "knowledge_extraction"]:
        src_mod = ROOT / mod
        if src_mod.exists():
            copy_dir_all(src_mod, PATEDA_SRC / mod, T)

    # selection/utils
    sel_utils = ROOT / "selection" / "utils"
    if sel_utils.exists():
        copy_dir_all(sel_utils, PATEDA_SRC / "selection" / "utils", T)

    # functions (including data subdirs; copy Python files only)
    _copy_functions(ROOT / "functions", PATEDA_SRC / "functions", T)

    # permutation utilities (distances, consensus — used by Mallows EDA)
    perm_src = ROOT / "permutation"
    if perm_src.exists():
        copy_dir_all(perm_src, PATEDA_SRC / "permutation", T)

    # root __init__.py
    copy_file(ROOT / "__init__.py", PATEDA_SRC / "__init__.py", T)

    print(f"  → {PATEDA_SRC}")


def _copy_functions(src: Path, dst: Path, T):
    """Recursively copy functions/ including instance data files."""
    for item in sorted(src.rglob("*")):
        if item.is_file():
            rel = item.relative_to(src)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if item.suffix == ".py":
                copy_file(item, dst_file, T)
            else:
                shutil.copy2(item, dst_file)


def _write_pateda_learning_init(dst: Path):
    """Generate a clean learning/__init__.py without NN imports."""
    content = '''"""Probabilistic model learning methods"""

from pateda.learning.fda import LearnFDA
from pateda.learning.umda import LearnUMDA
from pateda.learning.cumda import LearnCUMDA
from pateda.learning.cfda import (
    LearnCFDA,
    create_pairwise_chain_cliques,
    create_block_cliques,
    create_overlapping_windows_cliques,
)
from pateda.learning.bmda import LearnBMDA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.boa import LearnBOA
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.tree import LearnTreeModel
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.mnfda_r import LearnMNFDAR
from pateda.learning.mnfdag_r import LearnMNFDAGR
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.moa import LearnMOA
from pateda.learning.interaction_learning import (
    find_matrix_interactions_SAT,
    find_matrix_interactions_ising,
    find_matrix_interactions_nk,
    find_matrix_interactions_additive_decomposable,
    find_matrix_interactions_RNA_design,
)
from pateda.learning.affinity import (
    LearnAffinityFactorization,
    LearnAffinityFactorizationElim,
)
from pateda.learning.pbil import LearnPBIL
from pateda.learning.bsc import LearnBSC
from pateda.learning.mimic import LearnMIMIC
from pateda.learning.basic_gaussian import (
    learn_gaussian_univariate,
    learn_gaussian_full,
    LearnGaussianUnivariate,
    LearnGaussianFull,
)
from pateda.learning.mixture_gaussian import (
    learn_mixture_gaussian_univariate,
    learn_mixture_gaussian_full,
    learn_mixture_gaussian_em,
)
from pateda.learning.gmrf_eda import (
    learn_gmrf_eda,
    learn_gmrf_eda_lasso,
    learn_gmrf_eda_elasticnet,
    learn_gmrf_eda_lars,
)
from pateda.learning.mallows import LearnMallowsKendall, learn_mallows_kendall
from pateda.learning.histogram import LearnEHM, LearnNHM, learn_ehm, learn_nhm

try:
    from pateda.learning.vine_copula import (
        learn_vine_copula_cvine,
        learn_vine_copula_dvine,
        learn_vine_copula_auto,
    )
except ImportError:
    learn_vine_copula_cvine = None
    learn_vine_copula_dvine = None
    learn_vine_copula_auto = None

__all__ = [
    "LearnFDA", "LearnUMDA", "LearnCUMDA", "LearnCFDA",
    "create_pairwise_chain_cliques", "create_block_cliques",
    "create_overlapping_windows_cliques",
    "LearnBMDA", "LearnEBNA", "LearnBOA", "LearnMarkovChain",
    "LearnMixtureTrees", "LearnTreeModel",
    "LearnMNFDA", "LearnMNFDAG", "LearnMNFDAR", "LearnMNFDAGR",
    "LearnTreeModelR", "LearnMOA",
    "find_matrix_interactions_SAT", "find_matrix_interactions_ising",
    "find_matrix_interactions_nk",
    "find_matrix_interactions_additive_decomposable",
    "find_matrix_interactions_RNA_design",
    "LearnAffinityFactorization", "LearnAffinityFactorizationElim",
    "LearnPBIL", "LearnBSC", "LearnMIMIC",
    "learn_gaussian_univariate", "learn_gaussian_full",
    "LearnGaussianUnivariate", "LearnGaussianFull",
    "learn_mixture_gaussian_univariate", "learn_mixture_gaussian_full",
    "learn_mixture_gaussian_em",
    "learn_gmrf_eda", "learn_gmrf_eda_lasso",
    "learn_gmrf_eda_elasticnet", "learn_gmrf_eda_lars",
    "LearnMallowsKendall", "learn_mallows_kendall",
    "LearnEHM", "LearnNHM", "learn_ehm", "learn_nhm",
    "learn_vine_copula_cvine", "learn_vine_copula_dvine",
    "learn_vine_copula_auto",
]
'''
    (dst / "__init__.py").write_text(content)


def _write_pateda_sampling_init(dst: Path):
    """Generate a clean sampling/__init__.py without NN imports."""
    content = '''"""Sampling methods"""

from pateda.sampling.fda import SampleFDA
from pateda.sampling.partial import SamplePartialFDA
from pateda.sampling.cumda import SampleCUMDA, SampleCUMDARange
from pateda.sampling.cfda import (
    SampleCFDA,
    SampleCFDARange,
    SampleCFDAWeighted,
)
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.markov import SampleMarkovChain, SampleMarkovChainForward
from pateda.sampling.mixture_trees import SampleMixtureTrees, SampleMixtureTreesDirect
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.map_sampling import (
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleHybridMAP,
)
from pateda.sampling.basic_gaussian import (
    sample_gaussian_univariate,
    sample_gaussian_full,
    sample_gaussian_with_diversity_trigger,
    SampleGaussianUnivariate,
    SampleGaussianFull,
)
from pateda.sampling.mixture_gaussian import (
    sample_mixture_gaussian_univariate,
    sample_mixture_gaussian_full,
    sample_mixture_gaussian_em,
)
from pateda.sampling.gmrf_eda import sample_gmrf_eda
from pateda.sampling.mallows import SampleMallowsKendall, sample_mallows_kendall
from pateda.sampling.histogram import SampleEHM, SampleNHM, sample_ehm, sample_nhm

try:
    from pateda.sampling.vine_copula import (
        sample_vine_copula,
        sample_vine_copula_biased,
        sample_vine_copula_conditional,
    )
except ImportError:
    sample_vine_copula = None
    sample_vine_copula_biased = None
    sample_vine_copula_conditional = None

__all__ = [
    "SampleFDA", "SamplePartialFDA", "SampleCUMDA", "SampleCUMDARange",
    "SampleCFDA", "SampleCFDARange", "SampleCFDAWeighted",
    "SampleBayesianNetwork", "SampleMarkovChain", "SampleMarkovChainForward",
    "SampleMixtureTrees", "SampleMixtureTreesDirect", "SampleGibbs",
    "SampleInsertMAP", "SampleTemplateMAP", "SampleHybridMAP",
    "sample_gaussian_univariate", "sample_gaussian_full",
    "sample_gaussian_with_diversity_trigger",
    "SampleGaussianUnivariate", "SampleGaussianFull",
    "sample_mixture_gaussian_univariate", "sample_mixture_gaussian_full",
    "sample_mixture_gaussian_em", "sample_gmrf_eda",
    "SampleMallowsKendall", "sample_mallows_kendall",
    "SampleEHM", "SampleNHM", "sample_ehm", "sample_nhm",
    "sample_vine_copula", "sample_vine_copula_biased",
    "sample_vine_copula_conditional",
]
'''
    (dst / "__init__.py").write_text(content)


# ── Build pateda_nn package ───────────────────────────────────────────────────

def build_pateda_nn():
    print("Building pateda_nn package...")
    make_dirs(PATEDA_NN_SRC)

    T = rewrite_for_pateda_nn

    # learning – only NN modules
    dst_learn = PATEDA_NN_SRC / "learning"
    src_learn = ROOT / "learning"
    copy_dir_filtered(src_learn, dst_learn, only_stems=NN_LEARNING, transform=T)
    _write_pateda_nn_learning_init(dst_learn)

    # sampling – only NN modules
    dst_sample = PATEDA_NN_SRC / "sampling"
    src_sample = ROOT / "sampling"
    copy_dir_filtered(src_sample, dst_sample, only_stems=NN_SAMPLING, transform=T)
    _write_pateda_nn_sampling_init(dst_sample)

    # enhanced_edas (legacy TF-based + supporting utilities)
    dst_enh = PATEDA_NN_SRC / "legacy"
    src_enh = ROOT / "enhanced_edas"
    if src_enh.exists():
        copy_dir_all(src_enh, dst_enh)  # no import rewriting needed (standalone)
        _write_legacy_init(dst_enh)

    # root __init__.py
    _write_pateda_nn_init()

    print(f"  → {PATEDA_NN_SRC}")


def _write_pateda_nn_learning_init(dst: Path):
    content = '''"""Neural network-based probabilistic model learning methods (PyTorch)"""

from pateda_nn.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
)
from pateda_nn.learning.vae import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
)
from pateda_nn.learning.gan import learn_gan
from pateda_nn.learning.dbd import learn_dbd
from pateda_nn.learning.dae import learn_dae
from pateda_nn.learning.dendiff import learn_dendiff
from pateda_nn.learning.backdrive import learn_backdrive
from pateda_nn.learning.rbm import learn_rbm
from pateda_nn.learning.nn_eda import LearnNNEDA
from pateda_nn.learning.discrete_vae import (
    learn_discrete_vae,
    learn_discrete_extended_vae,
)
from pateda_nn.learning.discrete_gan import learn_discrete_gan
from pateda_nn.learning.discrete_dbd import (
    learn_discrete_dbd_cs,
    learn_discrete_dbd_cd,
)
from pateda_nn.learning.discrete_backdrive import learn_discrete_backdrive

__all__ = [
    "get_activation", "apply_weight_init",
    "compute_default_hidden_dims", "compute_default_batch_size",
    "learn_vae", "learn_extended_vae", "learn_conditional_extended_vae",
    "learn_gan", "learn_dbd", "learn_dae", "learn_dendiff",
    "learn_backdrive", "learn_rbm", "LearnNNEDA",
    "learn_discrete_vae", "learn_discrete_extended_vae",
    "learn_discrete_gan",
    "learn_discrete_dbd_cs", "learn_discrete_dbd_cd",
    "learn_discrete_backdrive",
]
'''
    (dst / "__init__.py").write_text(content)


def _write_pateda_nn_sampling_init(dst: Path):
    content = '''"""Neural network-based sampling methods (PyTorch)"""

from pateda_nn.sampling.vae import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
)
from pateda_nn.sampling.gan import sample_gan
from pateda_nn.sampling.dbd import sample_dbd
from pateda_nn.sampling.dae import sample_dae
from pateda_nn.sampling.dendiff import sample_dendiff
from pateda_nn.sampling.backdrive import sample_backdrive, sample_backdrive_adaptive
from pateda_nn.sampling.rbm import sample_rbm
from pateda_nn.sampling.nn_eda import SampleNNEDA
from pateda_nn.sampling.discrete_neural import (
    sample_discrete_vae,
    sample_discrete_gan,
    sample_discrete_backdrive,
)
from pateda_nn.sampling.discrete_dbd import (
    sample_discrete_dbd_cs,
    sample_discrete_dbd_cd,
)

__all__ = [
    "sample_vae", "sample_extended_vae", "sample_conditional_extended_vae",
    "sample_gan", "sample_dbd", "sample_dae", "sample_dendiff",
    "sample_backdrive", "sample_backdrive_adaptive",
    "sample_rbm", "SampleNNEDA",
    "sample_discrete_vae", "sample_discrete_gan", "sample_discrete_backdrive",
    "sample_discrete_dbd_cs", "sample_discrete_dbd_cd",
]
'''
    (dst / "__init__.py").write_text(content)


def _write_legacy_init(dst: Path):
    content = '''"""
Legacy TensorFlow-based EDA implementations.

These modules were developed prior to the PyTorch migration.
They require TensorFlow >= 2.0 and may not be actively maintained.
"""
'''
    (dst / "__init__.py").write_text(content)


def _write_pateda_nn_init():
    content = '''"""
pateda_nn — Neural Network EDA implementations (PyTorch)

Builds on the pateda package to provide EDA algorithms powered by deep
generative models: VAE, GAN, Diffusion, Backdrive, DBD, RBM, and DAE.

Requires:
    pateda >= 0.1.0
    torch >= 2.0.0
"""

__version__ = "0.1.0"
__author__ = "Roberto Santana"

from pateda_nn.learning import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
    learn_gan,
    learn_dbd,
    learn_dae,
    learn_dendiff,
    learn_backdrive,
    learn_rbm,
    LearnNNEDA,
)
from pateda_nn.sampling import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
    sample_gan,
    sample_dbd,
    sample_dae,
    sample_dendiff,
    sample_backdrive,
    sample_backdrive_adaptive,
    sample_rbm,
    SampleNNEDA,
)

__all__ = [
    "learn_vae", "learn_extended_vae", "learn_conditional_extended_vae",
    "learn_gan", "learn_dbd", "learn_dae", "learn_dendiff",
    "learn_backdrive", "learn_rbm", "LearnNNEDA",
    "sample_vae", "sample_extended_vae", "sample_conditional_extended_vae",
    "sample_gan", "sample_dbd", "sample_dae", "sample_dendiff",
    "sample_backdrive", "sample_backdrive_adaptive",
    "sample_rbm", "SampleNNEDA",
]
'''
    (PATEDA_NN_SRC / "__init__.py").write_text(content)


# ── Copy tests and examples ───────────────────────────────────────────────────

NN_TEST_PATTERNS = {
    "test_vae", "test_gan", "test_dbd", "test_dae", "test_rbm",
    "test_backdrive", "test_discrete_backdrive_variants",
    "test_discrete_backdrive_population_sampling",
    "test_backdrive_weight_transfer", "test_nn_utils",
    "test_dbd_variants", "test_dendiff_distributions",
    "test_update_data_per_epoch", "test_mi_based_dbd",
    "test_fix_isolated", "demo_backdrive_fix", "demo_weight_transfer_fix",
}

NN_EXAMPLE_PATTERNS = {
    "discrete_VAE_EDA", "discrete_VAE_EDA_RW", "discrete_GAN_EDA",
    "discrete_GAN_EDA_RW", "discrete_Backdrive_EDA", "discrete_Backdrive_EDA_RW",
    "discrete_DbD_EDA", "discrete_DbD_EDA_RW", "discrete_Dendiff_EDA",
    "discrete_Dendiff_EDA_RW", "vae_eda_example", "gan_eda_example",
    "dbd_eda_example", "dendiff_eda_example", "backdrive_eda_examples",
    "enhanced_vae_examples", "neural_eda_comparison",
    "complete_discrete_neural_eda_comparison",
    "complete_integer_neural_eda_comparison",
    "discrete_neural_eda_comparison", "nn_eda_example",
    "dendiff_quick_test", "dbd_quick_test", "dendiff_relu_comparison",
}


def copy_tests_and_examples():
    src_tests = ROOT / "tests"
    src_examples = ROOT / "examples"

    dst_pateda_tests = PACKAGES / "pateda" / "tests"
    dst_pateda_nn_tests = PACKAGES / "pateda_nn" / "tests"
    dst_pateda_examples = PACKAGES / "pateda" / "examples"
    dst_pateda_nn_examples = PACKAGES / "pateda_nn" / "examples"

    for f in sorted(src_tests.glob("*.py")):
        stem = f.stem
        if stem in NN_TEST_PATTERNS:
            copy_file(f, dst_pateda_nn_tests / f.name, rewrite_for_pateda_nn)
        else:
            copy_file(f, dst_pateda_tests / f.name, rewrite_for_pateda)

    for f in sorted(src_examples.glob("*.py")):
        stem = f.stem
        if stem in NN_EXAMPLE_PATTERNS:
            copy_file(f, dst_pateda_nn_examples / f.name, rewrite_for_pateda_nn)
        else:
            copy_file(f, dst_pateda_examples / f.name, rewrite_for_pateda)

    # conftest.py stubs
    for dst in [dst_pateda_tests, dst_pateda_nn_tests]:
        conftest = dst / "conftest.py"
        if not conftest.exists():
            conftest.write_text("# pytest conftest\n")

    # __init__.py stubs
    for dst in [dst_pateda_tests, dst_pateda_nn_tests]:
        touch_init(dst)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_pateda()
    build_pateda_nn()
    copy_tests_and_examples()
    print("Done.")

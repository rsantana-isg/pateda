"""
A context-free grammar that generates feasible EDA pipelines

An *EDA pipeline* is a consistent assembly of EDA components (seeding, selection,
learning, sampling, replacement, optional local search and mutation, stopping
condition) that together form a working EDA.  Following the grammar-based AutoML
literature (RECIPE, de Sa et al. 2017; PIPER, Marinescu et al. 2021), this module
defines a context-free grammar (CFG) whose language is the set of valid pipelines
and a sampler that draws random derivations from it.

The central difficulty is *consistency*: a learning method (LM) produces a model
of some type, and only some sampling methods (SM) can sample that type.  The
grammar solves this by organizing the model part into *typed blocks* -- one per
model type -- so that every complete derivation pairs an LM with a compatible
SM.  The **MODMOD / MODCONV** operators of :mod:`pateda.pipelines.model_operators`
appear inside these blocks and enlarge the set of feasible (LM, SM) combinations:
e.g. a Bayesian-network learner can reach the factorized samplers through the
``bn_to_factorized`` conversor, and a factorized model can be pruned or turned
into a forest / malign tree before sampling.

The grammar (BNF sketch, ``|`` = choice, capitalized names = non-terminals)::

    Pipeline   -> Seeding Selection ModelBlock Replacement LocalOpt Mutation Stop
    ModelBlock -> FacBlock | BNBlock | MarkovNetBlock | IntFDABlock
                | RegMarkovBlock | MarkovChainBlock
    FacBlock   -> FacLearner FacModMod SampleFDA
    BNBlock    -> BNLearner ( SampleBN | bn_to_factorized FacModMod SampleFDA )
    FacModMod  -> empty | prune_factorized | tree_to_forest | tree_to_malign
    ...

A derivation is a flat list of *terminals*; each terminal carries a *role*
(seeding, selection, learner, modop, sampler, ...) and a factory, so
:func:`build_pipeline` can reassemble it into a :class:`~pateda.core.EDA`.

Every complete derivation is *type-consistent by construction*; the remaining
infeasibility a random sample exhibits comes from run-time edge cases
(a mutation operator that assumes binary variables, a learner that needs more
data than a tiny population provides, ...), which is expected and measured by
the accompanying demo.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from pateda.core.components import EDAComponents
from pateda.pipelines.model_operators import ModifiedLearning


# ---------------------------------------------------------------------------
# Terminal symbols: a component (or operator) the grammar can emit
# ---------------------------------------------------------------------------

@dataclass
class Terminal:
    """A terminal of the grammar.

    Attributes:
        name: Unique terminal name (used in production rules).
        role: Pipeline role -- one of ``seeding``, ``selection``, ``learner``,
            ``modop`` (a MODMOD/MODCONV operator), ``sampler``, ``replacement``,
            ``local_opt``, ``mutation``, ``stop``, or ``noop``.
        category: Coverage category (usually the role; ``learner``/``sampler``).
        build: ``build(ctx) -> component`` factory; ``ctx`` carries ``pop_size``.
            ``None`` for ``modop`` / ``noop`` terminals.
        impl: Name of the pateda implementation this terminal covers (for the
            coverage report); may differ from ``name``.
    """
    name: str
    role: str
    category: str
    build: Optional[Callable[[Dict[str, Any]], Any]] = None
    impl: Optional[str] = None


def _t(name, role, build=None, impl=None, category=None):
    return Terminal(name=name, role=role, category=category or role,
                    build=build, impl=impl or name)


# ---- lazy imports keep grammar import light and tolerant of optional deps ---
def _seeding():
    from pateda.seeding import RandomInit
    return {"RandomInit": lambda c: RandomInit()}


def _selection():
    from pateda.selection import (
        TruncationSelection, TournamentSelection, BoltzmannSelection,
        ProportionalSelection, RankingSelection, StochasticUniversalSampling,
    )
    return {
        "Truncation": (lambda c: TruncationSelection(ratio=0.3), "TruncationSelection"),
        "Tournament": (lambda c: TournamentSelection(), "TournamentSelection"),
        "Boltzmann": (lambda c: BoltzmannSelection(), "BoltzmannSelection"),
        "Proportional": (lambda c: ProportionalSelection(), "ProportionalSelection"),
        "Ranking": (lambda c: RankingSelection(), "RankingSelection"),
        "SUS": (lambda c: StochasticUniversalSampling(), "StochasticUniversalSampling"),
    }


def _fac_learners():
    from pateda.learning import (
        LearnUMDA, LearnPBIL, LearnFDA, LearnCFDA, LearnCUMDA, LearnBMDA,
        LearnMIMIC, LearnMNFDA, LearnMNFDAG, LearnTreeModel,
        LearnTreeModelM, LearnAffinityFactorization, LearnAffinityFactorizationElim,
    )
    return {
        "LearnUMDA": lambda c: LearnUMDA(alpha=1.0),
        "LearnPBIL": lambda c: LearnPBIL(),
        "LearnFDA": lambda c: LearnFDA(),
        "LearnCFDA": lambda c: LearnCFDA(),
        "LearnCUMDA": lambda c: LearnCUMDA(),
        "LearnBMDA": lambda c: LearnBMDA(),
        "LearnMIMIC": lambda c: LearnMIMIC(),
        "LearnMNFDA": lambda c: LearnMNFDA(),
        "LearnMNFDAG": lambda c: LearnMNFDAG(),
        "LearnTreeModel": lambda c: LearnTreeModel(),
        "LearnTreeModelM": lambda c: LearnTreeModelM(),
        "LearnAffinityFactorization": lambda c: LearnAffinityFactorization(),
        "LearnAffinityFactorizationElim": lambda c: LearnAffinityFactorizationElim(),
    }


def _bn_learners():
    from pateda.learning import (
        LearnEBNA, LearnBOA, LearnHBOA, LearnLFDA, LearnPADA,
        LearnSARTRE, LearnBINOTEARS, LearnPCBN, LearnHSARTRE, LearnHBINOTEARS,
    )
    return {
        "LearnEBNA": lambda c: LearnEBNA(max_parents=3),
        "LearnBOA": lambda c: LearnBOA(max_parents=3),
        "LearnHBOA": lambda c: LearnHBOA(max_parents=4),
        "LearnLFDA": lambda c: LearnLFDA(max_parents=3),
        "LearnPADA": lambda c: LearnPADA(),
        # Alternative (non score-and-search) BN structure learners.  All return
        # a BayesianNetworkModel, so they pair with SampleBN exactly like EBNA.
        "LearnSARTRE": lambda c: LearnSARTRE(max_parents=4),
        "LearnBINOTEARS": lambda c: LearnBINOTEARS(max_parents=4),
        "LearnPCBN": lambda c: LearnPCBN(max_cond_set_size=3, max_parents=4),
        "LearnHSARTRE": lambda c: LearnHSARTRE(max_parents=6),
        "LearnHBINOTEARS": lambda c: LearnHBINOTEARS(max_parents=6),
    }


def _samplers():
    from pateda.sampling import (
        SampleFDA, SampleBayesianNetwork, SampleGibbs, SampleIntFDA,
        SampleRegularizedMarkov, SampleMarkovChain,
    )
    return {
        "SampleFDA": (lambda c: SampleFDA(n_samples=c["pop_size"]), "SampleFDA"),
        "SampleBN": (lambda c: SampleBayesianNetwork(n_samples=c["pop_size"]),
                     "SampleBayesianNetwork"),
        "SampleGibbs": (lambda c: SampleGibbs(n_samples=c["pop_size"]), "SampleGibbs"),
        "SampleIntFDA": (lambda c: SampleIntFDA(n_samples=c["pop_size"]), "SampleIntFDA"),
        "SampleRegularizedMarkov": (lambda c: SampleRegularizedMarkov(n_samples=c["pop_size"]),
                                    "SampleRegularizedMarkov"),
        "SampleMarkovChain": (lambda c: SampleMarkovChain(n_samples=c["pop_size"]),
                              "SampleMarkovChain"),
    }


def _other_learners():
    from pateda.learning import LearnMOA, LearnIntFDA, LearnRegularizedMarkov, LearnMarkovChain
    return {
        "LearnMOA": lambda c: LearnMOA(),
        "LearnIntFDA": lambda c: LearnIntFDA(),
        "LearnRegularizedMarkov": lambda c: LearnRegularizedMarkov(k=2, variant="rgk"),
        "LearnMarkovChain": lambda c: LearnMarkovChain(k=1, alpha=1.0),
    }


def _replacement():
    from pateda.replacement import (
        ElitistReplacement, GenerationalReplacement,
        RestrictedTournamentReplacement,
    )
    return {
        "Elitist": (lambda c: ElitistReplacement(), "ElitistReplacement"),
        "Generational": (lambda c: GenerationalReplacement(), "GenerationalReplacement"),
        # Niching replacement (hBOA-style); enables HSARTRE/HBINOTEARS-like
        # pipelines (BN learner + decision graph + niching) to be searched.
        "RTR": (lambda c: RestrictedTournamentReplacement(window_size=20),
                "RestrictedTournamentReplacement"),
    }


def _local_opts():
    from pateda.local_optimization import (
        DeterministicHillClimber, FirstImprovementHillClimber, StochasticHillClimber,
        SimulatedAnnealing, VariableNeighborhoodSearch,
        ReducedVariableNeighborhoodSearch, SubstructuralLocalSearch,
        DiscreteGreedySearch, DiscreteSimulatedAnnealing,
    )
    B = dict(subset_fraction=0.2, evaluation_budget=400)
    return {
        "DHC": (lambda c: DeterministicHillClimber(**B), "DeterministicHillClimber"),
        "FirstImpHC": (lambda c: FirstImprovementHillClimber(**B), "FirstImprovementHillClimber"),
        "StochHC": (lambda c: StochasticHillClimber(**B), "StochasticHillClimber"),
        "SA": (lambda c: SimulatedAnnealing(**B), "SimulatedAnnealing"),
        "VNS": (lambda c: VariableNeighborhoodSearch(**B), "VariableNeighborhoodSearch"),
        "RVNS": (lambda c: ReducedVariableNeighborhoodSearch(**B),
                 "ReducedVariableNeighborhoodSearch"),
        "Substructural": (lambda c: SubstructuralLocalSearch(subset_fraction=0.2,
                          evaluation_budget=400), "SubstructuralLocalSearch"),
        "DiscreteGreedy": (lambda c: DiscreteGreedySearch(trials=50), "DiscreteGreedySearch"),
        "DiscreteSA": (lambda c: DiscreteSimulatedAnnealing(), "DiscreteSimulatedAnnealing"),
    }


def _mutation():
    from pateda.mutation import RandomResetMutation, FrequencyBalanceMultivalueMutation
    return {
        "RandomReset": (lambda c: RandomResetMutation(mutation_prob=0.05), "RandomResetMutation"),
        "FreqBalance": (lambda c: FrequencyBalanceMultivalueMutation(alpha=0.1),
                        "FrequencyBalanceMultivalueMutation"),
    }


def _stop():
    from pateda.stop_conditions import MaxGenerations
    return {"MaxGenerations": (lambda c: MaxGenerations(max_gen=c["n_gen"]), "MaxGenerations")}


# ---------------------------------------------------------------------------
# Build the terminal registry
# ---------------------------------------------------------------------------

def _build_terminals() -> Dict[str, Terminal]:
    T: Dict[str, Terminal] = {}

    for name, mk in _seeding().items():
        T[name] = _t(name, "seeding", mk)
    for name, (mk, impl) in _selection().items():
        T[name] = _t(name, "selection", mk, impl)
    for name, mk in _fac_learners().items():
        T[name] = _t(name, "learner", mk, category="learner")
    for name, mk in _bn_learners().items():
        T[name] = _t(name, "learner", mk, category="learner")
    for name, mk in _other_learners().items():
        T[name] = _t(name, "learner", mk, category="learner")
    for name, (mk, impl) in _samplers().items():
        T[name] = _t(name, "sampler", mk, impl, category="sampler")
    for name, (mk, impl) in _replacement().items():
        T[name] = _t(name, "replacement", mk, impl)
    for name, (mk, impl) in _local_opts().items():
        T[name] = _t(name, "local_opt", mk, impl)
    for name, (mk, impl) in _mutation().items():
        T[name] = _t(name, "mutation", mk, impl)
    for name, (mk, impl) in _stop().items():
        T[name] = _t(name, "stop", mk, impl)

    # MODMOD / MODCONV operators (role 'modop', no component factory).
    for op, impl in [("prune_factorized", "prune_factorized"),
                     ("tree_to_forest", "tree_to_forest"),
                     ("tree_to_malign", "tree_to_malign"),
                     ("bn_to_factorized", "bn_to_factorized")]:
        T[op] = _t(op, "modop", None, impl, category="modop")

    # No-op terminals (empty optional slots).
    for noop in ("empty", "no_local_opt", "no_mutation"):
        T[noop] = _t(noop, "noop", None, category="noop")

    return T


TERMINALS: Dict[str, Terminal] = _build_terminals()


# ---------------------------------------------------------------------------
# Production rules
# ---------------------------------------------------------------------------

START = "Pipeline"

RULES: Dict[str, List[List[str]]] = {
    "Pipeline": [["Seeding", "Selection", "ModelBlock", "Replacement",
                  "LocalOptOpt", "MutationOpt", "Stop"]],

    "Seeding": [["RandomInit"]],

    "Selection": [["Truncation"], ["Tournament"], ["Boltzmann"],
                  ["Proportional"], ["Ranking"], ["SUS"]],

    "ModelBlock": [["FacBlock"], ["BNBlock"], ["MarkovNetBlock"],
                   ["IntFDABlock"], ["RegMarkovBlock"], ["MarkovChainBlock"]],

    "FacBlock": [["FacLearner", "FacModMod", "SampleFDA"]],

    "BNBlock": [["BNLearner", "SampleBN"],
                ["BNLearner", "bn_to_factorized", "FacModMod", "SampleFDA"]],

    "MarkovNetBlock": [["LearnMOA", "SampleGibbs"]],
    "IntFDABlock": [["LearnIntFDA", "SampleIntFDA"]],
    "RegMarkovBlock": [["LearnRegularizedMarkov", "SampleRegularizedMarkov"]],
    "MarkovChainBlock": [["LearnMarkovChain", "SampleFDA"],
                         ["LearnMarkovChain", "SampleMarkovChain"]],

    "FacModMod": [["empty"], ["prune_factorized"], ["tree_to_forest"], ["tree_to_malign"]],

    "FacLearner": [[n] for n in _fac_learners().keys()],
    "BNLearner": [[n] for n in _bn_learners().keys()],

    "Replacement": [["Elitist"], ["Generational"]],

    "LocalOptOpt": [["no_local_opt"], ["LocalOpt"]],
    "LocalOpt": [[n] for n in _local_opts().keys()],

    "MutationOpt": [["no_mutation"], ["RandomReset"], ["FreqBalance"]],

    "Stop": [["MaxGenerations"]],
}


# ---------------------------------------------------------------------------
# Derivation sampling
# ---------------------------------------------------------------------------

def sample_derivation(rng: Optional[np.random.Generator] = None,
                      start: str = START, max_expansions: int = 200) -> List[str]:
    """
    Randomly derive a complete pipeline: a flat list of terminal names.

    Expands the grammar depth-first, choosing a uniformly random production for
    every non-terminal, until only terminals remain.

    Args:
        rng: Random generator (``None`` -> fresh).
        start: Start symbol.
        max_expansions: Safety cap on the number of rule expansions.

    Returns:
        The ordered list of terminal names of the derivation.
    """
    rng = rng or np.random.default_rng()
    stack = [start]
    out: List[str] = []
    steps = 0
    while stack:
        steps += 1
        if steps > max_expansions:
            raise RuntimeError("derivation exceeded max_expansions")
        sym = stack.pop(0)
        if sym in RULES:
            prod = RULES[sym][int(rng.integers(0, len(RULES[sym])))]
            stack = list(prod) + stack
        else:
            out.append(sym)                     # terminal
    return out


# ---------------------------------------------------------------------------
# Pipeline specification and builder
# ---------------------------------------------------------------------------

@dataclass
class PipelineSpec:
    """A parsed pipeline: the terminals chosen for each role."""
    seeding: str
    selection: str
    learner: str
    operators: List[str]
    sampler: str
    replacement: str
    local_opt: Optional[str]
    mutation: Optional[str]
    stop: str
    terminals: List[str] = field(default_factory=list)

    def __str__(self):
        ops = (" -> " + " -> ".join(self.operators)) if self.operators else ""
        lo = self.local_opt or "-"
        mu = self.mutation or "-"
        return (f"{self.selection} | {self.learner}{ops} -> {self.sampler} | "
                f"{self.replacement} | LS={lo} | mut={mu}")


def parse_derivation(terminals: List[str]) -> PipelineSpec:
    """Group a flat terminal list into a :class:`PipelineSpec` by role."""
    roles = {"seeding": None, "selection": None, "learner": None,
             "sampler": None, "replacement": None, "local_opt": None,
             "mutation": None, "stop": None}
    operators: List[str] = []
    for name in terminals:
        term = TERMINALS[name]
        if term.role == "modop":
            operators.append(name)
        elif term.role == "noop":
            continue
        elif term.role in roles:
            roles[term.role] = name
    return PipelineSpec(
        seeding=roles["seeding"], selection=roles["selection"],
        learner=roles["learner"], operators=operators, sampler=roles["sampler"],
        replacement=roles["replacement"], local_opt=roles["local_opt"],
        mutation=roles["mutation"], stop=roles["stop"], terminals=list(terminals),
    )


def build_components(spec: PipelineSpec, pop_size: int, n_gen: int) -> EDAComponents:
    """Instantiate :class:`~pateda.core.components.EDAComponents` from a spec."""
    ctx = {"pop_size": pop_size, "n_gen": n_gen}

    base_learner = TERMINALS[spec.learner].build(ctx)
    if spec.operators:
        params = {"prune_factorized": {"K": 2}}
        learning = ModifiedLearning(base_learner, spec.operators, params)
    else:
        learning = base_learner

    return EDAComponents(
        seeding=TERMINALS[spec.seeding].build(ctx),
        selection=TERMINALS[spec.selection].build(ctx),
        learning=learning,
        sampling=TERMINALS[spec.sampler].build(ctx),
        replacement=TERMINALS[spec.replacement].build(ctx),
        local_opt=TERMINALS[spec.local_opt].build(ctx) if spec.local_opt else None,
        mutation=TERMINALS[spec.mutation].build(ctx) if spec.mutation else None,
        stop_condition=TERMINALS[spec.stop].build(ctx),
    )


def build_pipeline(terminals: List[str], n_vars: int, fitness_func: Callable,
                   cardinality: np.ndarray, pop_size: int = 100, n_gen: int = 10,
                   random_seed: Optional[int] = None):
    """
    Build a runnable :class:`~pateda.core.EDA` from a derivation.

    Returns ``(eda, spec)``.  Raises if the components cannot be assembled.
    """
    from pateda import EDA

    spec = parse_derivation(terminals)
    components = build_components(spec, pop_size, n_gen)
    eda = EDA(pop_size=pop_size, n_vars=n_vars, fitness_func=fitness_func,
              cardinality=np.asarray(cardinality), components=components,
              random_seed=random_seed)
    return eda, spec


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def grammar_terminals_by_category() -> Dict[str, List[str]]:
    """Return the implementations the grammar can emit, grouped by category."""
    by_cat: Dict[str, List[str]] = {}
    for term in TERMINALS.values():
        if term.role == "noop":
            continue
        by_cat.setdefault(term.category, []).append(term.impl)
    return {k: sorted(set(v)) for k, v in by_cat.items()}

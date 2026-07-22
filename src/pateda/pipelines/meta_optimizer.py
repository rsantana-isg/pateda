"""
A multi-objective (NSGA-II) meta-optimizer over EDA pipelines

The grammar in :mod:`pateda.pipelines.grammar` generates *feasible* EDA
pipelines; this module searches that space.  Each candidate is a whole EDA
pipeline (a :class:`~pateda.pipelines.grammar.PipelineSpec`), evaluated by
building it and running it on a target problem at a *fixed inner budget*
(population size and number of generations).  A candidate has two conflicting
objectives:

- **objective value** -- the best fitness the pipeline reaches (to *maximize*);
- **running time** -- the wall-clock time it spends (to *minimize*).

The meta-optimizer is a grammar-guided genetic algorithm with NSGA-II
selection, so instead of a single "best" pipeline it returns the **Pareto set**:
the pipelines that are not dominated on the (quality, time) trade-off -- from the
cheapest-but-weaker to the strongest-but-slower.  This mirrors the two-objective
search over pipelines discussed in the AutoML grammar literature (RECIPE uses an
NSGA-II selection over accuracy and pipeline size).

Genetic operators respect the grammar:

- **crossover** swaps whole slots between two parents -- selection, replacement,
  local search, mutation, and the *model block* (learner + model operators +
  sampler) as a unit -- so every child stays type-consistent (a learner is never
  paired with an incompatible sampler);
- **mutation** re-derives one slot from the grammar (e.g. a new model block, or a
  different local searcher).

Evaluations are cached by genotype signature, and infeasible pipelines are
dominated by every feasible one, so they are quickly discarded.

References
----------
- de Sa, A. et al. (2017). "RECIPE: A Grammar-based Framework..." EuroGP 2017.
- Deb, K. et al. (2002). "A fast and elitist multiobjective genetic algorithm:
  NSGA-II." IEEE TEC 6(2).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import numpy as np

from pateda import EDA
from pateda.pipelines.grammar import (
    TERMINALS, sample_derivation, parse_derivation, build_components, PipelineSpec,
)


# ---------------------------------------------------------------------------
# Module-level evaluation worker (picklable -> usable in worker processes)
# ---------------------------------------------------------------------------

def _evaluate_spec_worker(payload):
    """Build and run one pipeline; return ``(quality, runtime, feasible)``.

    Defined at module level so it (and its arguments) can be sent to a worker
    process.  The problem's ``fitness`` must be picklable when parallel
    evaluation (``n_jobs > 1``) is used -- a module-level function or a
    :func:`functools.partial` of one, not a lambda / local closure.
    """
    (spec, fitness, n_vars, cardinality, optimum,
     inner_pop, inner_gen, n_seeds) = payload
    qualities, times = [], []
    try:
        for s in range(n_seeds):
            components = build_components(spec, inner_pop, inner_gen)
            eda = EDA(pop_size=inner_pop, n_vars=n_vars, fitness_func=fitness,
                      cardinality=cardinality, components=components,
                      random_seed=1000 + s)
            t0 = time.perf_counter()
            stats, _ = eda.run(verbose=False)
            dt = time.perf_counter() - t0
            best = float(stats.best_fitness_overall)
            qualities.append(best / optimum if optimum else best)
            times.append(dt)
        return float(np.mean(qualities)), float(np.mean(times)), True
    except Exception:
        return float("-inf"), float("inf"), False


def _worker_put(idx, payload, q):
    """Worker entry: evaluate one pipeline and put ``(idx, result)`` on ``q``."""
    q.put((idx, _evaluate_spec_worker(payload)))


def _parallel_evaluate(payloads, n_jobs, timeout):
    """
    Evaluate many pipelines across up to ``n_jobs`` CPUs (one pipeline per CPU).

    Uses one process per pipeline with *bounded concurrency* and a *per-task
    timeout*: a pipeline that exceeds ``timeout`` seconds is terminated and
    reported as infeasible (it is too slow to be useful, and would otherwise
    keep a whole generation waiting on a single CPU).  This keeps all workers
    busy and prevents a slow / hanging pipeline from stalling the search.

    Returns a list of ``(quality, runtime, feasible)`` aligned with ``payloads``.
    """
    import multiprocessing as mp
    import queue as _queue

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    n = len(payloads)
    results: Dict[int, Tuple[float, float, bool]] = {}
    running: Dict[int, Tuple[Any, float]] = {}
    next_idx = 0

    while len(results) < n:
        # Launch new tasks up to the concurrency limit.
        while len(running) < n_jobs and next_idx < n:
            proc = ctx.Process(target=_worker_put, args=(next_idx, payloads[next_idx], q))
            proc.daemon = True
            proc.start()
            running[next_idx] = (proc, time.time())
            next_idx += 1
        # Collect any finished results.
        try:
            idx, res = q.get(timeout=0.2)
            if idx not in results:
                results[idx] = res
                proc, _ = running.pop(idx, (None, None))
                if proc is not None:
                    proc.join(0.1)
        except _queue.Empty:
            pass
        # Terminate tasks that overran the per-task timeout.
        if timeout:
            now = time.time()
            for idx, (proc, start) in list(running.items()):
                if now - start > timeout and idx not in results:
                    proc.terminate()
                    proc.join(0.5)
                    results[idx] = (float("-inf"), float(timeout), False)
                    running.pop(idx, None)
    return [results[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------

@dataclass
class MetaProblem:
    """The target problem the meta-optimizer tunes pipelines for."""
    fitness: Callable[[np.ndarray], float]
    n_vars: int
    cardinality: np.ndarray
    optimum: Optional[float] = None            # for quality normalization to [0,1]
    name: str = "problem"


# ---------------------------------------------------------------------------
# A pipeline individual (genotype + evaluation)
# ---------------------------------------------------------------------------

@dataclass
class PipelineIndividual:
    spec: PipelineSpec
    quality: float = -np.inf                   # objective value (maximize)
    runtime: float = np.inf                    # seconds (minimize)
    feasible: bool = False
    rank: int = 0
    crowd: float = 0.0

    def signature(self) -> Tuple:
        s = self.spec
        return (s.selection, s.learner, tuple(s.operators), s.sampler,
                s.replacement, s.local_opt, s.mutation)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MetaResult:
    pareto_front: List[PipelineIndividual]     # non-dominated pipelines
    evaluated: List[PipelineIndividual]        # every distinct pipeline evaluated
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def best_quality(self) -> PipelineIndividual:
        return max(self.pareto_front, key=lambda ind: (ind.quality, -ind.runtime))

    @property
    def fastest(self) -> PipelineIndividual:
        return min(self.pareto_front, key=lambda ind: (ind.runtime, -ind.quality))


# ---------------------------------------------------------------------------
# Meta-optimizer
# ---------------------------------------------------------------------------

class PipelineMetaOptimizer:
    """
    NSGA-II search over EDA pipelines (maximize quality, minimize time).
    """

    def __init__(
        self,
        problem: MetaProblem,
        inner_pop: int = 100,
        inner_gen: int = 15,
        meta_pop: int = 20,
        meta_gens: int = 8,
        n_eval_seeds: int = 1,
        tournament_size: int = 2,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.4,
        n_jobs: int = 1,
        eval_timeout: Optional[float] = 30.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            problem: The :class:`MetaProblem` to tune pipelines for.
            inner_pop, inner_gen: The *fixed inner budget* each pipeline runs at.
            meta_pop, meta_gens: Meta-GA population size and number of generations.
            n_eval_seeds: Inner runs averaged per pipeline evaluation.
            tournament_size: Meta-GA tournament size (NSGA-II binary tournament
                by default).
            crossover_prob, mutation_prob: Meta-GA operator probabilities.
            n_jobs: Number of worker processes for *parallel* pipeline evaluation
                (each pipeline runs on its own CPU).  ``1`` evaluates
                sequentially.  With ``n_jobs > 1`` the problem's ``fitness`` must
                be picklable (a module-level function or a ``functools.partial``,
                not a lambda).
            eval_timeout: Per-pipeline wall-time cap in seconds (parallel mode);
                a pipeline exceeding it is terminated and marked infeasible, so a
                slow / hanging pipeline never stalls a whole generation on one
                CPU.  ``None`` disables the cap.
            seed: Random seed for the meta search.
        """
        self.problem = problem
        self.inner_pop = inner_pop
        self.inner_gen = inner_gen
        self.meta_pop = meta_pop
        self.meta_gens = meta_gens
        self.n_eval_seeds = n_eval_seeds
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_jobs = max(1, int(n_jobs))
        self.eval_timeout = eval_timeout
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[Tuple, Tuple[float, float, bool]] = {}
        self._all: Dict[Tuple, PipelineIndividual] = {}   # every distinct pipeline

    # -- genotype construction / variation ------------------------------
    def _random_spec(self) -> PipelineSpec:
        return parse_derivation(sample_derivation(self.rng))

    def _random_model_block(self):
        terms = sample_derivation(self.rng, start="ModelBlock")
        learner, sampler, ops = None, None, []
        for t in terms:
            role = TERMINALS[t].role
            if role == "learner":
                learner = t
            elif role == "sampler":
                sampler = t
            elif role == "modop":
                ops.append(t)
        return learner, ops, sampler

    def _random_slot(self, nonterminal: str, role: str) -> str:
        for t in sample_derivation(self.rng, start=nonterminal):
            if TERMINALS[t].role == role:
                return t
        return None                            # e.g. no_local_opt / no_mutation

    def _mutate(self, spec: PipelineSpec) -> PipelineSpec:
        slot = self.rng.choice(["selection", "model", "replacement",
                                "local_opt", "mutation"])
        if slot == "model":
            learner, ops, sampler = self._random_model_block()
            return replace(spec, learner=learner, operators=ops, sampler=sampler)
        if slot == "selection":
            return replace(spec, selection=self._random_slot("Selection", "selection"))
        if slot == "replacement":
            return replace(spec, replacement=self._random_slot("Replacement", "replacement"))
        if slot == "local_opt":
            return replace(spec, local_opt=self._random_slot("LocalOptOpt", "local_opt"))
        return replace(spec, mutation=self._random_slot("MutationOpt", "mutation"))

    def _crossover(self, a: PipelineSpec, b: PipelineSpec) -> PipelineSpec:
        pick = lambda x, y: x if self.rng.random() < 0.5 else y
        take_model_from_a = self.rng.random() < 0.5
        model_src = a if take_model_from_a else b
        return PipelineSpec(
            seeding=a.seeding,
            selection=pick(a.selection, b.selection),
            learner=model_src.learner,
            operators=list(model_src.operators),
            sampler=model_src.sampler,
            replacement=pick(a.replacement, b.replacement),
            local_opt=pick(a.local_opt, b.local_opt),
            mutation=pick(a.mutation, b.mutation),
            stop=a.stop,
        )

    # -- evaluation -----------------------------------------------------
    def _payload(self, spec: PipelineSpec):
        p = self.problem
        return (spec, p.fitness, p.n_vars, p.cardinality, p.optimum,
                self.inner_pop, self.inner_gen, self.n_eval_seeds)

    def _evaluate(self, ind: PipelineIndividual) -> None:
        """Evaluate a single pipeline (sequential; used for tests / n_jobs=1)."""
        self._evaluate_many([ind])

    def _evaluate_many(self, inds: List[PipelineIndividual]) -> None:
        """Evaluate a batch of pipelines, reusing the cache and running the
        *uncached* ones in parallel (when ``n_jobs > 1``)."""
        todo: List[PipelineIndividual] = []
        for ind in inds:
            sig = ind.signature()
            if sig in self._cache:
                ind.quality, ind.runtime, ind.feasible = self._cache[sig]
            else:
                self._all.setdefault(sig, ind)   # register this distinct pipeline
                todo.append(ind)
        if not todo:
            return
        payloads = [self._payload(ind.spec) for ind in todo]

        if self.n_jobs > 1 and len(todo) > 1:
            results = _parallel_evaluate(payloads, self.n_jobs, self.eval_timeout)
        else:
            results = [_evaluate_spec_worker(pl) for pl in payloads]

        for ind, (q, t, feas) in zip(todo, results):
            ind.quality, ind.runtime, ind.feasible = q, t, feas
            self._cache[ind.signature()] = (q, t, feas)

    # -- NSGA-II machinery ---------------------------------------------
    @staticmethod
    def _dominates(a: PipelineIndividual, b: PipelineIndividual) -> bool:
        """``a`` dominates ``b``: no worse on both objectives, better on one.
        Objectives: quality (max), runtime (min)."""
        better_q = a.quality >= b.quality
        better_t = a.runtime <= b.runtime
        strict = a.quality > b.quality or a.runtime < b.runtime
        return better_q and better_t and strict

    def _non_dominated_sort(self, pop: List[PipelineIndividual]) -> List[List[int]]:
        n = len(pop)
        dominated: List[List[int]] = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts: List[List[int]] = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(pop[i], pop[j]):
                    dominated[i].append(j)
                elif self._dominates(pop[j], pop[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)
        f = 0
        while fronts[f]:
            nxt = []
            for i in fronts[f]:
                for j in dominated[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        pop[j].rank = f + 1
                        nxt.append(j)
            f += 1
            fronts.append(nxt)
        return [fr for fr in fronts if fr]

    @staticmethod
    def _crowding(pop: List[PipelineIndividual], front: List[int]) -> None:
        if not front:
            return
        for i in front:
            pop[i].crowd = 0.0
        for obj, sign in (("quality", 1.0), ("runtime", 1.0)):
            vals = np.array([getattr(pop[i], obj) for i in front], dtype=float)
            finite = np.isfinite(vals)
            order = [front[k] for k in np.argsort(vals)]
            pop[order[0]].crowd = np.inf
            pop[order[-1]].crowd = np.inf
            vmin, vmax = vals[finite].min() if finite.any() else 0.0, \
                vals[finite].max() if finite.any() else 1.0
            span = (vmax - vmin) or 1.0
            for k in range(1, len(order) - 1):
                prev_v = getattr(pop[order[k - 1]], obj)
                next_v = getattr(pop[order[k + 1]], obj)
                if np.isfinite(prev_v) and np.isfinite(next_v):
                    pop[order[k]].crowd += abs(next_v - prev_v) / span

    def _tournament(self, pop: List[PipelineIndividual]) -> PipelineIndividual:
        idx = self.rng.integers(0, len(pop), size=self.tournament_size)
        best = pop[idx[0]]
        for k in idx[1:]:
            c = pop[k]
            if (c.rank < best.rank) or (c.rank == best.rank and c.crowd > best.crowd):
                best = c
        return best

    # -- Pareto over all evaluated pipelines ---------------------------
    def _current_pareto(self) -> List[PipelineIndividual]:
        """Feasible non-dominated pipelines over *every* distinct pipeline
        evaluated so far, sorted by increasing runtime."""
        distinct = list(self._all.values())
        fronts = self._non_dominated_sort(distinct)
        for fr in fronts:
            self._crowding(distinct, fr)
        pareto = [distinct[i] for i in (fronts[0] if fronts else [])
                  if distinct[i].feasible]
        pareto.sort(key=lambda ind: ind.runtime)
        return pareto

    def _progress(self, gen, t_start) -> Dict[str, Any]:
        pareto = self._current_pareto()
        feasible = [i for i in self._all.values() if i.feasible]
        best = max((i.quality for i in feasible), default=float("nan"))
        return {
            "generation": gen,
            "elapsed": time.time() - t_start,
            "n_evaluated": len(self._all),
            "n_feasible": len(feasible),
            "best_objective": best,
            "pareto_size": len(pareto),
            "pareto_front": pareto,
        }

    # -- main loop ------------------------------------------------------
    def optimize(self, verbose: bool = True, callback: Optional[Callable] = None) -> MetaResult:
        """
        Run the meta-search.

        Args:
            verbose: Print a progress line after each meta-generation.
            callback: Optional ``callback(stats)`` invoked after every generation
                (and the initial one, ``generation = -1``).  ``stats`` is a dict
                with ``generation``, ``elapsed``, ``n_evaluated``, ``n_feasible``,
                ``best_objective``, ``pareto_size`` and ``pareto_front`` (the list
                of non-dominated :class:`PipelineIndividual` s over everything
                evaluated so far) -- handy for live logging or checkpointing.
        """
        t_start = time.time()
        if verbose and self.n_jobs > 1:
            print(f"  evaluating pipelines on up to {self.n_jobs} CPUs in parallel "
                  f"(per-pipeline timeout {self.eval_timeout}s)", flush=True)
        population = [PipelineIndividual(self._random_spec()) for _ in range(self.meta_pop)]
        self._evaluate_many(population)
        fronts = self._non_dominated_sort(population)
        for fr in fronts:
            self._crowding(population, fr)

        history = []
        init_stats = self._progress(-1, t_start)
        if verbose:
            self._print_progress(init_stats, "init")
        if callback:
            callback(init_stats)

        for gen in range(self.meta_gens):
            # Offspring via grammar-aware crossover + mutation.
            offspring = []
            while len(offspring) < self.meta_pop:
                pa, pb = self._tournament(population), self._tournament(population)
                child_spec = (self._crossover(pa.spec, pb.spec)
                              if self.rng.random() < self.crossover_prob
                              else replace(pa.spec, operators=list(pa.spec.operators)))
                if self.rng.random() < self.mutation_prob:
                    child_spec = self._mutate(child_spec)
                offspring.append(PipelineIndividual(child_spec))
            # Evaluate the whole offspring batch (in parallel when enabled).
            self._evaluate_many(offspring)

            # (mu + lambda) NSGA-II survival.
            combined = population + offspring
            fronts = self._non_dominated_sort(combined)
            new_pop: List[PipelineIndividual] = []
            for fr in fronts:
                self._crowding(combined, fr)
                if len(new_pop) + len(fr) <= self.meta_pop:
                    new_pop.extend(combined[i] for i in fr)
                else:
                    remaining = self.meta_pop - len(new_pop)
                    fr_sorted = sorted(fr, key=lambda i: combined[i].crowd, reverse=True)
                    new_pop.extend(combined[i] for i in fr_sorted[:remaining])
                    break
            population = new_pop
            fronts = self._non_dominated_sort(population)
            for fr in fronts:
                self._crowding(population, fr)

            # Per-generation progress over everything evaluated so far.
            stats = self._progress(gen, t_start)
            history.append({k: v for k, v in stats.items() if k != "pareto_front"})
            if verbose:
                self._print_progress(stats, f"gen {gen + 1}/{self.meta_gens}")
            if callback:
                callback(stats)

        # Final Pareto front over EVERY distinct pipeline evaluated (feasible ones).
        pareto = self._current_pareto()
        return MetaResult(pareto_front=pareto,
                          evaluated=list(self._all.values()), history=history)

    @staticmethod
    def _print_progress(stats: Dict[str, Any], tag: str) -> None:
        best = stats["best_objective"]
        best_s = f"{best:.3f}" if np.isfinite(best) else "n/a"
        pf = stats["pareto_front"]
        obj_lo = min((i.quality for i in pf), default=float("nan"))
        obj_hi = max((i.quality for i in pf), default=float("nan"))
        t_lo = min((i.runtime for i in pf), default=float("nan"))
        t_hi = max((i.runtime for i in pf), default=float("nan"))
        print(f"  [{tag:>12}] best obj={best_s}  Pareto={stats['pareto_size']:2d} "
              f"(obj {obj_lo:.1f}..{obj_hi:.1f}, time {t_lo:.1f}..{t_hi:.1f}s)  "
              f"feasible={stats['n_feasible']}/{stats['n_evaluated']}  "
              f"elapsed={stats['elapsed']:.0f}s", flush=True)

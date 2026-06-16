# perm_pateda — Roadmap

This roadmap organizes the development of `perm_pateda` into phases whose target
is **feature parity with the `perm_mateda` toolbox** described in:

> E. Irurozki, J. Ceberio, J. Santamaria, A. Mendiburu (2018). *Algorithm 989:
> perm_mateda — A Matlab Toolbox of EDAs for Permutation-based Combinatorial
> Optimization Problems.* ACM TOMS 44(4), Article 47.

`perm_mateda` provides, on top of the MATEDA-2.0 EDA engine:

* **Two distance-based probability models** — the **Mallows** model (MEDA, a
  single spread parameter θ) and the **Generalized Mallows** model (GMEDA, an
  (n−1)-dimensional spread vector θ).
* **Three distance-metrics** — **Kendall's-τ**, **Cayley**, and **Ulam** — giving
  **five EDAs** (GM is not defined for Ulam): MEDA-Kendall, MEDA-Cayley,
  MEDA-Ulam, GMEDA-Kendall, GMEDA-Cayley.
* **Central-permutation (consensus) estimators** — `Borda`, `SetMedianPermutation`,
  `BestPermutation` — followed by MLE of the spread parameter(s).
* **Multistage / Distances samplers** matched to each distance.
* **Four permutation problems** — TSP, **PFSP** (Permutation Flowshop Scheduling),
  LOP, and QAP.

The phases below add what is still missing relative to that scope, and then
extend it with real-instance loaders, an experimental harness, and docs.

---

## Phase 0 — Baseline (✅ done, ported from the original `pateda`)

Already available in `perm_pateda`:

* Distances: `kendall_distance`, `cayley_distance`, `ulam_distance`
  (+ inversion/cycle decomposition vectors).
* Consensus: `find_consensus_borda` (Borda), `find_consensus_median`
  (SetMedianPermutation).
* Models / EDAs:
  * MEDA-Kendall (`MallowsKendallEDA`), MEDA-Cayley (`MallowsCayleyEDA`),
    GMEDA-Kendall (`GMallowsKendallEDA`), GMEDA-Cayley (`GMallowsCayleyEDA`).
  * Edge / Node Histogram Models (`EHMEDA`, `NHMEDA`) — an extra beyond the paper.
* Problems: TSP, QAP, LOP.
* Seeding: `PermutationInit`.
* Plug-and-play wrappers + test suite for the Mallows/GMallows models.

---

## Phase 1 — Complete the five paper EDAs: **MEDA-Ulam**

Goal: add the Mallows model under the Ulam distance, the one paper EDA still
missing (GM-Ulam is intentionally excluded — Ulam has no distance-decomposition
vector, see §3.3 of the paper).

* `learning/mallows.py`: `LearnMallowsUlam`
  * central permutation σ₀ via the existing consensus estimators;
  * MLE of the global spread θ under the Ulam distance, using the number of
    permutations at each Ulam distance (Ferrers/RSK shape counts).
* `sampling/mallows.py`: `SampleMallowsUlam` — the **Distances sampler**
  (sample a target distance from the Ulam-Mallows distribution, then a random
  permutation at that distance via random Standard Young Tableaux / RSK).
* `algorithms/permutation.py`: `MallowsUlamEDA` wrapper; export it.
* Tests: distance/ξ-shape counts, learn→sample round-trip, convergence on LOP/TSP.

**Deliverable:** all five paper EDAs (Kendall/Cayley/Ulam × Mallows, Kendall/Cayley
× GMallows) available as wrappers.

---

## Phase 2 — Consensus strategies: **BestPermutation** + selectable σ₀

Goal: expose all three central-permutation estimators from the paper and let the
user choose per algorithm.

* `consensus.py`: `find_consensus_best` (BestPermutation — the sampled
  permutation with the best objective value).
* Give every Mallows/GMallows learner a `consensus="borda"|"setmedian"|"best"`
  parameter (default per distance, following the paper's recommendation that
  Borda is for Kendall's-τ), threading the objective/fitness through when
  `best` is requested.
* Tests comparing the three estimators on the same selected set.

---

## Phase 3 — Fourth problem: **PFSP**

Goal: add the Permutation Flowshop Scheduling Problem to match the paper's four
problems (TSP, PFSP, LOP, QAP).

* `functions/pfsp.py`: `PFSP` (m machines × n jobs processing-time matrix,
  makespan / total-flowtime objective), plus `create_random_pfsp` and an
  instance reader.
* Register it in `functions/__init__.py`.
* Tests on small instances with known makespan.

---

## Phase 4 — Real-instance loaders & experimental harness

Goal: reproduce the experimental study of §5 of the paper.

* Instance loaders for the standard benchmarks used in the paper:
  Taillard (PFSP), QAPLIB (QAP), LOLIB (LOP), TSPLIB (TSP).
* A `scripts/compare_permutation_edas.py`-style harness running the five MEDA/GMEDA
  variants on each problem, with the distance↔problem correspondence suggested by
  Ceberio et al. (2015).
* Statistical comparison (best/mean/std over seeds) and figures, consistent with
  the wider `pateda` experimental conventions.

---

## Phase 5 — Robustness, docs, and validation

* Numerical hardening of the histogram samplers (guard the
  `invalid value in divide` edge case when a position/edge has zero mass).
* API docs (Sphinx) and worked examples for each model × distance × problem.
* Validation of learning/sampling against reference outputs of the MATLAB
  `perm_mateda` toolbox (parameter estimates and sampled-distance histograms).
* Optional: local-search / repair hooks and additional seeding heuristics, as the
  paper notes these modules can be customized for real-world problems.

---

## Summary: paper coverage

| Component | Paper | perm_pateda status |
|---|---|---|
| MEDA-Kendall   | ✓ | ✅ Phase 0 |
| MEDA-Cayley    | ✓ | ✅ Phase 0 |
| MEDA-Ulam      | ✓ | ⏳ Phase 1 |
| GMEDA-Kendall  | ✓ | ✅ Phase 0 |
| GMEDA-Cayley   | ✓ | ✅ Phase 0 |
| Borda consensus        | ✓ | ✅ Phase 0 |
| SetMedian consensus    | ✓ | ✅ Phase 0 |
| BestPermutation consensus | ✓ | ⏳ Phase 2 |
| TSP | ✓ | ✅ Phase 0 |
| LOP | ✓ | ✅ Phase 0 |
| QAP | ✓ | ✅ Phase 0 |
| PFSP | ✓ | ⏳ Phase 3 |
| EHM / NHM histogram EDAs | — | ✅ extra |

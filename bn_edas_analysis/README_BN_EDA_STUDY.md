# BN-EDA comparison study — do model-quality metrics predict EDA performance?

This directory adds a second experiment on top of the offline BN-learning
benchmark (`eda_cluster_results.csv`). There, 19 `bayes_nets` structure-learning
algorithms were scored **offline** on fixed training sets by how well the learned
search distribution matches the target (correlation `sp`, log-likelihood `ll`, KL
`kl`, skeleton F1). Here we ask the follow-up question:

> When each of these BN learners is actually used **inside an EDA**, do those
> search-distribution quality metrics predict the EDA's real optimization
> performance? I.e. are correlation / likelihood / KL good **surrogates** of EDA
> success?

To answer it we run one EDA per BN learning algorithm on each problem, save the
full per-generation behaviour (model-quality metrics *and* fitness progress), and
correlate the two.

## Common EDA (everything shared except the BN learner)

| component | setting |
|-----------|---------|
| seeding | `RandomInit` |
| selection | `TruncationSelection(ratio=0.5)` — **T = 0.5** |
| learning weights | `selection_weighting="boltzmann"` — **weighted probabilities** |
| learning | `LearnEBNA(score_metric=ALGORITHM, max_parents=5, alpha=1.0)` — **the only variable** |
| sampling | `SampleBayesianNetwork` |
| replacement | `ElitistReplacement(n_elite=1)` |
| stop | `MaxGenerations(100)` — **100 generations** |
| population | `pop_size = 10 · n` (selected set `5 · n`) |

The BN learner is exactly the same code path exercised by the offline benchmark
(`bayes_nets.BayesianNetwork.fit(method=ALGORITHM, sample_weights=…)`), so the two
experiments are directly comparable.

## Algorithm selection (feasibility)

Learning is done **every generation** (100×), on a selected set of `5·n`
solutions rather than the benchmark's fixed 800. We therefore estimate the full
EDA cost per `(algorithm, problem)` from the benchmark's own measured learning
time:

```
est_hours = 100 · max_time(alg, problem) · (5·n / 800) / 3600
```

(the **worst** measured time over temperatures/splits — a single slow generation
stalls the whole run). Rather than pick a fixed subset, `launch_bn_eda.py`
includes **all 19 algorithms** and launches each `(algorithm, problem)` pair only
when `est_hours ≤ 40 h` (margin under the 48 h wall-time); pairs with no benchmark
time are skipped.

Because the population is `10·n` (selected `5·n`, ~6× fewer samples than the
benchmark's 800 on the smaller problems), learning is cheap and coverage is
broad. At this size the feasible experiment list is:

| coverage | algorithms |
|----------|-----------|
| all 33 problems | `univ_bn`, `k2`, `k2_mi`, `k2_mb`, `k2_ensemble`, `fi_k2`, `rfe_k2`, `bic`, `sartre`, `binotears`, `bounded_tw` (11) |
| 32 | `aic` (only `Ising_256` dropped) |
| 28 | `k2_plus`, `pc`, `stable_pc` |
| 27 | `k2_refine` |
| 26 | `stable_hc`, `dt`, `dmbbn` |

The 43 skipped pairs are almost all the seven largest (`n = 256`) instances for
the heaviest methods (`stable_hc`, `dt`, `dmbbn`, `pc`, `stable_pc`, `k2_refine`,
`k2_plus`) — e.g. `dt` still runs everywhere except `n = 256`, including the small
MaxClique instances it wins offline.

This yields **584 feasible (algorithm, problem) pairs × 5 seeds = 2920 jobs**
(vs. 520 / 2600 at `pop_size = 50·n`). The list is computed on the fly, so it
always matches the current `POP_MULT` / `MAX_EST_HOURS`: inspect the exact
per-algorithm counts and skip list with `python3 slurm/launch_bn_eda.py --report`,
and adjust the cut via the `BN_EDA_MAX_EST_HOURS` env var.

## Pipeline

```bash
# 1. run the grid on the cluster (idempotent; safe to re-launch)
python3 slurm/launch_bn_eda.py | head -400 | bash      # <= 400 jobs in flight
python3 slurm/launch_bn_eda.py | sed -n '401,800p' | bash
# ... -> results/bn_eda_cluster/bneda_<problem>_<algorithm>_s<seed>.json

# 2. aggregate the per-run JSONs into tidy CSVs
python3 scripts/extract_bn_eda.py
#   -> results/bn_eda_summary.csv        (one row per run)
#   -> results/bn_eda_trajectory.csv     (one row per run × generation)

# 3. surrogate-validity analysis (tables + figures)
python3 scripts/compare_bn_eda.py
#   -> results/bn_eda_analysis/tables/table_surrogate_validity.tex
#      results/bn_eda_analysis/tables/table_algorithm_summary.tex
#      results/bn_eda_analysis/figures/{fig_surrogate_rankcorr,
#                                       fig_surrogate_scatter,fig_perf_vs_cost}.pdf
```

A single run can be launched directly for testing:

```bash
python3 scripts/run_bn_eda.py 1 Ising_100 bic        # SEED PROBLEM ALGORITHM
```

## What is saved (per run)

`run_bn_eda.py` writes a self-describing JSON with the run config, a run-level
`summary`, and a per-generation `trajectory`. Everything needed to characterise
*why* an algorithm behaves as it does is stored **each generation**:

* **performance** — `best_fitness`, `mean_fitness`, `std_fitness`, `best`-so-far,
  `generation_found`, `auc_best`;
* **model quality (the surrogate metrics)** — Spearman correlation between the BN
  log-probability and the objective, on the selected set (`sp_sel`) and on the
  full population (`sp_pop`); mean log-likelihood (`ll_sel`/`ll_pop`); KL
  divergence (`kl_sel`/`kl_pop`); skeleton F1 vs the true interaction graph;
* **model structure & cost** — number of `edges`, per-generation `learn_time`,
  `n_selected`;
* **search state** — population diversity (mean per-variable normalised entropy).

## The question the analysis answers

`compare_bn_eda.py` ranks the algorithms **within each problem** by real EDA
`best_fitness` and by each surrogate metric, then reports the macro-averaged
Spearman rank-correlation between the two (and a top-1 "hit rate"). A high
correlation for, say, `sp_pop` would mean the cheap offline correlation metric is
a good surrogate for EDA performance; a low one would mean model-fit quality does
**not** translate into optimization success — the central hypothesis of the study.
`fig_perf_vs_cost.pdf` additionally places every algorithm on the
performance-vs-learning-cost plane (the efficiency trade-off).

## Files

| file | role |
|------|------|
| `scripts/run_bn_eda.py` | one EDA run; saves per-generation behaviour |
| `scripts/extract_bn_eda.py` | JSONs → `bn_eda_summary.csv` + `bn_eda_trajectory.csv` |
| `scripts/compare_bn_eda.py` | surrogate-validity tables + figures |
| `slurm/slurm_bn_eda.sh` | SLURM batch script (one job) |
| `slurm/launch_bn_eda.py` | feasibility-filtered sbatch generator |

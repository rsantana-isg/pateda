# EDA cluster benchmark — analysis of BN learning algorithms

Analysis of the SLURM grid launched by `slurm/launch_eval_eda_benchmark.py`
(runner: `scripts/gen_eval_eda_benchmark.py`), whose finished result files live
in `eda_eval_cluster/`. Each `.dat` file is one
`(problem, train_set, algorithm, seed, temperature)` run holding the 11 values
`f1 time [ll kl sp]×3 splits`, where **`sp` = Spearman correlation** between the
BN-predicted probability and the target Boltzmann probability of that split.

## Pipeline (two scripts)

```bash
# 1. raw result files  ->  tidy CSV  (results/eda_cluster_results.csv)
python3 scripts/extract_eda_cluster.py [results_dir] [out_csv]

# 2. CSV  ->  LaTeX tables + figures  (results/eda_cluster_analysis/)
python3 scripts/compare_eda_cluster.py [results_csv] [out_dir]
```

The current run covers **19 algorithms × 33 problems × 3 train-splits × 3
temperatures** (5624 finished runs; a handful of the largest jobs did not
finish and are simply absent).

## Priority metric and train/test design

* **Correlation is the priority metric.** Algorithms are ranked by their
  **average rank on the per-problem test-split Spearman `sp`** (blocked by
  problem — Demšar 2006). Pooled means are *not* used for ranking because the
  per-problem correlation ranges from ~0 (Braid, EqualProducts) to ~1 (OneMax),
  which swamps algorithm differences (a naive Kruskal–Wallis is non-significant,
  whereas the blocked **Friedman test is `p ≈ 5e-23`**).
* **The three splits are different selection-pressure regimes** (the objective
  distribution shifts from split 0 to split 2). For a run trained on split `t`:
  * **train** = correlation on split `t`;
  * **test**  = mean correlation on the two held-out splits (the "two test
    cases", whose probability ranking was never used for learning).
* All displayed `ρ` values are **macro-averaged** (per-problem mean, then mean
  over problems) so the same number is identical across every table/figure.

## Outputs

`tables/` (LaTeX, `booktabs`+`amsmath`; best value bold per column)

| file | content |
|------|---------|
| `table_main_ranking.tex` | algorithms ranked by average rank on test `ρ`; test/train `ρ`, F1, time |
| `table_train_vs_test.tex` | train `ρ`, test `ρ`, generalisation gap |
| `table_generalisation_matrix.tex` | 3×3 train-split × eval-split mean `ρ` |
| `table_by_group.tex` | test `ρ` by size group (Small/Medium/Large) |
| `table_by_family.tex` | test `ρ` per problem family |
| `table_by_temperature.tex` | test `ρ` per Boltzmann temperature |
| `table_best_per_problem.tex` | **best algorithm on each of the 33 problems** (winner, its `ρ` and F1, runner-up) |
| `table_best_per_family.tex` | best algorithm per problem family + instances won |
| `stats_summary.txt` | Friedman + Nemenyi critical-difference summary |

`figures/` (PDF, no titles — captions belong in the LaTeX document)

| file | content |
|------|---------|
| `fig_ranking_bar.pdf` | train vs test `ρ` per algorithm, ordered by rank |
| `fig_avg_rank_cd.pdf` | average-rank plot with Nemenyi critical difference |
| `fig_nemenyi_heatmap.pdf` | pairwise Nemenyi p-values (top-10) |
| `fig_train_vs_test.pdf` | train vs test `ρ` scatter (generalisation gap) |
| `fig_generalisation_matrix.pdf` | 3×3 regime-transfer heatmap |
| `fig_by_group.pdf` | test `ρ` per algorithm across size groups |
| `fig_family_heatmap.pdf` | algorithm × family test-`ρ` heatmap |
| `fig_by_temperature.pdf` | test `ρ` vs temperature (top-8) |
| `fig_corr_boxplot.pdf` | distribution of test `ρ` per algorithm |
| `fig_f1_vs_corr.pdf` | skeleton F1 vs test `ρ` (does structure predict ranking?) |
| `fig_best_per_problem.pdf` | **per-problem best `ρ`, bar coloured/labelled by the winning algorithm** |
| `fig_win_counts.pdf` | number of problems each algorithm wins |

## Headline findings

* **Best by test correlation (average rank):** `BIC-HC` (rank 5.4), then
  `SARTRE`, `HC-Stable`, `AIC-HC` — all within the Nemenyi critical difference
  (CD = 5.2) of the best. The Friedman omnibus is highly significant.
* **The K2 family generalises worst by this metric.** Plain `K2` and the
  objective-guided orderings (`FI-K2`, `RFE-K2`, `K2-MI`) reach *high train*
  `ρ` (~0.63) but low *test* `ρ` (~0.43) and the poorest average ranks — they
  over-fit the training regime.
* **Regime transfer is the dominant difficulty.** In the 3×3 matrix the diagonal
  (train) `ρ` is ~0.5–0.62 while the far off-diagonal (train split 0 → eval
  split 2) collapses to ~0.11: predicting a distant selection-pressure regime is
  much harder than an adjacent one.
* **`SARTRE` is the best value** — top-tier correlation *and* the best skeleton
  F1 (0.20) at low cost (~5 s), whereas `HC-Stable`, `DT` and `DMBBN` are far
  slower (hundreds–thousands of seconds).
* **Correlation is largely temperature-insensitive** for the leading methods; a
  few families (Braid, EqualProducts) are essentially unlearnable in ranking
  terms (`ρ ≈ 0` for every algorithm), while OneMax/Trap/Deceptive are easy.

## Best algorithm per problem (per-problem winners)

By per-problem test `ρ` (see `table_best_per_problem.tex`,
`table_best_per_family.tex`, `fig_best_per_problem.pdf`, `fig_win_counts.pdf`):

* **Wins per algorithm (of 33):** `Univ` 10, `BIC-HC` 5, `SARTRE` 5, `HC-Stable` 3,
  `DT` 3, `AIC-HC` 2, `K2-Ref` 2, `PC`/`BdTW`/`DMBBN` 1 each.
* **Winners by family:** OneMax & Deceptive3 → `Univ` (separable / additive, so an
  independent model already reproduces the ranking; OneMax `ρ ≈ 0.85`);
  Checkerboard → `AIC-HC`; Ising & UBQP → `BIC-HC`; MaxClique → `DT`; Trap → `BINO`.
* **Caveat — read winners with the `ρ` column.** For the unlearnable families
  (`Braid`, `EqualProducts`) every algorithm sits at `ρ ≈ 0`, so the per-problem
  "winner" there is a near-tie / noise (the runner-up `ρ` in the table is almost
  identical). The winners are meaningful only where the best `ρ` is clearly
  above zero.

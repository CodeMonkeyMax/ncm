# Phase 1 Candidate Sampling Report

**Status:** P1.1-P1.4 are complete for the initial held-out sweep on the RTX 3080. No Phase 2 selector or semantic API evaluation was run.

## Summary and decision

The sampler produced one nested K=64 candidate pool for each of the first 200 official test examples in XSum, WMT14 and Sudoku easy. Each pool contains 12,800 records, for 38,400 total candidates. All three runs completed within the recorded 2.5 GPU-hour budget: 83.0 minutes of end-to-end wall time, including 82.6 minutes of synchronized generation. One XSum candidate returned an empty generation; the sampler retained and recorded it rather than silently resampling.

Reference-scored oracle gains are large for XSum and WMT14, but they describe candidate availability, not a practical selection result. The stochastic first candidate is substantially worse than the greedy baseline on WMT14 and Sudoku. Sudoku reaches 100% reference-oracle success at K=64, but its pool has a 91.8% duplicate fraction and gains only two examples over greedy. An independently approved Phase 2 selector experiment could be justified for XSum first, with WMT14 treated cautiously; Sudoku is a low-priority candidate-scaling target. No Phase 2 work is authorized by this report.

## Frozen protocol and provenance

- Sampling configurations were selected using only the separate development data documented in the [Phase 0/1 plan](../implementation-phases-0-1.md), then frozen before held-out stochastic generation.
- Evaluation IDs are `0-199` from each task's pinned test split. The P0 greedy outputs and scores on these same prefixes were already available; only the stochastic configs were held back, selected on development data, and frozen before these pools were generated. The archived greedy outputs are paired baselines, and stochastic references are read after each generation run completes.
- Seed schedule: base seed `20261112`, replicate `0`, candidate seed derived from `(base seed, task, input ID, replicate, candidate rank)`. Candidate ranks 0-63 are one saved stream; K=1, 4 and 16 results are prefixes of K=64.
- References were not included in the source-only input JSONL. Dataset manifests record hashes and the P0 baseline source run. All candidate pools, manifests and offline score JSON are under ignored `runs/`.

| Task | Frozen sampling settings | Candidate run | Candidate count / errors | Generator SHA-256 prefix | Input SHA-256 prefix | Candidate SHA-256 prefix |
|---|---|---|---:|---|---|---|
| XSum | `temperature=0.7`, `top_k=100`, `top_p=0.95`; upstream penalties unchanged | `p13-xsum-test200-k64-v1` | 12,800 / 1 | `f152f390253d` | `8c9743739237` | `827976c0265a` |
| WMT14 | `temperature=0.7`, `top_k=100`, `top_p=0.95` | `p13-wmt14-test200-k64-v1` | 12,800 / 0 | `ae580cbbb400` | `e1a58256f90b` | `7c4c20eff598` |
| Sudoku easy | `temperature=2.1`, `top_k=0`, `top_p=0.95` | `p13-sudoku-easy-test200-k64-v1` | 12,800 / 0 | `1cefebcd1eae` | `e4fe478642ba` | `4275d47ca672` |

The model config hashes are `6f10a7f8b97a` (XSum), `6469ffcfcda5` (WMT14), and `3b419835923b` (Sudoku). The sampler adapter hash is `0381429bcf62` for all three pools. Full hashes and complete per-run provenance are in each run's `manifest.json` and `generation-checkpoint.json`.

The selected development configurations were each rerun over the same 50 inputs and seed. Candidate text and error fields matched exactly across the original and repeat runs for all three tasks.

## K=1 comparison with greedy

| Task | Metric | Greedy baseline | Stochastic K=1 | Paired K=1 minus greedy |
|---|---|---:|---:|---:|
| XSum | ROUGE-1 F (%) | 34.061 | 33.422 | -0.639 |
| XSum | ROUGE-2 F (%) | 12.443 | 11.662 | -0.781 |
| XSum | ROUGE-L F (%) | 27.475 | 26.897 | -0.579 |
| WMT14 | Corpus BLEU | 18.668 | 15.254 | -3.414 |
| WMT14 | Lowercase corpus BLEU | 19.449 | 16.020 | -3.428 |
| WMT14 | Corpus chrF | 46.177 | 43.766 | -2.411 |
| Sudoku easy | Exact and valid solution rate | 99.0% | 88.0% | -11.0 points |

XSum K=1's paired 95% bootstrap intervals versus greedy include zero for all three ROUGE measures. WMT14's sentence-level chrF difference is -2.112 points (95% paired bootstrap CI: -3.605 to -0.593); its corpus BLEU comparison is reported above. The Sudoku K=1 exact/valid difference is -11 points (95% CI: -15.5 to -7.0 points). Mean output length was 20.57 words for XSum K=1 versus 20.97 greedy, and 17.90 versus 17.47 for WMT14; WMT14's K=1 length ratio was 0.9112 versus 0.8893 greedy.

## Quality-versus-K curves

All confidence intervals are paired bootstrap intervals over the 200 input examples (5,000 resamples, seed `20261112`). Candidate outputs within a prompt are not treated as independent examples. Oracle scores choose the best candidate independently per metric using held-out references; this is an offline upper bound, not a deployable selector.

### XSum

ROUGE values are mean per-example F-measures in percent. Gains are ROUGE-L points.

| K | Mean sampled ROUGE-L | Oracle@K ROUGE-L | Oracle gain vs. rank 0 (95% CI) | Oracle gain vs. greedy (95% CI) | Prompts beating greedy |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.897 | 26.897 | 0.000 | -0.579 (-1.943, 0.790) | 43.0% |
| 4 | 26.989 | 33.741 | 6.844 (5.893, 7.806) | 6.265 (5.158, 7.427) | 78.0% |
| 16 | 27.033 | 38.696 | 11.799 (10.642, 12.971) | 11.221 (10.058, 12.396) | 95.0% |
| 64 | 26.793 | 42.816 | 15.920 (14.708, 17.203) | 15.341 (14.194, 16.521) | 97.5% |

At K=64, Oracle@K ROUGE-1/2/L is 49.901 / 26.826 / 42.816; mean sampled ROUGE-1/2/L is 33.418 / 11.868 / 26.793. The reference-ROUGE oracle does not establish factuality.

### WMT14

Oracle ranking uses sentence-level chrF, not corpus BLEU. The K=1 row also reports corpus metrics above; sentence-level chrF gains below must not be interpreted as corpus BLEU gains.

| K | Mean sampled sentence chrF | Oracle@K sentence chrF | Oracle gain vs. rank 0 (95% CI) | Oracle gain vs. greedy (95% CI) | Prompts beating greedy |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.862 | 43.862 | 0.000 | -2.112 (-3.605, -0.593) | 41.5% |
| 4 | 44.359 | 51.309 | 7.447 (6.313, 8.652) | 5.335 (4.120, 6.594) | 73.0% |
| 16 | 44.673 | 56.208 | 12.346 (11.068, 13.708) | 10.234 (9.046, 11.476) | 90.0% |
| 64 | 44.512 | 59.870 | 16.008 (14.597, 17.505) | 13.896 (12.525, 15.291) | 96.0% |

At K=64, 775 of 12,800 translations are duplicates (6.05%). The K=1 corpus BLEU loss makes the low-K operating point a concern even though reference-ranked sentence chrF shows substantial candidate availability.

### Sudoku easy

Rates are fractions of examples; gains are percentage points. The held-out greedy baseline solves 198/200 examples exactly.

| K | Mean sampled exact rate | Oracle@K exact rate | Oracle gain vs. rank 0 (95% CI) | Oracle gain vs. greedy (95% CI) | Prompts beating greedy |
|---:|---:|---:|---:|---:|---:|
| 1 | 88.0% | 88.0% | 0.0 | -11.0 (-15.5, -7.0) points | 0.0% |
| 4 | 90.0% | 99.0% | 11.0 (7.0, 15.5) points | 0.0 (0.0, 0.0) points | 0.0% |
| 16 | 91.2% | 99.5% | 11.5 (7.0, 16.0) points | 0.5 (0.0, 1.5) points | 0.5% |
| 64 | 91.0% | 100.0% | 12.0 (8.0, 16.5) points | 1.0 (0.0, 2.5) points | 1.0% |

The stochastic K=1 clue-preservation rate is 92.0%; at K=64, mean sampled exact/valid rate is 91.0% and mean clue preservation is 95.3%. Oracle@64 repairs the two greedy misses, but this reference-based result does not show that an available selector can find those candidates.

## Diversity, duplicates and generation failures

| Task | K | Prompts with ≥2 unique candidates | Mean unique candidates per prompt | Duplicate fraction |
|---|---:|---:|---:|---:|
| XSum | 4 | 200/200 | 4.00 | 0.13% |
| XSum | 16 | 200/200 | 16.00 | 0.03% |
| XSum | 64 | 200/200 | 63.93 | 0.11% |
| WMT14 | 4 | 199/200 | 3.94 | 1.50% |
| WMT14 | 16 | 200/200 | 15.41 | 3.69% |
| WMT14 | 64 | 200/200 | 60.12 | 6.05% |
| Sudoku easy | 4 | 65/200 | 1.38 | 65.63% |
| Sudoku easy | 16 | 139/200 | 2.24 | 86.00% |
| Sudoku easy | 64 | 194/200 | 5.23 | 91.83% |

The only generation failure was `xsum:test:53`, candidate rank 9: the model emitted an empty string. It is explicitly recorded, contributes zero lexical score, and remains in K≥16 pools. No candidate was silently discarded or resampled.

## Example-level audit

These are reference-scored diagnostics only; no candidate was deployed or selected.

- **XSum helps:** `xsum:test:88` has greedy ROUGE-L 41.18, stochastic rank 0 at 28.57, and a rank-10 candidate at 86.67. **XSum stalls after rank 0:** `xsum:test:105` is already at 83.33 ROUGE-L for rank 0 and no candidate in the 64-pool scores higher.
- **WMT14 helps by sentence chrF:** `wmt14:test:141` moves from greedy 17.84 and rank 0 at 30.64 to 66.00 at rank 63. **WMT14 stalls after rank 0:** `wmt14:test:74` has 93.22 sentence chrF at rank 0, already above greedy's 66.05, with no higher-scoring candidate in the pool.
- **Sudoku recovers greedy misses:** `sudoku-easy:test:39` and `sudoku-easy:test:106` are greedy failures; exact solutions first appear at ranks 56 and 11. Conversely, `sudoku-easy:test:7` is a greedy success but stochastic rank 0 is incorrect (rank 1 is exact). The 64-pool contains an exact solution for all 200 inputs.

## Compute, uncertainty and limitations

| Task | End-to-end wall time | Synchronized generation | Model load | Remaining run overhead | Offline analysis/scoring |
|---|---:|---:|---:|---:|---:|
| XSum | 29.38 min | 29.23 min | 5.52 s | 3.80 s | 13.52 s |
| WMT14 | 30.15 min | 29.96 min | 6.25 s | 4.86 s | 5.39 s |
| Sudoku easy | 23.51 min | 23.38 min | 2.90 s | 4.71 s | 7.58 s |
| **Total** | **83.04 min** | **82.57 min** | **14.67 s** | **13.37 s** | **26.49 s** |

“Remaining run overhead” is wall time minus synchronized generation and model loading; it includes setup and output serialization, not a disk-only measurement. Offline analysis time includes validation, scoring, bootstrapping and diversity calculations. All pools fit and completed on the 10 GB RTX 3080; no per-process peak VRAM series was recorded.

There was one stochastic replicate. Bootstrap intervals quantify input-sample uncertainty for these fixed pools, not seed-to-seed variation. XSum's requested manual factuality audit was not performed, and Jev/semantic API scoring remains separately gated; lexical ROUGE must not be described as a factuality result. WMT14 oracle ranking is sentence-level chrF. Sudoku's synthetic 40-clue development set understated the held-out K=1 degradation and is not representative evidence for the released test set.

K=256, extra replicates, semantic scoring, and a Phase 2 selector were not run. The available results justify proposing a separately approved Phase 2 selection study, prioritizing XSum and treating WMT14's lower K=1 BLEU as a constraint. Sudoku offers little measured incremental headroom over greedy at the current budget.

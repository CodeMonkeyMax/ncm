# Phase 0 Baseline Report

**Status:** P0.5 is complete on the available RTX 3080. Phase 1 has been approved to begin; RTX 3090 comparison is deferred and is not a gate. The measurements below remain RTX 3080 results and must not be presented as 3090 measurements.

## Summary

| Task | Inputs | Exact / task score | Other independent metrics | Upstream agreement |
|---|---:|---:|---|---|
| XSum | 200 (test indices 0–199) | — | ROUGE-1 34.061, ROUGE-2 12.443, ROUGE-L 27.475 (mean per-example F-measure, %) | Matched upstream output to two decimal places: 34.06 / 12.44 / 27.48 |
| Sudoku easy | 200 (test indices 0–199) | 198/200 exact solutions (99.00%) | 200/200 well-formed; 198/200 valid grids; 200/200 preserve clues; mean cell accuracy 99.907% | Matched upstream: 99.00% solved, 99.91% cell accuracy, 99.00% valid |
| WMT14 | 200 (test indices 0–199) | — | BLEU 18.668; lowercase BLEU 19.449; chrF 46.177; length ratio 0.8893 | Matched upstream: 18.67 / 19.45 / 46.18 / 0.889 |

These are fixed-prefix smoke baselines, not full-test-set reproduction or uncertainty estimates. XSum and WMT14 use the untouched released configuration (`temperature=1`, `top_k=1`, `top_p=0`), which is effectively greedy. They are not candidate-scaling results.

## Provenance and environment

- Upstream NCM commit: `c70274273f489abc73832b2b98584a8483b6b203`.
- Checkpoint and dataset revisions are pinned in `configs/phase0/artifacts.json`; run manifests include their file hashes. The XSum GPT-2 reference model resolved offline to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. WMT14's Qwen3 tokenizer revision is `c1899de289a04d12100db370d81485cdf75e47ca`; only tokenizer files were downloaded, not Qwen model weights.
- Python 3.12.10, PyTorch 2.15.0+cu132, CUDA 13.2, Transformers 5.22.2, `rouge_score` 0.1.2, sacreBLEU 2.6.0.
- Device: NVIDIA GeForce RTX 3080, 10,240 MiB, driver 610.47. The project target is an RTX 3090 (24 GB), which was unavailable in this run.
- Each run saved one candidate per input, stable input ordering, contexts, and references separately. Generation used staged inputs and empty reference placeholders; references were attached only after generation. This is application-level isolation, not an OS sandbox.

## Commands and archived runs

Measured runs used the harness with batch size 4:

```powershell
.\.local\venv\Scripts\python.exe scripts\baseline.py `
  --task xsum --n 200 --bs 4 --gpu 0 --run-id p05-xsum-n200-rtx3080-v1

.\.local\venv\Scripts\python.exe scripts\baseline.py `
  --task sudoku-easy --n 200 --bs 4 --gpu 0 --run-id p05-sudoku-n200-rtx3080-v1

.\.local\venv\Scripts\python.exe scripts\baseline.py `
  --task wmt14 --n 200 --bs 4 --gpu 0 --run-id p05-wmt14-n200-rtx3080-v1
```

Independent offline scoring was run on the saved candidates, contexts, and references:

```powershell
.\.local\venv\Scripts\python.exe scripts\score_baseline.py `
  --run-id p05-xsum-n200-rtx3080-v1 --output-id p05-xsum-independent-score-v1

.\.local\venv\Scripts\python.exe scripts\score_baseline.py `
  --run-id p05-sudoku-n200-rtx3080-v1 --output-id p05-sudoku-independent-score-v1

.\.local\venv\Scripts\python.exe scripts\score_baseline.py `
  --run-id p05-wmt14-n200-rtx3080-v1 --output-id p05-wmt14-independent-score-v1
```

Run manifests, logs, per-example outputs, and independent score JSON are under ignored `runs/`:

- XSum: `p05-xsum-n200-rtx3080-v1`; repeat: `p05-xsum-n200-rtx3080-repeat1`.
- Sudoku easy: `p05-sudoku-n200-rtx3080-v1`; repeat: `p05-sudoku-n50-rtx3080-repeat1`.
- WMT14: `p05-wmt14-n200-rtx3080-v1`; repeat subset: `p05-wmt14-n50-smoke`.
- Independent scoring: `p05-xsum-independent-score-v1`, `p05-sudoku-independent-score-v1`, `p05-wmt14-independent-score-v1`.

The generated outputs matched exactly on the first 200 XSum inputs (200/200), first 50 Sudoku inputs (50/50), and first 50 WMT14 inputs (50/50) across same-config runs. The model path is greedy and the conditional upstream evaluator does not explicitly seed each example; this is observed repeatability on this environment, not a cross-device determinism guarantee.

## Timing and compute cost

CUDA timings are synchronized. “Warm generation” excludes the first output-producing batch; “warm full batch” also includes decoding and text conversion. The run elapsed time is the manifest interval and includes setup/staging and scoring.

| Task | Cold generation phase | Warm generation only | Warm full batch | Scoring phase | Recorded run elapsed | Recorded elapsed per candidate |
|---|---:|---:|---:|---:|---:|---:|
| XSum, 200 | 27.20 s | 18.10 s / 196 = 0.0923 s/candidate | 19.49 s / 196 = 0.0994 s/candidate | 5.76 s | 41.74 s | 0.209 s |
| Sudoku easy, 200 | 12.22 s | 6.00 s / 196 = 0.0306 s/candidate | 6.54 s / 196 = 0.0334 s/candidate | 0.11 s | 20.06 s | 0.100 s |
| WMT14, 200 | 17.77 s | 8.00 s / 196 = 0.0408 s/candidate | 8.60 s / 196 = 0.0439 s/candidate | 0.29 s | 25.60 s | 0.128 s |

These per-candidate figures are local GPU elapsed time, not monetary or amortized hardware cost. Cold timing includes model loading and shard writing but excludes Python/CUDA process startup. The independent metrics replay is separate and does not regenerate candidates.

## Device utilization

`nvidia-smi` was sampled about every 0.5 seconds. Samples are whole-device measurements, not per-process attribution. XSum telemetry was collected during the identical 200-input repeat; the original XSum run’s monitor did not produce a CSV. Sudoku telemetry is from the measured 200-input run.

| Task / sample run | Samples | Mean / max GPU utilization | Mean / max device memory used | Mean / max power |
|---|---:|---:|---:|---:|
| XSum repeat, 200 | 64 | 56.08% / 99% | 2,799 / 3,813 MiB | 223.01 / 329.82 W |
| Sudoku easy, 200 | 24 | 21.17% / 44% | 2,041 / 2,246 MiB | 114.09 / 152.64 W |
| WMT14, 200 | 37 | 34.00% / 71% | 2,674 / 3,324 MiB | 162.88 / 282.64 W |

The harness-reported CUDA peak allocated/reserved memory was 1,037 / 1,346 MiB for XSum generation, 250 / 298 MiB for Sudoku generation, and 1,336 / 1,378 MiB for WMT14 generation. Device-wide and allocator figures have different scopes.

## Per-example audit

- Sudoku failures: `sudoku-easy:test:39` and `sudoku-easy:test:106` were not exact solutions and failed the independent valid-grid check. Both preserved all original clues; cell accuracy was 91.36% and 90.12%, respectively. The saved candidate/reference records allow full offline inspection without rerunning the model.
- XSum’s five lowest per-example ROUGE-L scores were for `xsum:test:24` (0), `:53` (0), `:160` (0.0465), `:196` (0.0541), and `:33` (0.0563). These lexical outliers are not factuality judgments. No manual factuality audit was performed, so no factuality claim is made.
- WMT14's largest source/reference whitespace-token length gap was `wmt14:test:12` (8 hypothesis tokens vs. 30 reference tokens; length ratio 0.267). This is a length diagnostic, not a sentence-level translation-quality judgment; the archived output preserves all examples for review.
- No IDs, hypothesis/reference counts, or scoring records were missing or mismatched. All 200 examples per task were scored.

## Deviations and remaining work

- The RTX 3080 differs from the planned RTX 3090, so its latency, capacity, and utilization must not be presented as 3090 measurements. A same-ID/configuration comparison on the target hardware is deferred until available.
- WMT14 now has a pinned selective artifact allowlist, an offline Qwen tokenizer cache, and a 200-example measured baseline. Its metric implementation has both fixture-level validation and a saved-run comparison against the unchanged upstream scorer.
- The 200-input results use test indices 0–199, one replicate, and one released sampling configuration. They are preliminary baselines; they do not establish full-test-set performance, uncertainty, stochastic diversity, or factual correctness.
- Phase 1 candidate generation is approved to begin. The RTX 3090 comparison is not a gate; Phase 2 selector work and semantic API evaluation remain outside the current approval.

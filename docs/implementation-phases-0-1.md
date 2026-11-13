# Implementation Plan — Phases 0 and 1

**Status:** P0-A, P0.3, P0.4, and P0.5 are complete on RTX 3080. Phase 1 is approved to begin; the RTX 3090 comparison is deferred and is not a gate. Separate XSum/WMT14 validation sources are pinned, and Sudoku development examples use a deterministic synthetic generator.
**Parent:** [Experiment Plan v2.1](experiment-plan.md)  
**Target:** One RTX 3090 (24 GB), local execution  
**Primary goal:** Reproduce official NCM baselines, then measure candidate quality and Oracle@K at controlled inference budgets.  
**Out of scope:** MBR/deployable verification selection, iterative refinement, fine-tuning, training from scratch, external compute, and general-purpose instruction following. **In scope (approval-gated):** offline semantic evaluation of saved candidate pools, after judge calibration and separate API spend approval. See [semantic evaluation design](semantic-evaluation.md).

## 0. Critical upstream findings (verified 2026-11-11)

The [official NCM repository](https://github.com/HazyResearch/ncm) provides `evaluate.py`, `scripts/eval_shard.py`, `scripts/eval_merge.py`, and checkpoint-specific configs, with weights in [ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints) and evaluation inputs in [ncm-datasets](https://huggingface.co/datasets/hazyresearch/ncm-datasets).

**Upstream sampling defaults are unsuitable for Phase 1:**
- XSum: `temperature=1.0`, `top_k=1`, `top_p=0.0`; effectively greedy.
- WMT14: same greedy settings.
- Sudoku easy: `temperature=0.0`, `top_k=0`, `top_p=1.0`; greedy.
- In `models/next_concept.py`, `temperature == 0` takes `argmax`; otherwise `top_k` and `top_p` constrain multinomial sampling. Changing the random seed alone is **not** sufficient to sample distinct candidates under these released defaults.
- Conditional `eval_shard.py` writes `idxs`, `hyps`, and `refs` to pickles but does **not** explicitly seed each input/candidate. Baseline `evaluate.py` can use `--n`, `--bs`, `--gpus`, `--temp`, `--top_k`, `--top_p` through pass-through arguments.
- The official Sudoku scorer checks exact solution match, solved-grid validity, and cell agreement; add **clue preservation** in our independent checker.
- HF artifacts are nested: e.g., `xsum/`, `wmt14/`, `sudoku/easy/`; dataset paths follow the analogous nesting. Do not assume `sudoku-easy/` exists as a storage folder just because it is a config name.

**Policy:** Preserve the official baseline config exactly for Phase 0. Tune and freeze a separate stochastic configuration for Phase 1; do not overwrite upstream files.

## 1. Build approach

### Working environment

- **Preferred:** Linux or Windows + WSL2 Ubuntu with a functioning NVIDIA CUDA/PyTorch installation. Other setups are allowed after a smoke test.
- Python 3.10+; a dedicated virtual environment.
- Record GPU, driver, CUDA, PyTorch, transformers, Hugging Face Hub and dependency versions. Pin the exact upstream commit SHA and HF checkpoint/dataset revisions or hashes.
- Clone the upstream code into a local, ignored directory. **Never vendor the large weights/data into our GitHub repository.**
- Treat model checkpoints and inputs as untrusted external files: download only official sources and record hashes; note upstream uses `torch.load(..., weights_only=False)` and pickle-based input files.
- Start with XSum and Sudoku easy; add WMT14 after the pipeline passes. Do not download the entire 8 GB checkpoint repository or ~23 GB dataset repository unnecessarily.

### Proposed repository layout (to be implemented, not present now)

```text
README.md
docs/
  experiment-plan.md
  implementation-phases-0-1.md
  semantic-evaluation.md     # semantic judge design; already documented
  results/                  # small reviewed reports; not raw generations
configs/
  phase0/                   # experiment manifests referencing upstream defaults
  phase1/                   # stochastic sampling manifests (no changed model weights)
src/ncm_experiments/
  upstream.py               # pinned-upstream model/eval adapter
  datasets.py               # stable ID and split management
  sampling.py               # controlled candidate generation
  metrics.py                # independently validated task metrics
  semantic_eval.py          # planned offline judge adapter, not part of generation
  profiling.py              # timed, synchronized measurements
  reporting.py              # aggregate results + bootstrap intervals
tests/
  test_metrics.py
  test_datasets.py
  test_sampling.py
  test_reproducibility.py
runs/                       # ignored, raw local outputs
.local/                     # ignored, upstream checkout and cached artifacts
```

These file/module names are intended interfaces, not authorization to implement them. Prefer a small CLI and plain JSONL/JSON artifacts to a workflow orchestration framework.

### Immutable run manifest

Every run must record:
- `run_id`, UTC timestamp, git SHAs (ours and upstream), environment lock/hash, device properties.
- Task, data revision, checkpoint revision/hash, and fixed input IDs.
- Generation parameters (temperature, top-k, top-p, CFG, batch size, any penalties) and seed schedule.
- Number of generated candidates per input, replicate number, output paths.
- Warmup policy, measured timing scope, peak VRAM, exceptions, retries.
- Raw candidates with `task`, `input_id`, `candidate_rank`, `seed`, `text`, generation duration and validity/metric fields.
- Explicit metric implementation/version and references stored separately from candidate generation.
- Stable task/context, original source, constraints and reference identifiers for offline semantic scoring; restrict reference access to offline evaluation. Store judge results separately, keyed by candidate ID.

Write append-safe intermediate artifacts. When candidate pools are introduced in P1, checkpoint completed batches and support resuming incomplete pools without overwriting valid outputs; the P0.3 K=1 adapter can replay completed raw results and retry scoring. Never commit dataset/test answers, checkpoint blobs, or raw generation dumps.

## 2. Phase 0 — Baseline reproduction and instrumentation

### P0.1 — Lock inputs and runtime

**Implementation tasks**
1. Pin an upstream NCM commit and checkout locally.
2. Create/lock the Python environment; verify CUDA visibility and free VRAM.
3. Download only XSum and Sudoku easy checkpoint + evaluation folders first; verify expected filenames and hashes. WMT14 files are now in the selective allowlist, including the recorded Qwen3 tokenizer files/revision. Download only each task’s pinned files.
4. Document local directory mappings and prerequisites (WSL2/NVIDIA drivers if applicable).

**Acceptance:** Model/tokenizer weights and the corresponding evaluation inputs load without modification to upstream source. Record revisions; no data downloads to Git.

### P0.2 — Run untouched upstream smoke tests

Start with official `evaluate.py` at **K=1/default sampling**, one GPU, `--n 10`, then `--n 50`; use a smaller `--bs` if needed. For example, from the upstream checkout:

```bash
python evaluate.py --config config/eval/xsum.yaml \
  --checkpoint-dir /path/to/artifacts/xsum \
  --data-dir /path/to/data/xsum \
  --gpus 0 --n 50 --bs 4 --out /path/to/runs/p0-xsum
```

For Sudoku easy, use config `config/eval/sudoku-easy.yaml` but checkpoint/data paths `artifacts/sudoku/easy` and `data/sudoku/easy`. Repeat for WMT14 once ready.

Capture the upstream score printout and the shard data files as read-only raw baseline artifacts. No claims of paper-level reproduction from a 50-item smoke test.

**Acceptance:** Successful score calculation, matching count of inputs/hypotheses/references, no silent malformed outputs or OOM, and one complete archived run manifest per task.

### P0.3 — Minimal local harness and schema

**Implementation tasks**
1. Implement a thin adapter that either launches the upstream evaluation entry point or calls the upstream model without forking its architecture.
2. Standardize stable IDs, candidate records and run manifests.
3. Add explicit input order and shard mapping.
4. Add isolated caches/results directories and a meaningful failure/reporting path.
5. Record per-stage timing: download/setup excluded from generation timing; warmup excluded; separately report model-load + generation (cold), generation only (warm), and scoring. Use `torch.cuda.synchronize()` around GPU timings.
6. Retain source/context and task constraints for offline semantic evaluation, without exposing references to generation. Phase 0 does not call Jev.

**Acceptance:** Our baseline harness and upstream evaluation agree on example IDs/output counts and task metrics for the same output records; raw results are reusable without rerunning the model.

### P0.4 — Independently validate scoring

**Implementation and validation (2026-11-12)**
- `scripts/metrics.py` scores saved P0.3 JSONL artifacts offline; `scripts/score_baseline.py` writes immutable, hashed score records under a new run ID. It supports the legacy P0.3 records by deriving stable example/candidate IDs when absent.
- Record validation rejects duplicate IDs, missing references/contexts, empty generations, inconsistent candidate ranks and unequal pool sizes. It joins records by stable IDs and scores out-of-order JSONL records in input/rank order.
- XSum implements upstream ROUGE-1/2/L mean F-measure with stemming. Sudoku independently reports exact reference match, strict 9×9 validity, clue preservation decoded from saved input token IDs, and cell accuracy. WMT14 implements upstream corpus BLEU, lowercase BLEU, chrF, whitespace-token length ratio and a sacreBLEU signature.
- The full test suite passes, including metric fixtures compared with the upstream ROUGE and sacreBLEU conventions. Re-scoring the saved 10-example runs matched upstream's displayed XSum ROUGE (30.08/8.40/24.28); Sudoku matched 10/10 exact and valid, with independently confirmed clue preservation (10/10). WMT14 has fixture-level validation only; no WMT14 baseline artifacts are present yet.
- Deliberate malformed-output differences: Sudoku validity requires the canonical nine-row format when row separators are present (upstream only removes separators before checking digits), and cell accuracy always uses 81 cells as the denominator (upstream's `zip` truncates malformed outputs). These stricter checks prevent malformed generations from receiving inflated validity or cell-accuracy scores. Exact-match semantics remain raw string equality, matching upstream.

To independently score an archived JSONL run:

```powershell
.\.local\venv\Scripts\python.exe scripts\score_baseline.py `
  --run-id p03-final-xsum-v1 --output-id p04-final-xsum
```

**Metrics**
- Sudoku: solved-grid exactness, 9×9 validity, **preservation of original clues**, cell accuracy.
- XSum: ROUGE-1/2/L with the upstream-compatible scoring convention; keep the source article and reference disjoint from candidate-generation scoring.
- WMT14: corpus sacreBLEU, chrF and length ratio using a recorded sacreBLEU signature.

**Tests**
- Fixtures for perfect, malformed and partially correct Sudoku grids, including valid grids that violate clues.
- ROUGE and BLEU fixture comparisons against the upstream scorer.
- Duplicate IDs, missing references, out-of-order shards, empty generations, and unequal pool sizes.
- Metrics must be computable offline solely from saved candidates and evaluation references.

**Acceptance:** All unit/integration tests pass; any metric deviation from upstream is explained, not silently accepted.

### P0.5 — Baseline runs and review

**Order:** Sudoku easy + XSum (first), then WMT14. After 10/50-case smoke tests, use approximately **200–500 fixed examples per task** as the initial measured baseline, increasing only if justified. Full official-test-set reproduction is optional and separately costed from measured throughput.

**Repeatability:** Rerun a fixed subset with identical configuration. Require deterministic candidate outputs when feasible or document any nondeterministic kernels and metric tolerances. Save example-level failure cases.

**Deliverable:** `docs/results/phase0-report.md` with environment, exact commands, output counts, scores, per-example audit, device utilization, cost per candidate and known deviations.

**Initial execution (2026-11-12 UTC):** The report contains measured 200-example XSum, Sudoku-easy, and WMT14 runs, independent score replays, nvidia-smi telemetry, and fixed-subset repeatability checks. All three produced exactly reproducible candidates on this RTX 3080; the same hardware results must not be treated as RTX 3090 measurements. WMT14 agrees with the independent corpus scorer. The user accepted RTX 3080 completion and approved moving to Phase 1; target-hardware comparison is deferred.

**Gate P0 → P1:**
- [x] Exact upstream versions and artifact hashes recorded.
- [x] At least XSum and Sudoku easy complete (WMT14 planned or successfully reproduced).
- [x] Saved outputs reproduce metrics without regeneration.
- [x] No mismatched IDs/refs, silent corruptions or unexplained evaluation failures in the measured runs.
- [x] End-to-end and warm generation latency measured on the available RTX 3080; RTX 3090 comparison deferred.
- [x] Phase 1 candidate-generation approach approved before implementation.

## 3. Phase 1 — Candidate scaling (no selector implementation)

**Primary question:** At what K does extra sampling stop producing materially better *possible* candidates?

### P1.1 — Implement controlled stochastic sampling

Extend the minimal upstream adapter to produce **K candidates per same fixed input**, without changing model weights or upstream baseline configs.

**Required controls**
- Explicit `temperature > 0`; `top_k > 1` and/or non-degenerate `top_p > 0` for meaningful sampling. In particular, do **not** leave XSum/WMT14 at `top_k=1`/`top_p=0.0`.
- For Sudoku, choose top-k compatible with the much smaller level vocabulary (or `top_k=0` for no truncation); avoid top-k values larger than the codebook.
- `K = 1, 4, 16, 64, 256`; use *nested prefixes* of the same saved candidate stream so Oracle@K is monotonically nondecreasing by construction.
- Unique deterministic seed derived from (base seed, task, input ID, replicate ID, candidate rank). First prioritize true per-candidate reproducibility; if batching changes RNG behavior, document and explicitly fix a batch schedule or redesign RNG isolation rather than silently treating index order as seed-independent.
- Reusable model in GPU memory, bounded batch sizes, checkpointing and ability to resume on failure.
- Output every candidate, including duplicates, malformed outputs and errors. Report duplicates separately; never silently resample to force uniqueness.

**Acceptance:** At least two distinct candidates appear for a substantial fraction of test prompts; repeated seeds/configs reproduce saved samples as specified; saved candidate records have complete IDs and provenance.

**Implementation status (2026-11-12):** `scripts/sample_candidates.py` keeps one model loaded, uses one input per generation call for candidate-isolated RNG, derives each seed from `(base seed, task, input ID, replicate, candidate rank)`, and atomically checkpoints one complete input pool at a time. It retains duplicate, malformed, empty, and errored candidates; real references are opened only after generation ends. Repeated Sudoku seeds matched exactly, and greedy outputs matched untouched upstream inference on 50 validation inputs per task. The first K=4 development pass produced distinct candidates for all 50 XSum/WMT14 inputs, but only 2/50 Sudoku inputs at the tested low-temperature setting; Sudoku configuration is not accepted yet.

### P1.2 — Calibrate sampling on development data only

Use a small, fixed **development set** separate from the held-out evaluation subset. Pilot a limited parameter matrix (for example 2 temperatures × 2 candidate-truncation choices), checking:
- Validity and baseline quality at K=1.
- Diversity (unique fraction, duplicate concentration, pairwise text/structure distance).
- Catastrophic degradation from excessive randomness.
- GPU/VRAM impact.

Choose **one primary stochastic configuration per task** before viewing held-out Oracle@K results. Keep the untouched greedy baseline separately. Pre-register any secondary settings as sensitivity analyses.

**Acceptance:** Stable, nontrivial diversity and viable single-candidate quality; sampling parameters frozen for held-out evaluation.

**Development inputs:** Fifty XSum examples come from the pinned EdinburghNLP validation split; fifty WMT14 examples come from the pinned `de-en` validation split. The NCM release contains no Sudoku validation split, so Sudoku uses a deterministic synthetic set of 50 unique, 40-clue puzzles. This Sudoku set is useful for sampler plumbing and initial calibration, but is not claimed to match the released test distribution.

**Calibration result (2026-11-13):** The grid used seed `20261112`, replicate 0, K=4, and 50 examples per task. The greedy baseline remains the untouched upstream configuration. The following primary sampling configurations are frozen for held-out evaluation:

| Task | Frozen generation settings | K=1 development quality vs. greedy | K=4 mean sampled quality | Diversity and errors |
|---|---|---|---|---|
| XSum | `temperature=0.7`, `top_k=100`, `top_p=0.95` (upstream penalties unchanged) | ROUGE-1/2/L: 36.72 / 14.32 / 28.43 vs. 36.53 / 14.28 / 29.36 | ROUGE-1/2/L: 36.10 / 13.63 / 28.69 | 50/50 prompts had 4 unique candidates; 0 errors |
| WMT14 | `temperature=0.7`, `top_k=100`, `top_p=0.95` | BLEU / chrF: 30.70 / 57.60 vs. 31.43 / 59.09 | BLEU / chrF: 30.68 / 58.18, pooled over candidates | 50/50 prompts had alternatives; mean 3.96 unique; 0 errors |
| Sudoku easy | `temperature=2.1`, `top_k=0`, `top_p=0.95` | 98.0% exact and valid vs. 100% greedy | 91.5% exact and valid across all candidate outputs | 15/50 prompts had alternatives (mean 1.34 unique); exact solution available in all 50 K=4 pools; 0 errors |

Across each frozen setting, a second run over the same examples and seed reproduced every candidate text and error field exactly. The XSum/WMT14 choice favors temperature 0.7: all prompts remain diverse while K=1 scores stay close to greedy. Sudoku's K=1 quality is strong and its diversity is modest but nontrivial; the 40-clue synthetic set is too easy to demonstrate oracle gains because greedy already solves every example. Treat Sudoku's calibration as a frozen procedural choice, not evidence that candidate scaling will help its released test distribution.

The selected development runs and exact-repeat runs are archived under `runs/`: `p12-xsum-t0p7-k100-v1` / `p12-xsum-repeat-v1`, `p12-wmt14-t0p7-k100-v1` / `p12-wmt14-repeat-v1`, and `p12-sudoku-temp21-v1` / `p12-sudoku-temp21-repeat-v1`. These development metrics do not use or reveal held-out stochastic results.

### P1.3 — Run a staged K sweep

Use ascending pools, persisting candidates once and computing all K-prefix metrics offline.

| Tier | Examples | K values | Purpose |
|---|---:|---|---|
| Smoke | 10–20 | 1, 4, 16 | Validate stochastic path and storage |
| Pilot | ~50 | 1, 4, 16, 64 | Estimate latency, diversity and curves |
| Main | ~200 (subject to measured cost) | 1, 4, 16, 64 | Primary inference-scaling analysis |
| Extension | ~25–50 initially | through 256 | Check saturation only if the 64-candidate curve justifies it |

Target several independently seeded pools for uncertainty assessment when cost permits (e.g. three replicates for a fixed representative subset). Fix input IDs and seeds before the run. Do not extrapolate a full 256-candidate sweep to every task or spend unlimited GPU time: **derive an explicit cap in GPU-hours from P0 measurements and seek approval if projected costs exceed it**.

**Acceptance:** Complete nested candidate sets, no unknown input loss, measured GPU usage, and a documented reason for running or skipping K=256.

### P1.4 — Compute candidate-scaling metrics

**Every task**
- K=1 stochastic quality against the official greedy baseline.
- Candidate diversity/duplicates by K.
- Mean sampled quality and **Oracle@K** on the same candidate pools.
- Percentage of prompts improving over the first sampled candidate (oracle improvement rate).
- Compute per input and marginal oracle gain per additional candidate.

**Sudoku:** Exact puzzle solution match, valid 9×9 board, preserves clues, success-within-K, per-difficulty results if expanded.

**XSum:** Per-item ROUGE-L (and ROUGE-1/2) Oracle@K, plus average length, duplicates and a *manually reviewed or independent* factuality error sample. Explicitly note that reference-ROUGE oracle quality is not guaranteed factuality.

**WMT14:** Corpus BLEU/chrF for selected single candidates; for reference-oracle analysis use clearly labeled *sentence-level* metrics (e.g., chrF) and acknowledge ranking/corpus metric differences.

**Statistics:** Paired bootstrap resampling over input examples (not over correlated candidate outputs), intervals for improvement curves, per-run replicate variability, and sensitivity to prompt length/difficulty. K=1 candidate must be the **same first candidate** used for all larger nested pools.

**Metric guardrail:** Oracle@K uses held-out references **offline only for analysis**. It must not be used to generate or select deployed outputs, and must not be misrepresented as MBR performance.

### P1.4a — Semantic correctness track (offline and approval-gated)

Follow [the Jev protocol](semantic-evaluation.md) on saved candidate pools. Keep the NCM checkpoint, stochastic generation, and existing conventional metrics unchanged.

1. Label development examples independently for reference equivalence, task correctness and material errors, including valid paraphrases, missing information, negation, entities and number changes. Calibrate Jev's probabilistic Noul answers and freeze the resolved judge version, rubrics, thresholds and uncertain/review policy **before** held-out scoring.
2. Request separate authorization for an API spend cap. Pilot only a representative subset of saved XSum and WMT14 candidates; Sudoku keeps its deterministic validator. Cache raw probabilities, model/version, rubric hash, input/candidate ID, usage, retries and latency.
3. Run the full saved pool only if the pilot demonstrates acceptable reliability and cost. Plot **Jev-estimated semantic pass-within-K** beside conventional reference Oracle@K and independently audited verified pass-within-K. Never claim a Jev pass is verified ground truth.
4. Audit false positives and false negatives, including **pool-level false-accept rates** as K increases. An illustrative *independent* 1% false-accept probability per wrong candidate yields apparent pool success near 15% at K=16, 47% at K=64 and 92% at K=256. Real judge errors may correlate.
5. Investigate lexical/semantic disagreements, threshold sensitivity and total cost. Treat inadequate calibration or missing spend approval as a documented limitation, not a positive result.

**Acceptance:** References stay out of generation and deployed selection; semantic scores are separately versioned and auditable; independent labels support any claimed semantic gains; judge error and spend are explicitly reported. This work package may be deferred if judge readiness or spend approval fails, while keeping a clear Phase 1 limitation.

### P1.5 — Report and decision

Produce `docs/results/phase1-report.md` containing:
1. Frozen configs, exact sample IDs, candidate totals, seed manifests and checkpoint hashes.
2. Quality-versus-K curves with uncertainty and diversity-versus-K curves.
3. Oracle gain against stochastic K=1 **and** against the official greedy baseline.
4. Separate scores for generation-only GPU time, end-to-end inference/selection-preparation time, and disk/scoring overhead.
5. Examples where candidate availability helps, stalls or deteriorates in validity.
6. A decision on which task and K budgets to carry into Phase 2 selection/MBR experiments.
7. A semantic appendix comparing lexical, Jev-estimated and independently verified results, model/rubric/threshold versions, false-pass rates and judge cost (or a documented gated deferral).

**Gate P1 → P2 (no automatic advance):**
- [ ] Non-degenerate, reproducible sampling confirmed.
- [ ] Nested K sweeps and offline scoring independently validated.
- [ ] Quality/compute and diversity curves produced with uncertainty.
- [ ] Reference leakage excluded from generation.
- [ ] If semantic evaluation ran: judge calibration, independent audit, pool-level false-pass analysis and spend accounting completed; otherwise explicitly document deferral.
- [ ] Evidence indicates whether additional candidate selection is worth testing (or documents a negative finding).
- [ ] A separate Phase 2 proposal/approval is recorded.

**Interpretation:** A higher Oracle@K is an *opportunity* for better selection; it does **not** demonstrate that a small NCM can replace a larger model, or that MBR can find the oracle answer.

## 4. Work packages in dependency order

| ID | Work package | Depends on | Reviewable completion artifact |
|---|---|---|---|
| P0-A | Pin upstream + environment + selective downloads | none | Version/artifact manifest |
| P0-B | Run official untouched evaluation smoke tests | P0-A | Raw outputs + scores |
| P0-C | Build minimal wrapper, result schema and timing | P0-B | Offline-replayable records |
| P0-D | Independently validate metrics and reproducibility | P0-C | Passing tests and audit |
| P0-E | Run measured baselines and write report | P0-D | Phase 0 report + decision |
| P1-A | Add stochastic sampling and deterministic seeds | **P0 approved** | Valid candidate pools + tests |
| P1-B | Calibrate sampling on separate development set | P1-A | Frozen sampling configs |
| P1-C | Run staged nested-K sweeps | P1-B | Candidate manifests and measured costs |
| P1-D | Offline conventional Oracle@K/diversity analysis | P1-C | Figures, tables, uncertainty |
| P1-D2 | Approval-gated Jev calibration + offline semantic evaluation | P1-C, judge/spend approval | Versioned audit and semantic-vs-lexical results (or documented deferral) |
| P1-E | Report results and propose next decision | P1-D, P1-D2 disposition | Phase 1 report + decision |

## 5. Minimal test plan

1. **Schema:** Exactly K records per completed (task, input, replicate); unique candidate IDs; all records have provenance.
2. **Reproduction:** A known manifest rerun matches saved candidates within documented reproducibility constraints.
3. **Nested K:** Candidate streams for K=1,4,16,64,256 share their initial prefix; Oracle@K never decreases across the same stored stream.
4. **Correctness:** Sudoku validity independently checks all rows, columns, boxes and given clues; multiple known fixtures.
5. **Isolation:** Reference texts or solved grids are never passed to candidate generation; test-only metrics are computed offline.
6. **Cost:** Time GPU work only after synchronization and warmup; distinguish cold-start, generation and scoring costs.
7. **Failure recovery:** Interrupted runs resume without silently duplicating candidate ranks or discarding unsuccessful candidates.
8. **Regression:** Vanilla upstream baseline outputs/metrics are unchanged by enabling experimental code.
9. **Semantic judge:** Independently labeled equivalence/validity/error fixtures; pin version/rubrics; reference isolation; threshold sensitivity and false-pass-by-K audits; scores reproduce from saved candidates without regenerating.

## 6. Decisions deliberately deferred

- Whether WMT14 participates in the initial full sweep or remains a secondary check until hardware measurements exist.
- Running K=256 for more than a small subset.
- Adding a separate large verifier, MBR, iterative feedback, training, or scaling model size.
- Additional hardware/rental costs.
- Jev model and thresholds until calibration, and any external API spend until separately approved; no judge-derived training in Phase 1.

## 7. External references

- [Official NCM code and evaluation instructions](https://github.com/HazyResearch/ncm)
- [Released checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints)
- [Released evaluation datasets](https://huggingface.co/datasets/hazyresearch/ncm-datasets)
- [NCM paper](https://arxiv.org/abs/2610.14162)

---

**Next approval request:** Authorize **P0-A only** (environment pinning, upstream clone, official artifact manifest/download setup). Do not begin coding or running experiments merely because this plan exists.

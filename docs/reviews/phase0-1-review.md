# Phase 0–1 Project Review

**Date:** 2026-11-13
**Reviewed at:** commit `a35a692` ("Phase 1 Complete")
**Scope:** all documents under `docs/`, all scripts under `scripts/`, the tests, and `configs/phase0/`.
**Method:** reading only. Nothing was executed: the review environment had no pytest, GPU, upstream checkout, or model artifacts, and `runs/` is not in the repository. Numbers quoted below come from the [Phase 0 report](../results/phase0-report.md) and [Phase 1 report](../results/phase1-report.md) and were not independently reproduced.

## Summary

The project's bookkeeping is stronger than its evidence so far. Provenance, isolation, and reporting discipline are very good. However, Phase 1 was passed on a gate that could not fail, and its headline numbers do not yet show that sampling buys anything a selector could capture.

The recommended next step is a CPU-only MBR pilot on the existing XSum and WMT14 pools, followed by fresh development pools before any selector is tuned.

## Strengths

- **Provenance.** Upstream commit, checkpoint, dataset, and tokenizer revisions are pinned with hashes. Runs record config, adapter, input, and output hashes.
- **Isolation.** References are kept out of generation inputs and attached only afterwards.
- **Reproducible sampling.** Each candidate has its own seed derived from `(base seed, task, input ID, replicate, rank)`, K values are nested prefixes of one stream, and repeat runs matched exactly.
- **Safe run handling.** Runs are resumable, chunks are validated on resume, and completed outputs are never silently overwritten.
- **Honest reporting.** Negative results are kept, the single empty generation is recorded rather than resampled, and the reports repeatedly state that oracle availability is not selection.

## Research concerns

Ordered by importance.

### R1. The Phase 1 gate cannot fail

The gate asks whether Oracle@K improves. Oracle@K is a maximum over nested prefixes, so it is nondecreasing by construction; the plan itself notes this.

Meanwhile, mean sampled quality is flat across K and below greedy on all three tasks:

| Task | Greedy | Mean sampled at K=64 | Oracle@64 |
|---|---:|---:|---:|
| XSum ROUGE-L | 27.475 | 26.793 | 42.816 |
| WMT14 sentence chrF | 45.974 (derived) | 44.512 | 59.870 |
| Sudoku easy exact | 99.0% | 91.0% | 100.0% |

The greedy WMT14 sentence-chrF figure is derived from the report's K=1 row (43.862 + 2.112).

A large oracle gain over a flat mean is what the maximum of 64 noisy scores looks like. It does not by itself show that better candidates exist in any sense a selector could detect.

**Recommendation:** replace the gate with one that can fail. Options:

- Compare the oracle gain against a noise baseline, for example the gain expected from the within-pool score spread alone.
- Pick the oracle candidate by one metric and score it on a different one, to see whether the "best" candidate is better in more than one sense.
- Go straight to a selector result (see R2), which is the quantity that matters.

### R2. The decisive test needs no GPU

MBR or consensus selection can run on the saved K=64 pools on CPU. That directly answers whether any oracle gain is capturable, and it is far cheaper than the generation already done.

Because the only K=64 pools are on test inputs (see R3), treat a first run on them as a pilot with a selector fixed in advance, not as a tuned result.

### R3. Selector development will consume the test set

The only K=64 pools are for test IDs 0–199. Development pools exist only at K=4 on 50 inputs. Any selector tuned on the existing K=64 pools is tuned on test data.

**Recommendation:** generate K=64 development pools before tuning any selector choice, such as the utility metric, pool size, or thresholds.

### R4. Sudoku is misread as "low priority"

Sudoku is the one task with a free, exact selector: a valid grid that preserves the clues is the solution for a uniquely solvable puzzle. Selection there is essentially solved, so Oracle@K is close to an achievable number rather than an upper bound.

The real problem is the ceiling. Greedy solves 198 of 200, so the full measured gain is two examples, with a confidence interval of 0.0 to 2.5 points. A sample of 200 cannot resolve that.

**Recommendation:** if upstream ships a harder Sudoku checkpoint, use it as the verification testbed. If not, treat Sudoku easy as a plumbing check and say so.

### R5. The Sudoku sampling configuration did not transfer

Temperature 2.1 was chosen on a synthetic set of 40-clue puzzles, where stochastic K=1 scored 98% against 100% greedy. On the held-out set, K=1 scored 88% against 99% greedy. The Phase 1 report acknowledges this.

The held-out pool is also 91.8% duplicates at K=64 (5.23 unique candidates per prompt), so the effective pool is small. The frozen configuration is neither accurate at K=1 nor diverse.

**Recommendation:** build a development set that matches the released test distribution before drawing any Sudoku conclusion.

### R6. Compute figures reflect the harness, not the method

The Phase 1 sampler generates one candidate per call. Derived from the reported totals over 12,800 candidates per task:

| Task | Phase 1, batch size 1 | Phase 0 warm generation, batch size 4 |
|---|---:|---:|
| XSum | 0.137 s/candidate | 0.092 s/candidate |
| WMT14 | 0.140 s/candidate | 0.041 s/candidate |
| Sudoku easy | 0.110 s/candidate | 0.031 s/candidate |

Global seeding per candidate (`torch.manual_seed`) is what prevents batching. Any quality-per-compute claim in later phases will overstate the cost of sampling by up to about 3.5x unless this is fixed.

**Recommendation:** implement batched sampling with per-candidate generators before Phase 4, and before any K=256 run.

### R7. There is no "bigger model" anchor

The research question is compute versus model size, but the comparison to a larger model is deferred to Phase 4. Without it, there is no way to judge whether a given selector gain is large or small.

**Recommendation:** obtain one cheap reference point now, such as a larger model's score on the same 200 inputs per task.

### R8. One seed replicate

The bootstrap intervals cover input sampling only, not seed-to-seed variation, as the Phase 1 report states. Add replicates on a subset before making any Phase 2 claim.

A smaller related point: `scripts/analyze_phase1.py` uses a different bootstrap seed for each K, so the intervals at different K are not paired with each other.

## Engineering concerns

### E1. Decoding logic is reimplemented, with no recorded parity check

`scripts/sample_candidates.py` (`pick_tokens`, line 395) re-creates upstream's frequency-penalty, deficit, and adjacency token-picking logic. The script has a `--greedy-baseline` mode, but neither report records that its output was compared with the Phase 0 upstream outputs.

**Recommendation:** run the adapter in greedy mode on the 200 test inputs per task and diff against the Phase 0 candidates. Record the result.

### E2. Results cannot be verified from the repository

Everything under `runs/` is ignored, and the report tables appear to be transcribed by hand.

**Recommendation:** commit the small analysis JSON files and run manifests (they contain no dataset text), and generate the report tables from them.

### E3. Test setup is undocumented

There is no CI, and pytest does not appear in `configs/phase0/requirements-lock.txt` or `docs/environment.md`.

**Recommendation:** pin the test dependency, document the command, and add a minimal CI job for the CPU-only tests.

### E4. Smaller items

- `digest`, `write_json`, and `read_jsonl` are duplicated across scripts; move them to one shared module.
- Documented commands are PowerShell-only.
- Commit messages such as "P1.2 Done" do not say what changed.
- The evaluation inputs are the first 200 test examples, not a random sample; note this wherever results are generalised.

## Recommended order of work

1. Run the adapter greedy parity check (E1).
2. Run a pre-specified MBR pilot on the existing XSum and WMT14 pools, CPU only (R2).
3. Replace the Phase 1 gate with one that can fail, and re-evaluate Phase 1 against it (R1).
4. Generate K=64 development pools for XSum and WMT14 (R3).
5. Add seed replicates on a subset (R8).
6. Implement batched sampling (R6).
7. Get a larger-model reference point (R7).
8. Decide what to do with Sudoku: harder checkpoint, or plumbing check only (R4, R5).
9. Commit analysis JSONs, pin pytest, add CI (E2, E3).

# Experiment Plan: Small-Model NCM with Test-Time Scaling

**Version:** 2.1  
**Status:** Proposed, not yet implemented  
**Initial hardware:** RTX 3090 (24 GB VRAM)  
**Approach:** Frozen pretrained models and inference-time experiments before new training

## 1. Research objective

Determine whether Next-Concept Modeling (NCM), combined with inexpensive candidate generation, consensus selection, verification and iterative refinement, can achieve capabilities ordinarily associated with larger language models.

The goal is **not** to maximize parameter count. It is to find the smallest model and most efficient inference strategy that produce a given level of capability.

### Core hypothesis

> A small NCM with sufficient inference-time computation can outperform a substantially larger single-pass model on selected tasks at comparable computational cost.

Supporting hypotheses:

- **H1 — Sampling:** More candidates increase the probability that at least one correct or high-quality answer appears.
- **H2 — Selection:** Consensus- or verifier-based selection can identify better candidates without requiring a large evaluator.
- **H3 — Iteration:** Generate–evaluate–revise cycles can outperform one-shot consensus at the same end-to-end compute budget.
- **H4 — Semantic correctness:** Reference-based metrics overlook genuinely correct alternatives, and calibrated lightweight judging can detect them without materially inflating success-within-K.

H1 and H2 can be tested with released checkpoints. H3 may require new conditioning mechanisms or training; this must be checked first.

## 2. Experimental principles

1. **Use released models first.** Do not train from scratch until inference experiments justify it.
2. **Control for compute.** Compare equal wall-clock time, GPU-seconds and relevant generation budgets, not just parameter count.
3. **Separate availability from selection.** A good answer somewhere in a pool does not mean the system can find it.
4. **Use independent evaluation.** Do not grade a method solely by the similarity or consensus criterion it optimizes.
5. **Use approval gates.** Every phase ends in a report and a continue/modify/stop decision.
6. **Distinguish reconstruction, generation, selection, and task correctness.** Improvements in one are not automatically improvements in the others.

## 3. Roadmap

### Phase 0 — Reproduce and instrument

**Objective:** Establish a trustworthy baseline on available hardware.

Use official HazyResearch checkpoints, starting with supported XSum summarization, WMT14 German-to-English translation, and Sudoku workloads, subject to checkpoint compatibility.

Tasks:
- Run the official inference/evaluation path.
- Reproduce single-generation quality metrics where feasible.
- Measure wall-clock latency, throughput, peak VRAM and GPU utilization on the RTX 3090.
- Record model/checkpoint versions, prompts, seeds, generation settings and raw outputs.
- Persist source context, task constraints, and references separately so candidate pools can be semantically evaluated offline without leaking reference answers into generation.
- Determine which checkpoints support conditional generation and whether arbitrary feedback conditioning is possible.
- Establish a reproducible held-out evaluation procedure.

**Deliverable:** Baseline report with reproducibility and resource measurements.

**Gate:** Proceed only if baseline results are credible and inference is practical locally.

### Phase 1 — Candidate scaling

**Objective:** Measure the value of extra generation *without changing the model*.

Evaluate candidate pool sizes:

| K | Role |
|---:|---|
| 1 | Single-generation baseline |
| 4 | Small sampling budget |
| 16 | Moderate pool |
| 64 | Large pool |
| 256 | Diminishing-returns test, if practical |

Measure:
- Average and best candidate quality.
- Candidate diversity and near-duplicate rates.
- **Oracle@K:** best possible selection from the generated pool, scored only for analysis with independently defined task criteria.
- Proportion of items with at least one *correct* candidate, for exactly checkable tasks.
- End-to-end generation cost and latency.
- **Semantic correctness:** On XSum and WMT14, compare conventional reference-based Oracle@K against Jev-estimated semantic pass-within-K and independently audited correctness. Keep Sudoku deterministic.
- Calibration error and judge costs, including whether false positives accumulate as K increases.

**Important:** Oracle@K is an upper bound on selection among observed candidates, *not* an achievable deployed selector. For open-ended tasks, report the limits of the oracle metric. Jev passes are **estimated**, not a verified oracle, and must never be conflated with correctness. See [semantic evaluation design](semantic-evaluation.md).

**Gate:** If larger pools do not materially increase potential answer quality, investigate generator limitations before investing in selectors.

### Phase 2 — Consensus and verification

**Objective:** Capture as much Oracle@K improvement as possible with an actual selector.

Compare:

| Selector | Method |
|---|---|
| Random | Choose a candidate uniformly from the pool |
| MBR | Choose the candidate with highest average utility/similarity to other sampled candidates |
| Diversity-aware MBR | Reduce duplicate/cluster dominance |
| Verifier | Rank candidates using independent task correctness/quality signals |
| Hybrid | Combine consensus, diversity and verification |

Use the same candidate pools across selectors to isolate selection quality.

For Sudoku, use exact rule validation. For translation and summarization, use task metrics and, where appropriate, separate checks for adequacy/factual consistency. Never claim that surface similarity alone verifies truth. Jev is one candidate verifier, but a *deployable* selector must not see held-out references that Phase 1's offline judge may use.

Questions:
- Does consensus predict correctness or popularity?
- When do correlated wrong candidates dominate?
- How much of Oracle@K is captured?
- How expensive is each selector, including verifier compute?
- Does diversity improve performance or merely spread outputs?

**Gate:** Selection must produce repeatable, meaningful quality gains at practical compute cost.

### Phase 3 — Iterative consensus

**Objective:** Test whether feedback and revision beat single-pass sampling at matched budgets.

**This is a new research extension, not merely reproduction of published MBR decoding.**

Candidate design:
1. Generate a pool of candidates.
2. Score and select promising candidates.
3. Identify disagreement, invalid constraints or unresolved requirements.
4. Supply feedback to a subsequent generation step.
5. Repeat until the budget is spent or a stopping rule fires.

Illustrative equal-output-count comparison:
- **Single-pass:** 64 candidates, one selection.
- **Iterative:** Four rounds of 16 candidates, feedback between rounds.
- **Adaptive:** Allocate candidate/round budgets based on uncertainty.

**Feasibility prerequisite:** Determine whether current task-specific NCM checkpoints can be conditioned on a prior draft and feedback. If not, treat any revision-capable fine-tuning or architecture changes as a separate approval gate. Do not silently substitute independent resampling for real revision.

Measure:
- Independent correctness improvement per round.
- Error correction versus reinforcement.
- Agreement versus accuracy.
- Convergence, stagnation, and failure modes.
- Total compute including feedback and evaluation overhead.

**Gate:** Iteration must outperform the best non-iterative method on a quality-versus-compute basis.

### Phase 4 — Compute allocation and model comparisons

**Objective:** Establish when model size beats additional inference, and vice versa.

Compare NCM models of different sizes **only where checkpoints, training data and tasks permit defensible comparisons**. Add appropriately trained autoregressive baselines for the same tasks.

Test strategies:
- Larger model with few samples.
- Smaller model with many samples.
- Smaller model with an independent verifier.
- Smaller model with iterative refinement (if feasible).

Evaluate both quality-at-fixed-compute and compute-at-fixed-quality. Report parameter count, training differences and serving optimizations as confounders.

**Gate:** Identify a real quality/cost Pareto advantage before proposing substantial new training.

### Phase 5 — General-purpose capability (conditional)

**Objective:** Test transfer from narrow tasks to useful general-purpose instruction following.

This phase is *not* authorized for immediate implementation.

Potential tasks:
- Short factual question answering.
- Multi-step arithmetic and logic.
- Structured extraction.
- Instruction following.
- Short conversational responses.
- Tool selection and structured output.

Prefer held-out tasks with deterministic validators where possible. If released models cannot support general instruction conditioning or feedback, propose a specific small-scale adaptation experiment and its compute budget for separate approval.

**Gate:** Continue only if evidence suggests a viable path to general-purpose capability, not just benchmark-specific optimization.

## 4. Evaluation protocol

Use separate development and held-out test sets. Begin with approximately 50 examples per task for pipeline validation; expand to hundreds where practical. Use multiple generation seeds and report uncertainty intervals.

| Dimension | Primary measurement |
|---|---|
| Correctness | Exact checks where possible; independent task metrics otherwise |
| Candidate potential | Oracle@K and success-within-K |
| Selector effectiveness | Fraction of Oracle@K gap recovered relative to a defined baseline |
| Efficiency | Quality vs. end-to-end GPU-seconds and latency |
| Reliability | Variance, failure rate and correlated mistakes |
| Scaling | Marginal gain per extra unit of compute |

Task measurements:
- **Summarization:** ROUGE-L plus factual consistency/adequacy checks.
- **Translation:** sacreBLEU and chrF, plus adequacy checks when needed.
- **Sudoku:** Exact solved-grid validity and compliance with clues.

Do not equate higher ROUGE/BLEU with demonstrated reasoning improvement. For selection, take care not to let test-set references leak into ranking methods.

**Phase 1 semantic evaluation addendum:** Use the [Jev semantic correctness protocol](semantic-evaluation.md) as a *separate offline evaluation track*. Distinguish reference equivalence from task correctness and material errors. Calibrate thresholds on development examples, audit held-out disagreements with independent labels, report pool-level false-pass rates at increasing K, and include the complete judge cost. Freeze the resolved Jev model version and scoring rubric before held-out analysis. No Jev score is ground truth simply because it is a probability.

## 5. Success and failure criteria

These are **provisional research thresholds**, not expected outcomes.

**Strong result**
- At least a 5% relative improvement in a primary task metric for practical selection versus its predeclared single-sample baseline, with uncertainty supporting a real effect.
- Iterative refinement beats the best non-iterative method at matched end-to-end compute.
- Small-model NCM occupies a favorable quality-versus-compute point relative to at least one appropriately chosen autoregressive baseline.

**Mixed result**
- Oracle@K improves, but selectors capture little of the potential gain.
- Consensus helps translation or summarization but not objectively checkable reasoning.
- Iteration helps quality but is less cost-effective than more sampling or a larger model.

**Negative result**
- Candidate quality saturates quickly.
- Incorrect modes dominate consensus.
- Verifier/revision cost erases the sampling advantage.
- Increasing model capacity is consistently more cost-effective.

Negative results should be recorded, not discarded.

## 6. Resource strategy

Start on the **RTX 3090 with 24 GB VRAM**. Do not assume extra GPUs, rented capacity or training from scratch.

| Milestone | Resource decision |
|---|---|
| Baselines | Current GPU |
| Sampling and MBR | Current GPU |
| Verification / iteration | Current GPU first; profile bottlenecks |
| External semantic judging | Estimate API spend; separately approve a cap before any calls |
| Cross-model comparisons | Additional compute only if justified |
| New general-purpose training | Separate proposal and explicit approval |

Track peak VRAM, runtime, disk use, utilization and model-loading overhead before forecasting later-phase costs.

## 7. Reporting and decision framework

Each phase should end with:
1. Hypothesis, dataset, checkpoints and controlled settings.
2. Quality-versus-compute results with uncertainty.
3. Comparisons to the defined baselines.
4. Failure analysis and unexpected observations.
5. A **continue / modify / stop** recommendation.

**No automatic progression.** Only Phase 0 should be considered for initial approval.

## 8. Final research question

> How much model capacity can we replace with efficient generation, consensus, verification, and iteration before the economics favor a larger model?

The first milestone is a **credible and reproducible measurement of small-model test-time scaling**, not a general-purpose chat assistant.

## References

- Official NCM repository: https://github.com/HazyResearch/ncm
- Official checkpoints: https://huggingface.co/hazyresearch/ncm-checkpoints
- Proposed semantic evaluation protocol: [semantic-evaluation.md](semantic-evaluation.md)

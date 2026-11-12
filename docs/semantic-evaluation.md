# Semantic Correctness Evaluation (Jev)

**Status:** Proposed; no implementation or API calls authorized
**Scope:** Offline evaluation of Phase 1 candidate pools; potential Phase 2 verifier and later training signal
**Related:** [Experiment plan](experiment-plan.md) · [Phases 0–1 implementation plan](implementation-phases-0-1.md)

## Research question

Does reference-based scoring undercount valid NCM outputs, and can a lightweight semantic judge reliably identify correct alternatives without inflating success as the candidate pool grows?

**Hypothesis H4:** Reference-aware semantic judging identifies genuinely correct candidate answers missed by lexical metrics. Those gains survive independent validation and remain meaningful at larger K.

Jev (TypeSafe AI) is a proposed judge, **not an assumed ground-truth oracle**. We must measure its errors before treating its outputs as correctness labels.

## Definitions: do not collapse these concepts

- **Reference equivalence:** The candidate expresses the same substantive meaning as a supplied reference. This does not alone guarantee faithfulness if the reference itself is problematic.
- **Task correctness:** The candidate satisfies the original task and is supported by the original input. A valid answer can differ substantively from the single reference.
- **Material error:** The candidate contains contradictions, unsupported facts, missing critical details, changed numbers, altered entities, or negation/causal reversals relevant to the task.
- **Verified correctness:** Deterministically checked or independently human-adjudicated result. This is distinct from Jev's prediction.

An alternate yet correct translation may pass task correctness without matching the reference exactly. A plausible but unsupported summary must fail task correctness even if it sounds fluent.

## Evaluation inputs and questions

For each saved NCM candidate, supply Jev with structured state:

- Task type and original instruction/constraints;
- Original source (article, source-language sentence, etc.);
- Reference answer (for offline, reference-aware evaluation only);
- Candidate prediction.

Evaluate three explicit **Noul** (binary-probability) questions:

1. **Equivalence:** Does the candidate preserve the reference's materially relevant meaning?
2. **Task validity:** Is the candidate a correct and adequate response to the original source/task, including legitimate alternatives?
3. **Material error:** Does the candidate introduce a substantive factual, logical, or constraint violation?

Use clearly specified task rubrics. Record raw yes probabilities, not only a pass/fail label. Compare three questions in one request against separate questions on a calibration subset, in case combined questions influence one another. Pin the **resolved Jev model version**, API/question schema and prompt/rubric hashes. Do not assume raw probabilities are calibrated.

**Illustrative policy (not thresholds):** pass only when task-validity probability exceeds a development-calibrated threshold and material-error probability is below its calibrated threshold. An indeterminate band triggers independent review; do not score it automatically as correct.

## Task-specific use

| Task | Primary correctness authority | What Jev might add |
|---|---|---|
| Sudoku | Deterministic checker for solved grid, all rules and original clues | None needed for Phase 1 |
| XSum | Source-grounded factuality and summary adequacy | Accept valid alternate summaries; detect unsupported details/critical omissions |
| WMT14 De→En | Fidelity to source and task constraints | Accept legitimate paraphrases of the translation; flag changed negation/entities/quantities |

Keep ROUGE, sacreBLEU and chrF as **separate reference metrics**, not semantic truth labels. Do not feed a held-out test reference into generation or a deployable selector.

## Calibration before interpreting Phase 1 scores

On a development/calibration set distinct from the held-out NCM evaluation inputs:

1. Build independently labeled pairs with exact matches, paraphrases, legitimate alternative answers, omissions, hallucinations, contradictions, negation shifts, number/entity swaps and causal reversals. Use some naturally occurring NCM outputs, not just synthetic examples.
2. Label reference equivalence **separately** from task correctness, and establish a review/adjudication policy for disputed cases.
3. Pilot Jev prompts/rubrics and measure discrimination, false-positive and false-negative rates, probability calibration and error categories.
4. Choose thresholds and uncertain/review handling on development data only; freeze prompts, model version and thresholds before held-out scoring.
5. Audit independently sampled held-out candidates, emphasizing Jev passes that disagree with lexical metrics and at least one set of Jev failures. Report estimates with uncertainty, rather than treating Jev as exact.

### False-positive amplification at K

If each wrong candidate has a 1% *independent* chance of being falsely accepted, an all-wrong pool appears to contain at least one passing candidate with probability

\[
P(\text{at least one false pass}) = 1 - (1 - 0.01)^K.
\]

Illustrative results: approximately 15% (K=16), 47% (K=64), and 92% (K=256).

The independence assumption is only an illustration; correlated predictions may change the actual rates. Audit **pool-level** false passes and report how estimated success changes with threshold and K. A monotone Jev-pass curve alone is not evidence of improving correctness.

## Phase 0 change: preserve the ability to evaluate offline

Do not change official model weights, generation defaults, or upstream metrics. Save and link separately:

- Stable input/candidate IDs, task type, full source/context, instruction/constraints and source/reference identifiers;
- Generated candidates, generation provenance and versioned immutable run manifests;
- References stored outside the candidate-generation inputs, with explicit read access limited to offline scoring.

Do not commit raw held-out sources, references, judge inputs, or predictions to GitHub. Phase 0 requires **schema readiness only**, not a Jev integration.

## Phase 1 change: add an offline semantic-evaluation work package

Keep original stochastic sampling, frozen checkpoints, K values (1, 4, 16, 64, and optional 256), nested candidate prefixes, and conventional metrics.

After saving candidate pools:

1. Run calibration and freeze the Jev protocol.
2. Score saved XSum and WMT14 candidates offline; use cached response records to prevent duplicate calls and document failures/retries.
3. Calculate reference-metric Oracle@K, **Jev-estimated semantic pass-within-K / Oracle@K**, and audited verified success-within-K **as distinct curves**.
4. Compare divergence between the curves and examine high-value disagreements.
5. Report total judge tokens, requests, latency, dollar cost, uncertainty and sensitivity to judge false positives/thresholds.

Score a representative subset first. Expand to full pools only after the calibration gate and a **separately approved evaluation spend cap**. If the judge is not adequately calibrated, report inconclusive semantic results rather than inventing a pass rate.

### Proposed stored judgment record

One record per candidate and pinned judge protocol, linked to the immutable generation artifact:

- \`candidate_id\`, \`task\`, \`input_id\`, \`judge_model_resolved\`, \`rubric_hash\`;
- \`p_equivalent\`, \`p_task_valid\`, \`p_material_error\`, \`decision\` (pass/fail/uncertain);
- request/token usage, latency, failure/retry status, and adjudication reference when available.

Never overwrite raw model outputs or conventional metric records with judge decisions.

## Phase 2 and beyond

- **Phase 2:** Evaluate Jev as one potential **reference-free deployable selector**, supplying only the original task/source and a candidate. Reference-aware offline scores are not available to a deployed selector. Compare selection performance against independent truth and account for judge cost.
- **Phase 3:** If useful, consider using validator feedback in iterative revision, contingent on the released checkpoint's ability to consume feedback; do not treat resampling as revision.
- **Future training:** Jev judgments might become preference/reward/filtered-supervision signals. Jev outputs do not directly backpropagate into NCM; optimization is a separate design and approval decision. Guard against reward hacking, judge overfitting, and feedback contamination.

## Reporting and stop criteria

Publish separate conventional, Jev-estimated and independently verified measures where the sample size permits, including confidence intervals and example-level failure analysis.

**Proceed** if semantic judging reliably identifies legitimate alternatives and the gain survives independent audits at practical cost.

**Modify** if it helps some tasks but exhibits predictable weaknesses (e.g., negation or source-grounding).

**Stop** treating Jev as a correctness oracle if false approvals or uncertainty make Oracle@K misleading.

**No code, model training, external spend, or subsequent phase is authorized by this document.**

## References

- TypeSafe AI Jev: https://docs.typesafe.ai/
- Official Jev API: https://api.typesafe.ai/v1/systemone
- NCM project experiment plan: [experiment-plan.md](experiment-plan.md)

# NCM: Small-Model Test-Time Scaling

An independent research project exploring whether **Next-Concept Modeling (NCM)** can trade inference-time computation for model size through candidate generation, Minimum Bayes Risk (MBR) selection, verification, and iterative refinement.

> **Status:** P0-A, the P0.3 baseline harness/schema, and P0.4 independent metric validation are complete. Official XSum and Sudoku-easy smoke tests ran on 10 and 50 inputs at batch size 4; independent scoring matched the saved 10-example upstream runs. Measured Phase 0 baselines remain. This repository is not affiliated with the original NCM researchers.

Local setup and hardware notes: [environment](docs/environment.md).

## Research question

**How much model capacity can we replace with efficient sampling, consensus, verification, and iteration before a larger model becomes the better investment?**

Instead of immediately training a large general-purpose NCM, the project starts with pretrained, task-specific checkpoints and measures what additional inference can actually buy.

## Plan

Read the [full experiment plan](docs/experiment-plan.md).

Implementation planning: [Phases 0–1 engineering plan](docs/implementation-phases-0-1.md).

Semantic correctness: [Jev evaluation design and safeguards](docs/semantic-evaluation.md).

| Phase | Focus | Decision gate |
|---|---|---|
| 0 | Reproduce and instrument official checkpoints | Confirm reliable local baselines |
| 1 | Candidate scaling + calibrated semantic evaluation (K = 1, 4, 16, 64, 256) | Does independently supported Oracle@K improve? |
| 2 | MBR, diversity-aware consensus, verification | Can selectors capture the improvement? |
| 3 | Iterative consensus / revision | Does iteration beat one-shot methods at equal compute? |
| 4 | Compute allocation and autoregressive baselines | Is there a meaningful quality-per-compute advantage? |
| 5 | General-purpose capability | Only if previous evidence justifies adaptation/training |

**Initial hardware:** RTX 3090 (24 GB VRAM). **Initial policy:** Frozen checkpoints, no training from scratch. Each phase requires review and explicit approval before moving on.

## Working principles

- Separate **candidate availability** (Oracle@K) from **candidate selection**.
- Compare at matched end-to-end compute and wall-clock cost.
- Evaluate with independent ground truth or task metrics, not consensus alone.
- Distinguish lexical/reference similarity, Jev-estimated semantic correctness, and independently verified correctness; calibrate judges before trusting pass rates.
- Regard convergence and correctness as different phenomena.
- Preserve negative results and report uncertainty.
- Do not implement later phases speculatively.

## References

- [NCM research repository](https://github.com/HazyResearch/ncm)
- [NCM pretrained checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints)

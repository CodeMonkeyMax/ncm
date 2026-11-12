# Local environment (P0-A)

Phase 0-A pins the official NCM checkout and the XSum / Sudoku-easy artifacts. Weights, datasets, the virtualenv, and raw runs stay in ignored directories.

## Hardware

| Planned | Observed on 2026-11-11 |
|---|---|
| RTX 3090, 24 GB | NVIDIA GeForce RTX 3080, 10240 MiB, driver 610.47 |

An earlier probe saw only about 173 MiB free because Ollama (`llama-server.exe`) was resident. The 2026-11-12 CUDA probe saw about 9097 MiB free. Stop Ollama before any smoke test if it is running again. A 10 GB card can still OOM at the official batch sizes (`xsum` 32, `sudoku-easy` 128); Phase 0 smoke tests should pass a smaller `--bs`.

WSL2 Ubuntu 24.04 is installed but stopped. This pin uses native Windows Python so it can see the NVIDIA driver directly. A WSL2 environment is acceptable later if `nvidia-smi` works inside it.

## Pins

| Item | Value |
|---|---|
| Upstream | [HazyResearch/ncm](https://github.com/HazyResearch/ncm) `c70274273f489abc73832b2b98584a8483b6b203` |
| Checkpoints | `hazyresearch/ncm-checkpoints` @ `d342ce8aaea58830eef3b8ad473664673e41f692` |
| Datasets | `hazyresearch/ncm-datasets` @ `6580ba41a4b0a976dde876fbe84d25327617b0e6` |
| Python | 3.12.10 venv at `.local/venv` |
| Torch | `2.15.0+cu132` (CUDA available on the RTX 3080; PyPI's default `torch` wheel is CPU-only) |
| Lockfile | [configs/phase0/requirements-lock.txt](../configs/phase0/requirements-lock.txt) |
| Allowlist | [configs/phase0/artifacts.json](../configs/phase0/artifacts.json) |

Official files are untrusted external inputs. Upstream evaluation uses `torch.load(..., weights_only=False)` and pickle. Download only the pinned official revisions and check hashes with `scripts/verify_p0a.py` before loading them.

## Directory map

| Role | Path |
|---|---|
| Upstream code (ignored) | `.local/upstream/ncm` |
| XSum checkpoint | `.local/artifacts/checkpoints/xsum` |
| Sudoku easy checkpoint | `.local/artifacts/checkpoints/sudoku/easy` |
| XSum eval inputs | `.local/artifacts/datasets/xsum` |
| Sudoku easy eval inputs | `.local/artifacts/datasets/sudoku/easy` |
| Raw run outputs (ignored) | `runs/` |

Do not assume a `sudoku-easy/` storage folder. The config name is `sudoku-easy`; the artifact folders are `sudoku/easy`.

The XSum eval config also names `validation_cond512_le48.pickle`. That file is not in the released dataset. The test split uses `passages.pickle` and `reference_summaries.pickle` only.

## Setup

From the repository root, with Python 3.12:

```powershell
py -3.12 -m venv .local\venv
.\.local\venv\Scripts\python -m pip install --upgrade pip
.\.local\venv\Scripts\python -m pip install -r .local\upstream\ncm\requirements.txt
.\.local\venv\Scripts\python -m pip install --force-reinstall "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu132
.\.local\venv\Scripts\python scripts\download_p0a.py
.\.local\venv\Scripts\python scripts\verify_p0a.py
```

The second `pip install` is required. Upstream `requirements.txt` only says `torch`, and the default Windows wheel is `2.15.0+cpu`. The CUDA 13.2 wheel matches driver 610.47.

`download_p0a.py` fetches only the allowlisted XSum and Sudoku-easy files (about 840 MB). It does not download WMT14 or the rest of the 8 GB / 23 GB repositories. Checkpoints use Hugging Face `repo_type=model`; datasets use `repo_type=dataset`.

## Verified

On 2026-11-12, `scripts/verify_p0a.py` passed for every allowlisted file. LFS weights and pickles matched the recorded sha256. The two `config.yaml` files matched size and git blob OID.

`torch.cuda.is_available()` was true (`2.15.0+cu132`, device RTX 3080). No model was loaded.

## Official smoke tests

On 2026-11-12 UTC, upstream `evaluate.py` completed for both tasks using `transformers==5.22.2`, one GPU, the official test split and default generation settings, and `--bs 4`. The pinned files were verified immediately before the runs. CUDA reported 9097 MiB free before loading; this is not a peak-memory measurement. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set.

| Task | Inputs | Official scores |
|---|---:|---|
| XSum | 10 | ROUGE-1/2/L: 30.08 / 8.40 / 24.28; genPPL: 104.92 |
| XSum | 50 | ROUGE-1/2/L: 32.76 / 12.16 / 26.48; genPPL: 91.40 |
| Sudoku easy | 10 | 10/10 solved; 100% cell agreement; 100% valid |
| Sudoku easy | 50 | 49/50 solved; 99.83% cell agreement; 98% valid |

Outputs and score logs are in ignored `runs/p0-{task}-n{10,50}-bs4/` directories. All four shards have the expected ordered indices, equal input/hypothesis/reference counts, and nonempty string hypotheses and references; all Sudoku hypotheses contain 81 cells. These small smoke results are not paper-level reproduction. XSum scoring also loads GPT-2 from Hugging Face; its model revision was not captured.

Both 50-input tests were subsequently rerun offline at batch size 4. Every saved index, hypothesis, and reference matched the earlier run exactly. XSum used the cached GPT-2 `main` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The measured run directories include score logs, raw shards, partial run records, and 200 ms `nvidia-smi` samples:

| Task | First sampled device usage | Maximum sampled device usage |
|---|---:|---:|
| XSum | 1561 MiB | 3164 MiB |
| Sudoku easy | 1644 MiB | 2084 MiB |

These are whole-device measurements including other GPU processes, not exact per-process peaks; transient peaks can fall between samples. Larger batch sizes were not tested. The records disclose missing synchronized stage timings, per-candidate timing, seed schedules, and standardized candidate/context records. P0.2 is not fully accepted until complete run manifests are archived.

## Baseline adapter

The Phase 0 adapter runs the pinned upstream shard and scorer as separate processes without changing upstream source. From the repository root, using a fresh run ID:

```powershell
.\.local\venv\Scripts\python.exe scripts\baseline.py --task xsum --n 10 --bs 4 --run-id xsum-smoke
.\.local\venv\Scripts\python.exe scripts\baseline.py --task sudoku-easy --n 10 --bs 4 --run-id sudoku-smoke
.\.local\venv\Scripts\python.exe scripts\rescore_baseline.py --run-id xsum-smoke --output-id xsum-rescore
```

Runs require verified pinned artifacts, a clean pinned upstream checkout, CUDA, and a pre-existing Hugging Face GPT-2 cache for XSum. The adapter runs offline and records the cached GPT-2 revision. It refuses to overwrite an existing run. Each ignored run directory retains generation and scoring logs, a versioned manifest, a write-once run specification, and separate candidate, context, and reference JSONL files. Schema version 2 assigns stable `example_id` and `candidate_id` values and writes `input-order.jsonl`, which maps each split-local input ID to its upstream index and shard position. Contexts preserve the original source token IDs.

The upstream shard script expects a reference file even during generation. The adapter stages the requested inputs with empty reference placeholders and passes that directory to the generation worker. Its `generation_shard.pkl` contains placeholders rather than actual references; the adapter attaches actual references afterward to create `shard_0.pkl` for the upstream scorer. Sudoku references are decoded with the upstream tokenizer. This prevents normal upstream generation code from reading actual references, but it is not an OS-level sandbox: the worker still has the user's filesystem permissions. The scoring shard and reference JSONL contain answers and remain ignored.

The timing worker synchronizes CUDA around the upstream batch function and the nested `NextConceptTransformer.generate` call. The full batch measurement includes VQ decoding, token selection, and text decoding, but not prefix preparation; warm totals exclude the first output-producing batch. It also records cold phase time from upstream entry through model loading, generation, and shard write, plus elapsed time until the first batch. Python/CUDA worker startup is excluded. Batch timings are shared by all candidates in the batch, not individual latency measurements; individual generation durations are therefore null in the normalized records. Peak allocated and reserved memory are PyTorch process measurements per phase, not whole-device peaks. The scorer loads GPT-2 separately for XSum.

On the 10-input validation runs using staged placeholders, indices, hypotheses, and subsequently attached references exactly matched the earlier official shards, and offline rescoring reproduced the printed official scores for both tasks.

New runs save a write-once `run-spec.json` and a generation checkpoint with hashes of the normalized records, input-order map, and raw shards. The run spec records the repository HEAD, a digest of the tracked diff, hashes of untracked non-ignored files, input-data hashes, environment lock, device properties, and adapter/profiler hashes. Its digest is copied into the mutable status manifest and validated during rescoring. If the run is interrupted after that checkpoint, or scoring fails, use `rescore_baseline.py` with the original run ID and a fresh output ID to validate the checkpoint and retry scoring without overwriting the original run. Stop the original process first to avoid competing GPU workloads. An interrupted manifest can still say `running`; the separate rescore record describes the retry.

The instrumented 10-input runs again matched the official IDs, hypotheses, references, and scores for both tasks. An injected XSum scoring failure after real generation and checkpointing was successfully retried offline without regenerating candidates. A forced process interruption was not verified.

**Remaining limitations:** Reference isolation is by staged data path, not filesystem access control. Per-candidate timings and metric fields are not measured; batching is timed as a whole, and independent metrics are P0.4. The upstream K=1 worker writes its shard only after all generation batches complete, so P0.3 can replay completed raw results and retry scoring but cannot resume an interrupted generation pass. Checkpointing partial multi-candidate pools belongs with P1 sampling; it must be implemented before launching large candidate sweeps.

## Not started

- WMT14 artifacts
- Independent metrics and Phase 1 sampling

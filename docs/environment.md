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

Runs require verified pinned artifacts, a clean pinned upstream checkout, CUDA, and a pre-existing Hugging Face GPT-2 cache for XSum. The adapter runs offline and records the cached GPT-2 revision. It refuses to overwrite an existing run. Each ignored run directory retains the upstream shard, generation and scoring logs, a manifest, and separate candidate, context, and reference JSONL files. Contexts preserve the original source token IDs; references also remain in the upstream shard.

The timing worker synchronizes CUDA around `NextConceptTransformer.generate`, excluding the first output-producing batch from warm generation timing. That measurement covers the transformer call only, not token decoding; the generation phase timing covers upstream model loading and the complete shard. Peak allocated and reserved memory are PyTorch process measurements per phase, not whole-device peaks. The scorer loads GPT-2 separately for XSum.

On the 10-input validation runs, normalized IDs, hypotheses, and references exactly matched the earlier official shards, and offline rescoring reproduced the printed official scores for both tasks. Five normalization tests passed.

**Remaining limitations:** The unchanged upstream generation script loads references during generation, so reference isolation is not yet enforced. Per-candidate timings and metric fields are not measured; batching is timed as a whole. Incomplete runs are preserved but cannot be resumed. Manifests flag a dirty worktree but do not capture its full contents, and per-stage timing excludes Python/CUDA worker startup. Independent metrics, complete immutable manifests, and the planned adapter acceptance therefore remain outstanding.

## Not started

- WMT14 artifacts
- Independent metrics and Phase 1 sampling

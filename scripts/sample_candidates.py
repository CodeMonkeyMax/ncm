"""Generate a reproducible candidate pool with one independent RNG stream per candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".local" / "upstream" / "ncm"
ARTIFACTS = ROOT / "configs" / "phase0" / "artifacts.json"
CHECKPOINTS = ROOT / ".local" / "artifacts" / "checkpoints"
RUNS = ROOT / "runs"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def candidate_seed(base_seed: int, task: str, input_id: int,
                   replicate_id: int, candidate_rank: int) -> int:
    payload = json.dumps(
        [base_seed, task, input_id, replicate_id, candidate_rank],
        separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no records")
    return rows


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to overwrite changed output: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def run_id_valid(run_id: str) -> bool:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return bool(run_id) and all(char in allowed for char in run_id)


def load_input_rows(path: Path, task: str, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for input_id, row in enumerate(rows):
        if row.get("task") != task or row.get("split") != split:
            raise ValueError(f"all candidate inputs must match the requested {task} {split} split")
        if row.get("input_id") != input_id:
            raise ValueError(f"{split} input IDs must be contiguous and in file order")
        source = row.get("source")
        if not isinstance(source, (str, list)) or not source:
            raise ValueError(f"{split} input {input_id} has no source")
        if isinstance(source, str) and not source.strip():
            raise ValueError(f"{split} input {input_id} has an empty source")
        if isinstance(source, list) and any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in source
        ):
            raise ValueError(f"{split} input {input_id} has invalid source token IDs")
    return rows


def load_reference_rows(
    path: Path, task: str, count: int, split: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != count:
        raise ValueError(f"expected {count} {split} references, found {len(rows)}")
    for input_id, row in enumerate(rows):
        example_id = f"{task}:{split}:{input_id}"
        if (row.get("task") != task or row.get("split") != split
                or row.get("input_id") != input_id or row.get("example_id") != example_id
                or not isinstance(row.get("text"), str) or not row["text"].strip()):
            raise ValueError(f"invalid or mismatched {split} reference {input_id}")
    return rows


def validate_candidate_chunk(path: Path, task: str, input_id: int, k: int,
                             replicate: int, base_seed: int,
                             greedy: bool, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != k:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {k}")
    for rank, row in enumerate(rows):
        if (row.get("task") != task or row.get("split") != split
                or row.get("input_id") != input_id
                or row.get("candidate_rank") != rank
                or row.get("candidate_id") != f"{task}:{split}:{input_id}:candidate:{rank}"
                or row.get("seed") != (
                    None if greedy else candidate_seed(base_seed, task, input_id, replicate, rank)
                )):
            raise ValueError(f"{path.name} has invalid candidate identity or seed at rank {rank}")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"{path.name} candidate {rank} has no text field")
        generation_error = row.get("generation_error")
        if generation_error is not None and (
            not isinstance(generation_error, str) or not generation_error.strip()
        ):
            raise ValueError(f"{path.name} candidate {rank} has an invalid generation error")
        if not row["text"].strip() and generation_error is None:
            raise ValueError(f"{path.name} empty candidate {rank} has no recorded failure")
    return rows


def context_for(task: str, input_row: dict[str, Any], split: str) -> dict[str, Any]:
    input_id = input_row["input_id"]
    context: dict[str, Any] = {
        "task": task,
        "input_id": input_id,
        "example_id": f"{task}:{split}:{input_id}",
        "constraints": "preserve clues and solve 9x9 grid" if task == "sudoku-easy" else
        "translate the source sentence" if task == "wmt14" else
        "summarize the source article",
    }
    if isinstance(input_row["source"], str):
        context["source_text"] = input_row["source"]
    else:
        context["source_token_ids"] = input_row["source"]
    return context


def prepare_model(config: Any, checkpoint: Path, manifest: dict[str, Any]):
    import torch

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))

    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer

    config.tokenizer.vqvae.checkpoint_path = str(checkpoint / config.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if config.checkpoints.get(name):
            config.tokenizer.pre_tokenizer[key] = str(checkpoint / config.checkpoints[name])
    tokenizer = VQVAETokenizer.from_config(config.tokenizer)
    tokenizer.vqvae = tokenizer.vqvae.cuda().eval()
    model = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=config.model).cuda()
    state = torch.load(checkpoint / config.checkpoints.generator, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    pre_tokenizer = None
    pre_tokenizer_model = None
    if config.tokenizer.pre_tokenizer.name == "hf":
        model_id = str(config.tokenizer.pre_tokenizer.model_id)
        if model_id == "gpt2":
            dev_spec = next(item for item in manifest["development_datasets"] if item["task"] == "xsum")
            revision = dev_spec["tokenizer"]["revision"]
        else:
            spec = manifest["pre_tokenizers"]
            if model_id != spec["repo"]:
                raise ValueError(f"unlisted HF pre-tokenizer model: {model_id}")
            revision = spec["revision"]
            tokenizer_path = ROOT / spec["local_dir"]
        if model_id == "gpt2":
            tokenizer_path = snapshot_download(
                model_id, revision=revision, local_files_only=True
            )
        pre_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        pre_tokenizer_model = {"id": model_id, "revision": revision}

    generation = config.generation
    allowance = None
    allowance_spec = generation.get("frequency_allowance")
    if allowance_spec:
        path, reference = str(allowance_spec).split(":", 1)
        allowance = torch.load(checkpoint / path, map_location="cuda", weights_only=False)
        for key in reference.split("."):
            allowance = allowance[key]
        allowance = allowance.double()
    return torch, model, tokenizer, pre_tokenizer, pre_tokenizer_model, allowance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("xsum", "sudoku-easy", "wmt14"), required=True)
    parser.add_argument("--inputs-jsonl", type=Path, required=True)
    parser.add_argument("--references-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--k", type=int, choices=(1, 4, 16, 64, 256), required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy-baseline", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not run_id_valid(args.run_id):
        parser.error("--run-id may contain only letters, digits, hyphens and underscores")
    if args.base_seed < 0 or args.replicate < 0 or args.gpu < 0:
        parser.error("--base-seed, --replicate, and --gpu must be nonnegative")
    supplied_sampling = (args.temperature, args.top_k, args.top_p)
    if args.greedy_baseline:
        if args.k != 1 or any(value is not None for value in supplied_sampling):
            parser.error("--greedy-baseline requires K=1 and no explicit sampling parameters")
    else:
        if any(value is None for value in supplied_sampling):
            parser.error("stochastic runs require --temperature, --top-k, and --top-p")
        if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0 or args.top_k == 1:
            parser.error("sampling requires temperature > 0, 0 < top-p <= 1, and top-k == 0 or > 1")
        if args.top_k == 0 and args.top_p <= 0:
            parser.error("top-k 0 requires positive top-p to avoid greedy output")
    inputs_path = args.inputs_jsonl.resolve()
    references_path = args.references_jsonl.resolve()
    if not inputs_path.is_file() or not references_path.is_file():
        parser.error("input and reference JSONL files must exist")
    dataset_manifest_path = inputs_path.parent / "dataset-manifest.json"
    if not dataset_manifest_path.is_file():
        parser.error("inputs must be accompanied by dataset-manifest.json")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if (dataset_manifest.get("task") != args.task
            or dataset_manifest.get("split") != args.split):
        parser.error(f"dataset-manifest.json does not describe the requested {args.split} task")
    if digest(inputs_path) != dataset_manifest.get("input_sha256"):
        raise ValueError(f"prepared {args.split} inputs do not match dataset-manifest.json")

    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    from verify_p0a import main as verify_artifacts

    manifest = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip() != manifest["upstream"]["commit"]:
        raise RuntimeError("upstream commit does not match artifact manifest")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=UPSTREAM, text=True).strip():
        raise RuntimeError("upstream checkout is modified")
    if verify_artifacts():
        raise RuntimeError("artifact verification failed")

    config_path = UPSTREAM / "config" / "eval" / f"{args.task}.yaml"
    config = OmegaConf.load(config_path)
    temperature = float(config.generation.temperature) if args.greedy_baseline else args.temperature
    top_k = int(config.generation.top_k) if args.greedy_baseline else args.top_k
    top_p = float(config.generation.top_p) if args.greedy_baseline else args.top_p
    vocabulary_limit = min(int(size) for size in config.model.vocab_size)
    if top_k > 0 and top_k > vocabulary_limit:
        parser.error(f"--top-k cannot exceed this task's smallest codebook ({vocabulary_limit})")
    input_rows = load_input_rows(inputs_path, args.task, args.split)
    if dataset_manifest.get("count") != len(input_rows):
        raise ValueError(f"prepared {args.split} input count does not match dataset-manifest.json")
    output = RUNS / args.run_id
    chunks = output / "checkpoint" / "candidate_chunks"
    run_spec = {
        "schema_version": 1,
        "run_id": args.run_id,
        "task": args.task,
        "split": args.split,
        "upstream_commit": manifest["upstream"]["commit"],
        "config_sha256": digest(config_path),
        "generator_sha256": digest(CHECKPOINTS / args.task / "generator.pt")
        if args.task != "sudoku-easy" else
        digest(CHECKPOINTS / "sudoku" / "easy" / "generator.pt"),
        "inputs_path": str(inputs_path),
        "inputs_sha256": digest(inputs_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": digest(dataset_manifest_path),
        "input_ids": [row["input_id"] for row in input_rows],
        "k": args.k,
        "base_seed": args.base_seed,
        "replicate": args.replicate,
        "greedy_baseline": args.greedy_baseline,
        "generation": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_scale": float(config.generation.get("cfg_scale", 0.0)),
            "frequency_penalty": float(config.generation.get("frequency_penalty", 0.0)),
            "frequency_deficit": float(config.generation.get("frequency_deficit", 0.0)),
            "adjacency_penalty": str(config.generation.get("adjacency_penalty", "") or ""),
        },
        "rng_schedule": None if args.greedy_baseline else
        "sha256(base_seed, task, input_id, replicate, candidate_rank); one input per generate call",
        "batch_size": 1,
        "references_path": str(references_path),
        "adapter_sha256": digest(Path(__file__)),
        "artifact_manifest_sha256": digest(ARTIFACTS),
    }
    run_spec_path = output / "run-spec.json"
    if args.resume:
        if not output.is_dir() or not run_spec_path.is_file():
            parser.error("--resume requires an existing run directory and run-spec.json")
        previous_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if previous_spec != run_spec:
            raise ValueError("resume specification differs from the original run")
        previous_manifest = output / "manifest.json"
        if previous_manifest.is_file() and json.loads(
            previous_manifest.read_text(encoding="utf-8")
        ).get("status") == "succeeded":
            raise ValueError("cannot resume an already completed candidate run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        chunks.mkdir(parents=True)
        write_json(run_spec_path, run_spec)
    chunks.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        **run_spec,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "exception": None,
        "candidate_errors": 0,
        "completed_inputs": 0,
    }
    write_json(output / "manifest.json", record)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        checkpoint_rel = {
            "xsum": Path("xsum"),
            "sudoku-easy": Path("sudoku") / "easy",
            "wmt14": Path("wmt14"),
        }[args.task]
        model_load_started = datetime.now(timezone.utc).isoformat()
        model_load_timer = time.perf_counter()
        torch, model, tokenizer, pre_tokenizer, pre_tokenizer_model, allowance = prepare_model(
            config, CHECKPOINTS / checkpoint_rel, manifest,
        )
        model_load_seconds = time.perf_counter() - model_load_timer
        config.generation.temperature = temperature
        config.generation.top_k = top_k
        config.generation.top_p = top_p
        prefix_len = int(config.model.prefix_len)
        eos = tokenizer.pre_tokenizer.eos_token_id
        pad = tokenizer.pre_tokenizer.pad_token_id
        cut_at_eos = bool(config.generation.get("cut_at_eos", True))
        adjacency = config.generation.get("adjacency_penalty")
        adjacency = [float(value) for value in str(adjacency).split(",")] if adjacency else None
        candidate_timing: dict[str, float] = {}
        device_info = subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], text=True
        ).strip().split(", ")
        if len(device_info) != 3:
            raise RuntimeError(f"unexpected nvidia-smi response: {device_info!r}")

        def prefix_for(row: dict[str, Any]):
            source = row["source"]
            ids = pre_tokenizer(source)["input_ids"] if isinstance(source, str) else list(source)
            return (ids[:prefix_len] + [pad] * prefix_len)[:prefix_len]

        def decode(ids: list[int]) -> str:
            tokens = list(ids)
            if cut_at_eos and eos is not None and eos in tokens:
                tokens = tokens[:tokens.index(eos)]
            return tokenizer.pre_tokenizer.decode(
                [token for token in tokens if token not in (eos, pad)],
                skip_special_tokens=True,
            )

        def pick_tokens(logits):
            frequency_penalty = float(config.generation.get("frequency_penalty", 0.0))
            frequency_deficit = float(config.generation.get("frequency_deficit", 0.0))
            if frequency_penalty <= 0 and not adjacency:
                return logits.argmax(-1)
            batch, steps, vocab = logits.shape
            adjusted = logits.double()
            allowed = (allowance[:vocab] * steps)[None, :] if allowance is not None else None
            adjacency_strength = adjacency[0] if adjacency else 0.0
            adjacency_window = int(adjacency[1]) if adjacency and len(adjacency) > 1 else 1
            adjacency_decay = adjacency[2] if adjacency and len(adjacency) > 2 else 1.0
            counts = torch.zeros(batch, vocab, dtype=torch.float64, device=adjusted.device)
            output_tokens = torch.empty(batch, steps, dtype=torch.long, device=adjusted.device)
            history = []
            for step_index in range(steps):
                step_logits = adjusted[:, step_index, :].clone()
                if frequency_penalty > 0:
                    excess = (counts - allowed).clamp_min(0.0) if allowed is not None else counts
                    step_logits -= frequency_penalty * excess
                    if frequency_deficit > 0 and allowed is not None:
                        step_logits += frequency_deficit * ((steps - step_index) / steps) * (
                            allowed - counts
                        ).clamp_min(0.0)
                for index, previous in enumerate(reversed(history[-adjacency_window:])):
                    step_logits.scatter_add_(
                        1, previous[:, None],
                        torch.full((batch, 1), -adjacency_strength * adjacency_decay ** index,
                                   dtype=torch.float64, device=adjusted.device),
                    )
                chosen = step_logits.argmax(-1)
                output_tokens[:, step_index] = chosen
                history.append(chosen)
                counts.scatter_add_(1, chosen[:, None], torch.ones(
                    batch, 1, dtype=torch.float64, device=adjusted.device
                ))
            return output_tokens

        def generate_one(row: dict[str, Any], seed: int | None) -> str:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            source_prefix = torch.tensor([prefix_for(row)], device="cuda")
            with torch.no_grad():
                generated = model.generate(
                    None, source_prefix,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    cfg_scale=float(config.generation.get("cfg_scale", 0.0)),
                    cfg_null_eos=eos,
                )
                _, level_z = tokenizer.vqvae.vq.idx_to_zhats(generated, return_zlevels=True)
                token_ids = pick_tokens(tokenizer.vqvae.decoder(level_z[-1])).tolist()
                return decode(token_ids[0])

        torch.cuda.synchronize()
        model_ready_utc = datetime.now(timezone.utc).isoformat()
        for input_row in input_rows:
            input_id = input_row["input_id"]
            chunk_path = chunks / f"input-{input_id:06d}.jsonl"
            if chunk_path.exists():
                validate_candidate_chunk(
                    chunk_path, args.task, input_id, args.k, args.replicate,
                    args.base_seed, args.greedy_baseline, args.split,
                )
                record["completed_inputs"] = input_id + 1
                continue
            rows = []
            for rank in range(args.k):
                seed = None if args.greedy_baseline else candidate_seed(
                    args.base_seed, args.task, input_id, args.replicate, rank
                )
                started = time.perf_counter()
                generation_error = None
                text = ""
                try:
                    torch.cuda.synchronize()
                    text = generate_one(input_row, seed)
                    torch.cuda.synchronize()
                except Exception as exc:
                    generation_error = f"{type(exc).__name__}: {exc}"
                    record["candidate_errors"] += 1
                    print(
                        f"candidate error {args.task}:{args.split}:{input_id}:{rank}: "
                        f"{generation_error}", file=sys.stderr, flush=True,
                    )
                if not text.strip() and generation_error is None:
                    generation_error = "EmptyGeneration: model produced no text"
                    record["candidate_errors"] += 1
                    print(
                        f"candidate error {args.task}:{args.split}:{input_id}:{rank}: "
                        f"{generation_error}", file=sys.stderr, flush=True,
                    )
                duration = time.perf_counter() - started
                candidate_id = f"{args.task}:{args.split}:{input_id}:candidate:{rank}"
                rows.append({
                    "candidate_id": candidate_id,
                    "task": args.task,
                    "split": args.split,
                    "input_id": input_id,
                    "example_id": f"{args.task}:{args.split}:{input_id}",
                    "candidate_rank": rank,
                    "seed": seed,
                    "text": text,
                    "generation_duration_seconds": duration,
                    "generation_batch_duration_seconds": duration,
                    "generation_batch_size": 1,
                    "validity": None,
                    "metrics": None,
                    "generation_error": generation_error,
                })
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            temporary = chunk_path.with_suffix(".jsonl.tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(chunk_path)
            record["completed_inputs"] = input_id + 1
            write_json(output / "manifest.json", record)

        candidate_rows = []
        chunk_hashes = {}
        for input_row in input_rows:
            input_id = input_row["input_id"]
            chunk_path = chunks / f"input-{input_id:06d}.jsonl"
            candidate_rows.extend(validate_candidate_chunk(
                chunk_path, args.task, input_id, args.k, args.replicate,
                args.base_seed, args.greedy_baseline, args.split,
            ))
            chunk_hashes[str(chunk_path.relative_to(output))] = digest(chunk_path)
        write_jsonl_once(output / "candidates.jsonl", candidate_rows)
        candidate_timing = {
            row["candidate_id"]: row["generation_duration_seconds"]
            for row in candidate_rows
        }
        record["candidate_errors"] = sum(
            row.get("generation_error") is not None for row in candidate_rows
        )

        contexts = [context_for(args.task, row, args.split) for row in input_rows]
        if digest(references_path) != dataset_manifest.get("reference_sha256"):
            raise ValueError(f"prepared {args.split} references do not match dataset-manifest.json")
        reference_rows = load_reference_rows(
            references_path, args.task, len(input_rows), args.split
        )
        references = [
            {"task": args.task, "split": args.split, "input_id": row["input_id"],
             "example_id": row["example_id"], "reference_id": row["reference_id"],
             "text": row["text"]}
            for row in reference_rows
        ]
        write_jsonl_once(output / "contexts.jsonl", contexts)
        write_jsonl_once(output / "references.jsonl", references)
        checkpoint = {
            "candidate_chunk_sha256": chunk_hashes,
            "candidates_sha256": digest(output / "candidates.jsonl"),
            "contexts_sha256": digest(output / "contexts.jsonl"),
            "references_sha256": digest(output / "references.jsonl"),
        }
        write_json(output / "generation-checkpoint.json", checkpoint)
        record.update({
            "status": "succeeded",
            "started_model_load_utc": model_load_started,
            "model_load_seconds": model_load_seconds,
            "warmup_policy": "first candidate is retained and included in per-candidate timing; excluded from warm total",
            "model_ready_utc": model_ready_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "references_sha256": digest(references_path),
            "pre_tokenizer_model": pre_tokenizer_model,
            "device": {"name": device_info[0], "memory_total_mib": int(device_info[1]),
                       "driver": device_info[2]},
            "candidate_timing_seconds": candidate_timing,
            "timings": {
                "generation_seconds": sum(candidate_timing.values()),
                "warm_generation_seconds": sum(list(candidate_timing.values())[1:]),
            },
            "outputs": {
                name: digest(output / name)
                for name in ("candidates.jsonl", "contexts.jsonl", "references.jsonl",
                             "generation-checkpoint.json", "run-spec.json")
            },
        })
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        write_json(output / "manifest.json", record)
    print(f"Candidate pool: {output}")


if __name__ == "__main__":
    main()

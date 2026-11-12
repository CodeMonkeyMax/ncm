"""Run an upstream phase with synchronized GPU timing, without editing upstream."""

from __future__ import annotations

import argparse
import functools
import json
import runpy
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("generation", "scoring"), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--timing-out", type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    sys.path.insert(0, str(args.upstream))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    batches: list[float] = []
    complete_batches: list[float] = []
    active_batch: float | None = None
    load_seconds: float | None = None
    if args.phase == "generation":
        from models.next_concept import NextConceptTransformer

        original = NextConceptTransformer.generate

        @functools.wraps(original)
        def timed_generate(*positional, **keyword):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return original(*positional, **keyword)
            finally:
                torch.cuda.synchronize()
                batches.append(time.perf_counter() - started)

        NextConceptTransformer.generate = timed_generate

    script = args.upstream / "scripts" / (
        "eval_shard.py" if args.phase == "generation" else "eval_merge.py"
    )
    sys.argv = [str(script), *upstream_args]
    started = time.perf_counter()
    def profile(frame, event, arg):
        nonlocal active_batch, load_seconds
        if frame.f_code.co_filename != str(script) or frame.f_code.co_name != "run":
            return
        if event == "call":
            torch.cuda.synchronize()
            active_batch = time.perf_counter()
            if load_seconds is None:
                load_seconds = active_batch - started
        elif event == "return" and active_batch is not None:
            torch.cuda.synchronize()
            complete_batches.append(time.perf_counter() - active_batch)
            active_batch = None

    try:
        if args.phase == "generation":
            sys.setprofile(profile)
        runpy.run_path(str(script), run_name="__main__")
        sys.setprofile(None)
        torch.cuda.synchronize()
        timing = {
            "phase_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "generate_batch_seconds": batches,
            "warm_generate_seconds": sum(batches[1:]) if len(batches) > 1 else None,
            "full_batch_seconds": complete_batches,
            "warm_full_generation_seconds": sum(complete_batches[1:]) if len(complete_batches) > 1 else None,
            "load_until_first_batch_seconds": load_seconds,
            "timing_scope": "full batch includes transformer, VQ decoding, token selection and text decoding; excludes prefix preparation",
            "warmup_policy": "first generated batch excluded from warm generation timing",
        }
        args.timing_out.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    finally:
        sys.setprofile(None)
        if args.phase == "generation":
            NextConceptTransformer.generate = original


if __name__ == "__main__":
    main()

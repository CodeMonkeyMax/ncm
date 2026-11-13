"""Download the P0-A official artifact allowlist only."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "phase0" / "artifacts.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for group in ("checkpoints", "datasets", "pre_tokenizers"):
        spec = manifest[group]
        local_dir = ROOT / spec["local_dir"]
        repo_type = "dataset" if group == "datasets" else "model"
        for item in spec["files"]:
            print(f"downloading {spec['repo']}@{spec['revision']}:{item['path']}")
            hf_hub_download(
                repo_id=spec["repo"],
                repo_type=repo_type,
                revision=spec["revision"],
                filename=item["path"],
                local_dir=local_dir,
            )
        if group == "pre_tokenizers":
            cached = snapshot_download(
                repo_id=spec["repo"],
                repo_type=repo_type,
                revision="main",
                allow_patterns=[item["path"] for item in spec["files"]],
            )
            if Path(cached).name != spec["revision"]:
                raise RuntimeError(
                    f"{spec['repo']} main resolved to {Path(cached).name}, "
                    f"not the pinned revision {spec['revision']}"
                )


if __name__ == "__main__":
    main()

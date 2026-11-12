"""Hash-verify the P0-A allowlist without loading models or pickles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "phase0" / "artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failed = False
    for group in ("checkpoints", "datasets"):
        spec = manifest[group]
        for item in spec["files"]:
            path = ROOT / spec["local_dir"] / item["path"]
            if not path.is_file():
                print(f"MISSING {path}")
                failed = True
                continue
            size = path.stat().st_size
            if size != item["size"]:
                print(f"SIZE {path}: {size} != {item['size']}")
                failed = True
                continue
            if "sha256" not in item:
                expected = item.get("git_oid")
                if expected is None:
                    print(f"OK size-only {path} ({size} bytes)")
                    continue
                digest = git_blob_oid(path)
                if digest != expected:
                    print(f"GIT-OID {path}: {digest} != {expected}")
                    failed = True
                    continue
                print(f"OK {path}")
                continue
            digest = sha256(path)
            if digest != item["sha256"]:
                print(f"HASH {path}: {digest} != {item['sha256']}")
                failed = True
                continue
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

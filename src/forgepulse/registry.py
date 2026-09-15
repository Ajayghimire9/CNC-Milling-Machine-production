from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

Stage = Literal["candidate", "validated", "production"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_artifact(model_path: str | Path, version: str, stage: Stage = "candidate") -> dict[str, str]:
    model = Path(model_path)
    manifest = {
        "model": model.name,
        "version": version,
        "stage": stage,
        "sha256": sha256_file(model),
    }
    output = model.with_suffix(".manifest.json")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

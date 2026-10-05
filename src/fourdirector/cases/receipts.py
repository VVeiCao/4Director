"""Immutable stage attempts: skip when receipts match, --force starts a new attempt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "4director-stage-receipt-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def forbid_repo_output(destination: Path, root: Path) -> None:
    """Keep generated meshes, controls, and videos out of the tracked inputs."""
    data_root = (root / "data").resolve()
    resolved = destination.expanduser().resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise SystemExit(
            f"refusing to write generated artifacts under {data_root}; "
            "data/ holds tracked inputs only. Use ./output/<case_id> instead."
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def attempt_root(run_root: Path, stage: str) -> Path:
    return run_root / "attempts" / stage


def current_receipt_path(run_root: Path, stage: str) -> Path:
    return run_root / "current" / stage / "receipt.json"


def next_attempt_index(run_root: Path, stage: str) -> int:
    root = attempt_root(run_root, stage)
    if not root.is_dir():
        return 1
    existing = [int(path.name) for path in root.iterdir() if path.name.isdigit()]
    return (max(existing) + 1) if existing else 1


def load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError(f"unsupported stage receipt: {path}")
    return payload


def outputs_exist(run_root: Path, receipt: dict[str, Any]) -> bool:
    for record in receipt.get("outputs", []):
        path = Path(record["path"])
        if not path.is_absolute():
            path = run_root / path
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            return False
    return True


def should_skip(
    run_root: Path,
    stage: str,
    identity: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any] | None:
    if force:
        return None
    receipt = load_receipt(current_receipt_path(run_root, stage))
    if receipt is None:
        return None
    if receipt.get("identity_sha256") != identity_digest(identity):
        return None
    if not outputs_exist(run_root, receipt):
        return None
    return receipt


def commit_attempt(
    run_root: Path,
    *,
    stage: str,
    identity: dict[str, Any],
    outputs: list[Path],
    extra: dict[str, Any] | None = None,
    skipped: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    # Stage entrypoints perform the skip check before doing work.  Rechecking
    # here would collapse a tamper-repair run into the old attempt after the
    # canonical output has been repaired to the same bytes.
    index = next_attempt_index(run_root, stage)
    destination = attempt_root(run_root, stage) / f"{index:04d}"
    destination.mkdir(parents=True)
    recorded = []
    for path in outputs:
        resolved = path.expanduser().resolve()
        recorded.append(
            {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "stage": stage,
        "attempt": index,
        "skipped": skipped,
        "identity": identity,
        "identity_sha256": identity_digest(identity),
        "outputs": recorded,
    }
    if extra:
        receipt["extra"] = extra
    write_json(destination / "receipt.json", receipt)
    current = run_root / "current" / stage
    current.mkdir(parents=True, exist_ok=True)
    write_json(current / "receipt.json", receipt)
    write_json(current / "attempt.json", {"attempt": index, "path": str(destination)})
    return receipt

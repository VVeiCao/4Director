#!/usr/bin/env python3
"""Run a command in a disposable, patched copy of the pinned Pixal3D commit."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

PIXAL3D_COMMIT = "28efad66fdcbd8174a8538d9baf71fe34fe4b6d2"
PATCH_SHA256 = "cfc32db915f08fc61a32f296889822e2c0fb3b805cd2d3928b49455ff983d2b9"
HELPER_SHA256 = "476756952e5f3edd13d667f2f1e7561778f6bd8386bc9fc1e10da85590b16234"
RELEASE_ROOT = Path(__file__).resolve().parents[2]
PATCH = RELEASE_ROOT / "third_party/patches/pixal3d.tracked.patch"
HELPER = Path(__file__).with_name("batch_davis_hike_sequence.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def verify_source(repository: Path) -> None:
    """Require the pinned commit, and the patch and helper this adapter was locked to."""
    if git(repository, "cat-file", "-e", f"{PIXAL3D_COMMIT}^{{commit}}", check=False).returncode:
        raise RuntimeError(
            f"{repository} does not contain Pixal3D commit {PIXAL3D_COMMIT}; "
            "run `git submodule update --init third_party/Pixal3D`"
        )
    for path, expected in ((PATCH, PATCH_SHA256), (HELPER, HELPER_SHA256)):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{path} SHA-256 {actual} != {expected}")


def run(repository: Path, command: list[str]) -> int:
    """Run ``command`` inside a patched copy of the pinned commit.

    The copy comes from ``git archive`` rather than a worktree, so the
    submodule is only read: concurrent runs cannot race on worktree
    bookkeeping, and an interrupted run leaves nothing registered behind.
    """
    repository = repository.expanduser().resolve()
    verify_source(repository)
    with tempfile.TemporaryDirectory(prefix="4director-pixal3d-") as temporary:
        tree = Path(temporary) / "Pixal3D"
        tree.mkdir()
        archive = subprocess.Popen(
            ["git", "-C", str(repository), "archive", "--format=tar", PIXAL3D_COMMIT],
            stdout=subprocess.PIPE,
        )
        extracted = subprocess.run(["tar", "-x", "-C", str(tree)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() != 0 or extracted.returncode != 0:
            raise RuntimeError(f"cannot extract Pixal3D {PIXAL3D_COMMIT} from {repository}")
        # Make the copy its own repository, so git apply cannot resolve the
        # patch against an enclosing work tree and silently skip every hunk.
        git(tree, "init", "-q")
        git(tree, "apply", str(PATCH))
        if git(tree, "apply", "--reverse", "--check", str(PATCH), check=False).returncode:
            raise RuntimeError(f"{PATCH} did not apply to the Pixal3D copy")
        environment = os.environ.copy()
        environment["PIXAL3D_ROOT"] = str(tree)
        environment["PIXAL3D_ADAPTER_ROOT"] = str(HELPER.parent)
        return subprocess.run(command, cwd=tree, env=environment, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=RELEASE_ROOT / "third_party/Pixal3D",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    return run(args.repository, command)


if __name__ == "__main__":
    raise SystemExit(main())

"""Safe subprocess execution for production stage adapters."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event


class ExternalCommandError(RuntimeError):
    """Base error for a failed external production stage."""


class ExternalCommandTimeout(ExternalCommandError):
    """Raised after a command exceeds its configured deadline."""


class ExternalCommandCancelled(ExternalCommandError):
    """Raised when a caller cancels a running command."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def validate_argv(argv: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    """Return an immutable argv while rejecting ambiguous command values."""

    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("command must be a non-empty argv sequence")
    normalized: list[str] = []
    for index, value in enumerate(argv):
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"argv[{index}] is not a string or path")
        item = os.fspath(value)
        if not item:
            raise ValueError(f"argv[{index}] is empty")
        if "\x00" in item:
            raise ValueError(f"argv[{index}] contains NUL")
        normalized.append(item)
    return tuple(normalized)


def _stop_process(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        process.wait()


class ExternalCommandRunner:
    """Execute argv-only commands with timeout and cooperative cancellation."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float = 3_600.0,
        poll_seconds: float = 0.1,
        termination_grace_seconds: float = 2.0,
    ) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.default_timeout_seconds = float(default_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.termination_grace_seconds = float(termination_grace_seconds)

    def run(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
        cancel_event: Event | None = None,
    ) -> CommandResult:
        command = validate_argv(argv)
        timeout = (
            self.default_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        working_directory = None if cwd is None else Path(cwd).expanduser()
        if working_directory is not None and not working_directory.is_dir():
            raise NotADirectoryError(working_directory)
        environment = None
        if env is not None:
            environment = {str(key): str(value) for key, value in env.items()}

        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            start_new_session=True,
        )
        stdout = ""
        stderr = ""
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _stop_process(process, self.termination_grace_seconds)
                    stdout, stderr = process.communicate()
                    raise ExternalCommandCancelled(
                        f"external command cancelled: {command[0]}"
                    )
                elapsed = time.monotonic() - started
                remaining = timeout - elapsed
                if remaining <= 0:
                    _stop_process(process, self.termination_grace_seconds)
                    stdout, stderr = process.communicate()
                    raise ExternalCommandTimeout(
                        f"external command exceeded {timeout:.3f}s: {command[0]}"
                    )
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(self.poll_seconds, remaining)
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            _stop_process(process, self.termination_grace_seconds)
            raise

        elapsed = time.monotonic() - started
        result = CommandResult(
            argv=command,
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
        )
        if result.returncode != 0:
            raise ExternalCommandError(
                f"external command failed with code {result.returncode}: "
                f"{command[0]}\n{stderr[-4_096:]}"
            )
        return result

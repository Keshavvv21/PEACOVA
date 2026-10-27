"""Real, process-isolated Python execution — CodingAgent's `run_code`
tool (agents/coding_agent.py). Before this, "coding agent" in this
codebase meant "generate text, ast.parse the fenced code block" —
a genuine syntax check, but never actually RUNNING anything, so a
snippet that parses fine and raises at line 1 was reported as fine.

Deliberately NOT a container/VM sandbox — no seccomp, no network
namespace, no filesystem jail beyond a throwaway cwd. This is
`subprocess.run` with Python's own `-I` isolated mode (ignores
PYTHONPATH/user site-packages, so a generated snippet can't accidentally
or deliberately import this app's own modules off sys.path), a stripped
environment, a fresh temporary working directory per call, and a hard
wall-clock timeout. Good enough to observe whether a real generated
snippet runs cleanly — not a defense against a user deliberately trying
to escape it (it can still open sockets, spawn processes, read/write
anywhere the OS-level file permissions of the running service account
allow). A real container-based sandbox (gVisor/Firecracker/Docker
--network=none) is real future work, not something to half-fake here.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass

_MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict:
        return {"stdout": self.stdout, "stderr": self.stderr,
                "exit_code": self.exit_code, "timed_out": self.timed_out, "ok": self.ok}


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> SandboxResult:
    """Runs `code` as a real, standalone Python process. Never raises —
    a crash or a timeout is a normal, reportable result (the caller's
    job to decide what that means), not an exception to catch. Output is
    truncated to the last _MAX_OUTPUT_CHARS characters so a runaway
    print loop can't blow up a chat reply or a log entry."""
    with tempfile.TemporaryDirectory(prefix="akanade-sandbox-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=cwd, capture_output=True, text=True, timeout=timeout,
                env={"PATH": "/usr/bin:/bin"},
            )
            return SandboxResult(stdout=proc.stdout[-_MAX_OUTPUT_CHARS:],
                                 stderr=proc.stderr[-_MAX_OUTPUT_CHARS:],
                                 exit_code=proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="ignore")
            stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="ignore")
            return SandboxResult(stdout=stdout[-_MAX_OUTPUT_CHARS:],
                                 stderr=stderr[-_MAX_OUTPUT_CHARS:],
                                 exit_code=None, timed_out=True)

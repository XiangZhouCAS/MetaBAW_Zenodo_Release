"""Route verbose startup diagnostics to the run directory, keeping progress visible."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import re
import shlex
import sys


def _detail_line(text: str) -> bool:
    line = re.sub(r"^\[\d+:\d{2}:\d{2}\]\s*", "", text.lstrip())
    if line.startswith("[INPUT]"):
        return not line.startswith(("[INPUT] Named manifests:", "[INPUT] Annotation:"))
    return line.startswith((
        "[OK]", "[CUDA ", "[GPU]", "[GPU AUTO]", "[RESOURCES]", "[WARNING]",
        "[COMEBIN CPU]", "[MEMORY ESTIMATE]", "[WORKFLOW DETAILS]",
        "[FAILURE POLICY]", "[CONFIG]", "[KEGG]", "[PREFLIGHT]",
    ))


class _StartupStream:
    def __init__(self, log, console):
        self.log = log
        self.console = console
        self.pending = ""

    def write(self, text):
        if not self.log.active:
            return self.console.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def _emit(self, text):
        self.log.record(text)
        if not _detail_line(text):
            self.console.write(text)

    def flush(self):
        if self.pending:
            self._emit(self.pending)
            self.pending = ""
        if self.log.handle is not None:
            self.log.handle.flush()
        self.console.flush()

    def __getattr__(self, name):
        # Preserve isatty()/fileno() for interactive dependency-install prompts.
        return getattr(self.console, name)


class StartupLog:
    def __init__(self, args, output: Path):
        self.args = args
        self.path = output / "start_info.txt"
        self.active = True
        self.handle = None
        self.stdout = _StartupStream(self, sys.stdout)
        self.stderr = _StartupStream(self, sys.stderr)

    def record(self, text):
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a", encoding="utf-8")
            self.handle.write(f"\n=== MetaBAW startup: {datetime.now(timezone.utc).isoformat()} ===\n")
            invocation = getattr(self.args, "_invocation", None)
            if invocation:
                self.handle.write(f"Command: {shlex.join(invocation)}\n")
        self.handle.write(text)
        self.handle.flush()

    def finish(self):
        self.stdout.flush()
        self.stderr.flush()
        self.active = False
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def capture_startup(output_position: int | None = None):
    """Wrap a command; nested runners share the log until finish_startup()."""
    def decorate(function):
        @wraps(function)
        def wrapped(args, *positional, **keywords):
            if getattr(args, "_startup_log", None) is not None or getattr(args, "dry_run", False):
                return function(args, *positional, **keywords)
            output = (
                positional[output_position - 1] if output_position is not None and len(positional) >= output_position
                else keywords.get("output", getattr(args, "output", None))
            )
            if output is None:
                return function(args, *positional, **keywords)
            log = StartupLog(args, Path(output).expanduser().resolve())
            args._startup_log = log
            try:
                with redirect_stdout(log.stdout), redirect_stderr(log.stderr):
                    try:
                        return function(args, *positional, **keywords)
                    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
                        if log.active and log.handle is not None:
                            log.record(f"[STARTUP ERROR] {type(exc).__name__}: {exc}\n")
                        raise
                    finally:
                        log.finish()
            finally:
                del args._startup_log
        return wrapped
    return decorate


def finish_startup(args) -> None:
    log = getattr(args, "_startup_log", None)
    if log is not None:
        log.finish()

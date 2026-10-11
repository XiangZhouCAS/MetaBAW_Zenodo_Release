"""Fresh private BAM indexes and supervised native MetaDecoder coverage.

This module also runs as a script under the interpreter from MetaDecoder's
console launcher. It must use only the standard library and no relative imports.
"""
from __future__ import annotations

import csv
import math
import multiprocessing.process
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback


BIN_SIZE = 2000


def metadecoder_executable() -> str | None:
    """Resolve MetaDecoder in MetaBAW's own environment, never Conda base."""
    prefix = Path(sys.prefix)
    for candidate in (prefix / "bin" / "metadecoder", prefix / "Scripts" / "metadecoder"):
        if not candidate.is_file():
            continue
        try:
            launcher = _python_launcher(str(candidate))
            # A venv Python may be a symlink to base. The executable's spelling
            # still controls sys.prefix, so matching only realpaths is unsafe.
            interpreter = Path(launcher[0]).absolute()
            owned_directories = {prefix.absolute() / "bin", prefix.absolute() / "Scripts"}
            if (interpreter.parent in owned_directories
                    and interpreter.resolve() == Path(sys.executable).resolve()):
                return str(candidate)
        except (OSError, ValueError):
            continue
    return None


def _python_launcher(executable: str) -> list[str]:
    """Keep the interpreter of the actual executable, not MetaBAW's Python."""
    with open(executable, "rb") as handle:
        first = handle.readline(4096).decode("utf-8", errors="replace").strip()
        second = handle.readline(4096).decode("utf-8", errors="replace").strip()
    if first.startswith("#!"):
        parts = shlex.split(first[2:])
        if parts and Path(parts[0]).name == "env":
            parts = parts[1:]
            if parts and parts[0] == "-S":
                parts = parts[1:]
            if parts:
                resolved = shutil.which(parts[0])
                if resolved:
                    parts[0] = resolved
        if parts and Path(parts[0]).name.lower().startswith("python"):
            return parts
        # pip/distlib uses a /bin/sh trampoline for long interpreter paths or
        # paths containing spaces. Parse only its standard exec-Python form.
        if parts == ["/bin/sh"] and second.startswith("'''exec' "):
            trampoline = shlex.split(second[len("'''exec' "):])
            if (len(trampoline) >= 3 and trampoline[-2:] == ["$0", "$@"]
                    and Path(trampoline[0]).name.lower().startswith("python")):
                return trampoline[:-2]
    raise ValueError(
        f"Cannot identify the Python interpreter of {executable}; reinstall "
        "MetaDecoder using its standard Python console launcher."
    )


def _bam_references(bams: tuple[Path, ...], samtools: str) -> dict[str, int]:
    # quickcheck validates the header/EOF, not every internal BAM block. Native
    # MetaDecoder reading and coverage validation below supply further checks.
    subprocess.run([samtools, "quickcheck", "-v", *(str(bam) for bam in bams)], check=True)
    shared: list[tuple[str, int]] | None = None
    for bam in bams:
        probe = subprocess.run([samtools, "view", "-H", str(bam)],
                               capture_output=True, text=True, check=True)
        references: list[tuple[str, int]] = []
        sort_order = None
        for line in probe.stdout.splitlines():
            fields = line.split("\t")
            tags = dict(field.split(":", 1) for field in fields[1:] if ":" in field)
            if fields[0] == "@HD":
                sort_order = tags.get("SO")
            elif fields[0] == "@SQ":
                name = tags.get("SN", "")
                length = int(tags.get("LN", "0"))
                if not name or length <= 0:
                    raise ValueError(f"Invalid BAM reference header: {bam}: {line}")
                references.append((name, length))
        if not references or len(dict(references)) != len(references):
            raise ValueError(f"Missing or duplicate BAM reference sequences: {bam}")
        if sort_order != "coordinate":
            raise ValueError(f"MetaDecoder requires coordinate-sorted BAMs: {bam}")
        if shared is not None and references != shared:
            raise ValueError(f"MetaDecoder BAM reference names/order/lengths differ: {bam}")
        shared = references
    return dict(shared or ())


def validate_coverage(path: Path, references: dict[str, int], samples: int) -> int:
    """Require every expected contig/bin, even bins with zero read coverage."""
    seen = {name: bytearray((length + BIN_SIZE - 1) // BIN_SIZE)
            for name, length in references.items()}
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        expected = ["sequence id", "bin index", "bin size"] + [
            f"coverage{index + 1}" for index in range(samples)]
        if next(reader, None) != expected:
            raise ValueError("Invalid MetaDecoder coverage header/sample count")
        for number, row in enumerate(reader, start=2):
            if len(row) != len(expected) or row[0] not in seen:
                raise ValueError(f"Invalid MetaDecoder coverage row {number}")
            name = row[0]
            index, size = int(row[1]), int(row[2])
            if not 1 <= index <= len(seen[name]):
                raise ValueError(f"Invalid MetaDecoder bin index at row {number}")
            if seen[name][index - 1]:
                raise ValueError(f"Duplicate MetaDecoder coverage bin at row {number}")
            expected_size = min(BIN_SIZE, references[name] - (index - 1) * BIN_SIZE)
            if size != expected_size:
                raise ValueError(f"Invalid MetaDecoder bin size at row {number}")
            if any(not math.isfinite(value) or value < 0
                   for value in map(float, row[3:])):
                raise ValueError(f"Non-finite or negative MetaDecoder coverage at row {number}")
            seen[name][index - 1] = 1
            rows += 1
    missing = sum(bins.count(0) for bins in seen.values())
    if missing:
        raise ValueError(f"Incomplete MetaDecoder coverage: {missing} expected bins missing")
    return rows


def run_coverage(bams: tuple[Path, ...], output: Path, marker: Path, threads: int) -> None:
    if not bams or threads < 1:
        raise ValueError("MetaDecoder coverage needs BAM inputs and positive threads")
    for bam in bams:
        if not bam.is_file():
            raise ValueError(f"Missing MetaDecoder BAM input: {bam}")
    originals = {bam.resolve() for bam in bams}
    if (output.resolve() in originals or marker.resolve() in originals
            or output.resolve() == marker.resolve()):
        raise ValueError("MetaDecoder output/marker must not overwrite an input BAM or each other")
    executable, samtools = metadecoder_executable(), shutil.which("samtools")
    if executable is None or samtools is None:
        raise ValueError("MetaDecoder coverage requires MetaDecoder in the MetaBAW environment and configured samtools")
    launcher = _python_launcher(executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    references = _bam_references(bams, samtools)
    # A unique owned directory makes old/partial .bam.index sidecars impossible
    # to reuse. Symlinks/hardlinks do not copy BAM data or touch BAM/BAI originals.
    with tempfile.TemporaryDirectory(prefix=".metabaw-coverage-", dir=output.parent) as directory:
        temporary = Path(directory)
        aliases = []
        for index, bam in enumerate(bams, start=1):
            alias = temporary / f"sample.{index}.bam"
            try:
                alias.symlink_to(bam.resolve())
            except OSError:
                try:
                    os.link(bam.resolve(), alias)
                except OSError as exc:
                    raise ValueError(f"Cannot create a private BAM link for {bam}; "
                                     "the workflow filesystem must support symlinks or hardlinks") from exc
            aliases.append(alias)
        pending = temporary / "coverage.tsv"
        command = [*launcher, "-u", str(Path(__file__).resolve()), "--guard-cli",
                   executable, "coverage", "-b", *(str(alias) for alias in aliases),
                   "-o", str(pending), "--threads", str(threads), "--bin_size", str(BIN_SIZE)]
        print(f"[METADECODER COVERAGE] Fresh private indexes; threads={threads}; "
              "original BAM/BAI/index files are unchanged.", flush=True)
        print(f"[COMMAND] {shlex.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError(f"MetaDecoder coverage failed (exit {result.returncode}); "
                               "no completion marker was written. Inspect the worker error "
                               "above; if BAM parsing still fails with fresh indexes, "
                               "validate/regenerate the affected BAM.")
        rows = validate_coverage(pending, references, len(bams))
        # Publish only after successful children and full output validation.
        os.replace(pending, output)
        marker.write_text(f"MetaDecoder coverage v2 validated: {rows} bins; "
                          f"{len(bams)} BAMs\n", encoding="utf-8")


def _guard_cli(executable: str, arguments: list[str]) -> None:
    """Fail promptly if any upstream worker dies, including during queue joins.

    Patch process lifecycle checks only in this short-lived interpreter. Do not
    replace MetaDecoder's coverage calculation or modify its installed files.
    Children remain in the workflow task's process group for cancellation.
    """
    base = multiprocessing.process.BaseProcess
    original_start, original_join, original_close = base.start, base.join, base.close
    owner_pid = os.getpid()
    processes = []
    lock = threading.RLock()
    stop, aborting = threading.Event(), threading.Event()

    def check():
        with lock:
            if aborting.is_set():
                return
            failed = next((process for process in processes
                           if process.exitcode not in (None, 0)), None)
            if failed is None:
                return
            aborting.set()
            children = list(processes)
            name, code = failed.name, failed.exitcode
            # Hold the lifecycle lock through shutdown: the native main thread
            # must not close process handles or start new workers meanwhile.
            print(f"[ERROR] MetaDecoder worker {name} failed with exit code {code}; "
                  "terminating remaining workers rather than accepting partial coverage.",
                  file=sys.stderr, flush=True)
            for child in children:
                if child.exitcode is None:
                    child.terminate()
            for child in children:
                original_join(child, timeout=0.2)
                if child.exitcode is None:
                    child.kill()
                    original_join(child, timeout=0.2)
            os._exit(1)

    def start(process):
        if os.getpid() != owner_pid:
            return original_start(process)
        with lock:
            original_start(process)
            processes.append(process)
        check()

    def join(process, timeout=None):
        if os.getpid() != owner_pid:
            return original_join(process, timeout)
        # Serialize waitpid/poll with the watchdog without holding the lifecycle
        # lock for an unbounded native join (which would prevent supervision).
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            interval = 0.05 if deadline is None else min(0.05, max(0, deadline - time.monotonic()))
            with lock:
                result = original_join(process, interval)
                check()
                if process.exitcode is not None:
                    return result
            if deadline is not None and time.monotonic() >= deadline:
                return result

    def close(process):
        if os.getpid() != owner_pid:
            return original_close(process)
        with lock:
            check()
            original_close(process)
            if process in processes:
                processes.remove(process)

    def monitor():
        while not stop.wait(0.1):
            check()

    def cleanup():
        # Also reap live workers when the native parent raises before its join
        # phase; otherwise multiprocessing's interpreter shutdown can hang.
        aborting.set()
        with lock:
            children = list(processes)
        for child in children:
            if child.exitcode is None:
                child.terminate()
        for child in children:
            original_join(child, timeout=0.2)
            if child.exitcode is None:
                child.kill()
                original_join(child, timeout=0.2)

    base.start, base.join, base.close = start, join, close
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        sys.argv = [executable, *arguments]
        try:
            runpy.run_path(executable, run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
            # Most pip console scripts call sys.exit(main()). Treat zero exits
            # exactly like a returned main function and check the children.
        check()
        with lock:
            if any(process.exitcode is None for process in processes):
                raise RuntimeError("MetaDecoder exited with active unfinished workers")
    except BaseException:
        # A native parent error can also leave a blocked Queue feeder. Reap the
        # workers, then bypass multiprocessing's potentially blocking atexit.
        traceback.print_exc()
        cleanup()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        cleanup()
        stop.set()
        watcher.join(timeout=1)
        base.start, base.join, base.close = original_start, original_join, original_close


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--guard-cli":
        raise SystemExit("This helper is launched by MetaBAW's coverage task.")
    _guard_cli(sys.argv[2], sys.argv[3:])

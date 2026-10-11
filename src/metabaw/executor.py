from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Sequence

from .model import Task, topological_order
from .refinement import read_empty_refinement
from .memory import MemorySnapshot, linux_memory_snapshot
from .resume import checkpoint_task_id, command_thread_counts, resume_command, resume_environment
from .relocation import ResumeLayout, portable_checkpoint, checkpoint_signature, external_locations
from .state import StateStore, TaskRecord
from .resources import GpuReservations, ResourcePlan, available_memory_gib
from .gpu_resources import GpuMonitor, process_tasks


@dataclass(frozen=True)
class RunResult:
    task_id: str
    return_code: int
    message: str
    memory_limit_exceeded: bool = False
    aborted_by_memory_limit: bool = False


MemorySampler = Callable[[Sequence[int]], int | MemorySnapshot]


def _linux_process_tree_memory_bytes(
    root_pids: Sequence[int],
    proc_root: Path = Path("/proc"),
) -> int:
    """Compatibility helper returning accounted bytes, not necessarily PSS.

    The executor uses the structured snapshot directly so RSS cannot be
    mistaken for measured PSS or an incomplete sum for a total upper bound.
    """
    return linux_memory_snapshot(root_pids, proc_root).upper_bound_bytes


class _WorkflowMemoryBudget:
    """Track accounted physical memory for every active task process tree."""

    def __init__(
        self,
        limit_gb: float | None,
        sampler: MemorySampler | None = None,
        diagnostic_path: Path | None = None,
        available_memory_provider=None,
        gpu_monitor: GpuMonitor | None = None,
        gpu_reservations=None,
    ) -> None:
        self.limit_gb = limit_gb
        self.limit_bytes = (
            max(1, int(limit_gb * 1024**3))
            if limit_gb is not None
            else None
        )
        self.event = threading.Event()
        self._sampler = sampler or linux_memory_snapshot
        self._lock = threading.Lock()
        self._roots: dict[int, str] = {}
        self.reason: str | None = None
        self.source_task: str | None = None
        self.observed_bytes: int | None = None
        self.observed_pss_partial = False
        self.observed_upper_bound_complete = True
        self.peak_bytes = 0
        self.peak_upper_bound_bytes = 0
        self.upper_bound_complete = True
        self.diagnostic_path = diagnostic_path
        self._candidate = None
        self._degraded_recorded = False
        self._no_address_space_recorded = False
        self._diagnostic_warning = False
        self._initial_limit_bytes = self.limit_bytes
        self._available_memory_provider = available_memory_provider
        self._task_limits: dict[str, int] = {}
        self.gpu_monitor = gpu_monitor
        self.gpu_reservations = gpu_reservations
        self.resource_detail = ""

    def _record_snapshot(self, snapshot: MemorySnapshot, event: str, roots) -> None:
        if self.diagnostic_path is None:
            return
        with self._lock:
            root_tasks = dict(self._roots)
        record = {
            "time": datetime.now().astimezone().isoformat(), "event": event,
            "limit_gib": self.limit_gb, "root_tasks": root_tasks,
            "root_pids": list(roots), "pss_bytes": snapshot.pss_bytes,
            "upper_bound_bytes": snapshot.upper_bound_bytes if snapshot.upper_bound_complete else None,
            "upper_bound_complete": snapshot.upper_bound_complete,
            "accounted_bytes": snapshot.upper_bound_bytes,
            "missing_pss_processes": snapshot.missing_pss, "supported": snapshot.supported,
            "processes": snapshot.processes,
        }
        try:
            self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            if not self._diagnostic_warning:
                print(f"[WARNING] Could not write memory diagnostics: {exc}", flush=True)
                self._diagnostic_warning = True

    def register(self, task_id: str, process_id: int) -> None:
        with self._lock:
            self._roots[process_id] = task_id

    def set_task_limit(self, task_id: str, limit_gb: float) -> None:
        self._task_limits[task_id] = max(1, int(limit_gb * 1024**3))

    def unregister(self, process_id: int) -> None:
        with self._lock:
            self._roots.pop(process_id, None)

    def active_roots(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._roots)

    @property
    def awaiting_confirmation(self) -> bool:
        return self._candidate is not None or bool(self.gpu_monitor and self.gpu_monitor.candidate)

    def _trigger(
        self,
        reason: str,
        *,
        source_task: str | None = None,
        observed_bytes: int | None = None,
    ) -> bool:
        with self._lock:
            if self.event.is_set():
                return False
            self.reason = reason
            self.source_task = source_task
            self.observed_bytes = observed_bytes
            self.event.set()
            return True

    def sample(self) -> int:
        if self.limit_bytes is None or self.event.is_set():
            return 0
        roots = self.active_roots()
        sample = self._sampler(roots) if roots else MemorySnapshot()
        if isinstance(sample, MemorySnapshot):
            usage = sample.pss_bytes
            upper = sample.upper_bound_bytes
            if sample.upper_bound_complete:
                self.peak_upper_bound_bytes = max(self.peak_upper_bound_bytes, upper)
            else:
                self.upper_bound_complete = False
            if not self._no_address_space_recorded and any(
                row["state"] == "no_address_space" for row in sample.processes
            ):
                self._record_snapshot(sample, "no_address_space_excluded", roots)
                self._no_address_space_recorded = True
            if (sample.missing_pss or not sample.supported) and not self._degraded_recorded:
                event = ("pss_unavailable_rss_is_upper_bound" if sample.upper_bound_complete
                         else "memory_measurement_incomplete")
                self._record_snapshot(sample, event, roots)
                self._degraded_recorded = True
            source = None
            with self._lock:
                root_tasks = dict(self._roots)
            pid_tasks = process_tasks(sample, root_tasks)
            if self._available_memory_provider is not None:
                available = self._available_memory_provider()
                if available is None:
                    self.resource_detail = "Current available RAM cannot be measured"
                    self._trigger("measurement_unavailable")
                    return usage
                self.limit_bytes = min(self._initial_limit_bytes, int(0.8 * (available * 1024**3 + usage)))
                self.limit_gb = self.limit_bytes / 1024**3
                if self.limit_bytes <= 0:
                    self.resource_detail = "Current RAM headroom is exhausted; no positive task share fits the 80% budget"
                    self._trigger("ram_unavailable")
                    return usage
            task_usage: dict[str, int] = {}
            task_upper: dict[str, int] = {}
            task_partial: set[str] = set()
            for row in sample.live:
                task_id = pid_tasks.get(row.get("pid"))
                if task_id is not None:
                    task_usage[task_id] = task_usage.get(task_id, 0) + (row["pss_bytes"] or 0)
                    task_upper[task_id] = task_upper.get(task_id, 0) + (
                        row["pss_bytes"] if row["pss_bytes"] is not None else (row["rss_bytes"] or 0))
                    if row["pss_bytes"] is None:
                        task_partial.add(task_id)
            over_task = next((task for task, used in task_usage.items()
                              if task in self._task_limits and used > self._task_limits[task]), None)
            unverified_task = next((task for task in task_partial if task in self._task_limits
                                    and task_upper[task] > self._task_limits[task]), None)
            if self.gpu_monitor is not None:
                try:
                    detail = self.gpu_monitor.sample(pid_tasks, self.gpu_reservations.reservations)
                    if detail:
                        self.resource_detail = detail
                        self._trigger("gpu_memory")
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    self.resource_detail = str(exc)
                    self._trigger("gpu_measurement_unavailable")
            if over_task is not None:
                kind, source = "task_memory_share", over_task
            elif usage > self.limit_bytes:
                kind = "workflow_memory"
            elif unverified_task or sample.unknown_memory or (
                not sample.supported and (sys.platform.startswith("linux") or self._available_memory_provider)
            ) or (
                sample.missing_pss and upper > self.limit_bytes
            ):
                kind = "measurement_unavailable"
                source = unverified_task
            else:
                kind = None
            # Sequential /proc reads are not an atomic snapshot. Confirm once
            # on the next scheduler poll before killing rapidly forking workers.
            if kind:
                self._record_snapshot(sample, kind + "_candidate", roots)
                candidate = (kind, roots, source)
                if self._candidate == candidate:
                    self.observed_pss_partial = bool(sample.missing_pss)
                    self.observed_upper_bound_complete = sample.upper_bound_complete
                    self._trigger(kind, source_task=source,
                                  observed_bytes=usage if kind in {"workflow_memory", "task_memory_share"} else upper)
                    self._record_snapshot(sample, kind + "_confirmed", roots)
                self._candidate = candidate
            else:
                self._candidate = None
        else:
            # Injectable samplers remain supported for tests/programmatic users.
            usage = max(0, int(sample))
            self.peak_upper_bound_bytes = max(self.peak_upper_bound_bytes, usage)
            if usage > self.limit_bytes:
                self._trigger("workflow_memory", observed_bytes=usage)
        with self._lock:
            self.peak_bytes = max(self.peak_bytes, usage)
        return usage

    def report_task_allocation_failure(self, task_id: str) -> None:
        self._trigger("task_allocation_failure", source_task=task_id)

    def abort_detail(self) -> str:
        with self._lock:
            reason = self.reason
            source_task = self.source_task
            observed = self.observed_bytes
        limit = (
            f"{self.limit_gb:g} GiB"
            if self.limit_gb is not None
            else "available host memory"
        )
        if reason == "workflow_memory" and observed is not None:
            metric = "measured workflow PSS lower bound" if self.observed_pss_partial else "total workflow PSS"
            return (
                f"{metric}={observed / 1024**3:.2f} GiB exceeded "
                f"--max-memory={limit}"
            )
        if reason == "task_memory_share":
            return f"task={source_task} exceeded its equal RAM share; the combined workflow budget is {limit}"
        if reason in {"gpu_memory", "gpu_measurement_unavailable"}:
            return f"GPU resource control: {self.resource_detail}; stopping task process groups safely"
        if reason == "ram_unavailable":
            return self.resource_detail
        if reason == "measurement_unavailable":
            if self.resource_detail:
                return self.resource_detail + "; stopping because the RAM resource ceiling cannot be verified"
            metric = ("conservative RSS-containing bound" if self.observed_upper_bound_complete
                      else "partial accounted amount")
            return (
                "PSS could not be read reliably for live task processes; "
                f"the {metric} is {(observed or 0) / 1024**3:.2f} GiB "
                f"and --max-memory={limit} cannot be verified. This is NOT a confirmed "
                "PSS breach. See per-process read errors in memory_diagnostics.jsonl; "
                "workflow stopped safely without increasing or disabling the limit"
            )
        if reason == "task_allocation_failure":
            return (
                f"task={source_task or 'unknown'} reported a CPU-memory allocation "
                f"failure under total workflow --max-memory={limit}"
            )
        return f"the total workflow memory budget ({limit}) was exceeded"


def _safe_log_name(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id) + ".log"


def _archive_retry_log(log_path: Path, attempt: int) -> Path | None:
    """Move a failed attempt log aside before the next attempt overwrites it."""
    if attempt < 1 or not log_path.is_file():
        return None
    candidate = log_path.with_name(
        f"{log_path.stem}.attempt-{attempt}{log_path.suffix}"
    )
    collision = 2
    while candidate.exists():
        candidate = log_path.with_name(
            f"{log_path.stem}.attempt-{attempt}.{collision}{log_path.suffix}"
        )
        collision += 1
    try:
        log_path.replace(candidate)
    except OSError:
        return None
    return candidate


def _local_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _elapsed_clock(started_at: float, now: float | None = None) -> str:
    total = max(0, int((time.monotonic() if now is None else now) - started_at))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_group_label(task: Task, sample_names: Sequence[str]) -> str:
    """Remove sample components from a task ID for concise cohort progress."""
    if task.sample is None:
        return task.id
    label = task.id
    for sample in sorted(sample_names, key=len, reverse=True):
        if label == sample:
            label = ""
            continue
        if label.startswith(sample + "."):
            label = label[len(sample) + 1 :]
        label = label.replace(f".{sample}.", ".")
        if label.endswith("." + sample):
            label = label[: -(len(sample) + 1)]
    return label.strip(".") or task.stage


def _sample_summary(samples: Sequence[str], maximum: int = 8) -> str:
    unique = list(dict.fromkeys(samples))
    shown = ",".join(unique[:maximum])
    if len(unique) > maximum:
        shown += f",+{len(unique) - maximum}"
    return shown


def _is_internal_sample_name(sample: str) -> bool:
    """Return whether SAMPLE is a workflow input aggregate, not a real sample."""
    return sample.startswith("semibin2_multisample_input")


def _terminal_text(value: str, maximum_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= maximum_chars:
        return compact
    return compact[: maximum_chars - 3] + "..."


def _unexpected_worker_result(
    task_id: str,
    log_path: Path,
    error: Exception,
) -> RunResult:
    """Convert an unexpected worker exception into a normal task failure."""
    error_text = str(error) or repr(error)
    diagnostic = f"{type(error).__name__}: {_terminal_text(error_text, 500)}"
    message = f"task execution worker raised an unexpected exception: {diagnostic}"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(
                "\n# executor_exception: "
                f"{diagnostic}\n"
                + "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                )
                + f"# finished: {_local_time()}\n# exit_code: 70\n"
            )
    except Exception as log_error:
        log_diagnostic = str(log_error) or repr(log_error)
        message += (
            f"; additionally could not append diagnostic information to {log_path}: "
            f"{type(log_error).__name__}: "
            f"{_terminal_text(log_diagnostic, 300)}"
        )
    return RunResult(task_id, 70, message)


def _failure_tail(log_path: Path, maximum_lines: int = 3, maximum_chars: int = 500) -> str:
    try:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if (
                line.strip()
                and not line.lstrip().startswith("#")
                and not re.match(
                    r"^(?:real|user|sys)\s+\d",
                    line.strip(),
                    flags=re.IGNORECASE,
                )
            )
        ]
    except OSError:
        return ""
    error_lines = [
        line
        for line in lines
        if re.search(
            r"(?:traceback|(?:runtime|import|module|value|type|memory)error:|"
            r"cuda out of memory|resource temporarily unavailable|cannot allocate memory|"
            r"insufficient memory|std::bad_alloc)",
            line,
            flags=re.IGNORECASE,
        )
    ]
    selected = [*error_lines[-2:], *lines[-maximum_lines:]]
    summary = " | ".join(dict.fromkeys(selected))
    if len(summary) > maximum_chars:
        summary = "..." + summary[-(maximum_chars - 3) :]
    return summary


def _memory_limit_exceeded(log_path: Path, return_code: int | None = None) -> bool:
    """Detect a CPU-memory allocation failure that requires a workflow abort."""
    try:
        lines = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return False
    # GPU exhaustion is handled separately through --max-gpu-memory. Do not
    # mistake it for a breach of the CPU --max-memory limit.
    cupy_trace = any("cupy" in line.lower() for line in lines)
    cpu_output = "\n".join(
        line for line in lines if "cuda out of memory" not in line.lower()
        and not ("outofmemoryerror" in line.lower() and
                 ("cuda" in line.lower() or cupy_trace))
    )
    if re.search(
        r"(?:\[morecore\]\s*insufficient memory|cannot allocate memory|"
        r"\bmemoryerror\b|std::bad_alloc|outofmemoryerror|"
        r"out of memory:\s*killed process|oom-kill|oom_kill)",
        cpu_output,
        flags=re.IGNORECASE,
    ):
        return True
    return return_code in {-9, 137} and bool(
        re.search(r"(?:out of memory|oom|\bkilled\b)", cpu_output, re.IGNORECASE)
    )


def _failure_diagnosis(log_path: Path) -> str:
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if ("cuda out of memory" in output.lower() or
        ("cupy" in output.lower() and "outofmemoryerror" in output.lower())):
        return (
            "GPU allocation failed under the assigned task share; reduce --task "
            "or the tool's batch size, or increase the aggregate --max-gpu-memory "
            "request only if it remains within 80% of currently free VRAM"
        )
    if "tf_data_private_threadpool creation via pthread_create() failed" in output:
        return (
            "TensorFlow could not create its prediction thread pool; the process reached "
            "a thread, process, or memory resource limit"
        )
    if "AF_UNIX path too long" in output:
        return (
            "Python multiprocessing could not create its Unix socket because the "
            "configured temporary path is too long"
        )
    if "EOFError" in output and "checkm2" in output.lower():
        return (
            "CheckM2's multiprocessing manager exited before the parent process could "
            "connect; an overlong TMPDIR Unix-socket path or a native worker crash is "
            "the likely cause"
        )
    if "libtorch_cpu.so" in output and "iJIT_NotifyEvent" in output:
        return (
            "LorBin's PyTorch runtime is incompatible with the installed Intel "
            "MKL version; run `metabaw check --all` to repair the "
            "isolated environment"
        )
    if "Something went wrong with running training network" in output:
        if "CUDA out of memory" in output:
            return (
                "COMEBin exhausted GPU memory while training; lower "
                "--batch-size or increase --max-gpu-memory"
            )
        return (
            "COMEBin failed while training its representation model; the specific "
            "Python error is included in the task log"
        )
    if "No module named 'pkg_resources'" in output or (
        'No module named "pkg_resources"' in output
    ):
        return (
            "CheckM requires pkg_resources, which was removed from setuptools 82; "
            "run `metabaw check --all` and install the offered "
            "setuptools<82 compatibility version"
        )
    if (
        "OMP: Error #34" in output
        or (
            "OMP:" in output
            and "unable to allocate necessary resources for OMP thread" in output
        )
    ):
        return (
            "OpenMP could not create a worker thread because nested numerical-library "
            "threads exhausted the process or thread resource limit"
        )
    if (
        "Failed to load consolidated database" in output
        and "Cannot allocate memory" in output
    ):
        return (
            "GTDB-Tk could not map its consolidated database into the process address "
            "space; a virtual-memory limit or insufficient host memory is the likely cause"
        )
    if _memory_limit_exceeded(log_path):
        return (
            "the task reported CPU-memory exhaustion under the total workflow "
            "--max-memory budget or available host memory; reduce "
            "--task/--threads or increase --max-memory"
        )
    return ""


def _output_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    try:
        return any(item.is_file() for item in path.rglob("*"))
    except OSError:
        return False


FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


def _fasta_validation_error(directory: Path, *, allow_empty_refinement: bool = False) -> str | None:
    if not directory.is_dir():
        return f"{directory} (FASTA bin directory is missing)"
    fasta_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and any(path.name.lower().endswith(suffix) for suffix in FASTA_SUFFIXES)
    )
    if allow_empty_refinement:
        try:
            empty = read_empty_refinement(directory)
        except ValueError as exc:
            return str(exc)
        if empty is not None:
            if fasta_files:
                return f"{directory} (empty-refinement evidence conflicts with FASTA bins)"
            return None
    if not fasta_files:
        return f"{directory} (no FASTA bins were generated)"
    for path in fasta_files:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        records = 0
        sequence_bases = 0
        current_has_sequence = False
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if records and not current_has_sequence:
                            return f"{path} (FASTA record has no sequence)"
                        if len(line) == 1:
                            return f"{path} (FASTA header is empty)"
                        records += 1
                        current_has_sequence = False
                    else:
                        if records == 0:
                            return f"{path} (sequence appears before the first FASTA header)"
                        sequence_bases += len(line)
                        current_has_sequence = True
        except OSError as exc:
            return f"{path} (cannot read FASTA: {exc})"
        if records == 0 or sequence_bases == 0 or not current_has_sequence:
            return f"{path} (FASTA contains no complete sequence record)"
    return None


def _validation_errors(task: Task) -> list[str]:
    return [
        error
        for directory in task.fasta_output_dirs
        if (error := _fasta_validation_error(
            directory, allow_empty_refinement=task.allow_empty_refinement,
        )) is not None
    ]


def _outputs_exist(task: Task) -> bool:
    required = all(_output_complete(path) for path in task.outputs)
    alternatives = (
        not task.output_alternatives
        or any(
            all(_output_complete(path) for path in output_set)
            for output_set in task.output_alternatives
        )
    )
    return required and alternatives and not _validation_errors(task)


def _is_same_or_within(path: Path, parent: Path) -> bool:
    try:
        # Ownership is lexical. Resolving a copied link would incorrectly
        # classify a generated BAM as an external input in the old directory.
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(parent)))
    except ValueError:
        return False
    return True


def _legacy_generated_inputs(
    ordered: Sequence[Task],
) -> dict[str, frozenset[Path]]:
    by_id = {task.id: task for task in ordered}
    upstream_outputs: dict[str, tuple[Path, ...]] = {}
    generated: dict[str, frozenset[Path]] = {}
    for task in ordered:
        candidates: list[Path] = []
        for dependency in task.deps:
            candidates.extend(upstream_outputs[dependency])
            candidates.extend(by_id[dependency].outputs)
        upstream_outputs[task.id] = tuple(dict.fromkeys(candidates))
        generated[task.id] = frozenset(
            path
            for path in task.inputs
            if any(_is_same_or_within(path, output) for output in candidates)
        )
    return generated


def _legacy_resource_resume_matches(
    task: Task,
    record: TaskRecord,
    dependency_records: Sequence[TaskRecord | None],
    generated_inputs: frozenset[Path],
) -> bool:
    """Validate old strict hashes using the original command in the task log.

    A hash alone cannot reveal old thread counts. Reconstructing and matching
    that hash also verifies current input metadata and old dependency hashes;
    merely seeing complete outputs or a similar command is not sufficient.
    """
    if not record.log_path or any(item is None for item in dependency_records):
        return False
    header: dict[str, str] = {}
    command_lines: list[str] = []
    header_bytes = 0
    complete = False
    try:
        with Path(record.log_path).open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                header_bytes += len(line)
                if header_bytes > 256 * 1024:
                    return False
                if line.startswith("# workflow_max_memory_gb:"):
                    complete = True
                    break
                if command_lines:
                    command_lines.append(line)
                elif line.startswith("# command: "):
                    command_lines.append(line[len("# command: "):])
                elif line.startswith("# ") and ": " in line:
                    key, value = line[2:].split(": ", 1)
                    header[key] = value.rstrip("\n")
    except OSError:
        return False
    if (not complete or not command_lines
            or checkpoint_task_id(header.get("task", "")) != checkpoint_task_id(task.id)
            or any(
        header.get(key) != value
        for key, value in (("stage", task.stage), ("cwd", str(task.cwd)))
    )):
        return False
    command = "".join(command_lines).removesuffix("\n")
    environment: dict[str, str] = {}
    overrides = header.get("environment_overrides", "")
    matches = list(re.finditer(r"(?:^| )([A-Za-z_][A-Za-z0-9_]*)=", overrides))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(overrides)
        environment[match.group(1)] = overrides[match.end():end]
    if overrides and not matches:
        return False
    if (resume_command(command) != resume_command(task.display_command())
            or resume_environment(environment) != resume_environment(task.env)):
        return False
    dependencies = [item.fingerprint for item in dependency_records if item is not None]
    cpu_candidates = {task.cpus, 1, *command_thread_counts(command)}
    for cpus in cpu_candidates:
        original = replace(task, command=command, cpus=cpus, env=environment)
        if record.fingerprint in {
            original.fingerprint(dependencies),
            original.fingerprint(dependencies, assume_missing_inputs=generated_inputs),
        }:
            return True
    return False


def _missing_outputs(task: Task) -> list[str]:
    missing = [
        (
            f"{path} (empty output directory)"
            if path.exists() and path.is_dir()
            else str(path)
        )
        for path in task.outputs
        if not _output_complete(path)
    ]
    if task.output_alternatives and not any(
        all(_output_complete(path) for path in output_set)
        for output_set in task.output_alternatives
    ):
        choices = " OR ".join(
            "[" + ", ".join(str(path) for path in output_set) + "]"
            for output_set in task.output_alternatives
        )
        missing.append(f"one complete alternative output set: {choices}")
    missing.extend(_validation_errors(task))
    return missing


def _remove_tolerated_outputs(task: Task) -> None:
    if not task.failure_tolerated:
        return
    paths = list(task.outputs)
    for output_set in task.output_alternatives:
        paths.extend(output_set)
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = 3.0,
) -> None:
    """Terminate the complete process group created for one workflow task."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_process(
    process: subprocess.Popen,
    memory_budget: _WorkflowMemoryBudget | None,
) -> tuple[int, bool]:
    """Wait for a task, terminating its process tree after a memory abort."""
    while True:
        try:
            return process.wait(timeout=0.2), False
        except subprocess.TimeoutExpired:
            if memory_budget is not None and memory_budget.event.is_set():
                _terminate_process_tree(process)
                return process.wait(), True


def _execute_task(
    task: Task,
    log_path: Path,
    shell_executable: str,
    max_memory_gb: float | None,
    temp_dir: Path | None,
    memory_budget: _WorkflowMemoryBudget | None = None,
) -> RunResult:
    task.cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(task.env)
    hook_directory = None
    if "METABAW_GPU_BUDGET_BYTES" in task.env:
        parent = temp_dir or log_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        hook_directory = Path(tempfile.mkdtemp(prefix=".metabaw-gpu-", dir=parent))
        shutil.copyfile(Path(__file__).with_name("gpu_limits.py"), hook_directory / "sitecustomize.py")
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(hook_directory), environment.get("PYTHONPATH"))))
        environment["CUPY_GPU_MEMORY_LIMIT"] = task.env.get("METABAW_GPU_ALLOCATOR_BYTES",
                                                           task.env["METABAW_GPU_BUDGET_BYTES"])
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "TMPDIR": str(temp_dir),
                "TMP": str(temp_dir),
                "TEMP": str(temp_dir),
            }
        )
    task_memory_gb = task.memory_limit_gb if task.memory_limit_gb is not None else max_memory_gb
    if task_memory_gb is not None and task.enforce_memory_limit:
        environment["METABAW_MAX_MEMORY_GB"] = f"{task_memory_gb:g}"
    started_at = _local_time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(
            f"# task: {task.id}\n# stage: {task.stage}\n# description: {task.description}\n"
            f"# started: {started_at}\n# cwd: {task.cwd}\n"
        )
        log.write(f"# temp_dir: {temp_dir if temp_dir is not None else 'system default'}\n")
        if task.env:
            overrides = " ".join(
                f"{name}={value}" for name, value in sorted(task.env.items())
            )
            log.write(f"# environment_overrides: {overrides}\n")
        log.write(f"# command: {task.display_command()}\n")
        log.write(
            "# workflow_max_memory_gb: "
            f"{max_memory_gb if max_memory_gb is not None else 'unlimited'}\n"
        )
        log.write(f"# task_memory_share_gb: {task_memory_gb}\n")
        log.write(f"# cpu_affinity: {','.join(map(str, task.cpu_affinity)) or 'not explicitly assigned'}\n")
        if hook_directory is not None:
            log.write(f"# gpu_python_limit_hook: {hook_directory}; inherited by isolated Python workers\n")
        log.write(
            "# workflow_memory_scope: combined measured PSS of active task process trees; "
            "smaps fallback; RSS is an upper bound only; exited PIDs excluded\n"
            "# per_process_address_limit: "
            f"{'enabled as a secondary guard' if task.enforce_memory_limit else 'disabled; task remains included in workflow memory accounting'}\n\n"
        )
        log.flush()
        try:
            process_options: dict[str, object] = {}
            if os.name == "posix":
                process_options["start_new_session"] = True
            elif os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            if isinstance(task.command, str):
                command = task.command
                if (
                    task_memory_gb is not None
                    and task.enforce_memory_limit
                    and os.name == "posix"
                ):
                    limit_kib = max(1, int(task_memory_gb * 1024 * 1024))
                    command = f"ulimit -v {limit_kib}\n{command}"
                launch = [shell_executable, "-c", command]
                if task.cpu_affinity and os.name == "posix":
                    launch = [sys.executable, "-B", str(Path(__file__).with_name("resource_launcher.py")),
                              ",".join(map(str, task.cpu_affinity)), "--", *launch]
                process = subprocess.Popen(
                    launch if task.cpu_affinity and os.name == "posix" else command,
                    cwd=task.cwd,
                    env=environment,
                    shell=not (task.cpu_affinity and os.name == "posix"),
                    executable=None if task.cpu_affinity and os.name == "posix" else shell_executable,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **process_options,
                )
            else:
                launch = list(task.command)
                if task.cpu_affinity and os.name == "posix":
                    launch = [sys.executable, "-B", str(Path(__file__).with_name("resource_launcher.py")),
                              ",".join(map(str, task.cpu_affinity)), "--", *launch]
                process = subprocess.Popen(
                    launch,
                    cwd=task.cwd,
                    env=environment,
                    shell=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **process_options,
                )
        except OSError as exc:
            if hook_directory is not None:
                shutil.rmtree(hook_directory)
            log.write(f"\nmetaBAW could not start the task: {exc}\n")
            log.write(f"# finished: {_local_time()}\n# exit_code: 127\n")
            return RunResult(task.id, 127, str(exc))
        if memory_budget is not None:
            memory_budget.register(task.id, process.pid)
        try:
            try:
                return_code, aborted = _wait_for_process(process, memory_budget)
            except Exception:
                _terminate_process_tree(process)
                raise
        finally:
            if memory_budget is not None:
                memory_budget.unregister(process.pid)
            if hook_directory is not None:
                shutil.rmtree(hook_directory)
        log.flush()
        memory_exceeded = (
            not aborted
            and return_code != 0
            and _memory_limit_exceeded(log_path, return_code)
        )
        if memory_exceeded:
            if memory_budget is not None:
                memory_budget.report_task_allocation_failure(task.id)
            limit = (
                f"{max_memory_gb:g} GiB"
                if max_memory_gb is not None
                else "the available host memory"
            )
            log.write(
                f"\n# memory_limit_exceeded: task reported a CPU-memory "
                f"allocation failure under total workflow --max-memory={limit}\n"
                "# workflow_abort: all other running task process groups are "
                "being terminated; pending tasks will not start\n"
            )
        elif aborted:
            detail = (
                memory_budget.abort_detail()
                if memory_budget is not None
                else "the workflow memory budget was exceeded"
            )
            log.write(
                "\n# workflow_abort: task process group was terminated because "
                f"{detail}\n"
            )
        log.write(f"\n# finished: {_local_time()}\n# exit_code: {return_code}\n")
        log.flush()
    if aborted:
        return RunResult(
            task.id,
            130,
            "terminated because the total workflow memory limit was exceeded",
            aborted_by_memory_limit=True,
        )
    if return_code != 0:
        message = f"command exited with code {return_code}"
        diagnosis = _failure_diagnosis(log_path)
        if diagnosis:
            message += f"; diagnosis: {diagnosis}"
        tail = _failure_tail(log_path)
        if tail:
            message += f"; last output: {tail}"
        return RunResult(
            task.id,
            return_code,
            message,
            memory_limit_exceeded=memory_exceeded,
        )
    missing = _missing_outputs(task)
    if missing:
        message = f"expected outputs were not created: {missing}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# validation_error: {message}\n")
        return RunResult(task.id, 98, message)
    return RunResult(task.id, 0, "ok")


class Executor:
    def __init__(
        self,
        state: StateStore,
        log_dir: Path,
        max_cpus: int,
        max_parallel: int,
        shell_executable: str = "/bin/bash",
        retries: int = 0,
        fail_fast: bool = True,
        max_memory_gb: float | None = None,
        show_progress: bool = False,
        temp_dir: Path | None = None,
        max_gpus: int = 0,
        started_at: float | None = None,
        memory_sampler: MemorySampler | None = None,
        memory_diagnostic_path: Path | None = None,
        resource_plan: ResourcePlan | None = None,
        gpu_device_budgets=(),
        gpu_share_bytes: int = 0,
        parallel_ready_tasks: bool = False,
        gpu_sampler=None,
        resume_layout: ResumeLayout | None = None,
    ):
        self.state = state
        self.log_dir = log_dir
        self.max_cpus = max_cpus
        self.max_parallel = max_parallel
        self.shell_executable = shell_executable
        self.retries = retries
        self.fail_fast = fail_fast
        self.max_memory_gb = max_memory_gb
        self.show_progress = show_progress
        self.temp_dir = temp_dir
        self.max_gpus = max(0, max_gpus)
        self.started_at = time.monotonic() if started_at is None else started_at
        self.memory_sampler = memory_sampler
        self.memory_diagnostic_path = memory_diagnostic_path or log_dir / "memory_diagnostics.jsonl"
        self.peak_memory_gb = 0.0
        self.peak_memory_upper_bound_gb = 0.0
        self.memory_pss_complete = True
        self.memory_upper_bound_complete = True
        self.memory_abort_reason: str | None = None
        self.resource_plan = resource_plan
        self.gpu_reservations = GpuReservations(tuple(gpu_device_budgets), gpu_share_bytes)
        self.gpu_monitor = (GpuMonitor(tuple(gpu_device_budgets), gpu_share_bytes, sampler=gpu_sampler)
                            if gpu_device_budgets else None)
        self.parallel_ready_tasks = parallel_ready_tasks
        self.resume_layout = resume_layout

    def _emit(
        self,
        event: str,
        label: str,
        position: int,
        total: int,
        detail: str,
    ) -> None:
        if not self.show_progress:
            return
        print(
            f"[{_elapsed_clock(self.started_at)}] "
            f"[{event} {position}/{total}] {label} - {detail}",
            flush=True,
        )

    def run(self, tasks: Sequence[Task], force: bool = False) -> dict[str, str]:
        ordered = topological_order(tasks)
        oversized_gpu_tasks = [
            task.id for task in ordered if task.gpus > self.max_gpus
        ]
        if oversized_gpu_tasks:
            raise ValueError(
                "GPU tasks exceed the detected GPU capacity: "
                + ", ".join(oversized_gpu_tasks)
            )
        by_id = {task.id: task for task in ordered}
        positions = {task.id: number for number, task in enumerate(ordered, start=1)}
        total = len(ordered)
        legacy_generated_inputs = _legacy_generated_inputs(ordered)
        if self.resume_layout is not None and not force:
            external_inputs = frozenset(path for task in ordered
                                       for path in external_locations(task, legacy_generated_inputs[task.id]))
            repaired = self.resume_layout.prepare(self.state, external_inputs)
            if self.show_progress and self.resume_layout.origins:
                print(f"[{_elapsed_clock(self.started_at)}] [RESUME RELOCATION] "
                      f"Checking copied/renamed workflow; repaired_internal_links={repaired}.", flush=True)
        fingerprints: dict[str, str] = {}
        resume_fingerprints: dict[str, str] = {}
        for task in ordered:
            fingerprints[task.id] = task.fingerprint([fingerprints[dep] for dep in task.deps])
            resume_fingerprints[task.id] = task.resume_fingerprint(
                [resume_fingerprints[dep] for dep in task.deps]
            )
        portable_checkpoints: dict[str, str] = {}
        portable_signatures: dict[str, str] = {}
        if self.resume_layout is not None:
            for task in ordered:
                checkpoint = portable_checkpoint(task, legacy_generated_inputs[task.id],
                    [portable_signatures[dep] for dep in task.deps], self.resume_layout.current)
                portable_checkpoints[task.id] = checkpoint
                portable_signatures[task.id] = checkpoint_signature(checkpoint)
        legacy_fingerprints: dict[str, str] = {}
        legacy_resume_fingerprints: dict[str, str] = {}
        for task in ordered:
            legacy_fingerprints[task.id] = task.fingerprint(
                [legacy_fingerprints[dependency] for dependency in task.deps],
                assume_missing_inputs=legacy_generated_inputs[task.id],
            )
            legacy_resume_fingerprints[task.id] = task.resume_fingerprint(
                [legacy_resume_fingerprints[dependency] for dependency in task.deps],
                assume_missing_inputs=legacy_generated_inputs[task.id],
            )

        cache_hits: set[str] = set()
        adopted_hits: set[str] = set()
        resource_hits: set[str] = set()
        relocation_hits: set[str] = set()
        records = {task.id: self.state.get(task.id) for task in ordered}
        output_states: dict[str, bool] = {}
        for task in ordered:
            record = records[task.id]
            outputs_exist = _outputs_exist(task)
            output_states[task.id] = outputs_exist
            fingerprint_matches = (
                record is not None
                and record.fingerprint
                in {fingerprints[task.id], legacy_fingerprints[task.id]}
            )
            if (
                not force
                and outputs_exist
                and bool(task.outputs or task.output_alternatives)
                # If a data dependency will rerun, its downstream output must
                # not be reused just because the old output still exists.
                and all(dependency in cache_hits for dependency in task.deps)
            ):
                if record is None:
                    cache_hits.add(task.id)
                    adopted_hits.add(task.id)
                    self.state.adopt(
                        task.id,
                        fingerprints[task.id],
                        "Adopted complete outputs during startup resume scan",
                        resume_fingerprint=resume_fingerprints[task.id],
                        portable_checkpoint=portable_checkpoints.get(task.id),
                    )
                    records[task.id] = self.state.get(task.id)
                elif record.status == "success":
                    resource_matches = record.resume_fingerprint in {
                        resume_fingerprints[task.id], legacy_resume_fingerprints[task.id],
                    }
                    if not fingerprint_matches and not resource_matches:
                        resource_matches = _legacy_resource_resume_matches(
                            task, record, [records[dep] for dep in task.deps],
                            legacy_generated_inputs[task.id],
                        )
                    relocated = False
                    if not fingerprint_matches and not resource_matches and self.resume_layout is not None:
                        relocated = self.resume_layout.matches(task, record, legacy_generated_inputs[task.id],
                            [records[dep] for dep in task.deps], portable_checkpoints[task.id])
                    if fingerprint_matches or resource_matches or relocated:
                        cache_hits.add(task.id)
                        if relocated:
                            relocation_hits.add(task.id)
                        elif not fingerprint_matches:
                            resource_hits.add(task.id)
                        if record.resume_fingerprint != resume_fingerprints[task.id]:
                            self.state.set_resume_fingerprint(task.id, resume_fingerprints[task.id])
                        if self.resume_layout is not None:
                            self.state.set_portable_checkpoint(task.id, portable_checkpoints[task.id])

        if self.resume_layout is not None:
            decisions = []
            for task in ordered:
                record = records[task.id]
                blocked = [dep for dep in task.deps if dep not in cache_hits]
                if task.id in cache_hits:
                    reason = ("adopted complete outputs" if task.id in adopted_hits else
                              "compatible relocated/helper checkpoint" if task.id in relocation_hits else
                              "compatible resource settings" if task.id in resource_hits else "matching checkpoint")
                elif force:
                    reason = "forced execution"
                elif blocked:
                    reason = "data dependency must rerun"
                elif record is not None and record.status != "success":
                    reason = "previous task status: " + record.status
                elif not bool(task.outputs or task.output_alternatives):
                    reason = "task has no reusable outputs"
                elif not output_states[task.id]:
                    reason = "expected outputs are missing, incomplete or invalid"
                elif record is None:
                    reason = "no previous task record"
                else:
                    reason = self.resume_layout.reasons.get(task.id, "input, command or checkpoint differs")
                decisions.append({"task_id": task.id, "previous_status": record.status if record else None,
                                  "action": "reuse" if task.id in cache_hits else "run",
                                  "reason": reason, "blocked_by": blocked})
            diagnostic = self.log_dir.parent / "resume_diagnostics.json"
            try:
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                diagnostic.write_text(json.dumps({"tasks": decisions}, indent=2) + "\n", encoding="utf-8")
                if self.show_progress:
                    invalidated = [row for row in decisions if row["action"] == "run" and
                                   row["previous_status"] == "success" and not row["blocked_by"]]
                    for row in invalidated[:3]:
                        print(f"[{_elapsed_clock(self.started_at)}] [RESUME INVALIDATED] "
                              f"{row['task_id']}: {row['reason']}.", flush=True)
                    if invalidated:
                        print(f"[{_elapsed_clock(self.started_at)}] [RESUME DETAILS] {diagnostic}", flush=True)
            except OSError as exc:
                if self.show_progress:
                    print(f"[RESUME WARNING] Could not save resume decisions: {exc}", flush=True)

        if self.show_progress:
            adopted = len(adopted_hits)
            adopted_detail = f", adopted={adopted}" if adopted else ""
            resource_detail = f", resource-compatible={len(resource_hits)}" if resource_hits else ""
            relocation_detail = f", relocated={len(relocation_hits)}" if relocation_hits else ""
            print(
                f"[{_elapsed_clock(self.started_at)}] [RESUME] "
                f"Reusing {len(cache_hits)}/{total} "
                f"complete tasks; {total - len(cache_hits)} will run"
                f"{adopted_detail}{resource_detail}{relocation_detail}.",
                flush=True,
            )

        statuses: dict[str, str] = {}
        pending: set[str] = set(by_id)
        attempts = {task_id: 0 for task_id in by_id}
        running: dict[Future[RunResult], tuple[str, int, int, float, Path]] = {}
        memory_budget = _WorkflowMemoryBudget(
            self.max_memory_gb,
            sampler=self.memory_sampler,
            diagnostic_path=self.memory_diagnostic_path,
            available_memory_provider=available_memory_gib if self.resource_plan else None,
            gpu_monitor=self.gpu_monitor,
            gpu_reservations=self.gpu_reservations,
        )
        resource_allocations: dict[str, tuple[tuple[int, ...], float]] = {}
        memory_notice_emitted = False
        active_samples: set[str] = set()
        used_cpus = 0
        used_gpus = 0
        failure_seen = False
        blocked_task_ids: list[str] = []
        sample_names = sorted(
            {
                task.sample
                for task in ordered
                if task.sample is not None
            },
            key=str,
        )
        progress_key_by_task: dict[str, str] = {}
        progress_members: dict[str, list[str]] = {}
        progress_labels: dict[str, str] = {}
        for task in ordered:
            if task.id in cache_hits:
                continue
            label = _progress_group_label(task, sample_names)
            key = f"{task.stage}\0{label}"
            progress_key_by_task[task.id] = key
            progress_members.setdefault(key, []).append(task.id)
            progress_labels[key] = label
        progress_positions = {
            key: index
            for index, key in enumerate(progress_members, start=1)
        }
        progress_order = list(progress_members)
        progress_total = len(progress_members)
        progress_started: set[str] = set()
        progress_finished: set[str] = set()
        progress_started_at: dict[str, float] = {}
        progress_failures: dict[
            str,
            list[tuple[str, str | None, Path]],
        ] = {}

        def announce_memory_abort() -> None:
            nonlocal memory_notice_emitted
            if memory_notice_emitted or not memory_budget.event.is_set():
                return
            memory_notice_emitted = True
            if self.show_progress:
                label = {"measurement_unavailable": "MEMORY ACCOUNTING ERROR",
                         "gpu_measurement_unavailable": "GPU ACCOUNTING ERROR",
                         "gpu_memory": "GPU MEMORY LIMIT"}.get(memory_budget.reason, "MEMORY LIMIT")
                print(
                    f"[{_elapsed_clock(self.started_at)}] [{label}] "
                    f"{memory_budget.abort_detail()}; terminating "
                    f"{len(running)} active task process group(s), disabling "
                    f"retries, and blocking pending tasks. Diagnostics={self.memory_diagnostic_path}",
                    flush=True,
                )

        def start_progress_group(
            task: Task,
            requested_cpus: int,
            requested_gpus: int,
        ) -> None:
            key = progress_key_by_task[task.id]
            if key in progress_started:
                return
            progress_started.add(key)
            progress_started_at[key] = time.monotonic()
            members = [by_id[task_id] for task_id in progress_members[key]]
            samples = [
                member.sample
                for member in members
                if (
                    member.sample is not None
                    and not _is_internal_sample_name(member.sample)
                )
            ]
            resources = [
                (
                    f"cpu={requested_cpus}/sample"
                    if samples
                    else f"cpu={requested_cpus}"
                )
            ]
            if requested_gpus:
                resources.append(
                    (
                        f"gpu={requested_gpus}/sample"
                        if samples
                        else f"gpu={requested_gpus}"
                    )
                )
            if samples:
                unique_samples = list(dict.fromkeys(samples))
                detail = (
                    f"running {len(members)} task(s) across "
                    f"{len(unique_samples)} sample(s) "
                    f"[{_sample_summary(unique_samples)}] "
                    f"({', '.join(resources)})"
                )
            else:
                detail = (
                    f"{_terminal_text(task.description or 'Run task', 180)} "
                    f"({', '.join(resources)})"
                )
            self._emit(
                "STEP",
                progress_labels[key],
                progress_positions[key],
                progress_total,
                detail,
            )

        def finish_ready_progress_groups() -> None:
            terminal_states = {"success", "failed", "blocked"}
            for key in progress_order:
                if key in progress_finished:
                    continue
                members = progress_members[key]
                if not all(statuses.get(task_id) in terminal_states for task_id in members):
                    continue
                if key not in progress_started:
                    progress_finished.add(key)
                    continue
                progress_finished.add(key)
                elapsed = _duration(
                    time.monotonic() - progress_started_at[key]
                )
                failed = progress_failures.get(key, [])
                if not failed:
                    continue
                failed_samples = list(
                    dict.fromkeys(
                        sample or task_id
                        for task_id, sample, _ in failed
                    )
                )
                detail = (
                    f"finished in {elapsed}; failed={len(failed)}, "
                    f"affected=[{_sample_summary(failed_samples)}]"
                )
                self._emit(
                    "FAIL",
                    progress_labels[key],
                    progress_positions[key],
                    progress_total,
                    detail,
                )

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while pending or running:
                made_progress = False
                memory_budget.sample()
                announce_memory_abort()
                if not memory_budget.event.is_set() and not memory_budget.awaiting_confirmation and not (
                    failure_seen and self.fail_fast
                ):
                    ready = [
                        by_id[task_id]
                        for task_id in pending
                        if (
                            all(
                                statuses.get(dep) in {"success", "skipped"}
                                for dep in by_id[task_id].scheduling_dependencies
                            )
                            or (
                                by_id[task_id].allow_failed_deps
                                and all(
                                    statuses.get(dep)
                                    in {"success", "skipped", "failed", "blocked"}
                                    for dep in by_id[task_id].scheduling_dependencies
                                )
                            )
                        )
                    ]
                    ready.sort(key=lambda task: (task.stage, task.priority, task.id))
                    active_progress_key = next(
                        (
                            key
                            for key in progress_order
                            if key not in progress_finished
                        ),
                        None,
                    )
                    for task in ready:
                        if task.id in cache_hits:
                            statuses[task.id] = "skipped"
                            pending.remove(task.id)
                            made_progress = True
                            continue
                        if not self.parallel_ready_tasks and progress_key_by_task[task.id] != active_progress_key:
                            continue
                        if task.sample is not None and task.sample in active_samples:
                            continue
                        requested = min(max(task.cpus, 1), self.max_cpus)
                        requested_gpus = max(0, task.gpus)
                        if (
                            len(running) >= self.max_parallel
                            or used_cpus + requested > self.max_cpus
                            or used_gpus + requested_gpus > self.max_gpus
                        ):
                            continue
                        runtime_task = task
                        if self.resource_plan is not None:
                            requested_memory = (memory_budget.limit_gb / self.max_parallel
                                                if task.sample is not None else memory_budget.limit_gb)
                            used_memory = sum(memory for ids, memory in resource_allocations.values())
                            if used_memory + requested_memory > memory_budget.limit_gb + 1e-9:
                                continue
                            occupied = {cpu for identifiers, memory in resource_allocations.values() for cpu in identifiers}
                            free = [cpu for cpu in self.resource_plan.cpu_ids if cpu not in occupied]
                            if len(free) < requested:
                                continue
                            affinity = tuple(free[:requested])
                            environment = dict(task.env)
                            # No CPU task may activate CUDA outside GPU accounting.
                            environment["CUDA_VISIBLE_DEVICES"] = ""
                            if requested_gpus:
                                if self.gpu_monitor is not None and self.gpu_reservations.refresh_idle_share(
                                    self.max_parallel, self.gpu_monitor.current_limits
                                ):
                                    self.gpu_monitor.share_bytes = self.gpu_reservations.share_bytes
                                    if self.show_progress:
                                        print(f"[GPU RESOURCE CAP] Current free VRAM changed; "
                                              f"equal task share={self.gpu_reservations.share_bytes / 1024**3:.3f} GiB.",
                                              flush=True)
                                if self.gpu_reservations.share_bytes <= 0:
                                    memory_budget.resource_detail = "No positive GPU task share fits current available VRAM"
                                    memory_budget._trigger("gpu_measurement_unavailable")
                                    break
                                device = self.gpu_reservations.acquire(
                                    task.id, self.gpu_monitor.current_limits if self.gpu_monitor else None)
                                if device is None:
                                    continue
                                share = self.gpu_reservations.share_bytes
                                # Leave room for CUDA contexts/library allocations,
                                # which are not included in the native allocator cap.
                                allocator = share - min(512 * 1024**2, share // 10)
                                environment.update({"CUDA_VISIBLE_DEVICES": device.uuid or device.index,
                                                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                                    "METABAW_GPU_BUDGET_BYTES": str(self.gpu_reservations.share_bytes),
                                                    "METABAW_GPU_ALLOCATOR_BYTES": str(allocator),
                                                    "METABAW_MAX_GPU_MEMORY": str(self.gpu_reservations.share_bytes)})
                            # CUDA drivers reserve large virtual address ranges;
                            # RLIMIT_AS is not an appropriate VRAM/RAM guard for them.
                            runtime_task = replace(task, cpu_affinity=affinity, memory_limit_gb=requested_memory,
                                                   enforce_memory_limit=task.enforce_memory_limit and not requested_gpus,
                                                   env=environment)
                            resource_allocations[task.id] = (affinity, requested_memory)
                            memory_budget.set_task_limit(task.id, requested_memory)
                        fingerprints[task.id] = task.fingerprint(
                            [fingerprints[dependency] for dependency in task.deps]
                        )
                        resume_fingerprints[task.id] = task.resume_fingerprint(
                            [resume_fingerprints[dependency] for dependency in task.deps]
                        )
                        attempts[task.id] += 1
                        log_path = self.log_dir / _safe_log_name(task.id)
                        self.state.start(task.id, fingerprints[task.id], log_path,
                                         resume_fingerprint=resume_fingerprints[task.id],
                                         portable_checkpoint=portable_checkpoints.get(task.id))
                        start_progress_group(task, requested, requested_gpus)
                        started = time.monotonic()
                        future = pool.submit(
                            _execute_task,
                            runtime_task,
                            log_path,
                            self.shell_executable,
                            self.max_memory_gb,
                            self.temp_dir,
                            memory_budget,
                        )
                        running[future] = (
                            task.id,
                            requested,
                            requested_gpus,
                            started,
                            log_path,
                        )
                        used_cpus += requested
                        used_gpus += requested_gpus
                        if task.sample is not None:
                            active_samples.add(task.sample)
                        pending.remove(task.id)
                        statuses[task.id] = "running"
                        made_progress = True

                if running:
                    done, _ = wait(running, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        (
                            task_id,
                            requested,
                            requested_gpus,
                            started,
                            log_path,
                        ) = running.pop(future)
                        task = by_id[task_id]
                        resource_allocations.pop(task_id, None)
                        self.gpu_reservations.release(task_id)
                        elapsed = _duration(time.monotonic() - started)
                        used_cpus -= requested
                        used_gpus -= requested_gpus
                        if task.sample is not None:
                            active_samples.discard(task.sample)
                        try:
                            result = future.result()
                        except Exception as error:
                            result = _unexpected_worker_result(
                                task_id,
                                log_path,
                                error,
                            )
                        if result.memory_limit_exceeded:
                            memory_budget.report_task_allocation_failure(task_id)
                            announce_memory_abort()
                            statuses[task_id] = "failed"
                            failure_seen = True
                            _remove_tolerated_outputs(task)
                            self.state.finish(
                                task_id,
                                "failed",
                                result.return_code,
                                result.message,
                            )
                            key = progress_key_by_task[task_id]
                            progress_failures.setdefault(key, []).append(
                                (task_id, task.sample, log_path)
                            )
                        elif result.aborted_by_memory_limit:
                            statuses[task_id] = "blocked"
                            _remove_tolerated_outputs(task)
                            self.state.finish(
                                task_id,
                                "blocked",
                                result.return_code,
                                result.message,
                            )
                            blocked_task_ids.append(task_id)
                        elif result.return_code == 0:
                            statuses[task_id] = "success"
                            self.state.finish(task_id, "success", 0, result.message)
                        elif attempts[task_id] <= max(
                            self.retries,
                            task.automatic_retries,
                        ):
                            archived_log = _archive_retry_log(
                                log_path,
                                attempts[task_id],
                            )
                            statuses.pop(task_id, None)
                            pending.add(task_id)
                            self.state.finish(task_id, "retry", result.return_code, result.message)
                            key = progress_key_by_task[task_id]
                            self._emit(
                                "RETRY",
                                progress_labels[key],
                                progress_positions[key],
                                progress_total,
                                (
                                    f"sample={task.sample or task.id}; attempt "
                                    f"{attempts[task_id]} failed after {elapsed}: "
                                    f"{_terminal_text(result.message, 500)}; "
                                    f"log={archived_log or log_path}"
                                ),
                            )
                        else:
                            statuses[task_id] = "failed"
                            failure_seen = True
                            _remove_tolerated_outputs(task)
                            self.state.finish(task_id, "failed", result.return_code, result.message)
                            key = progress_key_by_task[task_id]
                            progress_failures.setdefault(key, []).append(
                                (task_id, task.sample, log_path)
                            )
                            self._emit(
                                "ERROR",
                                progress_labels[key],
                                progress_positions[key],
                                progress_total,
                                (
                                    f"sample={task.sample or task.id} failed after {elapsed}: "
                                    f"{_terminal_text(result.message, 500)}; "
                                    f"log={log_path}"
                                ),
                            )
                        made_progress = True
                elif pending:
                    blocked = [
                        task_id
                        for task_id in pending
                        if (
                            not by_id[task_id].allow_failed_deps
                            and any(
                                statuses.get(dep) in {"failed", "blocked"}
                                for dep in by_id[task_id].scheduling_dependencies
                            )
                        )
                    ]
                    if memory_budget.event.is_set() or (
                        failure_seen and self.fail_fast
                    ):
                        blocked = list(pending)
                    if blocked:
                        for task_id in sorted(blocked, key=positions.get):
                            statuses[task_id] = "blocked"
                            pending.remove(task_id)
                            task = by_id[task_id]
                            _remove_tolerated_outputs(task)
                            blocked_task_ids.append(task_id)
                        made_progress = True
                finish_ready_progress_groups()
                if not made_progress:
                    time.sleep(0.05)
        self.peak_memory_gb = memory_budget.peak_bytes / 1024**3
        self.memory_upper_bound_complete = memory_budget.upper_bound_complete
        self.peak_memory_upper_bound_gb = (
            memory_budget.peak_upper_bound_bytes / 1024**3
            if self.memory_upper_bound_complete else None
        )
        self.memory_pss_complete = not memory_budget._degraded_recorded
        self.memory_abort_reason = (
            memory_budget.abort_detail()
            if memory_budget.event.is_set()
            else None
        )
        if self.show_progress and blocked_task_ids:
            preview = ", ".join(blocked_task_ids[:5])
            remaining = len(blocked_task_ids) - 5
            if remaining > 0:
                preview += f", +{remaining} more"
            print(
                f"[{_elapsed_clock(self.started_at)}] [BLOCKED] "
                f"{len(blocked_task_ids)} downstream "
                f"task(s) were not run: {preview}.",
                flush=True,
            )
        return statuses


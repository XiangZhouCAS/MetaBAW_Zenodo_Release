"""Linux process-tree memory snapshots with explicit measurement provenance."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import re
from typing import Sequence


GIB = 1024**3


def detected_total_memory_gib(
    proc_meminfo: Path = Path("/proc/meminfo"),
    cgroup_limit_paths: Sequence[Path] = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ),
) -> float | None:
    """Return the largest usable machine-memory budget visible to this process.

    Physical RAM is read from Linux procfs, POSIX sysconf, or the Windows API.
    A finite Linux cgroup memory ceiling is honored when it is smaller than the
    physical total. The result uses binary GiB, matching workflow accounting.
    """
    physical_bytes: int | None = None
    try:
        text = proc_meminfo.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
        if match:
            physical_bytes = int(match.group(1)) * 1024
    except OSError:
        pass

    if physical_bytes is None:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                physical_bytes = pages * page_size
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    if physical_bytes is None and os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                physical_bytes = int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    candidates = [physical_bytes] if physical_bytes and physical_bytes > 0 else []
    for path in cgroup_limit_paths:
        try:
            value = path.read_text(encoding="utf-8", errors="replace").strip()
            limit = int(value)
        except (OSError, TypeError, ValueError):
            continue
        # v1 commonly represents "unlimited" with a value near signed 64-bit
        # maximum; it is not a real usable memory ceiling.
        if 0 < limit < 2**60:
            candidates.append(limit)
    if not candidates:
        return None
    return round(min(candidates) / GIB, 3)


@dataclass(frozen=True)
class MemorySnapshot:
    processes: tuple[dict, ...] = ()
    supported: bool = True

    @property
    def live(self):
        return [row for row in self.processes if row["state"] == "live"]

    @property
    def pss_bytes(self) -> int:
        return sum(row["pss_bytes"] or 0 for row in self.live)

    @property
    def upper_bound_bytes(self) -> int:
        """Accounted sum; a total upper bound only if upper_bound_complete."""
        return sum(row["pss_bytes"] if row["pss_bytes"] is not None else (row["rss_bytes"] or 0)
                   for row in self.live)

    @property
    def upper_bound_complete(self) -> bool:
        return self.supported and not self.unknown_memory

    @property
    def missing_pss(self) -> int:
        return sum(row["pss_bytes"] is None for row in self.live)

    @property
    def unknown_memory(self) -> bool:
        return any(row["pss_bytes"] is None and row["rss_bytes"] is None for row in self.live)


def _identity(path: Path) -> tuple[int, str, str, str]:
    text = (path / "stat").read_text(encoding="utf-8", errors="replace")
    close = text.rfind(")")
    fields = text[close + 2:].split()
    if close < 0 or len(fields) < 20:
        raise ValueError("incomplete /proc stat record")
    # stat fields 4 (PPID), 22 (starttime), 3 (state), and 2 (comm).
    return int(fields[1]), fields[19], fields[0], text[text.find("(") + 1:close]


def _pss(path: Path) -> tuple[int | None, str, str]:
    errors = []
    rollup_no_mm = False
    try:
        text = (path / "smaps_rollup").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Pss:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
        if match:
            return int(match.group(1)) * 1024, "smaps_rollup", ""
        errors.append("smaps_rollup: no Pss field")
    except OSError as exc:
        rollup_no_mm = exc.errno == errno.ESRCH
        errors.append(f"smaps_rollup: {exc.__class__.__name__} (errno={exc.errno})")
    # Older kernels may provide smaps but not smaps_rollup. Sum actual PSS,
    # never Pss_Anon/Pss_File or RSS, and stream this potentially large file.
    try:
        total, found, empty = 0, False, True
        with (path / "smaps").open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                empty = False
                match = re.match(r"^Pss:\s+(\d+)\s+kB\s*$", line)
                if match:
                    total += int(match.group(1)) * 1024
                    found = True
        if found:
            return total, "smaps", "; ".join(errors)
        # Linux smaps_rollup returns ESRCH when the task or its mm is gone.
        # A PID can still have a non-zombie stat entry after releasing its mm.
        # Require a successfully read, truly empty smaps as corroboration; a
        # permission failure or malformed nonempty file is NOT zero memory.
        # A successful PSS fallback above wins if an exec race has recovered.
        if rollup_no_mm and empty:
            errors.append("smaps: empty; no user address space at measurement")
            return None, "no_address_space", "; ".join(errors)
        errors.append("smaps: no Pss fields")
    except OSError as exc:
        errors.append(f"smaps: {exc.__class__.__name__} (errno={exc.errno})")
    return None, "rss_upper_bound", "; ".join(errors)


def linux_memory_snapshot(root_pids: Sequence[int], proc_root: Path = Path("/proc")) -> MemorySnapshot:
    """Sample each live PID once; exclude vanished, zombie and recycled PIDs.

    RSS is only an upper bound when both PSS sources are unavailable. It must
    not be reported as PSS or used to assert a physical-memory breach. Tasks
    without a user address space are excluded only for the current snapshot;
    their descendants are still sampled and the PID is resampled next time.
    """
    roots = {int(pid) for pid in root_pids if int(pid) > 0}
    if not roots:
        return MemorySnapshot()
    if not proc_root.is_dir():
        return MemorySnapshot(supported=False)
    identities = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return MemorySnapshot(supported=False)
    for entry in entries:
        if entry.name.isdigit():
            try:
                identities[int(entry.name)] = _identity(entry)
            except (OSError, ValueError, IndexError):
                continue
    selected = set(roots)
    while True:
        expanded = selected | {pid for pid, identity in identities.items() if identity[0] in selected}
        if expanded == selected:
            break
        selected = expanded
    rows = []
    for pid in sorted(selected):
        path = proc_root / str(pid)
        row = {"pid": pid, "ppid": None, "starttime": None, "name": "unknown",
               "state": "live", "proc_state": None, "pss_bytes": None, "rss_bytes": None,
               "source": "unavailable", "error": ""}
        try:
            before = _identity(path)
            row.update(ppid=before[0], starttime=before[1], name=before[3], proc_state=before[2])
            if before[2] in {"Z", "X", "x"}:
                row["state"] = "exited"
            elif pid in identities and identities[pid][1] != before[1]:
                row["state"] = "pid_reused"
            else:
                try:
                    status = (path / "status").read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"^VmRSS:\s+(\d+)\s+kB\s*$", status, re.MULTILINE)
                    if match:
                        row["rss_bytes"] = int(match.group(1)) * 1024
                except OSError:
                    pass
                row["pss_bytes"], row["source"], row["error"] = _pss(path)
                after = _identity(path)
                row["proc_state"] = after[2]
                if after[1] != before[1]:
                    row["state"] = "pid_reused"
                elif after[2] in {"Z", "X", "x"}:
                    row["state"] = "exited"
                elif row["source"] == "no_address_space":
                    row["state"] = "no_address_space"
        except (FileNotFoundError, ProcessLookupError):
            row["state"] = "exited"
        except (OSError, ValueError, IndexError) as exc:
            # An unreadable but possibly live process is uncertainty, not zero.
            row.update(pss_bytes=None, rss_bytes=None, error=str(exc))
        if row["state"] != "live":
            row.update(pss_bytes=None, rss_bytes=None, source="excluded")
        rows.append(row)
    return MemorySnapshot(tuple(rows))

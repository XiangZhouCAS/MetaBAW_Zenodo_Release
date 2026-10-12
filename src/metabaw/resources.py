"""Conservative, shared resource budgets for independent binning tasks."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re

SAFETY_FRACTION = 0.8  # RAM and VRAM only; CPU uses the permitted capacity.
GIB = 1024**3


def cgroup_directories(root: Path = Path("/sys/fs/cgroup"),
                       membership: Path = Path("/proc/self/cgroup")) -> tuple[Path, ...]:
    """Read the current v2 hierarchy and its ancestors, without changing it."""
    root = root.resolve()
    try:
        path = next(line.split(":", 2)[2] for line in membership.read_text().splitlines()
                    if line.startswith("0::"))
        current = (root / path.lstrip("/")).resolve()
        if not current.is_relative_to(root):
            return (root,)
        return tuple([current, *(parent for parent in current.parents
                                 if parent.is_relative_to(root))])
    except (OSError, StopIteration):
        return (root,)


def available_memory_gib(meminfo: Path = Path("/proc/meminfo"),
                         cgroups: tuple[Path, ...] | None = None) -> float | None:
    """Available physical memory, limited by current cgroup headroom."""
    candidates = []
    try:
        text = meminfo.read_text()
        match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
        if match:
            candidates.append(int(match[1]) * 1024)
    except OSError:
        pass
    if not candidates and os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in (
                            "total", "available", "total_page", "available_page",
                            "total_virtual", "available_virtual", "extended")]]

        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            candidates.append(status.available)
    for directory in cgroups if cgroups is not None else cgroup_directories():
        try:
            limit = int((directory / "memory.max").read_text().strip())
            used = int((directory / "memory.current").read_text().strip())
            if 0 <= limit < 2**60:
                candidates.append(max(0, limit - used))
        except (OSError, ValueError):
            continue
    # Older v1 container mounts may be used without a v2 membership entry.
    try:
        base = Path("/sys/fs/cgroup/memory")
        limit = int((base / "memory.limit_in_bytes").read_text().strip())
        used = int((base / "memory.usage_in_bytes").read_text().strip())
        if 0 <= limit < 2**60:
            candidates.append(max(0, limit - used))
    except (OSError, ValueError):
        pass
    return min(candidates) / GIB if candidates else None


def available_cpus(cgroups: tuple[Path, ...] | None = None) -> tuple[tuple[int, ...], float]:
    """Permitted logical CPUs/CPU quota, not an instantaneous idle percentage."""
    try:
        identifiers = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        identifiers = tuple(range(os.cpu_count() or 1))
    capacity = float(len(identifiers))
    for directory in cgroups if cgroups is not None else cgroup_directories():
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if quota != "max":
                capacity = min(capacity, int(quota) / int(period))
        except (OSError, ValueError, ZeroDivisionError):
            continue
    return identifiers, capacity


@dataclass(frozen=True)
class ResourcePlan:
    cpu_ids: tuple[int, ...]
    available_cpu_capacity: float
    threads_requested: int
    threads: int
    memory_available_gib: float
    memory_requested_gib: float
    memory_gib: float
    slots: int

    @property
    def memory_per_task_gib(self) -> float:
        return self.memory_gib / self.slots


def plan_resources(threads: int, memory_gib: float, slots: int, workloads: int,
                   *, cpu_snapshot=None, memory_snapshot=None) -> ResourcePlan:
    identifiers, capacity = cpu_snapshot if cpu_snapshot is not None else available_cpus()
    free = memory_snapshot if memory_snapshot is not None else available_memory_gib()
    if free is None or free <= 0:
        raise ValueError("Cannot measure current available RAM; refusing an unverifiable 80% RAM budget")
    cpu_limit = min(len(identifiers), math.floor(capacity))
    if cpu_limit < 1:
        raise ValueError("Less than one CPU is permitted by affinity/quota; allocate at least one logical CPU")
    actual_threads = min(threads, cpu_limit)
    actual_slots = max(1, min(slots, workloads, actual_threads))
    return ResourcePlan(tuple(identifiers[:actual_threads]), capacity, threads, actual_threads,
                        free, memory_gib, min(memory_gib, free * SAFETY_FRACTION), actual_slots)


def memory_bytes(value: str, total_bytes: int) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?i?B?|%)?", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid GPU memory budget: {value}")
    amount, unit = float(match[1]), (match[2] or "").upper()
    if unit == "%":
        return int(total_bytes * amount / 100)
    return int(amount * (1024 ** {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit[0]]
                         if unit and unit != "B" else 1))


def equal_gpu_share(ceilings, aggregate_budget: int, slots: int) -> int:
    """Largest equal share fitting both the aggregate budget and real cards."""
    if not ceilings:
        return 0
    slots = max(1, slots)
    low, high = 0, min(max(0, aggregate_budget) // slots, max(ceilings))
    while low < high:
        middle = (low + high + 1) // 2
        if sum(limit // middle for limit in ceilings) >= slots:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class GpuBudget:
    index: str
    uuid: str | None
    total_bytes: int
    available_bytes: int
    limit_bytes: int


def plan_gpu_resources(devices, requested: str, slots: int, visible: str | None = None):
    """GPU memory is a workflow-wide budget, evenly shared by sample slots."""
    selected = list(devices)
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        selected = [device for token in tokens for device in devices
                    if token == device.index or (getattr(device, "uuid", None)
                       and device.uuid.startswith(token))]
    selected = list({device.index: device for device in selected}.values())
    # Unknown free memory is not a permission to allocate 80% of physical total.
    selected = [device for device in selected if device.memory_free_mib is not None
                and device.memory_free_mib > 0]
    if not selected:
        return (), 0
    free = sum(device.memory_free_mib * 1024**2 for device in selected)
    requested_bytes = memory_bytes(requested, sum(device.memory_total_mib * 1024**2 for device in selected))
    budget = min(requested_bytes, int(free * SAFETY_FRACTION))
    slots = max(1, slots)
    ceilings = {device.index: int(device.memory_free_mib * 1024**2 * SAFETY_FRACTION)
                for device in selected}
    # One process cannot combine VRAM from multiple cards. Find the largest
    # equal share that both the aggregate request and physical cards can fit.
    share = equal_gpu_share(tuple(ceilings.values()), budget, slots)
    if share <= 0:
        return (), 0
    counts = {device.index: 0 for device in selected}
    for _ in range(slots):
        choices = [(counts[index], -(limit - counts[index] * share), index)
                   for index, limit in ceilings.items() if limit - counts[index] * share >= share]
        counts[min(choices)[2]] += 1
    budgets = tuple(GpuBudget(device.index, getattr(device, "uuid", None),
                              device.memory_total_mib * 1024**2, device.memory_free_mib * 1024**2,
                              counts[device.index] * share)
                    for device in selected if counts[device.index])
    return budgets, share


class GpuReservations:
    """Reserve equal task shares on concrete GPUs; never invent virtual VRAM."""
    def __init__(self, devices: tuple[GpuBudget, ...], share_bytes: int):
        self.devices = devices
        self.share_bytes = share_bytes
        self.initial_share_bytes = share_bytes
        self.reservations: dict[str, str] = {}

    @property
    def capacity(self) -> int:
        return sum(device.limit_bytes // self.share_bytes for device in self.devices) if self.share_bytes > 0 else 0

    def acquire(self, task_id: str, current_limits: dict[str, int] | None = None) -> GpuBudget | None:
        if self.share_bytes <= 0:
            return None
        candidates = []
        for device in self.devices:
            count = sum(index == device.index for index in self.reservations.values())
            limit = min(device.limit_bytes, (current_limits or {}).get(device.index, device.limit_bytes))
            remaining = limit - count * self.share_bytes
            if remaining >= self.share_bytes:
                candidates.append((count, -remaining, device.index, device))
        if not candidates:
            return None
        device = min(candidates)[3]
        self.reservations[task_id] = device.index
        return device

    def refresh_idle_share(self, slots: int, current_limits: dict[str, int]) -> bool:
        """Only change equal shares between GPU batches, never mid-allocation."""
        if self.reservations:
            return False
        ceilings = tuple(min(device.limit_bytes, current_limits.get(device.index, device.limit_bytes))
                         for device in self.devices)
        share = equal_gpu_share(ceilings, self.initial_share_bytes * slots, slots)
        changed = share != self.share_bytes
        self.share_bytes = share
        return changed

    def release(self, task_id: str) -> None:
        self.reservations.pop(task_id, None)

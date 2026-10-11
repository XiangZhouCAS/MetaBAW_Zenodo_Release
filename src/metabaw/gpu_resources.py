"""NVML/nvidia-smi accounting for the GPU memory of complete task trees."""
from __future__ import annotations

import csv
import shutil
import subprocess
import time

from .resources import GpuBudget, SAFETY_FRACTION


def process_tasks(snapshot, root_tasks: dict[int, str]) -> dict[int, str]:
    # A parent may have released its address space while its workers remain
    # alive. Keep that ancestry link without counting the parent as memory.
    parents = {row["pid"]: row.get("ppid") for row in snapshot.processes
               if "pid" in row and row["state"] != "pid_reused"}
    result = {}
    for row in snapshot.live:
        if "pid" not in row:
            continue
        pid = row["pid"]
        current, seen = pid, set()
        while current not in root_tasks and current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
        if current in root_tasks:
            result[pid] = root_tasks[current]
    return result


def gpu_snapshot(executable: str):
    if not executable:
        raise ValueError("nvidia-smi is required for GPU task-tree resource accounting")
    def query(fields):
        command = [executable, fields, "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]

    devices = {}
    for row in query("--query-gpu=index,uuid,memory.free"):
        if len(row) != 3:
            raise ValueError("Invalid nvidia-smi device-memory record")
        devices[row[0]] = (row[1], int(float(row[2]) * 1024**2))
    processes = []
    for row in query("--query-compute-apps=pid,gpu_uuid,used_gpu_memory"):
        if len(row) != 3:
            raise ValueError("Invalid nvidia-smi process-memory record")
        processes.append((int(row[0]), row[1], int(float(row[2]) * 1024**2)))
    return devices, processes


class GpuMonitor:
    def __init__(self, devices: tuple[GpuBudget, ...], share_bytes: int, sampler=None):
        self.devices, self.share_bytes = devices, share_bytes
        self.executable = shutil.which("nvidia-smi")
        self.sampler = sampler or (lambda: gpu_snapshot(self.executable))
        self.current_limits = {device.index: device.limit_bytes for device in devices}
        self.last_sample = float("-inf")
        self.candidate = None

    def sample(self, pid_tasks: dict[int, str], reservations: dict[str, str], *, now=None):
        """Return a confirmed breach; also refresh admission limits per card."""
        now = time.monotonic() if now is None else now
        if now - self.last_sample < 1:
            return None
        self.last_sample = now
        devices, processes = self.sampler()
        uuid_index = {uuid: index for index, (uuid, free) in devices.items()}
        task_usage, card_usage = {}, {}
        for pid, uuid, used in processes:
            if pid not in pid_tasks:
                continue
            task = pid_tasks[pid]
            index = uuid_index.get(uuid)
            if task not in reservations:
                raise ValueError(f"Unreserved CUDA process {pid} in task {task}")
            if index != reservations[task]:
                raise ValueError(f"Task {task} used GPU {index} instead of assigned GPU {reservations[task]}")
            task_usage[task] = task_usage.get(task, 0) + used
            card_usage[index] = card_usage.get(index, 0) + used
        breach = None
        for device in self.devices:
            if device.index not in devices:
                raise ValueError(f"Assigned GPU {device.index} disappeared during execution")
            owned = card_usage.get(device.index, 0)
            free = devices[device.index][1]
            limit = min(device.limit_bytes, int((free + owned) * SAFETY_FRACTION))
            self.current_limits[device.index] = limit
            if owned > limit:
                breach = (f"GPU {device.index}: task-tree VRAM {owned / 1024**3:.3f} GiB "
                          f"exceeds current shared 80% budget {limit / 1024**3:.3f} GiB")
        for task, used in task_usage.items():
            if used > self.share_bytes:
                breach = (f"Task {task}: task-tree VRAM {used / 1024**3:.3f} GiB "
                          f"exceeds equal task share {self.share_bytes / 1024**3:.3f} GiB")
        if breach is not None and self.candidate == breach:
            return breach
        # Breach values vary as allocations grow; confirm the same affected
        # card/task on the next sample, not an exactly equal numeric reading.
        identity = breach.split(":", 1)[0] if breach else None
        previous = self.candidate.split(":", 1)[0] if self.candidate else None
        self.candidate = breach
        return breach if identity is not None and identity == previous else None

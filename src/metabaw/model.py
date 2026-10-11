from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import shlex
from typing import AbstractSet, Mapping, Sequence

from .resume import checkpoint_task_id, resume_command, resume_environment


Command = str | tuple[str, ...]


@dataclass(frozen=True)
class Task:
    id: str
    stage: str
    command: Command
    cwd: Path
    deps: tuple[str, ...] = ()
    wait_for: tuple[str, ...] = ()
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    output_alternatives: tuple[tuple[Path, ...], ...] = ()
    fasta_output_dirs: tuple[Path, ...] = ()
    allow_empty_refinement: bool = False
    sample: str | None = None
    automatic_retries: int = 0
    cpus: int = 1
    gpus: int = 0
    priority: int = 100
    allow_failed_deps: bool = False
    failure_tolerated: bool = False
    description: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    enforce_memory_limit: bool = True
    minimum_memory_gb: float = 0.0
    memory_requirement_hint: str = ""
    # Runtime-only allocations; not analysis-fingerprint inputs.
    cpu_affinity: tuple[int, ...] = ()
    memory_limit_gb: float | None = None

    def display_command(self) -> str:
        if isinstance(self.command, str):
            return self.command
        return shlex.join(self.command)

    @property
    def scheduling_dependencies(self) -> tuple[str, ...]:
        """Return data dependencies followed by ordering-only dependencies."""
        return tuple(dict.fromkeys((*self.deps, *self.wait_for)))

    def fingerprint(
        self,
        dependency_fingerprints: Sequence[str],
        assume_missing_inputs: AbstractSet[Path] = frozenset(),
        *,
        resource_independent: bool = False,
        input_metadata: Mapping[Path, Mapping[str, object]] | None = None,
    ) -> str:
        input_stats: list[dict[str, object]] = []
        for path in self.inputs:
            if input_metadata is not None and path in input_metadata:
                input_stats.append(dict(input_metadata[path], path=str(path)))
                continue
            if path in assume_missing_inputs:
                input_stats.append({"path": str(path), "missing": True})
                continue
            try:
                stat = path.stat()
                input_stats.append(
                    {
                        "path": str(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "is_dir": path.is_dir(),
                    }
                )
            except FileNotFoundError:
                input_stats.append({"path": str(path), "missing": True})
        payload = {
            "id": checkpoint_task_id(self.id),
            "stage": self.stage,
            "command": self.display_command(),
            "cwd": str(self.cwd),
            "inputs": input_stats,
            "outputs": [str(path) for path in self.outputs],
            "output_alternatives": [
                [str(path) for path in alternatives]
                for alternatives in self.output_alternatives
            ],
            "cpus": self.cpus,
            "priority": self.priority,
            "allow_failed_deps": self.allow_failed_deps,
            "failure_tolerated": self.failure_tolerated,
            "env": dict(sorted(self.env.items())),
            "enforce_memory_limit": self.enforce_memory_limit,
            "minimum_memory_gb": self.minimum_memory_gb,
            "memory_requirement_hint": self.memory_requirement_hint,
            "dependencies": list(dependency_fingerprints),
        }
        if self.allow_empty_refinement:
            payload["allow_empty_refinement"] = True
        if resource_independent:
            payload["resume_policy"] = "cpu-resources-v1"
            payload["command"] = resume_command(self.display_command())
            payload["env"] = resume_environment(self.env)
            for name in ("cpus", "priority", "enforce_memory_limit",
                         "minimum_memory_gb", "memory_requirement_hint"):
                payload.pop(name)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(raw).hexdigest()

    def resume_fingerprint(
        self,
        dependency_fingerprints: Sequence[str],
        assume_missing_inputs: AbstractSet[Path] = frozenset(),
    ) -> str:
        """Track data and analysis settings, not CPU scheduling settings."""
        return self.fingerprint(dependency_fingerprints, assume_missing_inputs,
                                resource_independent=True)


def topological_order(tasks: Sequence[Task]) -> list[Task]:
    by_id = {task.id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Task IDs must be unique")
    for task in tasks:
        missing = [
            dep
            for dep in task.scheduling_dependencies
            if dep not in by_id
        ]
        if missing:
            raise ValueError(f"Task {task.id!r} has unknown dependencies: {missing}")

    indegree = {
        task.id: len(task.scheduling_dependencies)
        for task in tasks
    }
    children: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        for dep in task.scheduling_dependencies:
            children[dep].append(task.id)

    def queue_key(task_id: str) -> tuple[str, int, str]:
        task = by_id[task_id]
        return task.stage, task.priority, task.id

    queue = sorted(
        (task_id for task_id, degree in indegree.items() if degree == 0),
        key=queue_key,
    )
    ordered: list[Task] = []
    while queue:
        task_id = queue.pop(0)
        ordered.append(by_id[task_id])
        for child in sorted(children[task_id], key=queue_key):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort(key=queue_key)
    if len(ordered) != len(tasks):
        cyclic = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"Task graph contains a cycle involving: {cyclic}")
    return ordered

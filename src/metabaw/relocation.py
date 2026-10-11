"""Portable checkpoints for copies/renames of an owned workflow directory.

Only output/temp roots are relocatable. External input metadata, software and
database locations and analysis options remain part of the cache key. Generated
inputs are identified by data dependencies, not simply by living under tmp.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Mapping, Sequence
from uuid import uuid4

from .model import Task
from .resume import checkpoint_task_id, current_task_id, resume_command, resume_environment
from .state import StateStore, TaskRecord


def _absolute(path: Path) -> Path:
    # Do not resolve symlinks: a copied internal link may still point at the old
    # directory (which may no longer exist).
    return Path(os.path.abspath(path))


@dataclass(frozen=True)
class Location:
    output: Path
    temporary: Path

    def __post_init__(self) -> None:
        if any(not path.is_absolute() or path.parent == path for path in (self.output, self.temporary)):
            raise ValueError("Workflow relocation requires specific absolute output/temp directories, not filesystem roots")

    def roots(self) -> dict[str, str]:
        return {"output": str(_absolute(self.output)), "temporary": str(_absolute(self.temporary))}

    @classmethod
    def parse(cls, value: object) -> Location | None:
        if not isinstance(value, dict):
            return None
        if not all(isinstance(value.get(key), str) and Path(value[key]).is_absolute()
                   for key in ("output", "temporary")):
            return None
        try:
            return cls(Path(value["output"]), Path(value["temporary"]))
        except ValueError:
            return None


def _replace_roots(value: str, replacements: Sequence[tuple[Path, str | None]]) -> str:
    variants: dict[str, str | None] = {}
    for root, replacement in replacements:
        for spelling in (str(_absolute(root)), _absolute(root).as_posix()):
            variants[spelling.rstrip("/\\")] = replacement
            # Nested COMEBin arguments and binner specifications contain JSON.
            variants[json.dumps(spelling)[1:-1].rstrip("/\\")] = replacement
    if not variants:
        return value
    pattern = re.compile(
        r"(?<![\w./\\-])(?:" + "|".join(re.escape(key) for key in sorted(variants, key=len, reverse=True))
        + r")(?=$|[/\\\s'\";,|&<>()\]\}:])"
    )
    return pattern.sub(lambda match: variants[match.group()] if variants[match.group()] is not None else match.group(), value)


def _normalize(value: str, location: Location, protected: Sequence[Path] = ()) -> str:
    return _replace_roots(value, ((location.temporary, "@METABAW_TMP@"),
                                 (location.output, "@METABAW_OUTPUT@"),
                                 *((path, None) for path in protected)))


def external_locations(task: Task, generated: frozenset[Path]) -> frozenset[Path]:
    """Include configured databases even when a tool does not declare them as inputs."""
    paths = {path for path in task.inputs if path not in generated}
    for name in ("GTDBTK_DATA_PATH", "CHECKM_DATA_PATH", "CHECKM2DB"):
        if task.env.get(name):
            paths.add(Path(task.env[name]))
    tokens = resume_command(task.display_command())
    for index, token in enumerate(tokens):
        option, separator, value = token.partition("=")
        if option in {"--database_path", "--db_file"}:
            if not separator and index + 1 < len(tokens):
                value = tokens[index + 1]
            if value:
                paths.add(Path(value))
    return frozenset(paths)


def _metadata(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "is_dir": path.is_dir()}
    except FileNotFoundError:
        return {"missing": True}


def _recipe(task: Task, generated: frozenset[Path], location: Location, *, version: int = 2) -> dict[str, object]:
    protected = tuple(external_locations(task, generated))
    recipe = {
        "id": checkpoint_task_id(task.id), "stage": task.stage,
        "command": [_normalize(token, location, protected) for token in
                    resume_command(task.display_command(), launcher_independent=version >= 2)],
        "cwd": _normalize(str(task.cwd), location),
        "inputs": [{"path": _normalize(str(path), location) if path in generated else str(path),
                    "generated": path in generated} for path in task.inputs],
        "outputs": [_normalize(str(path), location) for path in task.outputs],
        "alternatives": [[_normalize(str(path), location) for path in group]
                         for group in task.output_alternatives],
        "fasta": [_normalize(str(path), location) for path in task.fasta_output_dirs],
        "env": {key: _normalize(value, location, protected) for key, value in resume_environment(task.env).items()},
        "gpus": task.gpus, "deps": [checkpoint_task_id(dep) for dep in task.deps],
        "allow_failed_deps": task.allow_failed_deps, "failure_tolerated": task.failure_tolerated,
    }
    if task.allow_empty_refinement:
        recipe["allow_empty_refinement"] = True
    return recipe


def portable_checkpoint(task: Task, generated: frozenset[Path],
                        dependencies: Sequence[str], location: Location, *, version: int = 2) -> str:
    analysis = _recipe(task, generated, location, version=version)
    analysis["external_inputs"] = [dict(_metadata(path), path=str(path))
                                   for path in sorted(external_locations(task, generated), key=str)]
    analysis["dependencies"] = list(dependencies)
    raw = json.dumps(analysis, sort_keys=True, separators=(",", ":")).encode()
    return json.dumps({"version": version, "location": location.roots(),
                       "signature": sha256(raw).hexdigest()}, sort_keys=True)


def _checkpoint(value: str | None) -> dict | None:
    try:
        result = json.loads(value or "null")
    except (ValueError, TypeError):
        return None
    if (isinstance(result, dict) and type(result.get("version")) is int
            and result.get("version") in {1, 2}
            and Location.parse(result.get("location")) is not None
            and isinstance(result.get("signature"), str)
            and re.fullmatch(r"[a-f0-9]{64}", result["signature"])):
        return result
    return None


def checkpoint_signature(value: str) -> str:
    return json.loads(value)["signature"]


def _read_source_records(path: Path) -> dict[str, TaskRecord]:
    """Read stopped original-run proof without upgrading/writing its SQLite.

    immutable=1 also forbids sidecar creation. An uncheckpointed live WAL is
    deliberately not adopted; the original run must be stopped before copying.
    """
    if not path.is_file():
        return {}
    try:
        connection = sqlite3.connect(path.absolute().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(task_runs)")}
            if not {"task_id", "fingerprint", "status"}.issubset(columns):
                return {}
            return {row["task_id"]: TaskRecord(**{
                name: row[name] if name in columns else None for name in TaskRecord.__dataclass_fields__
            }) for row in connection.execute("SELECT * FROM task_runs")}
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return {}


class ResumeLayout:
    def __init__(self, output: Path, temporary: Path, previous_manifest: Mapping | None = None):
        self.current = Location(_absolute(output), _absolute(temporary))
        self.previous_manifest = previous_manifest or {}
        # Keep the immediate previous run as well as older relocation proof.
        # A partial restart can have a mixture of new and legacy task records.
        self.manifests = [self.previous_manifest]
        self.legacy_records: dict[str, dict[str, TaskRecord]] = {}
        self.reasons: dict[str, str] = {}
        result_files = self.previous_manifest.get("result_files", {})
        temporary_files = self.previous_manifest.get("temporary_files", {})
        self.previous: Location | None = None
        if isinstance(result_files, dict) and isinstance(temporary_files, dict):
            detail = result_files.get("workflow_details") or result_files.get("start_info")
            temporary_path = temporary_files.get("workspace")
            if isinstance(detail, str) and isinstance(temporary_path, str):
                self.previous = Location.parse({"output": str(Path(detail).parent), "temporary": temporary_path})
        self.origins: list[Location] = []
        self.add_origin(self.previous)

    def add_origin(self, origin: Location | None) -> None:
        if origin is not None and origin != self.current and origin not in self.origins:
            self.origins.append(origin)

    def remember(self, state: StateStore) -> None:
        """Keep legacy proof in SQLite before the CLI replaces its manifest.

        A second interruption during migration must not discard the original
        task descriptions. This is bounded metadata, not another final report.
        """
        raw = state.get_metadata("relocation_legacy_manifest")
        try:
            saved = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            saved = None
        if isinstance(saved, dict):
            original = ResumeLayout(self.current.output, self.current.temporary, saved)
            if original.previous is not None:
                if saved not in self.manifests:
                    self.manifests.append(saved)
                self.add_origin(self.previous)
                self.previous_manifest = saved
                self.previous = original.previous
                self.add_origin(self.previous)
                self._remember_records(state)
                return
        if self.previous is not None and self.previous != self.current:
            minimal = {key: self.previous_manifest.get(key) for key in
                       ("result_files", "temporary_files", "tasks")}
            state.set_metadata("relocation_legacy_manifest", json.dumps(minimal, sort_keys=True))
        self._remember_records(state)

    def _remember_records(self, state: StateStore) -> None:
        """Retain original dependency hashes before any copied task reruns."""
        raw = state.get_metadata("relocation_legacy_records")
        try:
            saved = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            saved = {}
        if isinstance(saved, dict):
            for root, rows in saved.items():
                if not isinstance(root, str) or not isinstance(rows, list):
                    continue
                try:
                    self.legacy_records[root] = {row["task_id"]: TaskRecord(**row) for row in rows}
                except (TypeError, KeyError):
                    continue
        current = {record.task_id: record for record in state.records()}
        if not any(record.portable_checkpoint is None for record in current.values()):
            return
        changed = False
        for origin in self.origins:
            key = str(origin.output)
            if key in self.legacy_records:
                continue
            # Already-buggy restarts may have overwritten the copy's parent
            # hashes. Prefer read-only original records when still available.
            baseline = _read_source_records(origin.temporary / "runtime/state.sqlite3")
            self.legacy_records[key] = baseline or current
            changed = True
        if changed:
            state.set_metadata("relocation_legacy_records", json.dumps({
                root: [asdict(record) for record in records.values()]
                for root, records in self.legacy_records.items()
            }, sort_keys=True))

    def rebase(self, path: Path, origin: Location) -> Path:
        absolute = _absolute(path)
        pairs = ((origin.temporary, self.current.temporary), (origin.output, self.current.output))
        # A copy may live inside its original directory. Do not rebase links
        # or log locators that have already been repaired into that copy.
        if any(old != new and new.is_relative_to(old) and absolute.is_relative_to(new)
               for old, new in pairs):
            return absolute
        for old, new in sorted(pairs, key=lambda pair: len(str(pair[0])), reverse=True):
            try:
                relative = absolute.relative_to(_absolute(old))
            except ValueError:
                continue
            return new / relative
        return absolute

    def prepare(self, state: StateStore, external_inputs: frozenset[Path] = frozenset()) -> int:
        """Repair internal links and log locators before checking outputs.

        Never traverse link directories or change an external database/read
        link. No old source directory, log contents or sequence data are edited.
        """
        records = list(state.records())
        self.remember(state)
        for record in records:
            saved = _checkpoint(record.portable_checkpoint)
            if saved:
                self.add_origin(Location.parse(saved["location"]))
        if not self.origins:
            return 0
        for record in records:
            if not record.log_path:
                continue
            for origin in self.origins:
                new_log = self.rebase(Path(record.log_path), origin)
                if new_log != _absolute(Path(record.log_path)) and new_log.is_file():
                    state.relocate_log(record.task_id, new_log)
                    break
        repaired = 0
        # Walk each physical owned root once. Temporary storage can be outside
        # the output root; an unchanged external tmp is not a copied tree.
        roots = [self.current.output, self.current.temporary]
        if self.current.temporary.is_relative_to(self.current.output):
            roots = [self.current.output]
        elif self.current.output.is_relative_to(self.current.temporary):
            roots = [self.current.temporary]
        for root in roots:
            if root.is_symlink():
                raise ValueError(f"Refusing relocation through a symlink root: {root}")
            for directory, dirs, files in os.walk(root, followlinks=False):
                parent = Path(directory)
                parent_stat = parent.stat()
                changed = False
                for name in (*dirs, *files):
                    link = parent / name
                    if not link.is_symlink():
                        continue
                    target = Path(os.readlink(link))
                    if not target.is_absolute():
                        target = _absolute(parent / target)
                    if any(_absolute(target).is_relative_to(_absolute(source)) for source in external_inputs):
                        continue
                    for origin in self.origins:
                        rebased = self.rebase(target, origin)
                        if rebased == _absolute(target):
                            continue
                        # A missing copied target must cause its producer to
                        # rerun, not silently keep reading the original run.
                        was_directory = link.is_dir() or rebased.is_dir()
                        pending = link.with_name(f".{link.name}.metabaw-relocate-{uuid4().hex}")
                        try:
                            pending.symlink_to(rebased, target_is_directory=was_directory)
                            os.replace(pending, link)
                        finally:
                            if pending.is_symlink():
                                pending.unlink()
                        repaired += 1
                        changed = True
                        break
                if changed:
                    os.utime(parent, ns=(parent_stat.st_atime_ns, parent_stat.st_mtime_ns))
        return repaired

    def matches(self, task: Task, record: TaskRecord, generated: frozenset[Path],
                dependencies: Sequence[TaskRecord | None], checkpoint: str) -> bool:
        saved = _checkpoint(record.portable_checkpoint)
        if record.portable_checkpoint is not None:
            # Corrupt/future checkpoints do not become permission to adopt.
            if not saved:
                self.reasons[task.id] = "invalid or unsupported portable checkpoint"
                return False
            if (saved["version"] == 2 and saved["location"] != self.current.roots()
                    and saved["signature"] == checkpoint_signature(checkpoint)):
                return True
            if saved["version"] == 1 and saved["location"] != self.current.roots():
                # Verify v1 with the original dependency signatures. Upstream
                # records are validated independently and must already be hits.
                prior = [_checkpoint(item.portable_checkpoint) if item else None for item in dependencies]
                if all(prior):
                    signatures = [item["signature"] for item in prior]
                    candidates = [(task, generated, self.current), *self._legacy_tasks(task, generated)]
                    for old, old_generated, location in candidates:
                        proof = portable_checkpoint(old, old_generated, signatures, location, version=1)
                        if saved["signature"] == checkpoint_signature(proof):
                            return True
            # Same-directory launcher changes still authenticate generated-file
            # metadata; portable hashes alone must not hide an edited BAM.
        return self._legacy_matches(task, record, generated, dependencies)

    def _legacy_tasks(self, task: Task, generated: frozenset[Path]):
        for manifest in self.manifests:
            location = ResumeLayout(self.current.output, self.current.temporary, manifest).previous
            items = manifest.get("tasks", [])
            if location is None or not isinstance(items, list):
                continue
            item = next((value for value in items if isinstance(value, dict)
                         and isinstance(value.get("id"), str)
                         and checkpoint_task_id(value["id"]) == checkpoint_task_id(task.id)), None)
            if item is None:
                continue
            try:
                old = replace(
                    task, stage=item.get("stage", task.stage), command=item["command"],
                    cwd=Path(item.get("cwd", location.temporary / "work")),
                    inputs=tuple(Path(value) for value in item["inputs"]),
                    outputs=tuple(Path(value) for value in item["outputs"]),
                    output_alternatives=tuple(tuple(Path(value) for value in group)
                                              for group in item.get("output_alternatives", [])),
                    fasta_output_dirs=tuple(Path(value) for value in item.get("fasta_output_directories", [])),
                    allow_empty_refinement=bool(item.get("allow_empty_refinement", False)),
                    deps=tuple(item["dependencies"]), cpus=int(item["cpus"]),
                    gpus=int(item.get("gpus", 0)), priority=int(item.get("priority", 100)),
                    allow_failed_deps=bool(item.get("allow_failed_dependencies", False)),
                    failure_tolerated=bool(item.get("failure_tolerated", False)),
                    env=item.get("environment", {}),
                    enforce_memory_limit=bool(item.get("memory_limit_enforced", True)),
                    minimum_memory_gb=float(item.get("minimum_memory_gb", 0)),
                    memory_requirement_hint=str(item.get("memory_requirement_hint", "")),
                )
                old_generated = frozenset(path for path in old.inputs if self.rebase(path, location) in generated)
                before, after = _recipe(old, old_generated, location), _recipe(task, generated, self.current)
                if before != after:
                    differences = [key for key in before if before[key] != after[key]]
                    self.reasons[task.id] = "analysis recipe differs: " + ", ".join(differences)
                    continue
                yield old, old_generated, location
            except (KeyError, TypeError, ValueError, OSError, AttributeError):
                self.reasons[task.id] = "unusable legacy task description"

    def _legacy_matches(self, task: Task, record: TaskRecord, generated: frozenset[Path],
                        dependencies: Sequence[TaskRecord | None]) -> bool:
        """Migrate old records only after reconstructing their original hash.

        Prefer still-present original intermediate metadata, otherwise use the
        copied files' metadata. Legacy copies that discarded timestamps may
        need one rerun; new portable checkpoints do not have this limitation.
        """
        if any(item is None for item in dependencies):
            return False
        for old, old_generated, location in self._legacy_tasks(task, generated):
            baseline = self.legacy_records.get(str(location.output), {})
            source_dependencies = [baseline.get(dep) or baseline.get(current_task_id(dep))
                                   or baseline.get(checkpoint_task_id(dep)) for dep in old.deps]
            variants = [dependencies]
            if all(source_dependencies):
                variants.append(source_dependencies)
            copied_metadata = {old_path: _metadata(new_path) for old_path, new_path in zip(old.inputs, task.inputs)
                               if old_path in old_generated}
            try:
                for records in variants:
                    hashes = [item.fingerprint for item in records]
                    if record.fingerprint in {
                        old.fingerprint(hashes), old.fingerprint(hashes, old_generated),
                        old.fingerprint(hashes, input_metadata=copied_metadata),
                    }:
                        return True
                    if record.resume_fingerprint and all(item.resume_fingerprint for item in records):
                        semantic = [item.resume_fingerprint for item in records]
                        if record.resume_fingerprint in {
                            old.resume_fingerprint(semantic), old.resume_fingerprint(semantic, old_generated),
                            old.fingerprint(semantic, resource_independent=True, input_metadata=copied_metadata),
                        }:
                            return True
                self.reasons[task.id] = "legacy input/dependency metadata could not be authenticated"
            except (TypeError, ValueError, OSError, AttributeError):
                self.reasons[task.id] = "legacy input/dependency metadata could not be read"
        self.reasons.setdefault(task.id, "no matching legacy task proof")
        return False

"""Task-name migration with synthetic workers and real files/SQLite only."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from metabaw.cli import _task_dict
from metabaw.executor import Executor, RunResult, _legacy_generated_inputs, _legacy_resource_resume_matches
from metabaw.model import Task, topological_order
from metabaw.relocation import ResumeLayout, checkpoint_signature, portable_checkpoint
from metabaw.resume import checkpoint_task_id, current_task_id
from metabaw.state import StateStore


OLD_ID = "04.bins.HMI_1"
NEW_ID = "04.export_refined_bins.HMI_1"


def chain(root: Path, source: Path, *, old: bool):
    work = root / "tmp" / "work"
    mapping, export, catalog = (work / name for name in ("mapping.tsv", "export.done", "catalog.done"))
    export_id = OLD_ID if old else NEW_ID
    return [
        Task("04.refine.magscot.HMI_1", "04_refinement", ("refine", str(source), str(mapping)), work,
             inputs=(source,), outputs=(mapping,), sample="HMI_1"),
        Task(export_id, "04_refinement", ("export", str(mapping), str(export)), work,
             deps=("04.refine.magscot.HMI_1",), inputs=(mapping,), outputs=(export,), sample="HMI_1"),
        Task("05.catalog.candidates", "05_quality", ("collect", str(export), str(catalog)), work,
             deps=(export_id,), inputs=(export,), outputs=(catalog,)),
    ]


def seed(root, tasks, version):
    layout = ResumeLayout(root, root / "tmp")
    state = StateStore(root / "tmp/runtime/state.sqlite3")
    # Write all outputs before fingerprints, as an already completed run does.
    for task in tasks:
        for output in task.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("complete\n", encoding="utf-8")
    strict, semantic, signatures = {}, {}, {}
    generated = _legacy_generated_inputs(tasks)
    try:
        for task in tasks:
            strict[task.id] = task.fingerprint([strict[dep] for dep in task.deps])
            semantic[task.id] = task.resume_fingerprint([semantic[dep] for dep in task.deps])
            checkpoint = portable_checkpoint(task, generated[task.id], [signatures[dep] for dep in task.deps],
                                             layout.current, version=version or 2)
            signatures[task.id] = checkpoint_signature(checkpoint)
            log = root / "tmp/runtime/logs" / (task.id + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"# task: {task.id}\n# stage: {task.stage}\n# cwd: {task.cwd}\n"
                           f"# command: {task.display_command()}\n# workflow_max_memory_gb: 2\n", encoding="utf-8")
            state.start(task.id, strict[task.id], log, semantic[task.id], checkpoint if version else None)
            state.finish(task.id, "success", 0)
        record = state.get(OLD_ID)
    finally:
        state.close()
    return record, {"result_files": {"workflow_details": str(root / "workflow_details.md")},
                    "temporary_files": {"workspace": str(root / "tmp")},
                    "tasks": [_task_dict(task) for task in tasks]}


def worker(task, log, *_args):
    for output in task.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("rerun\n", encoding="utf-8")
    log.write_text(f"# task: {task.id}\n", encoding="utf-8")
    return RunResult(task.id, 0, "synthetic worker")


@pytest.mark.parametrize("version", [1, 2])
def test_task_and_downstream_hashes_are_identical_after_name_change(tmp_path, version):
    old = chain(tmp_path / "output", tmp_path / "source.fa", old=True)
    new = chain(tmp_path / "output", tmp_path / "source.fa", old=False)
    location = ResumeLayout(tmp_path / "output", tmp_path / "output/tmp").current
    generated = _legacy_generated_inputs(old)
    for method in ("fingerprint", "resume_fingerprint"):
        previous_hashes = updated_hashes = []
        for previous, updated in zip(old, new):
            previous_hashes = [getattr(previous, method)(previous_hashes)]
            updated_hashes = [getattr(updated, method)(updated_hashes)]
            assert previous_hashes == updated_hashes
    previous_signatures = updated_signatures = []
    for previous, updated in zip(old, new):
        paths = generated[previous.id]
        before = portable_checkpoint(previous, paths, previous_signatures, location, version=version)
        after = portable_checkpoint(updated, paths, updated_signatures, location, version=version)
        assert before == after
        previous_signatures = [checkpoint_signature(before)]
        updated_signatures = [checkpoint_signature(after)]


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize("copied", [False, True])
def test_old_export_and_downstream_resume_without_execution(tmp_path, version, copied):
    source = tmp_path / "source.fa"
    source.write_text(">c\nACGT\n", encoding="utf-8")
    original = tmp_path / "original"
    old_record, manifest = seed(original, chain(original, source, old=True), version)
    root = tmp_path / "copy" if copied else original
    if copied:
        shutil.copytree(original, root)
    state = StateStore(root / "tmp/runtime/state.sqlite3")
    output = io.StringIO()
    try:
        migrated = state.get(NEW_ID)
        assert migrated == replace(old_record, task_id=NEW_ID)
        assert state.get(OLD_ID) is None
        old_log_text = Path(migrated.log_path).read_text(encoding="utf-8")
        with patch("metabaw.executor._execute_task", side_effect=worker) as execute, redirect_stdout(output):
            statuses = Executor(state, root / "tmp/runtime/logs", max_cpus=2, max_parallel=2,
                                show_progress=True, resume_layout=ResumeLayout(root, root / "tmp", manifest)).run(
                                    chain(root, source, old=False))
        assert not execute.called
        assert set(statuses.values()) == {"skipped"}
        assert state.get(NEW_ID).started_at == old_record.started_at
        assert Path(state.get(NEW_ID).log_path).read_text(encoding="utf-8") == old_log_text
        assert "Reusing 3/3 complete tasks" in output.getvalue()
    finally:
        state.close()


@pytest.mark.parametrize("invalid", ["changed_command", "missing_output", "failed_status"])
def test_renaming_does_not_hide_invalid_export_checkpoints(tmp_path, invalid):
    source = tmp_path / "source.fa"
    source.write_text(">c\nACGT\n", encoding="utf-8")
    root = tmp_path / "output"
    _, manifest = seed(root, chain(root, source, old=True), 2)
    tasks = chain(root, source, old=False)
    state = StateStore(root / "tmp/runtime/state.sqlite3")
    output = io.StringIO()
    try:
        if invalid == "changed_command":
            tasks[1] = replace(tasks[1], command=(*tasks[1].command, "--different-analysis"))
        elif invalid == "missing_output":
            tasks[1].outputs[0].unlink()
        else:
            state.finish(NEW_ID, "failed", 1, "synthetic failure")
        with patch("metabaw.executor._execute_task", side_effect=worker) as execute, redirect_stdout(output):
            statuses = Executor(state, root / "tmp/runtime/logs", max_cpus=2, max_parallel=2,
                                show_progress=True, resume_layout=ResumeLayout(root, root / "tmp", manifest)).run(tasks)
        assert [call.args[0].id for call in execute.call_args_list] == [NEW_ID, "05.catalog.candidates"]
        assert statuses[tasks[0].id] == "skipped"
        assert statuses[NEW_ID] == "success"
        assert "04.export_refined_bins -" in output.getvalue()
        assert Path(state.get(NEW_ID).log_path).name == NEW_ID + ".log"
    finally:
        state.close()


def test_existing_new_record_is_never_overwritten_by_old_alias(tmp_path):
    path = tmp_path / "state.sqlite3"
    state = StateStore(path)
    state.start(OLD_ID, "old-hash", tmp_path / "old.log")
    state.finish(OLD_ID, "success", 0)
    state.start(NEW_ID, "new-hash", tmp_path / "new.log")
    state.finish(NEW_ID, "failed", 1)
    state.close()
    state = StateStore(path)
    try:
        assert state.get(NEW_ID).status == "failed"
        assert state.get(NEW_ID).fingerprint == "new-hash"
        assert state.get(OLD_ID).fingerprint == "old-hash"
    finally:
        state.close()


def test_old_log_authenticates_resource_only_change_under_new_name(tmp_path):
    root = tmp_path / "output"
    old = Task(OLD_ID, "04_refinement", ("export", "--format", "fa"), root, cpus=1)
    log = tmp_path / "old.log"
    log.write_text(f"# task: {OLD_ID}\n# stage: {old.stage}\n# cwd: {old.cwd}\n"
                   f"# command: {old.display_command()}\n# workflow_max_memory_gb: 2\n", encoding="utf-8")
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        state.start(OLD_ID, old.fingerprint([]), log)
        state.finish(OLD_ID, "success", 0)
        record = state.get(OLD_ID)
        updated = replace(old, id=NEW_ID, cpus=4)
        assert _legacy_resource_resume_matches(updated, record, [], frozenset())
        assert not _legacy_resource_resume_matches(replace(updated, command=("different",)), record, [], frozenset())
    finally:
        state.close()


@pytest.mark.parametrize("name", ["04.bins", "04.bins.", "04.bins_extra.HMI_1", "03.bin.HMI_1"])
def test_alias_is_restricted_to_exact_refined_export_prefix(name):
    assert checkpoint_task_id(name) == name
    assert current_task_id(name) == name

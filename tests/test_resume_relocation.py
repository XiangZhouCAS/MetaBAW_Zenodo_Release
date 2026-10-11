"""Real task graphs/files and SQLite; upstream binning workers are synthetic."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from metabaw.cli import _task_dict
from metabaw.executor import Executor, RunResult, _legacy_generated_inputs
from metabaw.model import Task, topological_order
from metabaw.relocation import ResumeLayout, portable_checkpoint, checkpoint_signature
from metabaw.state import StateStore


def _manifest(root, tasks, temporary=None):
    return {"result_files": {"workflow_details": str(root / "workflow_details.md")},
            "temporary_files": {"workspace": str(temporary or root / "tmp")},
            "tasks": [_task_dict(task) for task in tasks]}


def _tasks(root, data, threads=8, temporary=None):
    temporary = temporary or root / "tmp"
    work = temporary / "work"
    contigs, index, bam, bins = (work / name for name in ("contigs.fa", "index.mmi", "reads.bam", "bins.done"))
    return [
        Task("prepare", "01_prepare", ("prepare", "--source", str(data / "source.fa"), "--output", str(contigs)),
             work, inputs=(data / "source.fa",), outputs=(contigs,)),
        Task("index", "02_mapping", f"minimap2 -t {threads} -d '{index}' '{contigs}'", work,
             deps=("prepare",), inputs=(contigs,), outputs=(index,), cpus=threads),
        Task("map", "02_mapping", f"minimap2 -t {threads} -ax map-ont '{index}' '{data / 'reads.fq'}' "
             f"| samtools sort -@ {threads} -o '{bam}'", work,
             deps=("index",), inputs=(contigs, index, data / "reads.fq"), outputs=(bam,), cpus=threads),
        Task("bin", "03_binning", f"LorBin bin --num_process {threads} -b '{bam}' -o '{bins}' --cuda", work,
             deps=("map",), inputs=(bam,), outputs=(bins,), cpus=threads, gpus=1),
    ]


def _data(tmp_path):
    data = tmp_path / "external data"
    data.mkdir()
    (data / "source.fa").write_text(">contig\nACGT\n", encoding="utf-8")
    (data / "reads.fq").write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    return data


def _complete(tasks):
    for task in tasks:
        for output in task.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("complete", encoding="utf-8")


def _seed(root, tasks, *, portable=True, temporary=None, missing_generated=False, checkpoint_version=2):
    temporary = temporary or root / "tmp"
    layout = ResumeLayout(root, temporary)
    state = StateStore(temporary / "runtime" / "state.sqlite3")
    generated = _legacy_generated_inputs(topological_order(tasks))
    strict, semantic, signatures = {}, {}, {}
    try:
        for task in topological_order(tasks):
            missing = generated[task.id] if missing_generated else frozenset()
            strict[task.id] = task.fingerprint([strict[dep] for dep in task.deps], missing)
            semantic[task.id] = task.resume_fingerprint([semantic[dep] for dep in task.deps], missing)
            checkpoint = portable_checkpoint(task, generated[task.id], [signatures[dep] for dep in task.deps], layout.current,
                                             version=checkpoint_version)
            signatures[task.id] = checkpoint_signature(checkpoint)
            log = temporary / "runtime" / "logs" / (task.id + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"# task: {task.id}\n# stage: {task.stage}\n# cwd: {task.cwd}\n"
                           f"# command: {task.display_command()}\n# workflow_max_memory_gb: 2\n", encoding="utf-8")
            state.start(task.id, strict[task.id], log, semantic[task.id], checkpoint if portable else None)
            state.finish(task.id, "success", 0)
    finally:
        state.close()
    manifest = _manifest(root, tasks, temporary)
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _worker(task, log, *_args):
    for output in task.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("rerun", encoding="utf-8")
    return RunResult(task.id, 0, "")


def _run(root, tasks, manifest=None, *, temporary=None):
    temporary = temporary or root / "tmp"
    state = StateStore(temporary / "runtime" / "state.sqlite3")
    progress = io.StringIO()
    try:
        layout = ResumeLayout(root, temporary, manifest)
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute, redirect_stdout(progress):
            statuses = Executor(state, temporary / "runtime" / "logs", max_cpus=8, max_parallel=4,
                                max_gpus=1, show_progress=True, resume_layout=layout).run(tasks)
        return statuses, [call.args[0].id for call in execute.call_args_list], list(state.records()), progress.getvalue()
    finally:
        state.close()


@pytest.mark.parametrize("portable", [True, False])
@pytest.mark.parametrize("operation", ["copy", "rename"])
def test_copy_or_rename_reuses_complete_chain_and_original_provenance(tmp_path, portable, operation):
    data = _data(tmp_path)
    original, moved = tmp_path / "old output", tmp_path / "renamed output"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks, portable=portable)
    old_state = StateStore(original / "tmp/runtime/state.sqlite3")
    old_record = old_state.get("map")
    old_state.close()
    if operation == "copy":
        shutil.copytree(original, moved)
    else:
        original.rename(moved)
    statuses, executed, records, progress = _run(moved, _tasks(moved, data, 2), manifest)
    assert set(statuses.values()) == {"skipped"}
    assert not executed
    assert "relocated=4" in progress
    record = next(item for item in records if item.task_id == "map")
    assert record.fingerprint == old_record.fingerprint
    assert record.started_at == old_record.started_at
    assert Path(record.log_path).is_relative_to(moved)
    assert str(original) in Path(record.log_path).read_text(encoding="utf-8")
    if operation == "copy":
        assert tasks[2].outputs[0].read_text(encoding="utf-8") == "complete"
        assert old_record.log_path == str(original / "tmp/runtime/logs/map.log")


def test_legacy_manifest_without_cwd_uses_original_work_directory(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks, portable=False)
    for item in manifest["tasks"]:
        item.pop("cwd")
    original.rename(moved)
    assert not _run(moved, _tasks(moved, data, 2), manifest)[1]


@pytest.mark.parametrize("checkpoint_version", [None, 1, 2])
@pytest.mark.parametrize("operation", ["copy", "same_location"])
def test_main_python_install_location_is_not_an_analysis_change(tmp_path, checkpoint_version, operation):
    data = _data(tmp_path)
    original = tmp_path / "old"
    moved = tmp_path / "copy" if operation == "copy" else original
    old = _tasks(original, data)
    old[0] = replace(old[0], command=(
        "/home/zhoux/anaconda3/envs/metabaw/bin/python", "-m", "metabaw.internal",
        "concat-fasta", "--input", str(data / "source.fa"), "--output", str(old[0].outputs[0]),
    ))
    _complete(old)
    manifest = _seed(original, old, portable=checkpoint_version is not None,
                     checkpoint_version=checkpoint_version or 2)
    if operation == "copy":
        shutil.copytree(original, moved)
    new = _tasks(moved, data, 2)
    new[0] = replace(new[0], command=(
        "/home/zhoux/miniconda3/envs/metabaw/bin/python3.11", "-m", "metabaw.internal",
        "concat-fasta", "--input", str(data / "source.fa"), "--output", str(new[0].outputs[0]),
    ))
    statuses, executed, _, _ = _run(moved, new, manifest)
    assert not executed
    assert set(statuses.values()) == {"skipped"}


def test_remaining_legacy_bins_survive_partial_restart_in_copy(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    old = _tasks(original, data)
    _complete(old)
    manifest = _seed(original, old, portable=False)
    original_state_bytes = (original / "tmp/runtime/state.sqlite3").read_bytes()
    shutil.copytree(original, moved)
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    try:
        # The buggy restart already overwrote upstream records in the copy,
        # but successful old bins and the original tree remain untouched.
        updated = _tasks(moved, data, 2)
        strict, semantic = {}, {}
        for task in updated[:3]:
            strict[task.id] = task.fingerprint([strict[dep] for dep in task.deps])
            semantic[task.id] = task.resume_fingerprint([semantic[dep] for dep in task.deps])
            state.start(task.id, strict[task.id], moved / "tmp/runtime/logs" / (task.id + ".log"), semantic[task.id])
            state.finish(task.id, "success", 0)
    finally:
        state.close()
    assert not _run(moved, updated, manifest)[1]
    assert (original / "tmp/runtime/state.sqlite3").read_bytes() == original_state_bytes


def test_v1_upgrade_validates_old_dependency_signatures_without_original_manifest(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    old = _tasks(original, data)
    _complete(old)
    _seed(original, old, checkpoint_version=1)
    shutil.copytree(original, moved)
    (moved / "run_manifest.json").unlink()
    assert not _run(moved, _tasks(moved, data, 2))[1]
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    assert all(json.loads(record.portable_checkpoint)["version"] == 2 for record in state.records())
    state.close()


def test_bad_restart_does_not_reuse_legacy_bins_when_original_reads_changed(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    old = _tasks(original, data)
    _complete(old)
    manifest = _seed(original, old, portable=False)
    shutil.copytree(original, moved)
    (data / "reads.fq").write_text("different reads", encoding="utf-8")
    _, executed, _, _ = _run(moved, _tasks(moved, data, 2), manifest)
    assert executed == ["map", "bin"]
    diagnostic = json.loads((moved / "tmp/runtime/resume_diagnostics.json").read_text())
    reasons = {row["task_id"]: row for row in diagnostic["tasks"]}
    assert reasons["map"]["action"] == "run"
    assert reasons["bin"]["blocked_by"] == ["map"]


def test_newly_executed_tasks_get_portable_checkpoints_without_manual_seeding(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    tasks = _tasks(original, data)
    statuses, executed, records, _ = _run(original, tasks)
    assert set(statuses.values()) == {"success"}
    assert executed == ["prepare", "index", "map", "bin"]
    assert all(record.portable_checkpoint for record in records)
    shutil.copytree(original, moved, copy_function=shutil.copy)
    shutil.rmtree(original)
    assert not _run(moved, _tasks(moved, data, 2))[1]


def test_copy_within_original_directory_can_resume_repeatedly(tmp_path):
    data = _data(tmp_path)
    original = tmp_path / "original"
    moved = original / "copy"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks)
    shutil.copytree(original, moved)
    assert not _run(moved, _tasks(moved, data, 2), manifest)[1]
    assert not _run(moved, _tasks(moved, data, 4), _manifest(moved, _tasks(moved, data, 2)))[1]
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    assert all(Path(record.log_path).is_relative_to(moved / "tmp/runtime/logs") for record in state.records())
    state.close()


def test_cli_reads_and_preserves_previous_manifest_before_replacing_it(tmp_path):
    from argparse import Namespace
    from metabaw.cli import _run_direct
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    tasks = _tasks(original, data)
    _complete(tasks)
    _seed(original, tasks, portable=False)
    original.rename(moved)
    args = Namespace(threads=8, task=4, threads_per_task=2, concurrent_sample_limit=4,
                     max_memory=2., retries=0, gpu=True, cuda_task_slots=1, delete_tmp_files=False)
    with patch("metabaw.executor._execute_task") as execute, redirect_stdout(io.StringIO()):
        result = _run_direct(args, _tasks(moved, data, 2), moved, moved / "tmp", {"module": "annotation"})
    assert result == 0
    execute.assert_not_called()
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    remembered = json.loads(state.get_metadata("relocation_legacy_manifest"))
    assert str(original) in remembered["result_files"]["workflow_details"]
    state.close()
    current = json.loads((moved / "run_manifest.json").read_text(encoding="utf-8"))
    assert current["temporary_files"]["workspace"] == str(moved / "tmp")


def test_portable_copy_does_not_require_generated_file_or_directory_timestamps(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    tasks = _tasks(original, data)
    directory = original / "tmp/work/collection"
    directory.mkdir(parents=True)
    (directory / "bin.fa").write_text(">c\nACGT\n", encoding="utf-8")
    tasks[2] = replace(tasks[2], inputs=(*tasks[2].inputs, directory))
    tasks[0] = replace(tasks[0], outputs=(*tasks[0].outputs, directory))
    _complete([replace(task, outputs=tuple(path for path in task.outputs if path != directory)) for task in tasks])
    _seed(original, tasks)
    shutil.copytree(original, moved, copy_function=shutil.copy)
    # Make every intermediate mtime explicitly different, including directories.
    for path in (moved / "tmp/work").rglob("*"):
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 999_000_000))
    # Removing the original is safe: resume must not rely on its continued existence.
    shutil.rmtree(original)
    new = _tasks(moved, data, 2)
    new_dir = moved / "tmp/work/collection"
    new[0] = replace(new[0], outputs=(*new[0].outputs, new_dir))
    new[2] = replace(new[2], inputs=(*new[2].inputs, new_dir))
    statuses, executed, _, _ = _run(moved, new)
    assert not executed
    assert set(statuses.values()) == {"skipped"}


@pytest.mark.parametrize("change, expected", [
    ("reads", ["map", "bin"]), ("contigs", ["prepare", "index", "map", "bin"]),
    ("preset", ["map", "bin"]), ("database", ["bin"]), ("gpu_mode", ["bin"]),
    ("missing_index", ["index", "map", "bin"]), ("failed", ["bin"]), ("running", ["bin"]),
])
@pytest.mark.parametrize("portable", [True, False])
def test_relocation_still_rejects_changed_inputs_options_and_incomplete_tasks(tmp_path, change, expected, portable):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    old = _tasks(original, data)
    _complete(old)
    manifest = _seed(original, old, portable=portable)
    shutil.copytree(original, moved)
    new = _tasks(moved, data, 2)
    if change in {"reads", "contigs"}:
        (data / ("reads.fq" if change == "reads" else "source.fa")).write_text("changed_input", encoding="utf-8")
    elif change == "preset":
        new[2] = replace(new[2], command=new[2].display_command().replace("map-ont", "map-hifi"))
    elif change == "database":
        new[3] = replace(new[3], env={"GTDBTK_DATA_PATH": str(tmp_path / "new_database")})
    elif change == "gpu_mode":
        new[3] = replace(new[3], command=new[3].display_command().replace(" --cuda", ""), gpus=0)
    elif change == "missing_index":
        new[1].outputs[0].unlink()
    else:
        state = StateStore(moved / "tmp/runtime/state.sqlite3")
        if change == "failed":
            state.finish("bin", "failed", 1)
        else:
            state.start("bin", state.get("bin").fingerprint, Path(state.get("bin").log_path))
        state.close()
    _, executed, _, _ = _run(moved, new, manifest)
    assert executed == expected


def test_multiple_moves_upgrade_checkpoints_and_no_longer_need_old_manifest(tmp_path):
    data = _data(tmp_path)
    first, second, third = (tmp_path / name for name in ("first", "second", "third"))
    tasks = _tasks(first, data)
    _complete(tasks)
    manifest = _seed(first, tasks, portable=False)
    first.rename(second)
    assert not _run(second, _tasks(second, data, 2), manifest)[1]
    second.rename(third)
    (third / "run_manifest.json").unlink()
    for path in (third / "tmp/work").glob("*"):
        os.utime(path, None)
    assert not _run(third, _tasks(third, data, 1))[1]
    assert not _run(third, _tasks(third, data, 4))[1]


def test_same_location_does_not_hide_modified_intermediate_input(tmp_path):
    data = _data(tmp_path)
    root = tmp_path / "output"
    tasks = _tasks(root, data)
    _complete(tasks)
    _seed(root, tasks)
    tasks[0].outputs[0].write_text("modified_contigs", encoding="utf-8")
    assert _run(root, _tasks(root, data, 2))[1] == ["index", "map", "bin"]


def test_legacy_copy_with_original_available_tolerates_new_copy_timestamps(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks, portable=False)
    shutil.copytree(original, moved, copy_function=shutil.copy)
    assert not _run(moved, _tasks(moved, data, 2), manifest)[1]


def test_legacy_proof_is_retained_when_manifest_is_replaced_before_migration(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    old = _tasks(original, data)
    _complete(old)
    manifest = _seed(original, old, portable=False)
    original.rename(moved)
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    ResumeLayout(moved, moved / "tmp", manifest).remember(state)
    state.close()
    updated = _manifest(moved, _tasks(moved, data, 2))
    (moved / "run_manifest.json").write_text(json.dumps(updated), encoding="utf-8")
    assert not _run(moved, _tasks(moved, data, 2), updated)[1]


def test_corrupt_checkpoint_does_not_bypass_input_validation(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks)
    shutil.copytree(original, moved)
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    state.set_portable_checkpoint("index", "broken json")
    state.close()
    assert _run(moved, _tasks(moved, data, 2), manifest)[1] == ["index", "map", "bin"]


@pytest.mark.parametrize("version", [[], {}, 3, "2", True])
def test_malformed_checkpoint_version_is_rejected_without_crashing(tmp_path, version):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "copy"
    tasks = _tasks(original, data)
    _complete(tasks)
    manifest = _seed(original, tasks)
    shutil.copytree(original, moved)
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    saved = json.loads(state.get("index").portable_checkpoint)
    saved["version"] = version
    state.set_portable_checkpoint("index", json.dumps(saved))
    state.close()
    assert _run(moved, _tasks(moved, data, 2), manifest)[1] == ["index", "map", "bin"]


def test_custom_external_tmp_can_be_relocated_together_with_output(tmp_path):
    data = _data(tmp_path)
    original, moved, tmp_old, tmp_new = (tmp_path / name for name in ("old", "new", "scratch-old", "scratch-new"))
    original.mkdir()
    tasks = _tasks(original, data, temporary=tmp_old)
    _complete(tasks)
    manifest = _seed(original, tasks, temporary=tmp_old)
    original.rename(moved)
    tmp_old.rename(tmp_new)
    assert not _run(moved, _tasks(moved, data, 2, temporary=tmp_new), manifest, temporary=tmp_new)[1]


def test_root_replacement_observes_boundaries_and_nested_json(tmp_path):
    from metabaw.relocation import _normalize
    root = tmp_path / "old result"
    layout = ResumeLayout(root, root / "tmp")
    normalized = _normalize(json.dumps({"bam": str(root / "tmp/work/a.bam"),
                                        "external": str(tmp_path / "old result-other/data")}), layout.current)
    assert "@METABAW_TMP@" in normalized
    assert str(tmp_path / "old result-other").replace("\\", "\\\\") in normalized


def test_internal_links_are_repaired_without_traversing_external_directories(tmp_path):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    tasks = _tasks(original, data)
    _complete(tasks)
    links = original / "tmp/work/links"
    links.mkdir()
    try:
        (links / "bam").symlink_to(tasks[2].outputs[0])
        (links / "external").symlink_to(data, target_is_directory=True)
        (data / "leave_me").symlink_to(tasks[2].outputs[0])
    except OSError:
        pytest.skip("Real symlinks require Windows developer mode/privileges")
    _seed(original, tasks)
    original.rename(moved)
    _, _, _, progress = _run(moved, _tasks(moved, data, 2))
    assert "repaired_internal_links=1" in progress
    assert (moved / "tmp/work/links/bam").resolve() == moved / "tmp/work/reads.bam"
    assert (moved / "tmp/work/links/external").resolve() == data
    assert os.readlink(data / "leave_me") == str(original / "tmp/work/reads.bam")


@pytest.mark.parametrize("mapper", ["minimap2", "bowtie2", "minibwa"])
@pytest.mark.parametrize("portable", [True, False])
def test_real_binning_graph_copy_reuses_mapping_and_every_selected_binner(tmp_path, mapper, portable):
    from metabaw.direct import DirectBinBuilder
    from metabaw.discovery import ReadSample, Analysis
    from test_resume_resources import _bin_options, _complete_plan_outputs

    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    (data / "R2.fq").write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("HMI_1", data / "reads.fq", None if mapper == "minimap2" else data / "R2.fq",
                        contigs=data / "source.fa")
    analysis = Analysis("HMI_1", (sample,), (data / "source.fa",), False)
    options = replace(_bin_options(tmp_path, 8, 8), outdir=original, workdir=original / "tmp/work",
                      align_tool=mapper, read_type="long" if mapper == "minimap2" else "short",
                      binners=("metabat2", "metadecoder", "vamb", "comebin", "semibin2", "lorbin"))
    old = DirectBinBuilder([analysis], options, tmp_path).build()
    new = DirectBinBuilder([analysis], replace(options, outdir=moved, workdir=moved / "tmp/work", threads=2), tmp_path).build()
    _complete_plan_outputs(old)
    manifest = _seed(original, old, portable=portable)
    shutil.copytree(original, moved)
    statuses, executed, _, _ = _run(moved, new, manifest)
    assert not executed
    for tool in options.binners:
        assert statuses[f"03.bin.{tool}.HMI_1"] == "skipped"
    assert statuses["02.index.HMI_1"] == "skipped"
    assert statuses["02.map.HMI_1.HMI_1"] == "skipped"


@pytest.mark.parametrize("checkpoint_version", [None, 1, 2])
def test_named_rumen_plan_can_move_main_python_conda_frontend_and_all_six_binners(tmp_path, checkpoint_version):
    from metabaw.direct import BINNER_ORDER
    from metabaw.discovery import ReadSample
    from metabaw.named_inputs import BinningProfile, NamedBinBuilder, named_analyses
    from test_resume_resources import _bin_options, _complete_plan_outputs

    data = _data(tmp_path)
    original, moved = tmp_path / "checkm_drep_output", tmp_path / "checkm_drep_output_bak"
    samples = [ReadSample(f"HMI_{i}", data / "reads.fq", contigs=data / "source.fa") for i in (1, 2)]
    analyses = named_analyses(samples, [], original / "tmp/work/assembly")
    profiles = {sample.name: BinningProfile("long", "minimap2", BINNER_ORDER) for sample in samples}
    options = replace(_bin_options(tmp_path, 64, 64), outdir=original, workdir=original / "tmp/work",
                      binners=BINNER_ORDER,
                      comebin_run_prefix=("/home/zhoux/anaconda3/condabin/conda", "run", "--name", "comebin"),
                      lorbin_run_prefix=("/home/zhoux/anaconda3/condabin/conda", "run", "--name", "lorbin"))
    with patch("metabaw.direct.sys.executable", "/home/zhoux/anaconda3/envs/metabaw/bin/python"), \
         patch("metabaw.direct.LORBIN_COMPAT_SCRIPT", Path("/old/site-packages/metabaw/lorbin_compat.py")):
        old = NamedBinBuilder(analyses, options, options.workdir, profiles).build()
    _complete_plan_outputs(old)
    manifest = _seed(original, old, portable=checkpoint_version is not None,
                     checkpoint_version=checkpoint_version or 2)
    shutil.copytree(original, moved)
    new_options = replace(options, outdir=moved, workdir=moved / "tmp/work", threads=32, total_threads=128,
                          comebin_run_prefix=("/home/zhoux/miniconda3/condabin/conda", "run", "--name", "comebin"),
                          lorbin_run_prefix=("/home/zhoux/miniconda3/condabin/conda", "run", "--name", "lorbin"))
    with patch("metabaw.direct.sys.executable", "/home/zhoux/miniconda3/envs/metabaw/bin/python3.11"), \
         patch("metabaw.direct.LORBIN_COMPAT_SCRIPT", Path("/new/site-packages/metabaw/lorbin_compat.py")):
        new = NamedBinBuilder(analyses, new_options, new_options.workdir, profiles).build()
    statuses, executed, _, _ = _run(moved, new, manifest)
    # A cheap marker created with python -c can rerun; arbitrary python -c
    # scripts are deliberately not hidden by helper launcher normalization.
    assert executed == ["03.binning.complete"]
    for sample in samples:
        for tool in BINNER_ORDER:
            assert statuses[f"03.bin.{tool}.{sample.name}"] == "skipped"
        assert statuses[f"01.prepare.{sample.name}"] == "skipped"
        assert statuses[f"02.index.{sample.name}"] == "skipped"
        assert statuses[f"02.map.{sample.name}.{sample.name}"] == "skipped"


@pytest.mark.parametrize("portable", [True, False])
def test_real_annotation_graph_relocates_taxonomy_abundance_niche_and_functions(tmp_path, portable):
    from metabaw.direct import AnnotationBuilder, AnnotationOptions
    from metabaw.discovery import ReadSample
    from test_resume_resources import _complete_plan_outputs

    data = _data(tmp_path)
    sources = tuple(data / f"MAG_{number}.fa" for number in range(4))
    for source in sources:
        source.write_text(">c\nACGT\n", encoding="utf-8")
    original, moved = tmp_path / "old", tmp_path / "new"
    mags = tuple(original / "tmp/work/genomes" / source.name for source in sources)
    options = AnnotationOptions(
        mag_dir=original / "tmp/work/genomes", mags=mags, mag_suffix="fa", mag_sources=sources,
        reads=tuple(ReadSample(f"S{number}", data / "reads.fq") for number in range(4)),
        output=original, output_suffix=".tsv", threads=8, total_threads=8, read_type="long",
        methods=("relative_abundance", "rpkm", "tpm", "mean"), place_species=False,
        niche_rank="family", niche_method="cv", no_niche=False, gtdbtk_data=data / "gtdb",
        run_kegg=True, run_cazy=True, run_hydrogenase=True,
        kegg_db=data / "kegg", dbcan_db=data / "dbcan", hydrogenase_db=data / "hydrogenase")
    old = AnnotationBuilder(options, original / "tmp/work").build()
    _complete_plan_outputs(old)
    for directory in {path for task in old for path in task.fasta_output_dirs}:
        for fasta in directory.glob("*.fa"):
            fasta.write_text(">c\nACGT\n", encoding="utf-8")
    manifest = _seed(original, old, portable=portable)
    original.rename(moved)
    new_options = replace(options, output=moved, threads=2,
                          mag_dir=moved / "tmp/work/genomes",
                          mags=tuple(moved / "tmp/work/genomes" / source.name for source in sources))
    new = AnnotationBuilder(new_options, moved / "tmp/work").build()
    statuses, executed, _, _ = _run(moved, new, manifest)
    assert not executed
    assert set(statuses.values()) == {"skipped"}
    for name in ("01.annotation.gtdbtk", "04.annotation.niche", "06.annotation.kegg", "07.annotation.cazy"):
        assert statuses[name] == "skipped"


@pytest.mark.parametrize("fail_repair", [False, True])
def test_atomic_link_repair_on_mocked_windows_symlink_api(tmp_path, fail_repair):
    # Exercise the full repair branch even without Windows symlink privileges.
    original, moved = tmp_path / "old", tmp_path / "new"
    work = moved / "tmp/work"
    work.mkdir(parents=True)
    link, bam = work / "link.bam", work / "reads.bam"
    link.write_text("original_link", encoding="utf-8")
    bam.write_text("bam", encoding="utf-8")
    manifest = _manifest(original, [])
    layout = ResumeLayout(moved, moved / "tmp", manifest)
    state = StateStore(moved / "tmp/runtime/state.sqlite3")
    is_symlink = Path.is_symlink
    def symbolic(path):
        return path == link or is_symlink(path)
    def create(path, target, **_kwargs):
        if fail_repair:
            raise OSError("cannot create symlink")
        path.write_text(str(target), encoding="utf-8")
    try:
        with patch.object(Path, "is_symlink", symbolic), patch.object(Path, "symlink_to", create), \
             patch("metabaw.relocation.os.readlink", return_value=str(original / "tmp/work/reads.bam")):
            if fail_repair:
                with pytest.raises(OSError, match="cannot create symlink"):
                    layout.prepare(state)
                assert link.read_text(encoding="utf-8") == "original_link"
            else:
                assert layout.prepare(state) == 1
                assert link.read_text(encoding="utf-8") == str(bam)
        assert not list(work.glob(".*.metabaw-relocate-*"))
    finally:
        state.close()


@pytest.mark.parametrize("bad_tasks", [None, "broken", [{"id": "prepare", "environment": None}]])
def test_malformed_legacy_manifest_cannot_grant_reuse_or_crash_scan(tmp_path, bad_tasks):
    data = _data(tmp_path)
    original, moved = tmp_path / "old", tmp_path / "new"
    old = _tasks(original, data)
    _complete(old)
    manifest = _seed(original, old, portable=False)
    original.rename(moved)
    manifest["tasks"] = bad_tasks
    assert _run(moved, _tasks(moved, data, 2), manifest)[1] == ["prepare", "index", "map", "bin"]


def test_databases_inside_owned_roots_are_not_mistaken_for_generated_assets(tmp_path):
    from metabaw.relocation import Location
    original, moved = tmp_path / "old", tmp_path / "new"
    db_old, db_new = original / "database.dmnd", moved / "database.dmnd"
    db_old.parent.mkdir()
    db_old.write_text("db", encoding="utf-8")
    db_new.parent.mkdir()
    shutil.copy2(db_old, db_new)
    old = Task("qc", "05_quality", f"checkm2 predict --database_path '{db_old}' -o '{original / 'qc'}'",
               original / "tmp/work", env={"GTDBTK_DATA_PATH": str(original / "gtdb")})
    new = replace(old, command=f"checkm2 predict --database_path '{db_new}' -o '{moved / 'qc'}'",
                  cwd=moved / "tmp/work", env={"GTDBTK_DATA_PATH": str(moved / "gtdb")})
    first = portable_checkpoint(old, frozenset(), [], Location(original, original / "tmp"))
    second = portable_checkpoint(new, frozenset(), [], Location(moved, moved / "tmp"))
    assert checkpoint_signature(first) != checkpoint_signature(second)

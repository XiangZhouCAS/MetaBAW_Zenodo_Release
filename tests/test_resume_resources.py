from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import re
import shlex
import sqlite3
import sys
from unittest.mock import patch

import pytest

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.cli import _thread_budget
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import Executor, RunResult, _legacy_generated_inputs
from metabaw.model import Task, topological_order
from metabaw.resume import command_thread_counts, resume_command
from metabaw.state import StateStore


@pytest.mark.parametrize("command", [
    "megahit --num-cpu-threads {n} -m 0.5",
    "flye --threads {n} --nano-hq reads.fq",
    "bowtie2-build --threads {n} contigs.fa index",
    "bowtie2 -p {n} -x index -1 R1.fq -2 R2.fq",
    "minimap2 -t {n} -p 0.8 -ax map-ont index.mmi reads.fq",
    "minibwa map -t{n} index reads.fq",
    "samtools sort -@{n} -o reads.bam -",
    "metabat2 -t {n} -m 1500 --seed 42",
    "vamb bin default -p {n} --minfasta 200000",
    "metadecoder cluster --threads={n} -f contigs.fa",
    "run_comebin.sh -t {n} -p bamset -b 1024 -d cuda",
    "SemiBin2 single_easy_bin --threads {n} --min-len 1500",
    "LorBin bin --num_process {n} --cuda",
    "python /installed/metabaw/lorbin_compat.py bin --num_process {n} --cuda",
    "parallel --jobs {n} --pipe 'prodigal -p meta -a proteins.faa'",
    "hmmsearch --cpu {n} --cut_nc markers.hmm proteins.faa",
    "DAS_Tool -t {n} --score_threshold 0.5",
    "metawrap bin_refinement -t {n} -c 50 -x 10",
    "checkm lineage_wf -t {n} -x fa bins quality",
    "conda run --no-capture-output --name checkm2 checkm2 predict --threads {n}",
    "gunc run --threads {n} --input_dir bins",
    "gtdbtk identify --cpus {n} --extension fa",
    "galah cluster --threads {n} --ani 99",
    "dRep dereplicate out -p {n} -comp 50 -con 10",
    "coverm genome --threads {n} --methods relative_abundance tpm",
    "exec_annotation --cpu {n} --format mapper --ko-list ko_list",
    "run_dbcan CAZyme_annotation --threads {n} --db_dir dbcan",
    "blastp -num_threads {n} -evalue 1e-50",
    "diamond blastp -p {n} -k 1",
    "python -B -m metabaw.internal rna-qc --threads {n} --trna-pass 18",
    "python -m metabaw.internal run-gtdbtk-classify --threads {n} --pplacer-threads 1",
])
def test_only_known_cpu_options_are_ignored(command):
    old, new = command.format(n=64), command.format(n=32)
    assert resume_command(old) == resume_command(new)
    assert 64 in command_thread_counts(old)


@pytest.mark.parametrize("old,new", [
    ("minimap2 -t 64 -p 0.8", "minimap2 -t 32 -p 0.9"),
    ("minimap2 -t 64 -x map-ont", "minimap2 -t 32 -x map-hifi"),
    ("prodigal -p single", "prodigal -p meta"),
    ("run_comebin.sh -t 64 -p bam1 -b 1024", "run_comebin.sh -t 32 -p bam2 -b 1024"),
    ("run_comebin.sh -t 64 -b 1024", "run_comebin.sh -t 32 -b 512"),
    ("LorBin --num_process 64 --cuda", "LorBin --num_process 32"),
    ("conda run -p /env1 vamb -p 64", "conda run -p /env2 vamb -p 32"),
    ("python -m metabaw.internal filter-quality --max-contamination 10",
     "python -m metabaw.internal filter-quality --max-contamination 5"),
    ("python -m metabaw.internal classify-niche-literature --method cv",
     "python -m metabaw.internal classify-niche-literature --method cv --cv-correct"),
    ("unknown-tool --threads 64", "unknown-tool --threads 32"),
    ("python -c 'print(64)'", "python -c 'print(32)'"),
    ("minimap2 -t 'broken", "minimap2 -t 'other"),
    ("python -m metabaw.internal helper --label ''", "python -m metabaw.internal helper --label"),
])
def test_analysis_options_and_unknown_commands_are_not_hidden(old, new):
    assert resume_command(old) != resume_command(new)


def test_nested_comebin_command_is_normalized_without_hiding_batch_size():
    def command(threads, batch=1024):
        nested = f"/opt/conda/bin/conda run -n comebin run_comebin.sh -t {threads} -p '/data/bam set' -b {batch}"
        return shlex.join((sys.executable, "-m", "metabaw.internal", "run-comebin", "--command", nested))
    assert resume_command(command(64)) == resume_command(command(32))
    assert resume_command(command(64)) != resume_command(command(32, 512))
    assert command_thread_counts(command(64)) == (64,)


@pytest.mark.parametrize("suffix", [
    "-m metabaw.internal concat-fasta --min-length 1500",
    "-u -m metabaw.internal run-gtdbtk-classify --threads 64",
    "/installed/metabaw/lorbin_compat.py bin --num_process 64 --cuda",
])
def test_main_helper_launcher_path_can_change_but_execution_hash_is_preserved(suffix, tmp_path):
    old = "/old/env/bin/python " + suffix
    new = "/new/env/bin/python3.11 " + suffix
    assert resume_command(old) != resume_command(new)
    assert resume_command(old, launcher_independent=True) == resume_command(new, launcher_independent=True)
    task = Task("helper", "01_prepare", old, tmp_path)
    assert task.fingerprint([]) != replace(task, command=new).fingerprint([])


@pytest.mark.parametrize("old,new", [
    ("/old/python -m external_package", "/new/python -m external_package"),
    ("/old/python /external/lorbin_compat.py bin", "/new/python /external/lorbin_compat.py bin"),
    ("conda run -n lorbin python /old/metabaw/lorbin_compat.py bin --cuda",
     "conda run -n different-lorbin python /new/metabaw/lorbin_compat.py bin --cuda"),
    ("/old/vamb bin default -p 64", "/new/vamb bin default -p 32"),
    ("python -m metabaw.internal helper --min-length 1500", "python3.11 -m metabaw.internal helper --min-length 2000"),
])
def test_launcher_compatibility_never_hides_external_software_or_analysis_changes(old, new):
    assert resume_command(old, launcher_independent=True) != resume_command(new, launcher_independent=True)


def test_conda_frontend_path_is_not_its_selected_environment():
    old = "/home/zhoux/anaconda3/condabin/conda run --name comebin run_comebin.sh -t 64 -b 1024"
    new = "/home/zhoux/miniconda3/condabin/conda run --name comebin run_comebin.sh -t 32 -b 1024"
    assert resume_command(old, launcher_independent=True) == resume_command(new, launcher_independent=True)
    assert resume_command(old, launcher_independent=True) != resume_command(new.replace("--name comebin", "--name other"), launcher_independent=True)


def test_semantic_fingerprint_ignores_cpu_allocation_and_thread_environment(tmp_path):
    old = Task("index", "02_mapping", "minimap2 -t 64 -d index.mmi contigs.fa", tmp_path,
               cpus=64, env={"OMP_NUM_THREADS": "64", "GTDBTK_DATA_PATH": "/db/v1"})
    new = replace(old, command="minimap2 -t 32 -d index.mmi contigs.fa", cpus=32,
                  priority=3, enforce_memory_limit=False, minimum_memory_gb=160,
                  env={"OMP_NUM_THREADS": "32", "GTDBTK_DATA_PATH": "/db/v1"})
    assert old.fingerprint([]) != new.fingerprint([])
    assert old.resume_fingerprint([]) == new.resume_fingerprint([])
    assert old.resume_fingerprint([]) != replace(new, env={"GTDBTK_DATA_PATH": "/db/v2"}).resume_fingerprint([])


def _log(task: Task, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = " ".join(f"{key}={value}" for key, value in sorted(task.env.items()))
    destination.write_text(
        f"# task: {task.id}\n# stage: {task.stage}\n# cwd: {task.cwd}\n"
        f"# environment_overrides: {environment}\n# command: {task.display_command()}\n"
        "# workflow_max_memory_gb: 1007\n", encoding="utf-8",
    )


def _seed(state: StateStore, tasks, *, legacy=False, missing_generated=False):
    ordered = topological_order(tasks)
    generated = _legacy_generated_inputs(ordered)
    strict, semantic = {}, {}
    for task in ordered:
        missing = generated[task.id] if missing_generated else frozenset()
        strict[task.id] = task.fingerprint([strict[dep] for dep in task.deps], missing)
        semantic[task.id] = task.resume_fingerprint([semantic[dep] for dep in task.deps], missing)
        log = task.cwd / "logs" / (task.id + ".log")
        _log(task, log)
        state.start(task.id, strict[task.id], log,
                    resume_fingerprint=None if legacy else semantic[task.id])
        state.finish(task.id, "success", 0)


def _chain(root, threads=64):
    source = root / "source.fa"
    source.write_text(">contig\nACGT\n", encoding="utf-8")
    contigs, index, bam, bins = (root / name for name in ("prepared.fa", "index.mmi", "reads.bam", "bins.done"))
    for result in (contigs, index, bam, bins):
        result.write_text("complete", encoding="utf-8")
    return [
        Task("prepare", "01_prepare", "prepare source.fa prepared.fa", root, inputs=(source,), outputs=(contigs,)),
        Task("index", "02_mapping", f"minimap2 -x map-ont -t {threads} -d index.mmi prepared.fa", root,
             deps=("prepare",), inputs=(contigs,), outputs=(index,), cpus=threads),
        Task("map", "02_mapping", f"minimap2 -t {threads} -ax map-ont index.mmi reads.fq | samtools sort -@ {threads} -o reads.bam", root,
             deps=("index",), inputs=(contigs, index), outputs=(bam,), cpus=threads),
        Task("bin", "03_binning", f"LorBin bin --num_process {threads} -b reads.bam --cuda", root,
             deps=("map",), inputs=(bam,), outputs=(bins,), cpus=threads),
    ]


def _changed_threads(tasks, threads=32):
    return [replace(task, command=task.display_command().replace(" 64", f" {threads}"),
                    cpus=threads if task.cpus == 64 else task.cpus) for task in tasks]


def _worker(task, log_path, *_args):
    _log(task, log_path)
    for output in task.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.is_dir():
            output.write_text("rerun", encoding="utf-8")
    return RunResult(task.id, 0, "")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("missing_generated", [False, True])
def test_threads_and_concurrency_reuse_whole_completed_chain(tmp_path, legacy, missing_generated):
    old = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, old, legacy=legacy, missing_generated=missing_generated)
        old_record = state.get("index")
        new = _changed_threads(old)
        progress = io.StringIO()
        with patch("metabaw.executor._execute_task") as execute, redirect_stdout(progress):
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4,
                                show_progress=True).run(new)
        execute.assert_not_called()
        assert statuses == {task.id: "skipped" for task in new}
        assert "resource-compatible=3" in progress.getvalue()
        upgraded = state.get("index")
        assert upgraded.resume_fingerprint is not None
        assert upgraded.fingerprint == old_record.fingerprint
        assert upgraded.started_at == old_record.started_at
        assert "-t 64" in Path(upgraded.log_path).read_text(encoding="utf-8")
        # After migration, logs are no longer required to identify CPU-only
        # changes; preserve user logs instead of asking them to delete state.
        Path(upgraded.log_path).unlink()
        with patch("metabaw.executor._execute_task") as execute:
            assert all(value == "skipped" for value in Executor(
                state, tmp_path / "logs", max_cpus=64, max_parallel=2,
            ).run(_changed_threads(old, 16)).values())
        execute.assert_not_called()
    finally:
        state.close()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("change", ["input", "preset", "missing_index", "failed_index", "interrupted_index"])
def test_real_changes_or_incomplete_tasks_rerun_only_affected_chain(tmp_path, legacy, change):
    tasks = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, tasks, legacy=legacy)
        new = _changed_threads(tasks)
        if change == "input":
            tasks[0].inputs[0].write_text(">different\nACGTACGT\n", encoding="utf-8")
        elif change == "preset":
            new[1] = replace(new[1], command=new[1].display_command().replace("map-ont", "map-hifi"))
        elif change == "missing_index":
            tasks[1].outputs[0].unlink()
        elif change == "failed_index":
            state.finish("index", "failed", 1)
        else:
            state.start("index", tasks[1].fingerprint([state.get("prepare").fingerprint]),
                        tmp_path / "logs" / "index.log")
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(new)
        executed = [call.args[0].id for call in execute.call_args_list]
        assert executed == (["prepare", "index", "map", "bin"] if change == "input" else ["index", "map", "bin"])
        assert statuses["prepare"] == ("success" if change == "input" else "skipped")
        # A second restart reuses newly completed tasks with fresh input stats.
        with patch("metabaw.executor._execute_task") as execute:
            assert all(value == "skipped" for value in Executor(
                state, tmp_path / "logs", max_cpus=128, max_parallel=4,
            ).run(new).values())
        execute.assert_not_called()
    finally:
        state.close()


def test_legacy_same_command_with_changed_input_is_not_migrated(tmp_path):
    tasks = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, tasks, legacy=True)
        tasks[1].inputs[0].write_text("modified-contigs", encoding="utf-8")
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(_changed_threads(tasks))
        assert [call.args[0].id for call in execute.call_args_list] == ["index", "map", "bin"]
    finally:
        state.close()


def test_old_sqlite_schema_is_upgraded_without_losing_records(tmp_path):
    database = tmp_path / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE task_runs (task_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
                           "status TEXT NOT NULL, started_at TEXT, ended_at TEXT, exit_code INTEGER, "
                           "log_path TEXT, message TEXT)")
        connection.execute("INSERT INTO task_runs VALUES ('index', 'old-hash', 'success', 'start', 'end', 0, 'old.log', 'old')")
    for _ in range(2):
        state = StateStore(database)
        try:
            record = state.get("index")
            assert record.fingerprint == "old-hash"
            assert record.status == "success"
            assert record.started_at == "start"
            assert record.resume_fingerprint is None
            assert list(state.records()) == [record]
        finally:
            state.close()


def test_adopted_outputs_can_resume_after_cpu_changes_without_logs(tmp_path):
    tasks = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        with patch("metabaw.executor._execute_task") as execute:
            executor = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4)
            assert all(value == "skipped" for value in executor.run(tasks).values())
            assert all(value == "skipped" for value in executor.run(_changed_threads(tasks)).values())
        execute.assert_not_called()
        assert state.get("index").log_path is None
        assert state.get("index").resume_fingerprint is not None
    finally:
        state.close()


def test_explicit_executor_force_still_reruns_complete_tasks(tmp_path):
    tasks = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, tasks)
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(tasks, force=True)
        assert all(value == "success" for value in statuses.values())
        assert execute.call_count == len(tasks)
    finally:
        state.close()


def test_unrelated_branch_is_reused_when_one_mapper_preset_changes(tmp_path):
    tasks = _chain(tmp_path)
    sibling_output = tmp_path / "unrelated.txt"
    sibling_output.write_text("complete", encoding="utf-8")
    tasks.append(Task("unrelated", "01_prepare", "unrelated-analysis", tmp_path,
                      outputs=(sibling_output,)))
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, tasks)
        new = _changed_threads(tasks)
        new[1] = replace(new[1], command=new[1].display_command().replace("map-ont", "map-hifi"))
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(new)
        assert statuses["unrelated"] == "skipped"
        assert [call.args[0].id for call in execute.call_args_list] == ["index", "map", "bin"]
    finally:
        state.close()


def test_ordering_only_dependency_does_not_invalidate_completed_output(tmp_path):
    result = tmp_path / "result.txt"
    result.write_text("complete", encoding="utf-8")
    gate = Task("gate", "01_prepare", "barrier", tmp_path)
    task = Task("result", "02_mapping", "minimap2 -t 64", tmp_path,
                wait_for=(gate.id,), outputs=(result,), cpus=64)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, [gate, task])
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(
                [gate, replace(task, command="minimap2 -t 32", cpus=32)],
            )
        assert statuses == {"gate": "success", "result": "skipped"}
        assert [call.args[0].id for call in execute.call_args_list] == ["gate"]
    finally:
        state.close()


def test_legacy_missing_log_does_not_bypass_analysis_validation(tmp_path):
    tasks = _chain(tmp_path)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, tasks, legacy=True)
        Path(state.get("index").log_path).unlink()
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4).run(_changed_threads(tasks))
        assert [call.args[0].id for call in execute.call_args_list] == ["index", "map", "bin"]
    finally:
        state.close()


def _bin_options(root, threads, total_threads):
    return BinOptions(
        outdir=root / "result", workdir=root / "work", threads=threads,
        total_threads=total_threads, read_type="long", align_tool="minimap2",
        flye_read_type="--nano-hq",
        binners=("comebin", "lorbin", "metadecoder", "vamb", "semibin2"),
        min_contig_length=1500, min_fasta_kbs=200, batch_size=1024,
        refiner="magscot", quality_control="checkm", min_completeness=50,
        max_contamination=10, min_quality_score=None, run_gunc=True,
        run_trna=True, trna_pass=18, run_rrna=True, rrna_pass=True,
        dereplicator="drep", environment=None, tag_contigs=True, gpu=True,
        max_gpu_memory="100%", assembly_strategy="default",
        magscot_dir=root / "MAGScoT", checkm2_db=None,
        gunc_db=root / "gunc.dmnd", gtdbtk_data=root / "gtdb",
        extra_args={},
    )


def _complete_plan_outputs(tasks):
    outputs = {path for task in tasks for path in task.outputs}
    outputs.update(path for task in tasks if task.output_alternatives
                   for path in task.output_alternatives[0])
    directories = {path for task in tasks for path in task.fasta_output_dirs}
    directories.update(path for path in outputs if any(path in other.parents for other in outputs))
    for directory in sorted(directories, key=lambda path: len(path.parts)):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test.fa").write_text(">contig\nACGT\n", encoding="utf-8")
    for output in sorted(outputs, key=lambda path: len(path.parts)):
        if output not in directories:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                output.write_text("test", encoding="utf-8")


@pytest.mark.parametrize("legacy", [False, True])
def test_fixing_global_qc_share_reuses_completed_32_thread_results(tmp_path, legacy):
    contigs = tmp_path / "HMI_1.contigs.fa"
    reads = tmp_path / "HMI_1.fastq"
    contigs.write_text(">contig\nACGT\n", encoding="utf-8")
    reads.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("HMI_1", reads, contigs=contigs)
    analysis = Analysis("HMI_1", (sample,), (contigs,), False)
    new = DirectBinBuilder([analysis], _bin_options(tmp_path, 32, 128), tmp_path).build()
    old = []
    global_ids = {"05.qc.checkm", "05.qc.gunc", "05.qc.rna_domain",
                  "05.qc.rna", "06.dereplicate.drep"}
    for task in new:
        if task.id in global_ids:
            if isinstance(task.command, str):
                command = re.sub(r"(--threads|--cpus|-t|-p) 128(?=\s|$)", r"\1 32", task.command)
            else:
                command = tuple("32" if value == "128" and task.command[i - 1] == "--threads"
                                else value for i, value in enumerate(task.command))
            task = replace(task, command=command, cpus=32)
        old.append(task)
    assert {task.cpus for task in new if task.id in global_ids} == {128}
    assert {task.cpus for task in old if task.id in global_ids} == {32}
    _complete_plan_outputs(old)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, old, legacy=legacy)
        with patch("metabaw.executor._execute_task") as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128,
                                max_parallel=4, max_gpus=1).run(new)
        execute.assert_not_called()
        assert all(value == "skipped" for value in statuses.values())
    finally:
        state.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_rumen_plan_reuses_mapping_and_all_binners_after_64_to_128_task4(tmp_path, legacy):
    assert _thread_budget(64, 1, 16) == (64, 1)
    assert _thread_budget(128, 4, 16) == (32, 4)
    contigs = tmp_path / "HMI_1.contigs.fa"
    reads = tmp_path / "HMI_1.fastq"
    contigs.write_text(">contig\nACGT\n", encoding="utf-8")
    reads.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("HMI_1", reads, contigs=contigs)
    analysis = Analysis("HMI_1", (sample,), (contigs,), False)
    old = DirectBinBuilder([analysis], _bin_options(tmp_path, 64, 64), tmp_path).build()
    new = DirectBinBuilder([analysis], _bin_options(tmp_path, 32, 128), tmp_path).build()
    _complete_plan_outputs(old)
    # Complete all synthetic outputs before recording their input metadata.
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, old, legacy=legacy)
        with patch("metabaw.executor._execute_task") as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4,
                                max_gpus=1).run(new)
        execute.assert_not_called()
        assert all(value == "skipped" for value in statuses.values())
        for tool in ("comebin", "lorbin", "metadecoder", "vamb", "semibin2"):
            assert statuses[f"03.bin.{tool}.HMI_1"] == "skipped"
        assert statuses["02.index.HMI_1"] == "skipped"
        assert statuses["02.map.HMI_1.HMI_1"] == "skipped"
    finally:
        state.close()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mapper", ["bowtie2", "minibwa"])
def test_actual_short_read_plan_also_reuses_index_bam_and_binners(tmp_path, legacy, mapper):
    contigs = tmp_path / "S1.contigs.fa"
    reads = tmp_path / "S1.fastq"
    contigs.write_text(">contig\nACGT\n", encoding="utf-8")
    reads.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("S1", reads, tmp_path / "S1.R2.fastq", contigs=contigs)
    analysis = Analysis("S1", (sample,), (contigs,), False)
    options = replace(_bin_options(tmp_path, 64, 64), read_type="short", align_tool=mapper,
                      binners=("metabat2", "metadecoder", "vamb", "comebin", "semibin2"))
    old = DirectBinBuilder([analysis], options, tmp_path).build()
    new = DirectBinBuilder([analysis], replace(options, threads=32, total_threads=128), tmp_path).build()
    _complete_plan_outputs(old)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, old, legacy=legacy)
        with patch("metabaw.executor._execute_task") as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4,
                                max_gpus=1).run(new)
        execute.assert_not_called()
        assert all(value == "skipped" for value in statuses.values())
        assert statuses["02.index.S1"] == "skipped"
        assert statuses["02.map.S1.S1"] == "skipped"
    finally:
        state.close()

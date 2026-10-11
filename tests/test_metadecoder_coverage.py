from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from metabaw import metadecoder_coverage as coverage
from metabaw.direct import DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.internal import build_parser, main
from metabaw.workflow_report import _method_calls, task_software
from test_resume_resources import _bin_options, _complete_plan_outputs, _seed, _worker
from metabaw.executor import Executor
from metabaw.state import StateStore


HEADER = "sequence id\tbin index\tbin size\tcoverage1\n"
VALID = HEADER + "ctg2\t1\t2000\t0\nctg1\t2\t1\t2.5\nctg1\t1\t2000\t1\n"
REFERENCES = {"ctg1": 2001, "ctg2": 2000}
BAM_HEADER = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:ctg1\tLN:2001\n@SQ\tSN:ctg2\tLN:2000\n"


def test_valid_coverage_allows_unordered_rows_and_zero_coverage(tmp_path):
    result = tmp_path / "coverage.tsv"
    result.write_text(VALID, encoding="utf-8")
    assert coverage.validate_coverage(result, REFERENCES, 1) == 3


@pytest.mark.parametrize("body,reason", [
    (HEADER, "expected bins missing"),
    (VALID.replace("ctg1\t2\t1\t2.5\n", ""), "expected bins missing"),
    (VALID + "ctg2\t1\t2000\t0\n", "Duplicate"),
    (VALID.replace("ctg2", "other"), "Invalid.*row"),
    (VALID.replace("ctg1\t2\t1", "ctg1\t3\t1"), "bin index"),
    (VALID.replace("ctg1\t2\t1", "ctg1\t2\t2000"), "bin size"),
    (VALID.replace("2.5", "nan"), "Non-finite"),
    (VALID.replace("2.5", "inf"), "Non-finite"),
    (VALID.replace("2.5", "-1"), "negative"),
    (VALID.replace("2.5", "oops"), "convert"),
    (VALID.replace("coverage1", "depth"), "header"),
    (VALID.replace("ctg2\t1\t2000\t0", "ctg2\t1\t2000"), "Invalid.*row"),
])
def test_invalid_coverage_is_rejected(tmp_path, body, reason):
    result = tmp_path / "coverage.tsv"
    result.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        coverage.validate_coverage(result, REFERENCES, 1)


def test_multiple_sample_columns_are_required(tmp_path):
    result = tmp_path / "coverage.tsv"
    result.write_text(VALID, encoding="utf-8")
    with pytest.raises(ValueError, match="sample count"):
        coverage.validate_coverage(result, REFERENCES, 2)
    result.write_text(VALID.replace("coverage1\n", "coverage1\tcoverage2\n").replace(
        "\t0\n", "\t0\t3\n").replace("\t2.5\n", "\t2.5\t0\n").replace(
        "\t1\n", "\t1\t0.5\n"), encoding="utf-8")
    assert coverage.validate_coverage(result, REFERENCES, 2) == 3


def _fixture(tmp_path):
    bam = tmp_path / "HMI 2.bam"
    bam.write_bytes(b"synthetic BAM")
    Path(str(bam) + ".index").write_bytes(b"stale custom index")
    Path(str(bam) + ".bai").write_bytes(b"samtools index")
    output, marker = tmp_path / "out" / "coverage.tsv", tmp_path / "out" / "coverage.v2.complete"
    output.parent.mkdir()
    output.write_text("old coverage", encoding="utf-8")
    marker.write_text("old marker", encoding="utf-8")
    return bam, output, marker


@pytest.mark.parametrize("threads", [1, 32])
def test_fresh_private_indexes_threads_and_atomic_publication(tmp_path, threads):
    bam, output, marker = _fixture(tmp_path)
    alias_paths = []

    def run(command, **kwargs):
        if command[1] == "quickcheck":
            return subprocess.CompletedProcess(command, 0)
        if command[1:3] == ["view", "-H"]:
            return subprocess.CompletedProcess(command, 0, BAM_HEADER, "")
        alias = Path(command[command.index("-b") + 1])
        alias_paths.append(alias)
        assert alias.read_bytes() == bam.read_bytes()
        assert not Path(str(alias) + ".index").exists()
        assert command[command.index("--threads") + 1] == str(threads)
        assert command[command.index("--bin_size") + 1] == "2000"
        assert command[:2] == ["/actual/metabaw/env/python3.11", "-u"]
        Path(str(alias) + ".index").write_bytes(b"new index")
        Path(command[command.index("-o") + 1]).write_text(VALID, encoding="utf-8")
        assert output.read_text(encoding="utf-8") == ("old coverage" if len(alias_paths) == 1 else VALID)
        assert not marker.exists()
        return subprocess.CompletedProcess(command, 0)

    with patch.object(coverage.shutil, "which", side_effect=lambda name: name), \
            patch.object(coverage, "metadecoder_executable", return_value="metadecoder"), \
            patch.object(coverage, "_python_launcher", return_value=["/actual/metabaw/env/python3.11"]), \
            patch.object(coverage.subprocess, "run", side_effect=run):
        coverage.run_coverage((bam,), output, marker, threads)
        coverage.run_coverage((bam,), output, marker, threads)
    assert output.read_text(encoding="utf-8") == VALID
    assert "3 bins" in marker.read_text(encoding="utf-8")
    assert alias_paths[0] != alias_paths[1]
    assert all(not path.parent.exists() for path in alias_paths)
    assert Path(str(bam) + ".index").read_bytes() == b"stale custom index"
    assert Path(str(bam) + ".bai").read_bytes() == b"samtools index"
    assert bam.read_bytes() == b"synthetic BAM"


@pytest.mark.parametrize("code,body", [(1, VALID), (0, HEADER)])
def test_worker_failure_or_partial_output_never_completes(tmp_path, code, body):
    bam, output, marker = _fixture(tmp_path)

    def run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(command, code)

    with patch.object(coverage.shutil, "which", side_effect=lambda name: name), \
            patch.object(coverage, "metadecoder_executable", return_value="metadecoder"), \
            patch.object(coverage, "_python_launcher", return_value=["python"]), \
            patch.object(coverage, "_bam_references", return_value=REFERENCES), \
            patch.object(coverage.subprocess, "run", side_effect=run):
        with pytest.raises((ValueError, RuntimeError), match="failed|Incomplete"):
            coverage.run_coverage((bam,), output, marker, 32)
    assert not marker.exists()
    assert output.read_text(encoding="utf-8") == "old coverage"
    assert not list(output.parent.glob(".metabaw-coverage-*"))


def test_bam_preflight_failure_clears_old_completion_marker(tmp_path):
    bam, output, marker = _fixture(tmp_path)
    with patch.object(coverage.shutil, "which", side_effect=lambda name: name), \
            patch.object(coverage, "metadecoder_executable", return_value="metadecoder"), \
            patch.object(coverage, "_python_launcher", return_value=["python"]), \
            patch.object(coverage.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "samtools quickcheck")):
        with pytest.raises(subprocess.CalledProcessError):
            coverage.run_coverage((bam,), output, marker, 32)
    assert not marker.exists()
    assert output.read_text(encoding="utf-8") == "old coverage"


@pytest.mark.parametrize("target", ["output", "marker", "same"])
def test_output_paths_cannot_overwrite_bam_or_each_other(tmp_path, target):
    bam, output, marker = _fixture(tmp_path)
    output = bam if target == "output" else output
    marker = bam if target == "marker" else output if target == "same" else marker
    with pytest.raises(ValueError, match="must not overwrite"):
        coverage.run_coverage((bam,), output, marker, 1)
    assert bam.read_bytes() == b"synthetic BAM"


@pytest.mark.parametrize("header,reason", [
    (BAM_HEADER.replace("coordinate", "queryname"), "coordinate-sorted"),
    (BAM_HEADER.replace("ctg2", "ctg1"), "duplicate"),
    (BAM_HEADER.replace("LN:2000", "LN:0"), "Invalid BAM"),
    ("@HD\tSO:coordinate\n", "Missing"),
])
def test_invalid_bam_headers_fail_before_native_coverage(header, reason):
    with patch.object(coverage.subprocess, "run", side_effect=[
        subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0, header, "")
    ]):
        with pytest.raises(ValueError, match=reason):
            coverage._bam_references((Path("input.bam"),), "samtools")


def test_multi_bam_reference_mismatch_is_rejected():
    with patch.object(coverage.subprocess, "run", side_effect=[
        subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0, BAM_HEADER, ""),
        subprocess.CompletedProcess([], 0, BAM_HEADER.replace("ctg1", "different"), ""),
    ]):
        with pytest.raises(ValueError, match="differ"):
            coverage._bam_references((Path("a.bam"), Path("b.bam")), "samtools")


def test_launcher_retains_the_executables_own_interpreter(tmp_path):
    launcher = tmp_path / "metadecoder"
    launcher.write_text("#!/opt/conda/bin/python3.12\n", encoding="utf-8")
    assert coverage._python_launcher(str(launcher)) == ["/opt/conda/bin/python3.12"]
    launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    with patch.object(coverage.shutil, "which", return_value="/env/bin/python3"):
        assert coverage._python_launcher(str(launcher)) == ["/env/bin/python3"]
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="interpreter"):
        coverage._python_launcher(str(launcher))
    launcher.write_text("#!/bin/sh\n'''exec' '/long env/bin/python3.12' \"$0\" \"$@\"\n", encoding="utf-8")
    assert coverage._python_launcher(str(launcher)) == ["/long env/bin/python3.12"]


@pytest.mark.parametrize("scenario", ["success", "timed_join", "failed_join", "failed_queue", "parent_error", "parent_error_queue", "unfinished"])
def test_real_guarded_child_processes_exit_without_hanging(tmp_path, scenario):
    # Real synthetic multiprocessing workers; neither MetaDecoder nor Linux BAM
    # data are needed to exercise joins, queue hangs and parent failure cleanup.
    launcher = tmp_path / "synthetic_cli.py"
    launcher.write_text(
        "import multiprocessing as mp, os, sys, time\n"
        "def worker(fail):\n"
        "    if fail: raise ValueError('synthetic BAM worker error')\n"
        "def idle(): time.sleep(20)\n"
        "def delayed(): time.sleep(0.3)\n"
        "if __name__ == '__main__':\n"
        "    scenario = sys.argv[1]\n"
        "    p = mp.Process(target=worker, args=(scenario.startswith('failed'),))\n"
        "    p.start()\n"
        "    if scenario == 'failed_queue':\n"
        "        q = mp.Queue(); q.put(b'x' * 2000000); q.close(); q.join_thread()\n"
        "    if scenario.startswith('parent_error'):\n"
        "        other = mp.Process(target=idle); other.start()\n"
        "        if scenario == 'parent_error_queue':\n"
        "            q = mp.Queue(); q.put(b'x' * 2000000)\n"
        "        raise RuntimeError('synthetic parent error')\n"
        "    if scenario == 'unfinished':\n"
        "        other = mp.Process(target=idle); other.start()\n"
        "    if scenario == 'timed_join':\n"
        "        other = mp.Process(target=delayed); other.start()\n"
        "        other.join(timeout=0.001); other.join(); other.close()\n"
        "    p.join(); p.close()\n"
        "    sys.exit(0)\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(Path(coverage.__file__).resolve()),
                             "--guard-cli", str(launcher), scenario],
                            capture_output=True, text=True, timeout=15)
    if scenario in ("success", "timed_join"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "synthetic" in result.stderr or "unfinished workers" in result.stderr
        if scenario.startswith("failed"):
            assert "MetaDecoder worker" in result.stderr


def test_internal_cli_routes_bams_threads_output_and_marker():
    arguments = ["run-metadecoder-coverage", "--bam", "a.bam", "--bam", "b.bam",
                 "--output", "coverage.tsv", "--completion-marker", "v2.complete", "--threads", "32"]
    assert build_parser().parse_args(arguments).threads == 32
    with patch("metabaw.internal.run_metadecoder_coverage") as run:
        main(arguments)
    run.assert_called_once_with((Path("a.bam"), Path("b.bam")), Path("coverage.tsv"),
                                Path("v2.complete"), 32)


def test_plan_accounts_for_coverage_threads_and_identifies_native_tools(tmp_path):
    contig, reads = tmp_path / "contigs.fa", tmp_path / "reads.fq"
    sample = ReadSample("HMI_2", reads, contigs=contig)
    tasks = DirectBinBuilder([Analysis("HMI_2", (sample,), (contig,), False)],
                             _bin_options(tmp_path, 32, 128), tmp_path).build()
    task = next(task for task in tasks if task.id == "03.coverage.metadecoder.HMI_2")
    assert task.cpus == 32
    assert task.command[task.command.index("--threads") + 1] == "32"
    assert task.outputs[1].name == "coverage.v2.complete"
    assert set(task_software(task)[0]) == {"MetaBAW", "metadecoder", "samtools"}
    assert any(tool == "metadecoder" for tool, options in _method_calls(task))
    assert any(tool == "metadecoder" and "--threads 32" in options
               for tool, options in _method_calls(task))


@pytest.mark.parametrize("existing_state", [True, False])
def test_upgrade_reruns_unvalidated_coverage_not_cached_mapping(tmp_path, existing_state):
    contigs, reads = tmp_path / "contigs.fa", tmp_path / "reads.fq"
    contigs.write_text(">ctg\nACGT\n", encoding="utf-8")
    reads.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("HMI_2", reads, contigs=contigs)
    tasks = DirectBinBuilder([Analysis("HMI_2", (sample,), (contigs,), False)],
                             _bin_options(tmp_path, 32, 128), tmp_path).build()
    _complete_plan_outputs(tasks)
    task = next(task for task in tasks if task.id == "03.coverage.metadecoder.HMI_2")
    task.outputs[1].unlink()
    old = [replace(value, outputs=value.outputs[:1], command="metadecoder coverage -b input.bam -o coverage.tsv", cpus=1)
           if value == task else value for value in tasks]
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        if existing_state:
            _seed(state, old)
        with patch("metabaw.executor._execute_task", side_effect=_worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128, max_parallel=4, max_gpus=1).run(tasks)
        assert statuses["02.index.HMI_2"] == "skipped"
        assert statuses["02.map.HMI_2.HMI_2"] == "skipped"
        assert statuses[task.id] == "success"
        assert statuses["03.bin.lorbin.HMI_2"] == "skipped"
        assert task.id in [call.args[0].id for call in execute.call_args_list]
    finally:
        state.close()

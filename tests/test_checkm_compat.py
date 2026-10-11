from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from metabaw import checkm_compat as compat
from metabaw.direct import DirectBinBuilder, internal, q
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import Executor, RunResult
from metabaw.internal import build_parser as internal_parser
from metabaw.model import Task
from metabaw.resume import command_thread_counts, resume_command
from metabaw.state import StateStore
from metabaw.workflow_report import task_software
from test_parallel_resources import options
from test_resume_resources import _bin_options, _complete_plan_outputs, _seed


@pytest.mark.parametrize("text,expected", [
    ("{'GC': 0.5, 'Mean contig length': 12500.5, 'bins': [1, 2]}",
     {"GC": 0.5, "Mean contig length": 12500.5, "bins": [1, 2]}),
    ("{'GC': np.float64(0.5), 'Mean contig length': numpy.float64(12500.5)}",
     {"GC": 0.5, "Mean contig length": 12500.5}),
    ("{'x': [np.float32(-0.0), np.float16(1.5e-4), np.int64(-3), numpy.uint32(4)]}",
     {"x": [-0.0, 1.5e-4, -3, 4]}),
    ("{np.int32(2): (np.bool_(True), numpy.bool(False)), 'literal': 'np.float64(9)'}",
     {2: (True, False), "literal": "np.float64(9)"}),
    ("  {'x': np.float64(-1.5e+3)}", {"x": -1500.0}),
    ("{'flags': [np.True_, numpy.False_]}", {"flags": [True, False]}),
])
def test_native_statistics_and_numpy_scalar_reprs_remain_literal_only(text, expected):
    assert compat.statistics_literal_eval(text) == expected


@pytest.mark.parametrize("text", [
    "{'x': __import__('os').system('touch unwanted')}",
    "{'x': np.float64(__import__('os').system('touch unwanted'))}",
    "{'x': np.load('somefile')}", "{'x': other.float64(0.5)}",
    "{'x': np.float64(value)}", "{'x': np.float64(1 + 2)}",
    "{'x': np.float64(x=0.5)}", "{'x': np.float64(1, 2)}",
    "{'x': np.float64(np.float64(0.5))}", "{'x': np.float64('nan')}",
    "{'x': np.float64(1e999)}", "{'x': np.float64(True)}",
    "{'x': np.int32(1.5)}", "{'x': np.uint64(-1)}",
    "{'x': np.bool_(1)}", "{'x': np.float128('0.1')}",
    "{'x': np.float64(0.5)",
])
def test_unknown_calls_nonfinite_and_corrupt_statistics_still_fail(text):
    with pytest.raises((ValueError, SyntaxError)):
        compat.statistics_literal_eval(text)


def test_ast_inputs_are_not_modified_and_global_literal_eval_stays_strict():
    tree = ast.parse("{'x': np.float64(0.5)}", mode="eval")
    before = ast.dump(tree)
    assert compat.statistics_literal_eval(tree) == {"x": 0.5}
    assert ast.dump(tree) == before
    with pytest.raises(ValueError):
        ast.literal_eval(tree)
    assert math.copysign(1, compat.statistics_literal_eval("np.float64(-0.0)")) == -1


def test_real_numpy_statistics_round_trip_without_changing_numeric_settings():
    numpy = pytest.importorskip("numpy")
    settings = numpy.get_printoptions()
    data = {"Mean contig length": numpy.mean([2000, 2500, 3001]),
            "# contigs": numpy.int64(3), "flag": numpy.bool_(True)}
    assert compat.statistics_literal_eval(repr(data)) == {
        "Mean contig length": float(data["Mean contig length"]), "# contigs": 3, "flag": True}
    assert numpy.get_printoptions() == settings


def test_standalone_child_reads_real_numpy_records_in_native_runtime(tmp_path, monkeypatch):
    numpy = pytest.importorskip("numpy")
    package = tmp_path / "checkm"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "resultsParser.py").write_text(
        "import ast\n"
        "def read_stats(path):\n"
        "    with open(path) as source:\n"
        "        return {row.split('\\t')[0]: ast.literal_eval(row.split('\\t')[1]) for row in source}\n",
        encoding="utf-8")
    executable = tmp_path / "bin" / "checkm"
    executable.parent.mkdir()
    report = tmp_path / "quality_report.tsv"
    stats = tmp_path / "bin_stats.analyze.tsv"
    native_arguments = ["lineage_wf", "-t", "128", "-x", "fa", "bins", "work", "--tab_table",
                        "-f", str(report), "--tmpdir", str(tmp_path / "ipc")]
    executable.write_text(
        f"#!{shlex.quote(sys.executable)}\n"
        "import sys\nfrom pathlib import Path\nimport numpy as np\n"
        "from checkm.resultsParser import read_stats\n"
        f"assert sys.argv[1:] == {native_arguments!r}\n"
        f"stats = Path({str(stats)!r})\n"
        "stats.write_text('MAG1\\t' + repr({'Mean contig length': np.mean([2000,2500,3001]), "
        "'count': np.int64(3), 'flag': np.bool_(True)}) + '\\n')\n"
        "row = read_stats(stats)['MAG1']\n"
        "assert row == {'Mean contig length': 2500.3333333333335, 'count': 3, 'flag': True}\n"
        f"Path({str(report)!r}).write_text('Bin Id\\tCompleteness\\tContamination\\nMAG1\\t90.0\\t2.0\\n')\n",
        encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    baseline = subprocess.run([sys.executable, str(executable), *native_arguments],
                              capture_output=True, text=True)
    if "np." in repr(numpy.mean([2000, 2500, 3001])):
        assert baseline.returncode != 0 and "malformed node or string" in baseline.stderr
    with patch.object(compat.shutil, "which", return_value=str(executable)):
        compat.run_checkm(shlex.join(["checkm", *native_arguments[:-2]]), tmp_path / "ipc")
    assert report.read_text().splitlines()[-1] == "MAG1\t90.0\t2.0"


def test_launcher_preserves_native_python_arguments_tmpdir_and_failures(tmp_path):
    executable = str(tmp_path / "native-env" / "bin" / "checkm")
    native_python = str(tmp_path / "native-env" / "bin" / "python3.11")
    command = "checkm lineage_wf -t 128 -x fa 'candidate bins' 'quality work' --tab_table -f 'report file.tsv'"
    with patch.object(compat.shutil, "which", return_value=executable), \
            patch("metabaw.metadecoder_coverage._python_launcher", return_value=[native_python]) as launcher, \
            patch.object(compat.subprocess, "run") as run:
        compat.run_checkm(command, tmp_path / "ipc")
    launcher.assert_called_once_with(executable)
    child = run.call_args.args[0]
    assert child[:5] == [native_python, str(Path(compat.__file__).resolve()),
                         "--native-executable", executable, "--"]
    assert child[5:] == shlex.split(command)[1:] + ["--tmpdir", str(tmp_path / "ipc")]
    assert run.call_args.kwargs == {"check": True}
    with patch.object(compat.shutil, "which", return_value=executable), \
            patch("metabaw.metadecoder_coverage._python_launcher", return_value=[native_python]), \
            patch.object(compat.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "checkm")):
        with pytest.raises(subprocess.CalledProcessError):
            compat.run_checkm(command, tmp_path / "ipc")


def test_missing_or_unrecognized_launcher_never_uses_unrelated_main_python(tmp_path):
    with patch.object(compat.shutil, "which", return_value=None):
        with pytest.raises(ValueError, match="not configured"):
            compat.run_checkm("checkm lineage_wf", tmp_path)
    with patch.object(compat.shutil, "which", return_value="/native/bin/checkm"), \
            patch("metabaw.metadecoder_coverage._python_launcher", side_effect=ValueError("unknown shebang")):
        with pytest.raises(ValueError, match="CheckM's Python interpreter"):
            compat.run_checkm("checkm lineage_wf", tmp_path)
    for command in ("other-tool lineage_wf", "checkm lineage_wf --tmpdir other"):
        with patch.object(compat.shutil, "which", return_value="/native/bin/checkm"), \
                patch("metabaw.metadecoder_coverage._python_launcher", return_value=["python"]):
            with pytest.raises(ValueError):
                compat.run_checkm(command, tmp_path)


def test_process_local_parser_patch_preserves_all_native_stats_and_global_ast(tmp_path):
    executable = str(tmp_path / "checkm")
    parser = SimpleNamespace(ast=ast)
    original = ast.literal_eval
    original_argv, original_path = list(sys.argv), list(sys.path)

    def native(path, run_name):
        assert path == executable and run_name == "__main__"
        assert sys.argv == [executable, "lineage_wf", "-t", "128"]
        assert parser.ast.literal_eval("{'Mean contig length': np.float64(2500.5)}") == {
            "Mean contig length": 2500.5}
        assert parser.ast.parse is ast.parse
        assert ast.literal_eval is original

    try:
        with patch.object(compat.importlib, "import_module", return_value=parser), \
                patch.object(compat.runpy, "run_path", side_effect=native), redirect_stdout(io.StringIO()):
            compat.main(["--native-executable", executable, "--", "lineage_wf", "-t", "128"])
    finally:
        sys.argv[:] = original_argv
        sys.path[:] = original_path


def test_unsupported_statistics_parser_is_not_silently_bypassed():
    with patch.object(compat.importlib, "import_module", return_value=SimpleNamespace()), \
            patch.object(compat.runpy, "run_path") as native:
        original_path = list(sys.path)
        try:
            with pytest.raises(ValueError, match="Unsupported CheckM"):
                compat._run_native("/native/bin/checkm", ["lineage_wf"])
        finally:
            sys.path[:] = original_path
        native.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_failed_old_checkm_reruns_qc_and_dependents_without_repeating_binning(tmp_path, legacy):
    contigs, reads = tmp_path / "S1.fa", tmp_path / "S1.fastq"
    contigs.write_text(">contig\nACGT\n", encoding="utf-8")
    reads.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
    sample = ReadSample("S1", reads, contigs=contigs)
    analysis = Analysis("S1", (sample,), (contigs,), False)
    tasks = DirectBinBuilder([analysis], _bin_options(tmp_path, 32, 128), tmp_path).build()
    old = []
    for task in tasks:
        if task.id == "05.qc.checkm":
            tokens = shlex.split(task.command)
            native = tokens[tokens.index("--command") + 1]
            wrapped = " ".join(q(arg) for arg in internal("run-checkm", "--command", native))
            command = task.command.replace(wrapped, native)
            assert command != task.command and "run-checkm" not in command
            task = replace(task, command=command)
        old.append(task)
    _complete_plan_outputs(old)
    affected = {"05.qc.checkm"}
    for task in tasks:
        if affected.intersection(task.deps):
            affected.add(task.id)
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, old, legacy=legacy)
        for task_id in affected:
            state.finish(task_id, "failed" if task_id == "05.qc.checkm" else "blocked", 1,
                         "previous CheckM statistics parsing failure")
        def worker(task, *args):
            if task.id == "05.qc.checkm":
                assert task.cpus == 128 and "run-checkm" in task.command
            return RunResult(task.id, 0, "synthetic repaired QC task")
        with patch("metabaw.executor._execute_task", side_effect=worker) as execute:
            statuses = Executor(state, tmp_path / "logs", max_cpus=128,
                                max_parallel=4, max_gpus=1).run(tasks)
        assert {call.args[0].id for call in execute.call_args_list} == affected
        assert all(value == "skipped" for task_id, value in statuses.items() if task_id not in affected)
        assert all(statuses[task_id] == "success" for task_id in affected)
    finally:
        state.close()


def test_generated_global_checkm_command_keeps_full_budget_and_resource_resume(tmp_path):
    opts = replace(options(tmp_path, "metabat2", gpu=False), quality_control="checkm")
    catalog = (Task("catalog", "05_quality", ("true",), tmp_path), tmp_path / "candidates")
    task, _ = DirectBinBuilder([], opts, tmp_path)._quality(catalog)
    assert task.cpus == 128 and task.sample is None
    assert "run-checkm" in task.command and "checkm lineage_wf" in task.command
    assert '-t 128' in task.command and '--tmpdir "$checkm_ipc_link"' in task.command
    assert "checkm" in task_software(task)[0]
    changed = task.command.replace("-t 128 ", "-t 64 ")
    assert resume_command(task.command) == resume_command(changed)
    assert 128 in command_thread_counts(task.command)
    assert resume_command(task.command) != resume_command(task.command.replace("-x fa", "-x fna"))
    args = internal_parser().parse_args(["run-checkm", "--command", "checkm lineage_wf", "--tmpdir", str(tmp_path)])
    assert args.external_command == "checkm lineage_wf" and args.tmpdir == tmp_path

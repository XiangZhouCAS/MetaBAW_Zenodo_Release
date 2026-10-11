"""Synthetic DAS Tool/MetaWRAP outcomes; no upstream Linux tools are run."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from metabaw.cli import _task_dict
from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import Executor, _fasta_validation_error, _outputs_exist
from metabaw.internal import (
    collect_fasta, normalize_refined_fasta, run_dastool_refinement,
    run_metawrap_refinement, write_metawrap_checkm_quality,
)
from metabaw.model import Task
from metabaw.refinement import (
    EMPTY_REFINEMENT_MARKER, read_empty_refinement, write_empty_refinement,
)
from metabaw.relocation import Location, portable_checkpoint
from metabaw.state import StateStore


DAS_EMPTY = "No bins with bin-score >0.5 found. Adjust score_threshold to report bins with lower quality.\n"
DAS_HEADER = "bin\tbin_set\tbin_score\tSCG_completeness\tSCG_redundancy\n"
DAS_EVAL = DAS_HEADER + "b1\tmetabat2\t0.3\t40\t2\n"
META_HEADER = "bin\tcompleteness\tcontamination\n"


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source_bins"
    source.mkdir()
    (source / "HMI_1_MetaBAT2_1.fa").write_text(">c1\nACGT\n", encoding="utf-8")
    mapping = tmp_path / "binner.tsv"
    mapping.write_text("b1\tc1\tmetabat2\n", encoding="utf-8")
    assembly = tmp_path / "assembly.fa"
    assembly.write_text(">c1\nACGT\n", encoding="utf-8")
    root = tmp_path / "refinement"
    root.mkdir()
    return source, mapping, assembly, root


def das_paths(case):
    source, mapping, assembly, root = case
    prefix = root / "sample"
    bins = root / "sample_DASTool_bins"
    complete = root / "dastool.complete"
    spec = json.dumps({"name": "metabat2", "mapping": str(mapping), "bins": str(source)})
    return [spec], assembly, prefix, bins, complete, 4, ""


def meta_paths(case):
    source, _mapping, _assembly, root = case
    return ([f"metabat2={source}"], "[]", root, root / "metawrap_50_10_bins",
            root / "metawrap_50_10_bins.stats", root / "metawrap.complete", 4, 50, 10, "")


def write_das(prefix, message=DAS_EMPTY, evaluation=DAS_EVAL):
    Path(str(prefix) + "_DASTool.log").write_text(message, encoding="utf-8")
    if evaluation is not None:
        Path(str(prefix) + "_allBins.eval").write_text(evaluation, encoding="utf-8")


@pytest.mark.parametrize("exit_code,evaluation", [(0, DAS_EVAL), (1, DAS_EVAL), (1, DAS_HEADER)])
def test_das_native_no_bin_abort_is_a_valid_empty_result(case, monkeypatch, exit_code, evaluation):
    args = das_paths(case)
    def native(command, check):
        assert not check
        assert "--write_bins" in command and "--write_bin_evals" in command
        assert command[command.index("-t") + 1] == "4"
        write_das(args[2], evaluation=evaluation)
        return subprocess.CompletedProcess(command, exit_code)
    monkeypatch.setattr("metabaw.internal.subprocess.run", native)
    run_dastool_refinement(*args)
    assert read_empty_refinement(args[3])["tool"] == "das_tool"
    assert args[4].is_file()
    assert list(args[3].glob("*.fa")) == []
    assert (case[0] / "HMI_1_MetaBAT2_1.fa").is_file()


@pytest.mark.parametrize("message,evaluation,code", [
    ("Database is missing\n", DAS_EVAL, 1),
    (DAS_EMPTY, None, 1),
    (DAS_EMPTY, "invalid\n", 1),
    (DAS_EMPTY, DAS_HEADER + "b1\ta\t0.5\t50\t0\n", 1),
    (DAS_EMPTY, DAS_HEADER + "b1\ta\tnan\t50\t0\n", 1),
    (DAS_EMPTY, DAS_EVAL, 7),
    (DAS_EMPTY, DAS_EVAL, -9),
    (DAS_EMPTY + "Unexpected failure\n", DAS_EVAL, 1),
    ("Finished\n", DAS_HEADER, 0),
])
def test_das_errors_missing_or_contradictory_evidence_still_fail(case, monkeypatch, message, evaluation, code):
    args = das_paths(case)
    def native(command, check):
        write_das(args[2], message, evaluation)
        return subprocess.CompletedProcess(command, code)
    monkeypatch.setattr("metabaw.internal.subprocess.run", native)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        run_dastool_refinement(*args)
    assert not args[4].exists()


def test_das_stale_evidence_and_selected_exports_are_not_adopted(case, monkeypatch):
    args = das_paths(case)
    write_das(args[2])
    args[3].mkdir()
    write_empty_refinement(args[3], "das_tool")
    args[4].write_text("old\n")
    def native(command, check):
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr("metabaw.internal.subprocess.run", native)
    with pytest.raises(ValueError, match="no FASTA"):
        run_dastool_refinement(*args)
    assert not args[4].exists()
    assert not Path(str(args[2]) + "_allBins.eval").exists()
    def contradictory(command, check):
        write_das(args[2])
        Path(str(args[2]) + "_DASTool_contig2bin.tsv").write_text("c1\tb1\n")
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr("metabaw.internal.subprocess.run", contradictory)
    with pytest.raises(subprocess.CalledProcessError):
        run_dastool_refinement(*args)
    assert not args[4].exists()


@pytest.mark.parametrize("tool", ["das_tool", "metawrap"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_nonempty_native_results_preserve_bins_and_do_not_hide_errors(case, monkeypatch, tool, exit_code):
    if tool == "das_tool":
        args = das_paths(case)
        bins, complete = args[3:5]
        runner = run_dastool_refinement
    else:
        args = meta_paths(case)
        bins, stats, complete = args[3:6]
        runner = run_metawrap_refinement
    def native(command, check):
        bins.mkdir(parents=True)
        (bins / "bin.1.fa").write_text(">c1\nACGT\n")
        if tool == "metawrap":
            stats.write_text(META_HEADER + "bin.1\t80\t2\n")
        if check and exit_code:
            raise subprocess.CalledProcessError(exit_code, command)
        return subprocess.CompletedProcess(command, exit_code)
    monkeypatch.setattr("metabaw.internal.subprocess.run", native)
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            runner(*args)
        assert not complete.exists()
    else:
        runner(*args)
        assert complete.exists()
        assert read_empty_refinement(bins) is None
        normalize_refined_fasta(bins, bins, "refined", [case[0]])
        assert (bins / "HMI_1_MetaBAT2_1.fa").read_text() == ">c1\nACGT\n"
        assert _fasta_validation_error(bins, allow_empty_refinement=True) is None


@pytest.mark.parametrize("mode", ["empty", "missing_dir", "missing_stats", "bad_stats", "nonempty_stats", "native_error"])
def test_metawrap_only_accepts_current_empty_dir_and_valid_empty_stats(case, monkeypatch, mode):
    args = meta_paths(case)
    args[3].mkdir()
    (args[3] / "stale.fa").write_text(">stale\nAAAA\n")
    args[4].write_text(META_HEADER)
    args[5].write_text("old\n")
    def native(command, check):
        assert check
        assert not args[3].exists() and not args[4].exists()
        assert command[command.index("-c") + 1] == "50"
        assert command[command.index("-x") + 1] == "10"
        args[2].mkdir()
        if mode != "missing_dir":
            args[3].mkdir()
        if mode != "missing_stats":
            text = "invalid\n" if mode == "bad_stats" else META_HEADER
            if mode == "nonempty_stats":
                text += "b1\t80\t2\n"
            args[4].write_text(text)
        if mode == "native_error":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr("metabaw.internal.subprocess.run", native)
    if mode == "empty":
        run_metawrap_refinement(*args)
        assert args[5].exists()
        assert read_empty_refinement(args[3])["tool"] == "metawrap"
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            run_metawrap_refinement(*args)
        assert not args[5].exists()
    assert (case[0] / "HMI_1_MetaBAT2_1.fa").exists()


@pytest.mark.parametrize("tool", ["das_tool", "metawrap"])
def test_normalization_empty_proof_and_nonempty_fasta_validation(case, tool):
    source, _mapping, _assembly, root = case
    bins = root / "bins"
    bins.mkdir()
    marker = bins / ".metabaw.naming-v2.complete"
    task = Task("publish", "04_refinement", "true", root,
                outputs=(bins, bins / "manifest.tsv", marker),
                fasta_output_dirs=(bins,), allow_empty_refinement=True)
    with pytest.raises(ValueError, match="No refined FASTA"):
        normalize_refined_fasta(bins, bins, "sample", [source], completion_marker=marker)
    write_empty_refinement(bins, tool)
    normalize_refined_fasta(bins, bins, "sample", [source], completion_marker=marker)
    assert (bins / "manifest.tsv").read_text() == "source\tpublished\tprovenance\n"
    assert _outputs_exist(task)
    # The same marker cannot bypass mandatory binner FASTA validation.
    assert not _outputs_exist(replace(task, allow_empty_refinement=False))
    (bins / "bad.fa").write_text(">c1\n")
    assert "conflicts" in _fasta_validation_error(bins, allow_empty_refinement=True)
    (bins / EMPTY_REFINEMENT_MARKER).unlink()
    assert "no complete" in _fasta_validation_error(bins, allow_empty_refinement=True)
    (bins / "bad.fa").unlink()
    (bins / EMPTY_REFINEMENT_MARKER).write_text('{"bin_count":0}')
    assert "Invalid" in _fasta_validation_error(bins, allow_empty_refinement=True)


@pytest.mark.parametrize("tool", ["das_tool", "metawrap"])
@pytest.mark.parametrize("native_error", [False, True])
def test_strict_workflow_empty_result_resume_and_real_child_exit(case, tool, native_error):
    source, mapping, assembly, root = case
    if tool == "das_tool":
        args = das_paths(case)
        bins, complete = args[3:5]
        native = (
            "from pathlib import Path; "
            f"Path({str(args[2]) + '_DASTool.log'!r}).write_text({DAS_EMPTY!r}); "
            f"Path({str(args[2]) + '_allBins.eval'!r}).write_text({DAS_EVAL!r}); "
            f"raise SystemExit({7 if native_error else 1})"
        )
        call = f"m.run_dastool_refinement({args[0]!r}, Path({str(assembly)!r}), Path({str(args[2])!r}), Path({str(bins)!r}), Path({str(complete)!r}), 4, '')"
        outputs = (complete, bins)
    else:
        args = meta_paths(case)
        bins, stats, complete = args[3:6]
        native = (
            "from pathlib import Path; "
            f"Path({str(bins)!r}).mkdir(parents=True); "
            f"Path({str(stats)!r}).write_text({META_HEADER!r}); "
            f"raise SystemExit({7 if native_error else 0})"
        )
        call = f"m.run_metawrap_refinement({args[0]!r}, '[]', Path({str(root)!r}), Path({str(bins)!r}), Path({str(stats)!r}), Path({str(complete)!r}), 4, 50, 10, '')"
        outputs = (complete, stats, bins)
    wrapper = (
        "from pathlib import Path\nimport sys\nfrom metabaw import internal as m\n"
        "real_run=m.subprocess.run\n"
        f"def native(command, **kwargs):\n return real_run([sys.executable, '-c', {native!r}], **kwargs)\n"
        "m.subprocess.run=native\n" + call
    )
    marker = bins / ".metabaw.naming-v2.complete"
    candidates = root.parent / "candidates"
    helper = (sys.executable, "-B", "-m", "metabaw.internal")
    tasks = [
        Task("refine", "04_refinement", (sys.executable, "-B", "-c", wrapper), root.parent,
             inputs=(source, mapping, assembly), outputs=outputs,
             fasta_output_dirs=(bins,), allow_empty_refinement=True),
        Task("publish", "04_refinement", (*helper, "normalize-refined-fasta",
             "--source-dir", str(bins), "--output-dir", str(bins), "--prefix", "sample",
             "--source", str(source), "--completion-marker", str(marker)), root.parent,
             deps=("refine",), inputs=(source,),
             outputs=(bins, bins / "manifest.tsv", marker),
             fasta_output_dirs=(bins,), allow_empty_refinement=True),
        Task("catalog", "05_quality", (*helper, "collect-fasta", "--source", str(bins),
             "--source", str(source), "--output-dir", str(candidates)), root.parent,
             deps=("publish",), inputs=(bins, source), outputs=(candidates,)),
    ]
    state = StateStore(root.parent / "state.sqlite3")
    try:
        executor = Executor(state, root.parent / "logs", 2, 2, fail_fast=True)
        first = executor.run(tasks)
        if native_error:
            assert first == {"refine": "failed", "publish": "blocked", "catalog": "blocked"}
            assert not complete.exists()
        else:
            assert set(first.values()) == {"success"}
            assert set(executor.run(tasks).values()) == {"skipped"}
            assert list(bins.glob("*.fa")) == []
            assert [p.name for p in candidates.glob("*.fa")] == ["HMI_1_MetaBAT2_1.fa"]
    finally:
        state.close()


def test_empty_samples_mix_with_nonempty_metawrap_quality_and_all_empty_stops(case):
    source, _mapping, _assembly, root = case
    empty = root / "empty"
    write_empty_refinement(empty, "metawrap")
    normalize_refined_fasta(empty, empty, "empty", [])
    empty_stats = root / "empty.stats"
    empty_stats.write_text(META_HEADER)
    full = root / "full"
    shutil.copytree(source, full)
    normalize_refined_fasta(full, full, "full", [source])
    full_stats = root / "full.stats"
    full_stats.write_text(META_HEADER + "HMI_1_MetaBAT2_1\t80\t2\n")
    candidates = root / "candidates"
    collect_fasta([empty, full], candidates)
    quality = root / "quality.tsv"
    write_metawrap_checkm_quality(candidates, [empty_stats, full_stats],
                                  [empty / "manifest.tsv", full / "manifest.tsv"], quality)
    assert quality.read_text().splitlines()[1] == "HMI_1_MetaBAT2_1\t80.0\t2.0"
    with pytest.raises(ValueError, match="No refined FASTA"):
        collect_fasta([empty], root / "all_empty")


@pytest.mark.parametrize("tool", ["das_tool", "metawrap"])
@pytest.mark.parametrize("policy", ["strict", "best-effort"])
def test_task_graph_only_refinement_allows_certified_empty_output(case, tool, policy):
    source, mapping, assembly, root = case
    options = BinOptions(
        outdir=root / "results", workdir=root / "results" / "tmp" / "work",
        threads=4, read_type="short", align_tool="bowtie2", flye_read_type="--nano-raw",
        binners=("metabat2",), min_contig_length=1500, min_fasta_kbs=200, batch_size=1024,
        refiner=tool, quality_control="checkm2", min_completeness=50, max_contamination=10,
        min_quality_score=None, run_gunc=False, run_trna=False, trna_pass=None,
        run_rrna=False, rrna_pass=False, dereplicator="galah", environment=None,
        tag_contigs=False, gpu=False, max_gpu_memory="4G", assembly_strategy="default",
        magscot_dir=None, checkm2_db=None, gunc_db=None, gtdbtk_data=None, extra_args={},
        failure_policy=policy,
    )
    analysis = Analysis("HMI_1", (ReadSample("HMI_1", root / "R1", root / "R2", assembly),), (assembly,), False)
    builder = DirectBinBuilder([], options, root)
    prepare = Task("prepare", "01_prepare", "true", root)
    binner = Task("binner", "03_binning", "true", root)
    result, _bins = builder._refine(analysis, prepare, assembly, [("metabat2", binner, mapping, source)])
    run = next(t for t in builder.tasks if t.id.startswith("04.refine."))
    assert run.allow_failed_deps == (policy == "best-effort")
    assert run.allow_empty_refinement and result.allow_empty_refinement
    assert _bins in run.outputs
    assert _bins not in result.inputs
    assert result.deps == (run.id,) and not result.allow_failed_deps
    assert _task_dict(run)["allow_empty_refinement"] is True


def test_empty_policy_is_preserved_in_portable_resume_keys(tmp_path):
    task = Task("refine", "04_refinement", "true", tmp_path,
                fasta_output_dirs=(tmp_path / "bins",), allow_empty_refinement=True)
    old = replace(task, allow_empty_refinement=False)
    location = Location(tmp_path, tmp_path / "tmp")
    assert task.resume_fingerprint([]) != old.resume_fingerprint([])
    assert portable_checkpoint(task, frozenset(), [], location) != portable_checkpoint(old, frozenset(), [], location)


@pytest.mark.parametrize("tool", ["das_tool", "metawrap"])
def test_copied_empty_completion_can_be_adopted_without_native_rerun(tmp_path, tool):
    original = tmp_path / "original"
    bins = original / "bins"
    write_empty_refinement(bins, tool)
    marker = bins / ".metabaw.naming-v2.complete"
    normalize_refined_fasta(bins, bins, "sample", [], completion_marker=marker)
    copied = tmp_path / "renamed"
    shutil.copytree(original, copied)
    copied_bins = copied / "bins"
    task = Task("publish", "04_refinement",
                (sys.executable, "-c", "raise AssertionError('must reuse copied result')"), copied,
                outputs=(copied_bins, copied_bins / "manifest.tsv", copied_bins / marker.name),
                fasta_output_dirs=(copied_bins,), allow_empty_refinement=True)
    state = StateStore(copied / "state.sqlite3")
    try:
        statuses = Executor(state, copied / "logs", 1, 1).run([task])
        assert statuses == {"publish": "skipped"}
    finally:
        state.close()

"""Refinement command/quality regressions; no upstream binners are launched."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import shlex
from unittest.mock import patch

import pytest

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.internal import main, run_metawrap_refinement


BRANCHES = (
    ("magscot", "checkm", True),
    ("magscot", "checkm2", True),
    ("metawrap", "checkm", False),
    ("metawrap", "checkm2", True),
    ("das_tool", "checkm", False),
    ("das_tool", "checkm2", False),
)


def options_for(root: Path, refiner: str, quality: str, **overrides) -> BinOptions:
    options = BinOptions(
        outdir=root / "results", workdir=root / "results" / "tmp" / "work",
        threads=8, read_type="short", align_tool="bowtie2", flye_read_type="--nano-raw",
        binners=("metabat2", "vamb"), min_contig_length=1500, min_fasta_kbs=200,
        batch_size=1024, refiner=refiner, quality_control=quality,
        min_completeness=50.0, max_contamination=10.0, min_quality_score=27.0,
        run_gunc=False, run_trna=False, trna_pass=None, run_rrna=False, rrna_pass=False,
        dereplicator="galah", environment=None, tag_contigs=False, gpu=False,
        max_gpu_memory="4G", assembly_strategy="existing-contigs",
        magscot_dir=root / "MAGScoT", checkm2_db=None, gunc_db=None, gtdbtk_data=None,
        extra_args={},
    )
    return replace(options, **overrides)


def graph(root: Path, options: BinOptions):
    contigs = root / "contigs.fa"
    sample = ReadSample("HMI_1", root / "R1.fastq", root / "R2.fastq", contigs)
    analysis = Analysis("HMI_1", (sample,), (contigs,), False)
    return {t.id: t for t in DirectBinBuilder([analysis], options, root).build()}


def value(command, flag):
    tokens = shlex.split(command) if isinstance(command, str) else list(command)
    return tokens[tokens.index(flag) + 1]


@pytest.mark.parametrize("refiner,quality,_relaxed", BRANCHES)
def test_all_refiners_use_explicit_export_task_name(tmp_path, refiner, quality, _relaxed):
    from metabaw.executor import _progress_group_label

    tasks = graph(tmp_path, options_for(tmp_path, refiner, quality))
    task = tasks["04.export_refined_bins.HMI_1"]
    assert task.deps == (f"04.refine.{refiner}.HMI_1",)
    assert _progress_group_label(task, ["HMI_1"]) == "04.export_refined_bins"
    assert "04.export_refined_bins.HMI_1" in tasks["05.catalog.candidates"].deps
    assert not any(name.startswith("04.bins.") for name in tasks)
    assert not any(dep.startswith("04.bins.") for item in tasks.values() for dep in item.deps)


@pytest.mark.parametrize("refiner,quality,relaxed", BRANCHES)
@pytest.mark.parametrize("com,con", [(50, 10), (75, 5), (92.5, 7.25), (100, 100), (0, 0)])
def test_relative_refinement_thresholds_keep_final_qc_unchanged(tmp_path, refiner, quality, relaxed, com, con):
    options = options_for(tmp_path, refiner, quality, min_completeness=com, max_contamination=con)
    tasks = graph(tmp_path, options)
    expected_com = com * 0.8 if relaxed else com
    expected_con = min(100, con * 1.2) if relaxed else con
    assert options.refinement_thresholds_relaxed is relaxed
    assert options.refinement_thresholds == pytest.approx((expected_com, expected_con))
    refine = tasks[f"04.refine.{refiner}.HMI_1"]
    if refiner == "magscot":
        native = json.loads(value(refine.command, "--native-command-json"))
        assert float(value(native, "--threshold")) == pytest.approx(expected_com / 100)
        assert float(value(native, "--max_cont")) == pytest.approx(expected_con / 100)
        assert "--min-quality-score" not in native
    elif refiner == "metawrap":
        assert float(value(refine.command, "--min-completeness")) == pytest.approx(expected_com)
        assert float(value(refine.command, "--max-contamination")) == pytest.approx(expected_con)
        bins = Path(value(refine.command, "--output-bins"))
        stats = Path(value(refine.command, "--stats"))
        assert bins.name == f"metawrap_{expected_com:g}_{expected_con:g}_bins"
        assert stats.name == bins.name + ".stats"
        assert bins in refine.outputs and stats in refine.outputs
        publish = tasks["04.export_refined_bins.HMI_1"]
        assert Path(value(publish.command, "--source-dir")) == bins
        assert Path(value(publish.command, "--output-dir")) == bins
        assert "--min-quality-score" not in refine.command
        if quality == "checkm":
            qc = tasks["05.qc.checkm"]
            assert "metawrap-checkm-quality" in qc.command
            assert str(stats) in qc.command and stats in qc.inputs
            assert bins / "manifest.tsv" in qc.inputs
        else:
            assert "checkm2 predict" in tasks["05.qc.checkm2"].command
    else:
        assert "--min-completeness" not in refine.command
        assert "--max-contamination" not in refine.command
    final_filter = tasks["05.qc.filter"]
    assert float(value(final_filter.command, "--min-completeness")) == com
    assert float(value(final_filter.command, "--max-contamination")) == con
    assert float(value(final_filter.command, "--min-quality-score")) == 27
    assert options.min_completeness == com and options.max_contamination == con


@pytest.mark.parametrize("refiner,quality,relaxed", BRANCHES)
@pytest.mark.parametrize("dereplicator", ["galah", "drep"])
def test_dereplication_uses_original_thresholds(tmp_path, refiner, quality, relaxed, dereplicator):
    options = options_for(tmp_path, refiner, quality, dereplicator=dereplicator)
    command = graph(tmp_path, options)[f"06.dereplicate.{dereplicator}"].command
    com_flag, con_flag = ("--min-completeness", "--max-contamination") if dereplicator == "galah" else ("-comp", "-con")
    assert float(value(command, com_flag)) == 50
    assert float(value(command, con_flag)) == 10


@pytest.mark.parametrize("refiner,quality,relaxed", BRANCHES)
def test_policy_change_only_invalidates_affected_refinement_not_upstream(tmp_path, refiner, quality, relaxed):
    options = options_for(tmp_path, refiner, quality)
    updated = graph(tmp_path, options)
    with patch.object(BinOptions, "refinement_thresholds", property(
        lambda self: (self.min_completeness, self.max_contamination)
    )):
        original = graph(tmp_path, options)
    assert original.keys() == updated.keys()
    for name, task in updated.items():
        if task.stage in {"01_prepare", "02_mapping", "03_binning"} or name.startswith(
            ("04.combine.", "04.prodigal.", "04.hmm.", "04.markers.")
        ):
            assert task.resume_fingerprint([]) == original[name].resume_fingerprint([])
    name = f"04.refine.{refiner}.HMI_1"
    assert (updated[name].resume_fingerprint([]) != original[name].resume_fingerprint([])) is relaxed


@pytest.mark.parametrize("refiner,quality", [(r, q) for r, q, relaxed in BRANCHES if relaxed])
def test_generated_final_filter_rejects_refinement_only_mags_and_keeps_boundary(tmp_path, refiner, quality):
    options = options_for(tmp_path, refiner, quality, min_quality_score=None)
    tasks = graph(tmp_path, options)
    command = tasks["05.qc.filter"].command
    bins = Path(value(command, "--bins-dir"))
    bins.mkdir(parents=True)
    scores = {"low_com": (45, 2), "high_con": (60, 12), "boundary": (50, 10), "good": (80, 5)}
    for name in scores:
        (bins / f"{name}.fa").write_text(">contig\nAAAA\n")
    report = Path(value(command, "--checkm2"))
    report.parent.mkdir(parents=True)
    report.write_text("Name\tCompleteness\tContamination\n" + "".join(
        f"{name}\t{com}\t{con}\n" for name, (com, con) in scores.items()
    ))
    # Run the actual generated MetaBAW filter, not a mocked quality decision.
    main(list(command[3:]))
    retained = Path(value(command, "--output-dir"))
    assert {p.stem for p in retained.glob("*.fa")} == {"boundary", "good"}
    with Path(value(command, "--summary")).open(newline="") as handle:
        rows = {row["bin"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert "completeness" in rows["low_com"]["reason"]
    assert "contamination" in rows["high_con"]["reason"]


@pytest.mark.parametrize("quality,expected_com,expected_con", [("checkm", "50", "10"), ("checkm2", "40", "12")])
def test_managed_metawrap_passes_thresholds_once_and_finds_matching_native_paths(tmp_path, quality, expected_com, expected_con):
    options = options_for(tmp_path, "metawrap", quality)
    refine = graph(tmp_path, options)["04.refine.metawrap.HMI_1"]
    source = tmp_path / "binner"
    source.mkdir()
    (source / "input.fa").write_text(">c\nAAAA\n")
    output_root = Path(value(refine.command, "--output-root"))
    bins = Path(value(refine.command, "--output-bins"))
    stats = Path(value(refine.command, "--stats"))
    marker = Path(value(refine.command, "--completion-marker"))

    def native(command, **kwargs):
        assert value(command, "-c") == expected_com
        assert value(command, "-x") == expected_con
        assert kwargs["check"] is True
        # Synthetic upstream output, named from the actual native arguments.
        native_bins = output_root / f"metawrap_{value(command, '-c')}_{value(command, '-x')}_bins"
        native_bins.mkdir(parents=True)
        (native_bins / "bin.1.fa").write_text(">c\nAAAA\n")
        (output_root / (native_bins.name + ".stats")).write_text("Name\tCompleteness\tContamination\nbin.1\t80\t2\n")

    with patch("metabaw.internal.subprocess.run", side_effect=native):
        run_metawrap_refinement(
            [f"metabat2={source}"], "[]", output_root, bins, stats, marker, 8,
            float(value(refine.command, "--min-completeness")),
            float(value(refine.command, "--max-contamination")), "",
        )
    assert marker.is_file() and stats.is_file() and (bins / "bin.1.fa").is_file()

from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch

import pytest

from metabaw.cli import build_parser, command_annotation
from metabaw.internal import classify_niche_literature, merge_coverm_taxonomy
from metabaw.internal import main as internal_main

METRICS = ("relative_abundance", "rpkm", "tpm", "mean")


def plan(tmp_path, *options):
    genomes, reads = tmp_path / "genomes.txt", tmp_path / "reads.tsv"
    genome_rows, read_rows = [], []
    for i in range(4):
        genome, read = tmp_path / f"M{i}.fa", tmp_path / f"S{i}.fastq"
        genome.write_text(">c\nACGT\n")
        read.write_text("@r\nACGT\n+\nIIII\n")
        genome_rows.append(str(genome))
        read_rows.append(f"S{i}\t{read}")
    genomes.write_text("\n".join(genome_rows) + "\n")
    reads.write_text("\n".join(read_rows) + "\n")
    args = build_parser().parse_args([
        "annotation", "--input_reads_files", str(reads), "--input_genome_files", str(genomes),
        "-o", str(tmp_path / "result"), *options,
    ])
    with patch("metabaw.cli._preflight_annotation"), patch("metabaw.cli._run_direct", return_value=0) as run, redirect_stdout(io.StringIO()):
        command_annotation(args)
    return args, run.call_args.args[1], run.call_args.args[4]


@pytest.mark.parametrize("criterion", ["cv", "occupancy"])
@pytest.mark.parametrize("metric", METRICS)
def test_cli_routes_every_criterion_and_abundance(tmp_path, criterion, metric):
    args, tasks, payload = plan(tmp_path, "--niche_classify_method", criterion,
                                "--niche_abundance", metric, "--methods", metric)
    assert args.niche_classify_method == criterion
    assert payload["coverm_methods"] == [metric]
    assert payload["niche_classify_method"] == criterion
    assert payload["niche_abundance"] == metric
    assert payload["niche"]["classify_method"] == criterion
    assert payload["niche"]["abundance"] == metric
    by_id = {task.id: task for task in tasks}
    niche = by_id["04.annotation.niche"].command
    assert niche[niche.index("--abundance-method") + 1] == metric
    assert niche[niche.index("--method") + 1] == criterion
    mode = "raw_cv" if criterion == "cv" else "occupancy"
    marker = tmp_path / "result" / "niche" / f"niche_classification.v3.{mode}.complete"
    assert niche[niche.index("--completion-marker") + 1] == str(marker)
    assert marker in by_id["04.annotation.niche"].outputs
    assert "--cv-correct" not in niche
    assert "--niche-output" in by_id["03.annotation.merge"].command
    support_tasks = [t for t in tasks if t.id.startswith("02.annotation.coverm.") and " count " in t.command]
    assert len(support_tasks) == 4
    assert all("niche_support" in str(t.outputs[0]) for t in support_tasks)


@pytest.mark.parametrize("removed", ["count", "length", "trimmed_mean", "variance", "covered_fraction", "covered_bases", "reads_per_base"])
def test_only_four_public_abundance_methods(tmp_path, removed):
    with pytest.raises(ValueError, match="accepts only"):
        plan(tmp_path, "--methods", removed, "--no-niche")


def test_defaults_alias_and_no_niche_support(tmp_path):
    args, tasks, payload = plan(tmp_path)
    assert tuple(args.methods) == METRICS
    assert payload["niche_classify_method"] == "cv"
    assert payload["niche_abundance"] == "relative_abundance"
    assert args.cv_correct is False and payload["cv_correct"] is False
    _, _, payload = plan(tmp_path, "--niche-method", "prevalence")
    assert payload["niche_classify_method"] == "occupancy"
    _, tasks, _ = plan(tmp_path, "--no-niche", "--methods", "mean")
    assert "04.annotation.niche" not in {t.id for t in tasks}
    assert not any(" count " in t.display_command() for t in tasks)
    with pytest.raises(ValueError, match="matching --niche_abundance"):
        plan(tmp_path, "--niche_abundance", "tpm", "--methods", "mean")


@pytest.mark.parametrize("metric", METRICS)
def test_cv_correct_is_passed_to_task_and_recorded_in_manifest(tmp_path, metric):
    args, tasks, payload = plan(tmp_path, "--cv-correct", "--niche_abundance", metric)
    assert args.cv_correct is True
    assert payload["cv_correct"] is True and payload["niche"]["cv_correct"] is True
    assert "cv_correct=True" in payload["niche"]["message"]
    task = next(t for t in tasks if t.id == "04.annotation.niche")
    assert "--cv-correct" in task.command
    marker = tmp_path / "result" / "niche" / "niche_classification.v3.corrected_cv.complete"
    assert marker in task.outputs
    assert "corrected-cv" in task.description


def test_cv_correct_is_invalid_with_occupancy(tmp_path):
    with pytest.raises(ValueError, match="--cv-correct requires"):
        plan(tmp_path, "--cv-correct", "--niche_classify_method", "occupancy")


def inputs(tmp_path):
    # Same four genomes; metric profiles differ intentionally, with known CVs.
    metrics = {
        "relative_abundance": [[1, 1, 1, 1], [4, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        "rpkm": [[1, 1, 1, 1], [1, 1, 1, 1], [9, 0, 0, 0], [0, 0, 0, 0]],
        "tpm": [[2, 2, 2, 2], [100, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]],
        "mean": [[10, 10, 10, 10], [1, 1, 1, 1], [40, 0, 0, 0], [0, 0, 0, 0]],
    }
    matrix = tmp_path / "matrix.tsv"
    with matrix.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Genome"] + [f"S{s}|{metric}" for s in range(4) for metric in (*METRICS, "count")])
        for genome in range(4):
            w.writerow([f"M{genome}"] + [value for s in range(4) for value in
                                       [*(metrics[m][genome][s] for m in METRICS), 100 if genome < 3 else 0]])
    return matrix, metrics


def classify(tmp_path, matrix, criterion, metric):
    dest = tmp_path / (criterion + "_" + metric)
    dest.mkdir(exist_ok=True)
    classify_niche_literature(
        matrix, tmp_path, dest / "niche.tsv", dest / "abundance.tsv",
        dest / "assignments.tsv", dest / "provenance.json",
        [f"S{i}=default" for i in range(4)], "strain", criterion, .01, 20, .2, .8, 1,
        abundance_method=metric,
    )
    with (dest / "niche.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows, json.loads((dest / "provenance.json").read_text())


@pytest.mark.parametrize("criterion", ["cv", "occupancy"])
@pytest.mark.parametrize("metric", METRICS)
def test_real_niche_calculation_uses_selected_matrix(tmp_path, criterion, metric):
    matrix, metrics = inputs(tmp_path)
    rows, provenance = classify(tmp_path, matrix, criterion, metric)
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert row["method"] == criterion and row["abundance_metric"] == metric
        assert provenance["selected_criterion"] == criterion and provenance["abundance_metric"] == metric
        if i == 3:
            assert row["niche"] == "unclassified_low_support"
        else:
            values = metrics[metric][i]
            cv = statistics.stdev(values) / statistics.fmean(values)
            assert float(row["raw_si_cv"]) == pytest.approx(cv)
            assert float(row["corrected_si"]) == pytest.approx(cv - math.sqrt(4 / 400))
            if criterion == "occupancy":
                occupancy = sum(v > 0 for v in values) / 4
                assert float(row["overall_occupancy"]) == occupancy
                assert row["niche"] == ("generalist" if occupancy >= .8 else "specialist")
            else:
                # Only three supported taxa: floor(3/4)=0 per tail, so all
                # supported taxa remain intermediate rather than forcing a tail.
                assert row["niche"] == "intermediate"
    if metric == "relative_abundance":
        assert "without renormalization" in provenance["parameters"]["detection_scale"]
    else:
        assert "sum selected MAG abundance" in provenance["parameters"]["detection_scale"]


def test_changing_abundance_changes_cv_and_occupancy_results(tmp_path):
    matrix, _ = inputs(tmp_path)
    for method in ("cv", "occupancy"):
        relative, _ = classify(tmp_path, matrix, method, "relative_abundance")
        rpkm, _ = classify(tmp_path, matrix, method, "rpkm")
        if method == "cv":
            assert float(relative[1]["raw_si_cv"]) > float(rpkm[1]["raw_si_cv"])
            assert relative[1]["niche"] == rpkm[1]["niche"] == "intermediate"
        else:
            assert relative[1]["niche"] == "specialist"
            assert rpkm[1]["niche"] == "generalist"


def test_zero_support_is_reported_not_a_failed_workflow(tmp_path):
    matrix, _ = inputs(tmp_path)
    text = matrix.read_text().splitlines()
    for i in range(1, len(text)):
        fields = text[i].split("\t")
        text[i] = fields[0] + "\t" + "\t".join("0" for _ in fields[1:])
    matrix.write_text("\n".join(text) + "\n")
    for method in ("cv", "occupancy"):
        rows, provenance = classify(tmp_path, matrix, method, "relative_abundance")
        assert all(row["niche"] == "unclassified_low_support" for row in rows)
        assert provenance["parameters"]["community_mean_corrected_si"] is None


def test_merge_hides_support_counts_from_public_abundance(tmp_path):
    raw = tmp_path / "raw.tsv"
    raw.write_text("Genome\tRelative Abundance (%)\tTPM\tRead Count\nM0\t1\t1000000\t100\n")
    public, private = tmp_path / "public", tmp_path / "private" / "support.tsv"
    (tmp_path / "taxonomy").mkdir()
    (tmp_path / "taxonomy" / "gtdbtk.bac120.summary.tsv").write_text(
        "user_genome\tclassification\nM0\td__Bacteria;p__P;c__C;o__O;f__F;g__G;s__S\n"
    )
    merge_coverm_taxonomy([f"S0={raw}"], tmp_path / "taxonomy", public, ".tsv", private)
    assert not (public / "coverm_counts.tsv").exists()
    assert "|count" not in (public / "coverm_all_metrics.tsv").read_text()
    assert "|count" in private.read_text()
    manifest = json.loads((public / "manifest.json").read_text())
    assert "count" not in manifest
    assert "relative_abundance" in manifest and "tpm" in manifest


def test_changed_niche_settings_cannot_reuse_old_results(tmp_path):
    _, _, payload = plan(tmp_path)
    output = tmp_path / "result"
    (output / "niche").mkdir(parents=True)
    (output / "run_manifest.json").write_text(json.dumps(payload))
    plan(tmp_path)  # unchanged settings remain resumable
    for options in (("--niche_classify_method", "occupancy"),
                    ("--niche_abundance", "tpm"), ("--no-niche",), ("--cv-correct",)):
        with pytest.raises(ValueError, match="use a new -o"):
            plan(tmp_path, *options)


def test_legacy_manifest_defaults_to_uncorrected_cv_on_resume(tmp_path):
    _, _, payload = plan(tmp_path)
    payload.pop("cv_correct")
    payload["niche"].pop("cv_correct")
    output = tmp_path / "result"
    (output / "niche").mkdir(parents=True)
    (output / "run_manifest.json").write_text(json.dumps(payload))
    _, _, current = plan(tmp_path)
    assert current["cv_correct"] is False
    with pytest.raises(ValueError, match="use a new -o"):
        plan(tmp_path, "--cv-correct")


@pytest.mark.parametrize("correct", [False, True])
def test_internal_cli_routes_cv_correction_and_removes_group_columns(tmp_path, correct):
    matrix, _ = inputs(tmp_path)
    command = [
        "classify-niche-literature", "--abundance", str(matrix),
        "--taxonomy-dir", str(tmp_path), "--output", str(tmp_path / "niche.tsv"),
        "--aggregated-output", str(tmp_path / "aggregated.tsv"),
        "--assignments-output", str(tmp_path / "assignments.tsv"),
        "--provenance-output", str(tmp_path / "provenance.json"),
        "--rank", "strain", "--method", "cv", "--detection-percent", "0.01",
        "--min-total-reads", "20", "--min-prevalence", "0.2",
        "--core-prevalence", "0.8", "--min-core-datasets", "1",
    ]
    # Multiple group labels verify that all prevalence::* fields are removed,
    # not just the automatically generated prevalence::default field.
    for i in range(4):
        command.extend(("--sample-dataset", f"S{i}={'default' if i < 2 else 'other'}"))
    if correct:
        command.append("--cv-correct")
    internal_main(command)
    with (tmp_path / "niche.tsv").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    assert "core_datasets" not in fields and "required_core_datasets" not in fields
    assert not any(field.startswith("prevalence::") for field in fields)
    assert all(row["correct_cv_classification"] == ("TRUE" if correct else "FALSE") for row in rows)


def test_occupancy_detection_respects_percentage_units_and_ignores_unmapped(tmp_path):
    matrix, _ = inputs(tmp_path)
    with matrix.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = list(rows[0])
    for row in rows:
        for field in fields[1:]:
            row[field] = "100" if field.endswith("|count") else "0.001"
    rows.append({field: "unmapped" if field == "Genome" else "NA" for field in fields})
    with matrix.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    relative, _ = classify(tmp_path, matrix, "occupancy", "relative_abundance")
    assert all(float(r["overall_occupancy"]) == 0 for r in relative)
    assert all(r["niche"] == "unclassified_low_support" for r in relative)
    for metric in ("rpkm", "tpm", "mean"):
        normalized, _ = classify(tmp_path, matrix, "occupancy", metric)
        assert len(normalized) == 4
        assert all(float(r["overall_occupancy"]) == 1 for r in normalized)
        assert all(r["niche"] == "generalist" for r in normalized)

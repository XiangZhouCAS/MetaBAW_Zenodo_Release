from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import Executor
from metabaw.internal import (
    collect_fasta, finalize_magscot_mapping, materialize_refined,
    run_magscot_refinement,
)
from metabaw.model import Task
from metabaw.state import StateStore


NO_BINS = "No bins with score >= 0.5 were found in the dataset.\n"
SCORES = "bin\tCompleteness\tContamination\tscore\nbin.1\t0.3\t0.1\t0.25\n"


def _native_fixture(root: Path, message: str = NO_BINS, exit_code: int = 0) -> str:
    """A synthetic native process; it does not run R or upstream MAGScoT."""
    return json.dumps([
        sys.executable,
        "-c",
        "from pathlib import Path; "
        f"Path({str(root / 'sample.scores.out')!r}).write_text({SCORES!r}); "
        f"print({message!r}, end='', flush=True); raise SystemExit({exit_code})",
    ])


class MAGScoTCoordinationTests(unittest.TestCase):
    def test_single_available_binner_is_preserved_after_score_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.1\tcontig_2\tmetabat2\n",
                encoding="utf-8",
            )
            mapping = root / "sample.refined.contig_to_bin.out"

            finalize_magscot_mapping(combined, mapping)

            with mapping.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(
                rows,
                [
                    ["binnew", "contig"],
                    ["bin.1", "contig_1"],
                    ["bin.1", "contig_2"],
                ],
            )

    def test_multiple_binners_without_native_no_bin_evidence_still_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.2\tcontig_2\tvamb\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "without.*mapping",
            ):
                finalize_magscot_mapping(combined, root / "missing.tsv")

    def test_native_no_bins_creates_empty_mapping_not_original_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\nbin.2\tcontig_1\tvamb\n",
                encoding="utf-8",
            )
            mapping = root / "sample.refined.contig_to_bin.out"
            log = root / "magscot.native.log"
            log.write_text(NO_BINS, encoding="utf-8")
            (root / "sample.scores.out").write_text(SCORES, encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                finalize_magscot_mapping(combined, mapping, run_log=log)

            self.assertEqual(mapping.read_text(encoding="utf-8"), "binnew\tcontig\n")
            self.assertIn("0 refined bins", output.getvalue())
            self.assertIn("metabat2", combined.read_text(encoding="utf-8"))
            self.assertIn("vamb", combined.read_text(encoding="utf-8"))

    def test_native_empty_result_requires_a_current_score_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc2\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            log = root / "native.log"
            log.write_text(NO_BINS)
            for scores in (None, "not a score table\n"):
                with self.subTest(scores=scores):
                    if scores is not None:
                        (root / "sample.scores.out").write_text(scores)
                    with self.assertRaisesRegex(ValueError, "score table"):
                        finalize_magscot_mapping(combined, mapping, run_log=log)
                    self.assertFalse(mapping.exists())

    def test_header_only_output_without_native_evidence_is_not_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc2\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            mapping.write_text("binnew\tcontig\n")
            with self.assertRaisesRegex(ValueError, "without.*mapping"):
                finalize_magscot_mapping(combined, mapping)

    def test_empty_refinement_clears_stale_bins_and_other_samples_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assembly = root / "contigs.fa"
            assembly.write_text(">contig_1\nAAAA\n")
            mapping = root / "empty.tsv"
            mapping.write_text("binnew\tcontig\n")
            bins = root / "empty_bins"
            bins.mkdir()
            (bins / "stale.fa").write_text(">contig_1\nAAAA\n")
            marker = bins / ".metabaw.naming-v2.complete"
            marker.write_text("stale\n")
            sources = root / "original_bins"
            sources.mkdir()
            original = sources / "LMI_6_vamb_1.fa"
            original.write_text(">contig_1\nAAAA\n")

            materialize_refined(
                assembly, mapping, bins, "LMI_6_magscot", [sources],
                completion_marker=marker,
            )

            self.assertEqual(list(bins.glob("*.fa")), [])
            self.assertEqual((bins / "manifest.tsv").read_text(), "bin\tfile\tcontigs\n")
            self.assertEqual(marker.read_text(), "complete\n")
            self.assertTrue(original.is_file())
            candidates = root / "candidates"
            collect_fasta([bins, sources], candidates)
            self.assertEqual([path.name for path in candidates.glob("*.fa")], [original.name])

    def test_all_empty_samples_do_not_enter_quality_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            (empty / "manifest.tsv").write_text("bin\tfile\tcontigs\n")
            with self.assertRaisesRegex(ValueError, "No refined FASTA bins"):
                collect_fasta([empty], root / "candidates")

    def test_malformed_mapping_is_not_accepted_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assembly = root / "assembly.fa"
            assembly.write_text(">c1\nAAAA\n")
            mapping = root / "mapping.tsv"
            for content in ("", "garbage\n", "binnew\tcontig\nb1\t\n", "binnew\tcontig\nb1\n"):
                with self.subTest(content=content):
                    mapping.write_text(content)
                    marker = root / "bins" / "naming.complete"
                    marker.parent.mkdir(exist_ok=True)
                    marker.write_text("stale\n")
                    with self.assertRaises(ValueError):
                        materialize_refined(
                            assembly, mapping, marker.parent, "sample_magscot", [],
                            completion_marker=marker,
                        )
                    self.assertFalse(marker.exists())

    def test_managed_native_no_bins_removes_stale_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc2\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            mapping.write_text("binnew\tcontig\nstale\tc1\n")
            stats = root / "sample.refined.out"
            stats.write_text("old statistics\n")
            scores = root / "sample.scores.out"
            scores.write_text("old scores\n")
            log = root / "native.log"
            log.write_text("old output\n")

            run_magscot_refinement(combined, mapping, log, _native_fixture(root))

            self.assertEqual(mapping.read_text(), "binnew\tcontig\n")
            self.assertEqual(log.read_text(), NO_BINS)
            self.assertEqual(scores.read_text(), SCORES)
            self.assertFalse(stats.exists())

    def test_native_error_is_not_swallowed_even_if_it_prints_no_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc2\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_magscot_refinement(
                    combined, mapping, root / "native.log",
                    _native_fixture(root, exit_code=7),
                )
            self.assertEqual(error.exception.returncode, 7)
            self.assertFalse(mapping.exists())

    def test_zero_exit_input_abort_and_missing_output_remain_failures(self) -> None:
        for binners, message in (
            ("b1\tc1\tmetabat2\nb2\tc2\tvamb\n", "Aborting execution!\n"),
            ("b1\tc1\tmetabat2\n", "Scoring bins only.\nAborting execution!\n"),
        ):
            with self.subTest(binners=binners):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    combined = root / "all_binners.tsv"
                    combined.write_text(binners)
                    mapping = root / "sample.refined.contig_to_bin.out"
                    (root / "sample.scores.out").write_text(SCORES)
                    native = json.dumps([sys.executable, "-c", f"print({message!r}, end='')"])
                    with self.assertRaisesRegex(ValueError, "mapping|score table"):
                        run_magscot_refinement(combined, mapping, root / "native.log", native)
                    self.assertFalse(mapping.exists())
                    self.assertFalse((root / "sample.scores.out").exists())

    def test_managed_single_binner_score_only_retains_original_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb1\tc2\tmetabat2\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            run_magscot_refinement(
                combined, mapping, root / "native.log",
                _native_fixture(root, "Scoring bins only.\n"),
            )
            self.assertEqual(mapping.read_text(), "binnew\tcontig\nb1\tc1\nb1\tc2\n")

    def test_strict_workflow_continues_other_samples_and_reuses_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc1\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            assembly = root / "assembly.fa"
            assembly.write_text(">c1\nAAAA\n")
            empty_bins = root / "LMI_6_bins"
            marker = empty_bins / "naming.complete"
            valid_bins = root / "HMI_1_bins"
            valid_bins.mkdir()
            (valid_bins / "HMI_1_magscot_1.fa").write_text(">c1\nAAAA\n")
            candidates = root / "candidates"
            helper = (sys.executable, "-B", "-m", "metabaw.internal")
            tasks = [
                Task(
                    "04.refine.magscot.LMI_6", "04_refinement",
                    (*helper, "run-magscot", "--combined", str(combined),
                     "--mapping", str(mapping), "--run-log", str(root / "native.log"),
                     "--native-command-json", _native_fixture(root)),
                    root, inputs=(combined,), outputs=(mapping,),
                ),
                Task(
                    "04.export_refined_bins.LMI_6", "04_refinement",
                    (*helper, "materialize-refined", "--assembly", str(assembly),
                     "--mapping", str(mapping), "--output-dir", str(empty_bins),
                     "--prefix", "LMI_6_magscot", "--completion-marker", str(marker)),
                    root, deps=("04.refine.magscot.LMI_6",),
                    inputs=(assembly, mapping),
                    outputs=(empty_bins, empty_bins / "manifest.tsv", marker),
                ),
                Task(
                    "05.catalog.candidates", "05_quality",
                    (*helper, "collect-fasta", "--source", str(empty_bins),
                     "--source", str(valid_bins), "--output-dir", str(candidates)),
                    root, deps=("04.export_refined_bins.LMI_6",), inputs=(empty_bins, valid_bins),
                    outputs=(candidates, candidates / "manifest.tsv"),
                ),
            ]
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(
                    state=state, log_dir=root / "logs", max_cpus=2,
                    max_parallel=2, fail_fast=True,
                )
                first = executor.run(tasks)
                output = io.StringIO()
                with redirect_stdout(output):
                    second = executor.run(tasks)
            finally:
                state.close()

            self.assertEqual(set(first.values()), {"success"})
            self.assertEqual(set(second.values()), {"skipped"})
            self.assertEqual(list(empty_bins.glob("*.fa")), [])
            self.assertEqual([path.name for path in candidates.glob("*.fa")], ["HMI_1_magscot_1.fa"])

    def test_strict_workflow_still_blocks_downstream_after_native_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc1\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            helper = (sys.executable, "-B", "-m", "metabaw.internal")
            tasks = [
                Task(
                    "04.refine.magscot.LMI_6", "04_refinement",
                    (*helper, "run-magscot", "--combined", str(combined),
                     "--mapping", str(mapping), "--run-log", str(root / "native.log"),
                     "--native-command-json", _native_fixture(root, exit_code=7)),
                    root, inputs=(combined,), outputs=(mapping,),
                ),
                Task(
                    "04.export_refined_bins.LMI_6", "04_refinement",
                    (sys.executable, "-c", "raise AssertionError('must not run')"),
                    root, deps=("04.refine.magscot.LMI_6",),
                ),
            ]
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state=state, log_dir=root / "logs", max_cpus=1,
                    max_parallel=1, fail_fast=True,
                ).run(tasks)
            finally:
                state.close()
            self.assertEqual(statuses[tasks[0].id], "failed")
            self.assertEqual(statuses[tasks[1].id], "blocked")
            self.assertFalse(mapping.exists())

    def test_managed_nonempty_mapping_is_not_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text("b1\tc1\tmetabat2\nb2\tc1\tvamb\n")
            mapping = root / "sample.refined.contig_to_bin.out"
            expected = "binnew\tcontig\nrefined.1\tc1\n"
            command = json.dumps([
                sys.executable, "-c",
                "from pathlib import Path; "
                f"Path({str(mapping)!r}).write_text({expected!r}); print('Refinement complete')",
            ])
            run_magscot_refinement(combined, mapping, root / "native.log", command)
            self.assertEqual(mapping.read_text(), expected)

    def test_existing_refined_mapping_is_validated_and_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.2\tcontig_2\tvamb\n",
                encoding="utf-8",
            )
            mapping = root / "sample.refined.contig_to_bin.out"
            expected = "binnew\tcontig\nrefined.1\tcontig_1\n"
            mapping.write_text(expected, encoding="utf-8")

            finalize_magscot_mapping(combined, mapping)

            self.assertEqual(mapping.read_text(encoding="utf-8"), expected)

    def test_refined_bins_wait_for_successful_magscot_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "contigs.fa"
            read1 = root / "reads.1.fastq.gz"
            read2 = root / "reads.2.fastq.gz"
            sample = ReadSample("A606", read1, read2, contigs)
            analysis = Analysis("A606", (sample,), (contigs,), False)
            options = BinOptions(
                outdir=root / "results",
                workdir=root / "results" / "tmp" / "work",
                threads=8,
                read_type="short",
                align_tool="bowtie2",
                flye_read_type="--nano-raw",
                binners=("metabat2",),
                min_contig_length=1500,
                min_fasta_kbs=200,
                batch_size=1024,
                refiner="magscot",
                quality_control="checkm2",
                min_completeness=50.0,
                max_contamination=10.0,
                min_quality_score=None,
                run_gunc=False,
                run_trna=False,
                trna_pass=None,
                run_rrna=False,
                rrna_pass=False,
                dereplicator="galah",
                environment=None,
                tag_contigs=False,
                gpu=False,
                max_gpu_memory="4G",
                assembly_strategy="default",
                magscot_dir=root / "MAGScoT",
                checkm2_db=None,
                gunc_db=None,
                gtdbtk_data=None,
                extra_args={},
            )
            builder = DirectBinBuilder([], options, root)
            prepare = Task("01.prepare.A606", "01_prepare", "true", root)
            map_task = Task("03.publish.metabat2.A606", "03_binning", "true", root)
            mapping = root / "metabat2.tsv"
            public_bins = root / "results" / "bin_files" / "metabat2" / "A606"

            result, _bins = builder._refine(
                analysis,
                prepare,
                contigs,
                [("metabat2", map_task, mapping, public_bins)],
            )
            refine = next(
                task for task in builder.tasks if task.id == "04.refine.magscot.A606"
            )

            self.assertIn("run-magscot", refine.display_command())
            self.assertIn("--run-log", refine.display_command())
            native = json.loads(refine.command[refine.command.index("--native-command-json") + 1])
            self.assertEqual(native[0], "Rscript")
            self.assertEqual(native[native.index("--threshold") + 1], "0.4")
            self.assertEqual(native[native.index("--max_cont") + 1], "0.12")
            self.assertFalse(refine.failure_tolerated)
            self.assertEqual(result.deps, (refine.id,))
            self.assertFalse(result.allow_failed_deps)


if __name__ == "__main__":
    unittest.main()

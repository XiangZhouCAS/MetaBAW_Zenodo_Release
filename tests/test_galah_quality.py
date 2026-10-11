from __future__ import annotations

import csv
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.internal import write_drep_genome_info
from metabaw.model import Task


class GalahQualityAdapterTests(unittest.TestCase):
    def _run_adapter(self, report_text: str) -> list[list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bins = root / "bins"
            bins.mkdir()
            (bins / "A606_MetaBAT2_19.fa").write_text(
                ">contig_1\nACGT\n",
                encoding="utf-8",
            )
            report = root / "quality.tsv"
            report.write_text(report_text, encoding="utf-8")
            output = root / "galah_genome_info.csv"

            write_drep_genome_info(
                bins,
                report,
                output,
                strip_extension=True,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                return list(csv.reader(handle))

    def test_metawrap_three_column_report_is_normalized_for_galah(self) -> None:
        rows = self._run_adapter(
            "Name\tCompleteness\tContamination\n"
            "A606_MetaBAT2_19\t99.44\t0.0\n"
        )
        self.assertEqual(
            rows,
            [
                ["genome", "completeness", "contamination"],
                ["A606_MetaBAT2_19", "99.44", "0.0"],
            ],
        )

    def test_checkm2_report_is_normalized_for_galah(self) -> None:
        rows = self._run_adapter(
            "Name\tCompleteness\tContamination\tCompleteness_Model_Used\n"
            "A606_MetaBAT2_19\t99.44\t0.0\tNeural Network (Specific Model)\n"
        )
        self.assertEqual(rows[1], ["A606_MetaBAT2_19", "99.44", "0.0"])

    def test_checkm_fourteen_column_report_is_normalized_for_galah(self) -> None:
        rows = self._run_adapter(
            "Bin Id\tMarker lineage\t# genomes\t# markers\t# marker sets\t0\t1\t2\t3\t4\t5+\tCompleteness\tContamination\tStrain heterogeneity\n"
            "A606_MetaBAT2_19\troot\t1\t10\t5\t0\t0\t0\t0\t0\t0\t99.44\t0.0\t0.0\n"
        )
        self.assertEqual(rows[1], ["A606_MetaBAT2_19", "99.44", "0.0"])

    def test_drep_mode_keeps_the_fasta_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bins = root / "bins"
            bins.mkdir()
            (bins / "A606_MetaBAT2_19.fa").write_text(
                ">contig_1\nACGT\n",
                encoding="utf-8",
            )
            report = root / "quality.tsv"
            report.write_text(
                "Name\tCompleteness\tContamination\n"
                "A606_MetaBAT2_19\t99.44\t0.0\n",
                encoding="utf-8",
            )
            output = root / "drep_genome_info.csv"

            write_drep_genome_info(bins, report, output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[1][0], "A606_MetaBAT2_19.fa")

    def test_galah_command_uses_the_normalized_genome_info_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                quality_control="checkm",
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
            filtered_task = Task(
                id="05.qc.filter",
                stage="05_quality",
                command=("true",),
                cwd=root,
            )

            task, _output = builder._dereplicate(
                filtered_task,
                root / "filtered_bins",
                root / "metawrap_quality.tsv",
            )
            command = task.display_command()

            self.assertIn("drep-genome-info", command)
            self.assertIn("--strip-extension", command)
            self.assertIn("--genome-info", command)
            self.assertNotIn("--checkm-tab-table", command)
            self.assertNotIn("--checkm2-quality-report", command)

    def test_quality_tools_use_short_ipc_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = BinOptions(
                outdir=root / "results" / "a-very-long-analysis-name-for-checkm2",
                workdir=(
                    root
                    / "results"
                    / "a-very-long-analysis-name-for-checkm2"
                    / "tmp"
                    / "work"
                ),
                threads=10,
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
                checkm2_db=root / "uniref100.KO.1.dmnd",
                gunc_db=None,
                gtdbtk_data=None,
                extra_args={},
                checkm2_run_prefix=("conda", "run", "--name", "checkm2"),
            )
            builder = DirectBinBuilder([], options, root)
            catalog_task = Task(
                id="05.catalog.candidates",
                stage="05_quality",
                command=("true",),
                cwd=root,
            )

            task, _report = builder._quality(
                (catalog_task, options.outdir / "quality_control_files" / "candidate_bins")
            )
            command = task.display_command()

            self.assertIn("checkm2_ipc_link=/tmp/mbw-checkm2-$$", command)
            self.assertIn('TMPDIR="$checkm2_ipc_link"', command)
            self.assertIn('--tmpdir "$checkm2_ipc_link"', command)
            self.assertIn("--force", command)
            self.assertIn("--threads 10", command)

            checkm_options = replace(
                options,
                quality_control="checkm",
                checkm2_db=None,
            )
            checkm_builder = DirectBinBuilder([], checkm_options, root)
            checkm_task, _checkm_report = checkm_builder._quality(
                (
                    catalog_task,
                    checkm_options.outdir
                    / "quality_control_files"
                    / "candidate_bins",
                )
            )
            checkm_command = checkm_task.display_command()

            self.assertIn("checkm_ipc_link=/tmp/mbw-checkm-$$", checkm_command)
            self.assertIn('TMPDIR="$checkm_ipc_link"', checkm_command)
            self.assertIn('--tmpdir "$checkm_ipc_link"', checkm_command)
            self.assertIn("checkm lineage_wf", checkm_command)
            self.assertIn("-t 10", checkm_command)


if __name__ == "__main__":
    unittest.main()

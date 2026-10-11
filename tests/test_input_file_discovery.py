from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from metabaw.discovery import (
    ReadSample,
    attach_contigs,
    discover_contigs_from_file,
    discover_reads_from_file,
)


class InputFileDiscoveryTests(unittest.TestCase):
    def test_read_list_supports_bom_comments_blanks_relative_paths_and_mixed_fastq_suffixes(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reads = root / "read files"
            reads.mkdir()
            paths = {
                name: reads / name
                for name in (
                    "t1.R1.fq.gz",
                    "t1.R2.fastq.gz",
                    "t2_1.fastq",
                    "t2_2.fq",
                )
            }
            for path in paths.values():
                path.write_bytes(b"reads")
            manifest = root / "reads.list"
            manifest.write_text(
                "\ufeff#reads_files\n\n"
                "  read files/t1.R1.fq.gz , read files/t1.R2.fastq.gz  \n"
                "read files/t2_1.fastq,read files/t2_2.fq\n",
                encoding="utf-8",
            )

            samples = discover_reads_from_file(manifest)

            self.assertEqual([sample.name for sample in samples], ["t1", "t2"])
            self.assertEqual(
                samples[0].reads,
                (paths["t1.R1.fq.gz"].resolve(), paths["t1.R2.fastq.gz"].resolve()),
            )
            self.assertEqual(
                samples[1].reads,
                (paths["t2_1.fastq"].resolve(), paths["t2_2.fq"].resolve()),
            )

    def test_long_read_numeric_suffixes_remain_distinct_samples(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            read_paths = [root / "HMI_1.fastq.gz", root / "HMI_2.fq.gz"]
            for path in read_paths:
                path.write_bytes(b"reads")
            manifest = root / "reads.list"
            manifest.write_text(
                "HMI_1.fastq.gz\nHMI_2.fq.gz\n",
                encoding="utf-8",
            )

            samples = discover_reads_from_file(manifest)

            self.assertEqual([sample.name for sample in samples], ["HMI_1", "HMI_2"])
            self.assertTrue(all(sample.read2 is None for sample in samples))

    def test_contig_list_accepts_common_fasta_suffixes_and_infers_association_keys(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contig_dir = root / "contigs"
            contig_dir.mkdir()
            t1 = contig_dir / "t1.contigs.fa"
            t2 = contig_dir / "t2.assembly.fasta.gz"
            t1.write_text(">t1\nACGT\n", encoding="utf-8")
            t2.write_bytes(b"contigs")
            manifest = root / "contigs.list"
            manifest.write_text(
                "\ufeff #contig_files\ncontigs/t2.assembly.fasta.gz\n\ncontigs/t1.contigs.fa\n",
                encoding="utf-8",
            )

            contigs = discover_contigs_from_file(manifest)
            samples = [
                ReadSample("t1", root / "t1.fastq.gz"),
                ReadSample("t2", root / "t2.fastq.gz"),
            ]
            attached = attach_contigs(samples, contigs)

            self.assertEqual(contigs, [t2.resolve(), t1.resolve()])
            self.assertEqual(
                [sample.contigs for sample in attached],
                [t1.resolve(), t2.resolve()],
            )

    def test_missing_manifest_is_reported(self) -> None:
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.list"
            with self.assertRaisesRegex(FileNotFoundError, "Read input list file not found"):
                discover_reads_from_file(missing, "short")

    def test_manifest_must_be_a_regular_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "must be a regular file"):
                discover_contigs_from_file(root)

    def test_comment_only_manifest_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "reads.list"
            manifest.write_text("# reads\n\n  # another comment\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains no data rows"):
                discover_reads_from_file(manifest, "short")

    def test_missing_or_non_file_listed_path_reports_manifest_line(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_manifest = root / "missing.list"
            missing_manifest.write_text("# reads\nmissing.fastq.gz\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, r"missing\.list:2: listed path not found"):
                discover_reads_from_file(missing_manifest, "short")

            directory = root / "sample.fa"
            directory.mkdir()
            directory_manifest = root / "directory.list"
            directory_manifest.write_text("sample.fa\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"directory\.list:1: listed path is not a regular file"):
                discover_contigs_from_file(directory_manifest)

    def test_unsupported_suffix_is_rejected_with_line_context(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            read = root / "sample.txt"
            read.write_bytes(b"reads")
            manifest = root / "reads.list"
            manifest.write_text("sample.txt\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"reads\.list:1: unsupported file suffix"):
                discover_reads_from_file(manifest, "short")

    def test_duplicate_resolved_path_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            read = root / "sample.fastq.gz"
            read.write_bytes(b"reads")
            manifest = root / "reads.list"
            manifest.write_text(
                "sample.fastq.gz\n./sample.fastq.gz\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"reads\.list:2: duplicate listed path .* first listed on line 1",
            ):
                discover_reads_from_file(manifest, "short")

    def test_single_path_is_long_even_with_a_numeric_mate_like_suffix(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            read2 = root / "sample_R2.fastq.gz"
            read2.write_bytes(b"reads")
            manifest = root / "reads.list"
            manifest.write_text("sample_R2.fastq.gz\n", encoding="utf-8")

            samples = discover_reads_from_file(manifest)
            self.assertEqual(samples, [ReadSample("sample_R2", read2.resolve())])

    def test_pair_row_uses_explicit_order_and_separator_sample_name(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            read1 = root / "SRR5024276_1.fastq.gz"
            read2 = root / "SRR5024276_2.fastq.gz"
            read1.touch()
            read2.touch()
            manifest = root / "reads.list"
            manifest.write_text(f"{read1},{read2}\n", encoding="utf-8")
            samples = discover_reads_from_file(manifest, separator="_")
            self.assertEqual(
                samples, [ReadSample("SRR5024276", read1.resolve(), read2.resolve())]
            )

            # An explicit row assigns mates even when filenames have no mate suffix.
            forward = root / "sample.forward.fq"
            reverse = root / "sample.reverse.fq"
            forward.touch()
            reverse.touch()
            manifest.write_text("sample.forward.fq,sample.reverse.fq\n", encoding="utf-8")
            sample = discover_reads_from_file(manifest)[0]
            self.assertEqual(sample.reads, (forward.resolve(), reverse.resolve()))
            self.assertEqual(sample.name, "sample")

    def test_mixed_row_layouts_and_explicit_type_conflicts_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("S1_1.fq", "S1_2.fq", "HMI_1.fq"):
                (root / name).touch()
            manifest = root / "reads.list"
            manifest.write_text("S1_1.fq,S1_2.fq\nHMI_1.fq\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"reads\.list:2: mixed"):
                discover_reads_from_file(manifest)
            for row, conflict in (("S1_1.fq,S1_2.fq", "long"), ("HMI_1.fq", "short")):
                with self.subTest(conflict=conflict):
                    manifest.write_text(row + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"--type {conflict} conflicts"):
                        discover_reads_from_file(manifest, conflict)
                    matching = "short" if conflict == "long" else "long"
                    self.assertEqual(len(discover_reads_from_file(manifest, matching)), 1)

    def test_malformed_rows_and_reused_mates_report_the_manifest_line(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "S1.fq").touch()
            (root / "S2.fq").touch()
            manifest = root / "reads.list"
            for row in ("S1.fq,", ",S1.fq", "S1.fq,,S2.fq", "S1.fq,S2.fq,S1.fq"):
                with self.subTest(row=row):
                    manifest.write_text("# reads\n" + row + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, r"reads\.list:2: expected one sample per line"):
                        discover_reads_from_file(manifest)
            for row in ("S1.fq,./S1.fq", "S1.fq,S2.fq\n./S1.fq,S2.fq"):
                with self.subTest(row=row):
                    manifest.write_text(row + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate listed path"):
                        discover_reads_from_file(manifest)

    def test_rows_are_not_paired_across_lines_and_duplicate_names_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("S1_1.fq", "S1_2.fq"):
                (root / name).touch()
            manifest = root / "reads.list"
            manifest.write_text("S1_1.fq\nS1_2.fq\n", encoding="utf-8")
            samples = discover_reads_from_file(manifest)
            self.assertEqual([sample.name for sample in samples], ["S1_1", "S1_2"])
            self.assertTrue(all(sample.read2 is None for sample in samples))
            with self.assertRaisesRegex(ValueError, r"reads\.list:2: duplicate sample name 'S1'"):
                discover_reads_from_file(manifest, separator="_")


if __name__ == "__main__":
    unittest.main()

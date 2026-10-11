from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from metabaw.internal import (
    build_parser,
    materialize_bins,
    materialize_refined,
    normalize_refined_fasta,
)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">{identifier}\n{sequence}\n" for identifier, sequence in records),
        encoding="utf-8",
    )


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle, delimiter="\t"))


class InternalMaterializationNamingTests(unittest.TestCase):
    def test_materialize_bins_uses_natural_bin_order_and_writes_marker(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            assembly = root / "assembly.fa"
            mapping = root / "mapping.tsv"
            output = root / "bins"
            marker = output / ".metabaw.naming-v2.complete"
            _write_fasta(
                assembly,
                [("contig1", "AAAA"), ("contig2", "CCCC"), ("contig10", "GGGG")],
            )
            mapping.write_text(
                "bin.10\tcontig10\n"
                "bin.2\tcontig2\n"
                "bin.1\tcontig1\n",
                encoding="utf-8",
            )

            materialize_bins(
                assembly,
                mapping,
                output,
                "reads_to_contigs_MetaBAT2",
                completion_marker=marker,
            )

            self.assertEqual(marker.read_text(encoding="utf-8"), "complete\n")
            self.assertIn(">contig1\n", (output / "reads_to_contigs_MetaBAT2_1.fa").read_text())
            self.assertIn(">contig2\n", (output / "reads_to_contigs_MetaBAT2_2.fa").read_text())
            self.assertIn(">contig10\n", (output / "reads_to_contigs_MetaBAT2_3.fa").read_text())
            self.assertEqual(
                _read_tsv(output / "manifest.tsv"),
                [
                    ["bin", "file", "contigs"],
                    ["bin.1", "reads_to_contigs_MetaBAT2_1.fa", "1"],
                    ["bin.2", "reads_to_contigs_MetaBAT2_2.fa", "1"],
                    ["bin.10", "reads_to_contigs_MetaBAT2_3.fa", "1"],
                ],
            )

    def test_failed_materialization_removes_stale_completion_marker(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            assembly = root / "assembly.fa"
            mapping = root / "mapping.tsv"
            marker = root / "state" / "naming.complete"
            _write_fasta(assembly, [("contig1", "AAAA")])
            mapping.write_text("bin.1\tmissing_contig\n", encoding="utf-8")
            marker.parent.mkdir(parents=True)
            marker.write_text("stale\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no matching contigs"):
                materialize_bins(
                    assembly,
                    mapping,
                    root / "bins",
                    "dataset_MetaBAT2",
                    completion_marker=marker,
                )

            self.assertFalse(marker.exists())

    def test_materialize_refined_naturally_numbers_only_changed_bins(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            assembly = root / "assembly.fa"
            mapping = root / "refined.tsv"
            source = root / "source_bins"
            output = root / "refined_bins"
            marker = root / "state" / "refined.naming.complete"
            _write_fasta(
                assembly,
                [("contig1", "AAAA"), ("contig2", "CCCC"), ("contig10", "GGGG")],
            )
            mapping.write_text(
                "bin.10\tcontig10\n"
                "bin.2\tcontig2\n"
                "bin.1\tcontig1\n",
                encoding="utf-8",
            )
            _write_fasta(source / "dataset_MetaBAT2_9.fa", [("contig2", "CCCC")])

            materialize_refined(
                assembly,
                mapping,
                output,
                "reads_to_contigs_MAGScoT",
                [source],
                completion_marker=marker,
            )

            self.assertTrue(marker.is_file())
            self.assertEqual(
                _read_tsv(output / "manifest.tsv"),
                [
                    ["bin", "file", "contigs"],
                    ["bin.1", "reads_to_contigs_MAGScoT_1.fa", "1"],
                    ["bin.2", "dataset_MetaBAT2_9.fa", "1"],
                    ["bin.10", "reads_to_contigs_MAGScoT_2.fa", "1"],
                ],
            )

    def test_normalize_refined_fasta_uses_natural_source_order_and_marker(self) -> None:
        with TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            refined = root / "refined"
            source = root / "source_bins"
            marker = refined / ".metabaw.naming-v2.complete"
            _write_fasta(refined / "refined.10.fa", [("contig10", "GGGG")])
            _write_fasta(refined / "refined.2.fa", [("contig2", "CCCC")])
            _write_fasta(refined / "refined.1.fa", [("contig1", "AAAA")])
            _write_fasta(source / "dataset_MetaBAT2_9.fa", [("contig2", "CCCC")])

            normalize_refined_fasta(
                refined,
                refined,
                "reads_to_contigs_DASTool",
                [source],
                completion_marker=marker,
            )

            self.assertTrue(marker.is_file())
            self.assertEqual(
                _read_tsv(refined / "manifest.tsv"),
                [
                    ["source", "published", "provenance"],
                    ["refined.1.fa", "reads_to_contigs_DASTool_1.fa", "refined"],
                    ["refined.2.fa", "dataset_MetaBAT2_9.fa", "unchanged"],
                    ["refined.10.fa", "reads_to_contigs_DASTool_2.fa", "refined"],
                ],
            )

    def test_materialization_parsers_accept_optional_completion_marker(self) -> None:
        parser = build_parser()
        cases = {
            "materialize-bins": [
                "--assembly",
                "assembly.fa",
                "--mapping",
                "mapping.tsv",
                "--output-dir",
                "bins",
                "--prefix",
                "sample_MetaBAT2",
            ],
            "materialize-refined": [
                "--assembly",
                "assembly.fa",
                "--mapping",
                "mapping.tsv",
                "--output-dir",
                "bins",
                "--prefix",
                "sample_MAGScoT",
            ],
            "normalize-refined-fasta": [
                "--source-dir",
                "raw_bins",
                "--output-dir",
                "bins",
                "--prefix",
                "sample_DASTool",
            ],
        }
        for command, arguments in cases.items():
            with self.subTest(command=command):
                parsed = parser.parse_args(
                    [command, *arguments, "--completion-marker", "naming.complete"]
                )
                self.assertEqual(parsed.completion_marker, Path("naming.complete"))


if __name__ == "__main__":
    unittest.main()

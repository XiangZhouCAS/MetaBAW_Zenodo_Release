from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metabaw.internal import run_metawrap_refinement


class MetaWrapStagingTests(unittest.TestCase):
    def test_only_fasta_files_are_passed_to_metawrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "published_bins"
            source.mkdir()
            (source / "sample_SemiBin2_1.fa").write_text(
                ">contig_1\nACGT\n",
                encoding="utf-8",
            )
            (source / "manifest.tsv").write_text(
                "source\tpublished\nraw.tsv\tsample_SemiBin2_1.fa\n",
                encoding="utf-8",
            )
            output_root = root / "refinement" / "sample"
            output_bins = output_root / "metawrap_50_10_bins"
            stats = output_root / "metawrap_50_10_bins.stats"
            completion = output_root / "metawrap.complete"

            def fake_run(command: list[str], check: bool) -> None:
                self.assertTrue(check)
                staged_input = Path(command[command.index("-A") + 1])
                self.assertEqual(
                    [path.name for path in staged_input.iterdir()],
                    ["sample_SemiBin2_1.fa"],
                )
                output_bins.mkdir(parents=True)
                (output_bins / "bin.1.fa").write_text(
                    ">contig_1\nACGT\n",
                    encoding="utf-8",
                )
                stats.write_text(
                    "bin\tcompleteness\tcontamination\n",
                    encoding="utf-8",
                )

            with patch("metabaw.internal.subprocess.run", side_effect=fake_run):
                run_metawrap_refinement(
                    [f"semibin2={source}"],
                    json.dumps(["conda", "run", "--name", "metawrap"]),
                    output_root,
                    output_bins,
                    stats,
                    completion,
                    10,
                    50.0,
                    10.0,
                    "",
                )

            staged_root = output_root.with_name(f"{output_root.name}_inputs")
            self.assertTrue((staged_root / "manifest.tsv").is_file())
            self.assertTrue(completion.is_file())
            self.assertNotIn("manifest.fa", {path.name for path in output_bins.iterdir()})


if __name__ == "__main__":
    unittest.main()

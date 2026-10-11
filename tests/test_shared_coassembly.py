from __future__ import annotations

from pathlib import Path
import unittest

from metabaw.discovery import ReadSample, build_analyses


class SharedCoassemblyTests(unittest.TestCase):
    def test_multi_mode_runs_one_analysis_for_a_shared_coassembly(self) -> None:
        assembly = Path("/data/CAMI_high.fa")
        samples = [
            ReadSample("S001", Path("/data/S001.fq.gz"), contigs=assembly),
            ReadSample("S002", Path("/data/S002.fq.gz"), contigs=assembly),
        ]

        analyses = build_analyses(samples, "multi")

        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0].name, "CAMI_high")
        self.assertEqual(analyses[0].samples, tuple(samples))
        self.assertEqual(analyses[0].contigs, (assembly,))
        self.assertTrue(analyses[0].cross_mapped)

    def test_single_mode_rejects_a_shared_coassembly_with_actionable_help(self) -> None:
        assembly = Path("/data/strain_madness.fa")
        samples = [
            ReadSample("sample_0", Path("/data/0.fq.gz"), contigs=assembly),
            ReadSample("sample_1", Path("/data/1.fq.gz"), contigs=assembly),
        ]

        with self.assertRaisesRegex(ValueError, "--multi"):
            build_analyses(samples, "single")


if __name__ == "__main__":
    unittest.main()

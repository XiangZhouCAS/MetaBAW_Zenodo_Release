from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from metabaw.internal import stage_bams


class BamStagingTests(unittest.TestCase):
    def test_stage_bams_publishes_complete_atomic_bamset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mapping" / "S1.bam"
            source.parent.mkdir()
            source.write_bytes(b"BAM")
            Path(str(source) + ".bai").write_bytes(b"BAI")
            output = root / "mapping" / "bamset"

            stage_bams([f"S1={source}"], output)

            self.assertEqual((output / "S1.bam").read_bytes(), b"BAM")
            self.assertEqual((output / "S1.bam.bai").read_bytes(), b"BAI")
            self.assertFalse(any(output.parent.glob(".bamset.staging-*")))

    def test_stage_bams_rejects_missing_bam_index_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mapping" / "S1.bam"
            source.parent.mkdir()
            source.write_bytes(b"BAM")
            output = root / "mapping" / "bamset"

            with self.assertRaisesRegex(FileNotFoundError, "BAM index"):
                stage_bams([f"S1={source}"], output)

            self.assertFalse(output.exists())
            self.assertFalse(any(output.parent.glob(".bamset.staging-*")))


if __name__ == "__main__":
    unittest.main()

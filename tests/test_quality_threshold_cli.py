from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from metabaw.cli import build_parser, command_bin
from metabaw.discovery import ReadSample


class QualityThresholdCliTests(TestCase):
    @staticmethod
    def _parse(extra: list[str]):
        return build_parser().parse_args(
            ["bin", "--input_reads_files", "reads.tsv", "--input_contig_files", "contigs.tsv", *extra]
        )

    def test_quality_threshold_option_names_and_defaults(self):
        args = self._parse([])
        self.assertEqual(args.con, 10.0)
        self.assertEqual(args.com, 50.0)

        explicit = self._parse(["--con", "5", "--com", "70"])
        self.assertEqual(explicit.con, 5.0)
        self.assertEqual(explicit.com, 70.0)

    def test_quality_thresholds_map_to_their_full_meanings(self):
        args = self._parse(["--con", "5", "--com", "70"])
        captured = {}

        class FakeBuilder:
            def __init__(self, _analyses, options, _workdir, _profiles):
                captured["options"] = options

            def build(self):
                return []

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            args.output = str(output)
            with (
                patch("metabaw.cli.load_named_samples", return_value=[ReadSample("S1", Path("r1.fq"), Path("r2.fq"), Path("contigs.fa"), preserve_name=True)]),
                patch("metabaw.cli._preflight_bin", return_value=None),
                patch("metabaw.cli.NamedBinBuilder", FakeBuilder),
                patch("metabaw.cli._run_direct", return_value=None),
            ):
                command_bin(args)

        self.assertEqual(captured["options"].max_contamination, 5.0)
        self.assertEqual(captured["options"].min_completeness, 70.0)

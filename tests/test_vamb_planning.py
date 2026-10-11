from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample


class VambPlanningTests(unittest.TestCase):
    def test_vamb_disables_inapplicable_default_binsplitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "A606.fa"
            sample = ReadSample(
                "A606",
                root / "A606.1.fastq.gz",
                root / "A606.2.fastq.gz",
                contigs,
            )
            analysis = Analysis("A606", (sample,), (contigs,), False)
            options = BinOptions(
                outdir=root / "results",
                workdir=root / "results" / "tmp" / "work",
                threads=4,
                read_type="short",
                align_tool="bowtie2",
                flye_read_type="--nano-raw",
                binners=("vamb",),
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
                extra_args={"vamb": "-e 30 -q 10 20"},
            )

            tasks = DirectBinBuilder([analysis], options, root).build()
            task = next(item for item in tasks if item.id == "03.bin.vamb.A606")
            command = task.display_command()

            self.assertIn("--seed 42 -o -p 4", command)
            self.assertIn("-e 30 -q 10 20", command)
            self.assertIn("vamb_ipc_link=/tmp/mbw-vamb-$$", command)
            self.assertIn('TMPDIR="$vamb_ipc_link"', command)
            self.assertIn('ln -s', command)
            self.assertIn('trap \'rm -f "$vamb_ipc_link"\' EXIT', command)


if __name__ == "__main__":
    unittest.main()

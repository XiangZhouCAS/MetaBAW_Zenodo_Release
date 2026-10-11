from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import build_parser, command_annotation, _run_direct
from metabaw.internal import main as internal_main, stage_annotation_genomes, merge_coverm_taxonomy
from metabaw.named_inputs import read_genome_files, read_named_reads


class NamedAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "other directory").mkdir()
        self.a = self.root / "MAG_A.fa"
        self.a.write_text(">c1 description\nACGTACGT\n")
        self.b = self.root / "other directory" / "MAG_B.FNA.GZ"
        self.b.write_bytes(gzip.compress(b">c1 independent record\nGGGGAAAA\n"))
        (self.root / "unlisted.fa").write_text(">unused\nAAAA\n")
        for name in ("machine_1.fastq.gz", "machine_2.fastq", "ont.fq.gz"):
            (self.root / name).touch()
        self.reads = self.root / "reads.tsv"
        self.reads.write_text("\ufeff#sample\treads\nA_clean\tmachine_1.fastq.gz,machine_2.fastq\nL1\tont.fq.gz\n", encoding="utf-8")
        self.genomes = self.root / "genomes.txt"
        self.genomes.write_text("\ufeff#MAGs\nMAG_A.fa\n\nother directory/MAG_B.FNA.GZ\n", encoding="utf-8")

    def args(self, extra=()):
        return build_parser().parse_args([
            "annotation", "--input_reads_files", str(self.reads),
            "--input_genome_files", str(self.genomes), "-o", str(self.root / "result"),
            *extra,
        ])

    def plan(self, extra=()):
        args = self.args(extra)
        with (patch("metabaw.cli._preflight_annotation"),
              patch("metabaw.cli._run_direct", return_value=0) as run,
              redirect_stdout(io.StringIO())):
            self.assertEqual(command_annotation(args), 0)
        return run.call_args.args[1], run.call_args.args[4]

    def test_genome_list_is_path_only_relative_bom_and_gzip(self):
        self.assertEqual(read_genome_files(self.genomes), [self.a, self.b])

    def test_genome_list_rejects_duplicate_path_or_basename(self):
        for text in ("MAG_A.fa\n./MAG_A.fa\n", "MAG_A.fa\nother directory/MAG_A.fna\n"):
            (self.root / "other directory" / "MAG_A.fna").write_text(">c\nAAAA\n")
            self.genomes.write_text(text)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "duplicate genome"):
                read_genome_files(self.genomes)

    def test_invalid_genome_rows_fail_before_preflight_or_outputs(self):
        for text in ("#empty\n", "A\tMAG_A.fa\n", "missing.fa\n", "ont.fq.gz\n", ".\n"):
            self.genomes.write_text(text)
            with self.subTest(text=text), patch("metabaw.cli._preflight_annotation") as preflight:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    command_annotation(self.args())
                preflight.assert_not_called()
            self.assertFalse((self.root / "result").exists())

    def test_old_flags_and_abbreviations_rejected_in_both_help_modes(self):
        for flag in ("-p", "-r", "-s", "-f", "--type", "--path", "--reads",
                     "--suffix", "--read-suffix", "--separate-sample-name",
                     "--dry-run", "--force", "--input_reads", "--input_genome"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                self.args((flag, "value"))
            self.assertEqual(stopped.exception.code, 2)
        for help_flag in ("-h", "--help"):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                build_parser().parse_args(["annotation", help_flag])
            self.assertEqual(stopped.exception.code, 0)
            text = output.getvalue()
            for flag in ("--input_reads_files", "--input_genome_files"):
                self.assertIn(flag, text)
            for flag in ("--path", "--read-suffix", "--separate-sample-name", "--type"):
                self.assertNotIn(flag, text)
            self.assertNotIn("--full-help", text)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            build_parser().parse_args(["annotation", "--full-help"])
        self.assertEqual(stopped.exception.code, 2)

    def test_mixed_samples_use_correct_coverm_mapper_and_preserve_names(self):
        tasks, payload = self.plan()
        by_id = {t.id: t for t in tasks}
        self.assertEqual(payload["read_type"], "mixed")
        self.assertEqual(payload["samples"], ["A_clean", "L1"])
        self.assertEqual(payload["mags"], [str(self.a), str(self.b)])
        self.assertEqual(payload["input_sources"]["genomes"]["kind"], "path_list")
        self.assertEqual([s["mapper"] for s in payload["input_samples"]], ["minimap2-sr", "minimap2-ont"])
        for task in tasks:
            if task.id.startswith("02.annotation.coverm."):
                self.assertIn("00.annotation.genomes", task.deps)
                if task.sample == "A_clean":
                    self.assertIn("--mapper minimap2-sr", task.display_command())
                    self.assertIn("--coupled", task.display_command())
                else:
                    self.assertIn("--mapper minimap2-ont", task.display_command())
                    self.assertIn("--single", task.display_command())
        self.assertIn("--input", by_id["03.annotation.merge"].command)
        self.assertTrue(any(arg.startswith("A_clean=") for arg in by_id["03.annotation.merge"].command))

    def test_staged_genomes_reach_taxonomy_and_shared_functional_searches(self):
        tasks, payload = self.plan()
        by_id = {t.id: t for t in tasks}
        stage = by_id["00.annotation.genomes"]
        self.assertEqual(stage.inputs, (self.a, self.b))
        self.assertNotIn(self.root / "unlisted.fa", stage.inputs)
        for name in ("MAG_A", "MAG_B"):
            self.assertIn(stage.id, by_id[f"05.annotation.prodigal.{name}"].deps)
            staged = Path(next(g["staged"] for g in payload["genomes"] if g["genome"] == name))
            self.assertIn(staged, by_id[f"05.annotation.prodigal.{name}"].inputs)
        self.assertIn(stage.id, by_id["01.annotation.gtdbtk"].deps)
        self.assertEqual(sum(t.id == "06.annotation.kegg" for t in tasks), 1)
        self.assertEqual(sum(t.id == "07.annotation.cazy" for t in tasks), 1)
        internal_main(list(stage.command[3:]))
        directory = Path(payload["genomes"][0]["staged"]).parent
        self.assertEqual({p.name for p in directory.glob("*.fa")}, {"MAG_A.fa", "MAG_B.fa"})
        self.assertEqual((directory / "MAG_A.fa").read_text(), self.a.read_text())
        self.assertIn("GGGGAAAA", (directory / "MAG_B.fa").read_text())
        self.assertTrue((directory / ".metabaw.complete").is_file())

    def test_staging_invalid_fasta_preserves_previous_result_and_sources(self):
        output = self.root / "staged"
        original = self.a.read_bytes()
        stage_annotation_genomes([f"MAG_A={self.a}"], output)
        self.b.write_bytes(gzip.compress(b">empty\n"))
        with self.assertRaisesRegex(ValueError, "Empty sequence"):
            stage_annotation_genomes([f"MAG_B={self.b}"], output)
        self.assertEqual(self.a.read_bytes(), original)
        self.assertEqual((output / "MAG_A.fa").read_bytes(), original)
        self.assertEqual(list(self.root.glob(".staged.staging-*")), [])

    def test_staging_omits_old_genomes_and_does_not_overwrite_its_input(self):
        output = self.root / "staged"
        stage_annotation_genomes([f"MAG_A={self.a}"], output)
        stage_annotation_genomes([f"MAG_B={self.b}"], output)
        self.assertFalse((output / "MAG_A.fa").exists())
        self.assertTrue((output / "MAG_B.fa").exists())
        with self.assertRaisesRegex(ValueError, "inside the staging output"):
            stage_annotation_genomes([f"MAG_B={output / 'MAG_B.fa'}"], output)

    def test_named_reads_are_shared_with_binning_and_single_sample_is_long(self):
        self.reads.write_text("Chosen_name\tont.fq.gz\n")
        self.assertEqual(read_named_reads(self.reads)[0].name, "Chosen_name")
        tasks, payload = self.plan(("--no-niche",))
        self.assertEqual(payload["read_type"], "long")
        self.assertTrue(all("--mapper minimap2-ont" in t.display_command()
                            for t in tasks if t.id.startswith("02.annotation.coverm.")))
        tasks, payload = self.plan()
        self.assertFalse(payload["niche"]["enabled"])
        self.assertNotIn("04.annotation.niche", {task.id for task in tasks})

    def counted_inputs(self, genomes, samples):
        genome_rows = []
        for number in range(genomes):
            path = self.root / f"counted_MAG{number}.fa"
            path.write_text(">c\nACGT\n")
            genome_rows.append(path.name)
        self.genomes.write_text("\n".join(genome_rows) + "\n")
        read_rows = []
        for number in range(samples):
            # Two paths still count as ONE named reads sample.
            pair = [self.root / f"counted_S{number}_{mate}.fastq" for mate in (1, 2)]
            for path in pair:
                path.touch()
            read_rows.append(f"S{number}\t{pair[0].name},{pair[1].name}")
        self.reads.write_text("\n".join(read_rows) + "\n")

    def test_niche_requires_both_counts_above_three(self):
        for genomes, samples in ((1, 4), (4, 1), (2, 5), (5, 2), (3, 3),
                                 (3, 4), (4, 3), (4, 4), (5, 5)):
            with self.subTest(genomes=genomes, samples=samples):
                self.counted_inputs(genomes, samples)
                tasks, payload = self.plan()
                eligible = genomes > 3 and samples > 3
                self.assertEqual(payload["niche"]["enabled"], eligible)
                self.assertEqual("04.annotation.niche" in {t.id for t in tasks}, eligible)
                self.assertEqual(payload["niche"]["genome_count"], genomes)
                self.assertEqual(payload["niche"]["read_sample_count"], samples)
                self.assertIn(f"read_samples={samples}", payload["niche"]["message"])
                if not eligible:
                    self.assertIn("Niche classification will not run", payload["niche"]["message"])
                self.assertTrue(any(t.id.startswith("02.annotation.coverm.") for t in tasks))
                self.assertIn("06.annotation.kegg", {t.id for t in tasks})

    def test_niche_explicit_disable_and_method_requirements(self):
        self.counted_inputs(4, 4)
        tasks, payload = self.plan(("--no-niche", "--methods", "mean"))
        self.assertFalse(payload["niche"]["requested"])
        self.assertFalse(payload["niche"]["enabled"])
        self.assertIn("--no-niche", payload["niche"]["reason"])
        self.assertNotIn("04.annotation.niche", {t.id for t in tasks})
        with self.assertRaisesRegex(ValueError, "requires CoverM methods"):
            self.plan(("--methods", "mean"))
        self.counted_inputs(3, 4)
        tasks, payload = self.plan(("--methods", "mean"))
        self.assertFalse(payload["niche"]["enabled"])
        self.assertTrue(any(t.id.startswith("02.annotation.coverm.") for t in tasks))

    def test_niche_skip_decision_is_logged_on_disk_and_in_manifest(self):
        tasks, payload = self.plan()
        args = self.args(("--max-memory", "1"))
        output = self.root / "logged"
        with (patch("metabaw.cli.Executor") as executor,
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            # The existing memory guard prevents external commands; startup
            # still persists the niche decision, even on a pre-execution stop.
            code = _run_direct(args, tasks, output, output / "tmp", payload)
        self.assertEqual(code, 1)
        executor.assert_not_called()
        log = output / "niche_status.log"
        self.assertIn("[NICHE] Skipped", log.read_text())
        self.assertIn("genomes=2; read_samples=2", log.read_text())
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertFalse(manifest["niche"]["enabled"])
        self.assertEqual(manifest["result_files"]["niche_status_log"], str(log))
        details = output / "workflow_details.md"
        self.assertEqual(manifest["result_files"]["workflow_details"], str(details))
        self.assertNotIn("resolved_parameters", manifest["result_files"])
        self.assertFalse((output / "resolved_parameters.tsv").exists())
        report = details.read_text(encoding="utf-8")
        recorded = {task["id"]: task for task in manifest["tasks"]}
        for task in tasks:
            self.assertEqual(recorded[task.id]["command"], task.display_command())
        for software in ("GTDB-Tk", "CoverM", "KofamScan (KEGG)", "dbCAN (CAZy)", "BLASTP"):
            self.assertIn(software, report)
        self.assertIn("Niche classification: skipped", report)
        self.assertIn("requires genomes > 3", report)
        self.assertIn("KEGG annotation", report)
        self.assertIn("Hydrogenase annotation", report)
        self.assertIn("# Step1", report)
        self.assertEqual(report.count("Software: CoverM"), 1)
        self.assertIn("[MEMORY ESTIMATE]", (output / "start_info.txt").read_text())

    def test_reused_taxonomy_still_depends_on_genome_staging_for_coverage(self):
        taxonomy = self.root / "existing_gtdb"
        taxonomy.mkdir()
        (taxonomy / "gtdbtk.bac120.summary.tsv").write_text(
            "user_genome\tclassification\nMAG_A\tUnclassified Bacteria\nMAG_B\tUnclassified Bacteria\n")
        tasks, payload = self.plan(("--gtdbtk_res", str(taxonomy)))
        self.assertTrue(payload["gtdbtk_classification_skipped"])
        self.assertNotIn("01.annotation.gtdbtk", {t.id for t in tasks})
        self.assertTrue(all("00.annotation.genomes" in t.deps
                            for t in tasks if t.id.startswith("02.annotation.coverm.")))

    def test_merge_uses_manifest_sample_names_not_fastq_basenames(self):
        taxonomy = self.root / "taxonomy"
        taxonomy.mkdir()
        (taxonomy / "gtdbtk.bac120.summary.tsv").write_text(
            "user_genome\tclassification\nMAG_A\tUnclassified Bacteria\n")
        profile = self.root / "profile.tsv"
        profile.write_text("Genome\tmachine_1 Relative Abundance (%)\nMAG_A\t2.5\n")
        output = self.root / "merge"
        merge_coverm_taxonomy([f"A_clean={profile}", f"L1={profile}"], taxonomy, output, ".tsv")
        header = (output / "coverm_all_metrics.tsv").read_text().splitlines()[0]
        self.assertIn("A_clean|", header)
        self.assertIn("L1|", header)
        self.assertNotIn("machine_1", header)


if __name__ == "__main__":
    unittest.main()

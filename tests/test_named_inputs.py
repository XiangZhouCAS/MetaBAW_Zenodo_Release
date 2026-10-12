from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from metabaw.cli import build_parser, command_bin, _run_direct, _write_binner_summary
from metabaw.dependencies import bin_requirements
from metabaw.direct import BINNER_ORDER
from metabaw.internal import (
    concatenate_fastas,
    materialize_bins,
    report_coassembly_provenance,
    rename_fasta_records,
    stage_bams,
)
from metabaw.named_inputs import (
    load_named_samples, read_named_reads, read_named_contigs, read_multi_names,
    named_analyses, read_coassembly_groups, named_coassembly_analyses,
    eligible_binners,
)


class NamedInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        # File basenames deliberately bear no relation to declared sample names.
        for name in ("machine a.fastq", "machine b.fastq.gz", "x.fq.gz", "y.fq",
                     "nanopore.fastq.gz", "u.fastq", "v.fastq"):
            (self.root / name).touch()
        (self.root / "assembly.fasta").write_text(">c\n" + "ACGT" * 400 + "\n")
        self.reads = self.root / "reads.tsv"
        self.reads.write_text(
            "\ufeff#sample\treads\n\n"
            "A_clean\tmachine a.fastq,machine b.fastq.gz\n"
            "B\tx.fq.gz,y.fq\n"
            "L\tnanopore.fastq.gz\n"
            "outside\tu.fastq,v.fastq\n", encoding="utf-8")
        self.contigs = self.root / "contigs.tsv"
        self.contigs.write_text("".join(
            f"{name}\tassembly.fasta\n" for name in ("L", "B", "outside", "A_clean")))
        self.multi = self.root / "multi.txt"
        self.multi.write_text("#selected\nA_clean\nB\nL\n")
        self.coassembly = self.root / "coassembly.tsv"
        self.coassembly.write_text(
            "#sample name\tgroup\nA_clean\tgroup1\nB\tgroup1\n"
        )

    def plan(self, grouped=False, extra=(), include_contigs=True):
        args = build_parser().parse_args([
            "bin", "--input_reads_files", str(self.reads),
            *(["--input_contig_files", str(self.contigs)] if include_contigs else []),
            "--no-gpu", "-t", "8", "-o", str(self.root / "results"),
            *(["--multi-files", str(self.multi)] if grouped else []), *extra,
        ])
        with (patch("metabaw.cli._preflight_bin") as preflight,
              patch("metabaw.cli._run_direct", return_value=0) as run,
              redirect_stdout(io.StringIO())):
            self.assertEqual(command_bin(args), 0)
        preflight.assert_called_once()
        return args, run.call_args.args[1], run.call_args.args[4]

    def test_named_identity_mates_order_and_relative_paths(self):
        samples = load_named_samples(self.reads, self.contigs)
        self.assertEqual([s.name for s in samples], ["A_clean", "B", "L", "outside"])
        self.assertEqual(samples[0].read1, self.root / "machine a.fastq")
        self.assertEqual(samples[0].read2, self.root / "machine b.fastq.gz")
        self.assertIsNone(samples[2].read2)
        self.assertEqual(samples[0].contigs, self.root / "assembly.fasta")
        analyses = named_analyses(samples, [])
        self.assertEqual(analyses[0].name, "A_clean")
        self.assertEqual(analyses[0].public_name, "A_clean")
        self.assertTrue(all(not a.cross_mapped for a in analyses))

    def test_refinement_and_final_quality_thresholds_recorded_separately(self):
        for refiner, quality, relaxation in (
            ("magscot", "checkm", 0.2), ("magscot", "checkm2", 0.2),
            ("metawrap", "checkm", 0.0), ("metawrap", "checkm2", 0.2),
            ("das_tool", "checkm2", None),
        ):
            with self.subTest(refiner=refiner, quality=quality):
                _args, _tasks, payload = self.plan(extra=(
                    "--tools", "metabat2", "vamb", "--refinement", refiner,
                    "--quality-control", quality, "--com", "75", "--con", "5",
                    "--quality-score", "27",
                ))
                self.assertEqual(payload["quality_thresholds"], {
                    "minimum_completeness": 75, "maximum_contamination": 5,
                    "minimum_quality_score": 27,
                })
                if relaxation is None:
                    self.assertIsNone(payload["refinement_thresholds"])
                else:
                    self.assertEqual(payload["refinement_thresholds"], {
                        "minimum_completeness": 60 if relaxation else 75,
                        "maximum_contamination": 6 if relaxation else 5,
                        "relative_relaxation": relaxation,
                    })

    def test_bad_read_rows_fail_with_source_line(self):
        for row in ("x.fastq", "X\tx.fq.gz,", "X\tx.fq.gz,y.fq,u.fastq",
                    "X\tx.fq.gz\textra", "../X\tx.fq.gz", "X__mbw_long\tx.fq.gz",
                    "X\tx.fq.gz,x.fq.gz", "X\tx.fq.gz\nX\ty.fq",
                    "X\tx.fq.gz\nY\tx.fq.gz"):
            with self.subTest(row=row):
                self.reads.write_text(row)
                with self.assertRaisesRegex(ValueError, "reads.tsv:[12]:"):
                    read_named_reads(self.reads)

    def test_missing_non_fastq_and_empty_inputs(self):
        self.reads.write_text("X\tmissing.fastq\n")
        with self.assertRaises(FileNotFoundError):
            read_named_reads(self.reads)
        self.reads.write_text("X\tassembly.fasta\n")
        with self.assertRaisesRegex(ValueError, "unsupported suffix"):
            read_named_reads(self.reads)
        self.reads.write_text("#nothing\n\n")
        with self.assertRaisesRegex(ValueError, "contains no samples"):
            read_named_reads(self.reads)

    def test_contig_names_must_match_and_be_unique(self):
        self.contigs.write_text("A_clean\tassembly.fasta\nExtra\tassembly.fasta\n")
        with self.assertRaisesRegex(ValueError, "absent from --input_reads_files"):
            load_named_samples(self.reads, self.contigs)
        self.contigs.write_text("X\tassembly.fasta\nX\tassembly.fasta\n")
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            read_named_contigs(self.contigs)

    def test_multi_names_validation(self):
        samples = load_named_samples(self.reads, self.contigs)
        for text in ("A_clean\nA_clean\n", "A_clean\nunknown\n", "A_clean\n",
                     "A_clean\tx.fq.gz\tassembly.fasta\n", "#empty\n"):
            with self.subTest(text=text):
                self.multi.write_text(text)
                with self.assertRaises(ValueError):
                    read_multi_names(self.multi, samples)

    def test_coassembly_group_manifest_and_analysis_plan(self):
        samples = load_named_samples(self.reads, self.contigs)
        groups = read_coassembly_groups(self.coassembly, samples)
        self.assertEqual(groups, {"group1": ("A_clean", "B")})
        analyses = named_coassembly_analyses(
            samples, groups, self.root / "work" / "coassembly"
        )
        combined = analyses[0]
        self.assertEqual(combined.name, "coassembly_group1")
        self.assertEqual(combined.public_name, "group1_A_clean-B")
        self.assertEqual(combined.plan_id, "group1")
        self.assertEqual(
            [sample.name for sample in combined.assembly_samples],
            ["A_clean", "B"],
        )
        self.assertEqual(
            [analysis.name for analysis in analyses[1:]], ["L", "outside"]
        )

    def test_coassembly_group_manifest_rejects_invalid_membership(self):
        samples = load_named_samples(self.reads, self.contigs)
        for text, message in (
            ("A_clean\tgroup1\n", "at least two"),
            ("A_clean\tgroup1\nunknown\tgroup1\n", "unknown samples"),
            ("A_clean\tbad/group\nB\tbad/group\n", "invalid coassembly group"),
            ("A_clean\tgroup1\nA_clean\tgroup2\n", "duplicate sample"),
            ("A_clean group1\nB group1\n", "TAB-separated"),
        ):
            with self.subTest(text=text):
                self.coassembly.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    read_coassembly_groups(self.coassembly, samples)

    def test_coassembly_fasta_headers_use_the_declared_group_name(self):
        samples = load_named_samples(self.reads, self.contigs)
        groups = read_coassembly_groups(self.coassembly, samples)
        analysis = named_coassembly_analyses(samples, groups, self.root / "assemblies")[0]
        raw = self.root / "megahit.fa"
        renamed = self.root / "renamed.fa"
        raw.write_text(">k141_1\nAAAA\n>k141_2 description\nCCCC\n")
        rename_fasta_records(raw, renamed, analysis.plan_id)
        self.assertEqual(renamed.read_text().splitlines(), [
            ">group1_1", "AAAA", ">group1_2", "CCCC",
        ])

    def test_coassembly_strategy_builds_megahit_and_keeps_unlisted_independent(self):
        args, tasks, payload = self.plan(
            extra=(
                "--assembly-strategy", "coassembly",
                "--coassembly-file", str(self.coassembly),
                "--tools", "metabat2", "vamb",
            )
        )
        self.assertEqual(args.mode, "coassembly")
        self.assertEqual(payload["assembly_strategy"], "coassembly")
        self.assertEqual(payload["coassembly_groups"], {"group1": ["A_clean", "B"]})
        self.assertIn("megahit", {item.executable for item in bin_requirements(args)})
        by_id = {task.id: task for task in tasks}
        assembly = by_id["01.coassemble.coassembly_group1"]
        self.assertIn("megahit", assembly.display_command())
        self.assertIn(str(self.root / "machine a.fastq"), assembly.display_command())
        self.assertIn(str(self.root / "x.fq.gz"), assembly.display_command())
        self.assertIn("--prefix group1", assembly.display_command())
        self.assertIn("01.prepare.L", by_id)
        self.assertIn("01.prepare.outside", by_id)
        self.assertNotIn("01.prepare.A_clean", by_id)
        self.assertNotIn("01.prepare.B", by_id)
        self.assertIn("07.report.coassembly", by_id)
        combined = next(item for item in payload["analyses"] if item["combined"])
        self.assertEqual(combined["name"], "coassembly_group1")
        self.assertEqual(combined["samples"], ["A_clean", "B"])
        self.assertEqual(combined["public_name"], "group1_A_clean-B")
        self.assertEqual(
            combined["contigs"],
            [str(self.root / "results" / "coassembly" / "assemblies" /
                 "coassembly_group1" / "group1.contigs.ok.fa")],
        )
        self.assertIn(
            "--prefix group1_A_clean-B_metabat2",
            by_id["03.publish.metabat2.coassembly_group1"].display_command(),
        )

    def test_coassembly_option_validation(self):
        for extra, message in (
            (("--assembly-strategy", "coassembly"), "requires --coassembly-file"),
            (("--coassembly-file", str(self.coassembly)), "requires --assembly-strategy"),
            (("--assembly-strategy", "coassembly", "--coassembly-file",
              str(self.coassembly), "--multi-files", str(self.multi)),
             "cannot be combined"),
        ):
            with self.subTest(extra=extra), patch("metabaw.cli._preflight_bin") as preflight:
                with self.assertRaisesRegex(ValueError, message):
                    self.plan(extra=extra)
                preflight.assert_not_called()

    def test_all_samples_coassembled_without_contig_manifest(self):
        (self.root / "long_mate.fastq").touch()
        self.reads.write_text(
            self.reads.read_text(encoding="utf-8").replace(
                "L\tnanopore.fastq.gz\n", "L\tnanopore.fastq.gz,long_mate.fastq\n"
            ), encoding="utf-8",
        )
        self.coassembly.write_text(
            "A_clean\tgroup1\nB\tgroup1\nL\tgroup2\noutside\tgroup2\n"
        )
        args, tasks, payload = self.plan(
            include_contigs=False,
            extra=("--assembly-strategy", "coassembly", "--coassembly-file",
                   str(self.coassembly), "--tools", "metabat2"),
        )
        self.assertIsNone(args.input_contig_files)
        self.assertIsNone(payload["input_sources"]["contigs"])
        self.assertTrue(all(item["contigs"] is None for item in payload["input_samples"]))
        self.assertEqual(
            {item["name"] for item in payload["analyses"]},
            {"coassembly_group1", "coassembly_group2"},
        )
        self.assertTrue(all(item["combined"] for item in payload["analyses"]))
        by_id = {task.id: task for task in tasks}
        for group in ("group1", "group2"):
            self.assertIn("megahit", by_id[f"01.coassemble.coassembly_{group}"].display_command())
            self.assertEqual(
                by_id[f"01.prepare.coassembly_{group}"].deps,
                (f"01.coassemble.coassembly_{group}",),
            )
        for sample in ("A_clean", "B", "L", "outside"):
            self.assertNotIn(f"01.prepare.{sample}", by_id)
        self.assertTrue(all("None" not in task.display_command() for task in tasks))

    def test_partial_coassembly_needs_only_ungrouped_contigs(self):
        self.contigs.write_text("L\tassembly.fasta\noutside\tassembly.fasta\n")
        _args, tasks, payload = self.plan(
            extra=("--assembly-strategy", "coassembly", "--coassembly-file", str(self.coassembly)),
        )
        samples = {item["sample"]: item for item in payload["input_samples"]}
        for name in ("A_clean", "B"):
            self.assertIsNone(samples[name]["contigs"])
        by_id = {task.id: task for task in tasks}
        for name in ("L", "outside"):
            self.assertEqual(samples[name]["contigs"], str(self.root / "assembly.fasta"))
            self.assertIn(str(self.root / "assembly.fasta"), by_id[f"01.prepare.{name}"].display_command())

    def test_partial_coassembly_auto_assembles_unlisted_samples_without_contigs(self):
        args, tasks, payload = self.plan(
            include_contigs=False,
            extra=("--assembly-strategy", "coassembly", "--coassembly-file", str(self.coassembly)),
        )
        by_id = {task.id: task for task in tasks}
        self.assertIn("01.assemble.L", by_id)
        self.assertIn("flye --meta --nano-raw", by_id["01.assemble.L"].display_command())
        self.assertIn("01.assemble.outside", by_id)
        self.assertIn("megahit", by_id["01.assemble.outside"].display_command())
        self.assertEqual(args.required_assemblers, ["flye", "megahit"])
        self.assertEqual(payload["required_assemblers"], ["flye", "megahit"])

    def test_partial_contig_manifest_auto_assembles_missing_and_rejects_unknown(self):
        self.contigs.write_text("L\tassembly.fasta\n")
        _args, tasks, _payload = self.plan(
            extra=("--assembly-strategy", "coassembly", "--coassembly-file", str(self.coassembly))
        )
        self.assertIn("01.assemble.outside", {task.id for task in tasks})
        self.contigs.write_text(
            "L\tassembly.fasta\noutside\tassembly.fasta\nExtra\tassembly.fasta\n"
        )
        with patch("metabaw.cli._preflight_bin") as preflight:
            with self.assertRaisesRegex(ValueError, "absent from --input_reads_files"):
                self.plan(extra=("--assembly-strategy", "coassembly",
                                 "--coassembly-file", str(self.coassembly)))
            preflight.assert_not_called()

    def test_reads_only_auto_assembles_each_sample_with_matching_software(self):
        args, tasks, payload = self.plan(include_contigs=False)
        by_id = {task.id: task for task in tasks}
        for name in ("A_clean", "B", "outside"):
            command = by_id[f"01.assemble.{name}"].display_command()
            self.assertIn("megahit", command)
            self.assertIn(f"{name}_contig_ok.fa", command)
            self.assertIn(f"--prefix {name}", command)
        command = by_id["01.assemble.L"].display_command()
        self.assertIn("flye --meta --nano-raw", command)
        self.assertIn("L_contig_ok.fa", command)
        self.assertIn("--prefix L", command)
        self.assertEqual(args.required_assemblers, ["flye", "megahit"])
        self.assertEqual(payload["effective_assembly"], "individual")
        self.assertTrue(
            {"flye", "megahit"}
            <= {item.executable for item in bin_requirements(args)}
        )

    def test_all_flye_read_modes_are_forwarded_and_set_mapping_preset(self):
        self.reads.write_text("L\tnanopore.fastq.gz\n", encoding="utf-8")
        modes = {
            "--pacbio-raw": "map-pb",
            "--pacbio-corr": "map-pb",
            "--pacbio-hifi": "map-hifi",
            "--nano-raw": "map-ont",
            "--nano-corr": "map-ont",
            "--nano-hq": "map-ont",
        }
        for option, minimap_preset in modes.items():
            with self.subTest(option=option):
                args, tasks, payload = self.plan(
                    include_contigs=False,
                    extra=(option, "--tools", "vamb"),
                )
                by_id = {task.id: task for task in tasks}
                self.assertIn(
                    f"flye --meta {option}",
                    by_id["01.assemble.L"].display_command(),
                )
                self.assertIn(
                    f"minimap2 -x {minimap_preset}",
                    by_id["02.index.L"].display_command(),
                )
                self.assertEqual(args.flye_read_type, option)
                self.assertEqual(payload["flye_read_type"], option)
                self.assertEqual(args.required_assemblers, ["flye"])

    def test_long_read_group_is_coassembled_by_flye(self):
        self.reads.write_text(
            "L1\tnanopore.fastq.gz\nL2\tu.fastq\n",
            encoding="utf-8",
        )
        self.coassembly.write_text("L1\trumen\nL2\trumen\n")
        args, tasks, payload = self.plan(
            include_contigs=False,
            extra=("--assembly-strategy", "coassembly", "--coassembly-file",
                   str(self.coassembly), "--pacbio-corr", "--tools", "vamb"),
        )
        by_id = {task.id: task for task in tasks}
        command = by_id["01.coassemble.coassembly_rumen"].display_command()
        self.assertIn("flye --meta --pacbio-corr", command)
        self.assertIn(str(self.root / "nanopore.fastq.gz"), command)
        self.assertIn(str(self.root / "u.fastq"), command)
        self.assertIn("rumen.contigs.ok.fa", command)
        self.assertIn("--prefix rumen", command)
        publish = by_id["03.publish.vamb.coassembly_rumen"].display_command()
        self.assertIn("--prefix rumen_L1-L2_vamb", publish)
        self.assertEqual(args.required_assemblers, ["flye"])
        self.assertIn("flye", {item.executable for item in bin_requirements(args)})
        self.assertEqual(payload["analyses"][0]["assembly_software"], "Flye")

    def test_reads_only_multisample_semibin_waits_for_each_generated_assembly(self):
        _args, tasks, _payload = self.plan(
            grouped=True,
            include_contigs=False,
            extra=("--tools", "semibin2"),
        )
        by_id = {task.id: task for task in tasks}
        self.assertEqual(len(by_id), len(tasks))
        prepare = next(
            task for task in tasks
            if task.id.startswith("01.prepare.__mbw_semibin2_multisample_input")
        )
        self.assertEqual(
            set(prepare.deps),
            {"01.assemble.A_clean", "01.assemble.B"},
        )
        self.assertIn("01.assemble.L", by_id)
        self.assertIn("01.assemble.outside", by_id)
        for name in ("A_clean", "B", "L", "outside"):
            self.assertEqual(by_id[f"01.assemble.{name}"].sample, name)
            self.assertEqual(by_id[f"01.assemble.{name}"].cpus, _args.threads_per_task)
            self.assertEqual(
                by_id[f"01.prepare.{name}"].deps,
                (f"01.assemble.{name}",),
            )

    def test_binning_writes_all_step_details_before_execution(self):
        args, tasks, payload = self.plan(extra=(
            "--tools", *BINNER_ORDER, "--gunc", "--trna", "--rrna",
            "--assembly-strategy", "coassembly", "--coassembly-file", str(self.coassembly),
        ))
        args.max_memory = 1
        # Trigger a known memory floor so startup is tested without external tools.
        tasks[0] = replace(tasks[0], minimum_memory_gb=2)
        output = self.root / "documented"
        output.mkdir()
        (output / "resolved_parameters.tsv").write_text(
            "parameter\tvalue\nlegacy\ttrue\n",
            encoding="utf-8",
        )
        with (patch("metabaw.cli.Executor") as executor,
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            self.assertEqual(_run_direct(args, tasks, output, output / "tmp", payload), 1)
        executor.assert_not_called()
        report = (output / "workflow_details.md").read_text(encoding="utf-8")
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        recorded = {task["id"]: task for task in manifest["tasks"]}
        for task in tasks:
            self.assertEqual(recorded[task.id]["command"], task.display_command())
        self.assertIn("# Step1\n\nCoassembly", report)
        self.assertEqual(report.count("Software: MetaBAT2"), 1)
        self.assertNotIn(str(output), report)
        for software in ("MEGAHIT", "MetaBAT2", "MetaDecoder", "VAMB", "COMEBin", "SemiBin2",
                         "MAGScoT", "CheckM2", "GUNC", "tRNAscan-SE", "barrnap", "galah"):
            self.assertIn(software, report)
        self.assertNotIn("LorBin", report)  # No long-read sample is eligible here.
        for flag in ("--threshold", "--max_cont", "--min-completeness"):
            self.assertIn(flag, report)
        self.assertEqual(manifest["workflow_details"]["path"], str(output / "workflow_details.md"))
        self.assertEqual(manifest["result_files"]["start_info"], str(output / "start_info.txt"))
        self.assertNotIn("resolved_parameters", manifest["result_files"])
        self.assertFalse((output / "resolved_parameters.tsv").exists())
        self.assertIn("[MEMORY ESTIMATE]", (output / "start_info.txt").read_text())

    def test_coassembly_rejects_mixed_read_group(self):
        self.coassembly.write_text("A_clean\tgroup1\nL\tgroup1\n")
        with patch("metabaw.cli._preflight_bin") as preflight:
            with self.assertRaisesRegex(ValueError, "cannot mix paired short"):
                self.plan(extra=("--assembly-strategy", "coassembly",
                                 "--coassembly-file", str(self.coassembly)))
            preflight.assert_not_called()

    def test_coassembly_provenance_uses_generic_origin(self):
        bins = self.root / "bins"
        bins.mkdir()
        (bins / "coassembly_group1_metabat2_1.fa").write_text(">c\nAAAA\n")
        (bins / "outside_metabat2_1.fa").write_text(">c\nAAAA\n")
        report = self.root / "bin_provenance.tsv"
        report_coassembly_provenance(
            bins,
            report,
            "fa",
            ["coassembly_group1=A_clean,B|A_clean,B"],
            ["outside"],
        )
        with report.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        by_mag = {row["mag"]: row for row in rows}
        self.assertEqual(
            by_mag["coassembly_group1_metabat2_1.fa"]["assembly_origin"],
            "coassembly",
        )
        self.assertEqual(
            by_mag["outside_metabat2_1.fa"]["assembly_origin"],
            "individual_assembly",
        )

    def test_coassembly_provenance_matches_public_group_sample_bin_prefix(self):
        bins = self.root / "public_bins"
        bins.mkdir()
        filename = "group1_A_clean-B_metabat2_1.fa"
        (bins / filename).write_text(">group1_1\nAAAA\n")
        report = self.root / "public_provenance.tsv"
        report_coassembly_provenance(
            bins,
            report,
            "fa",
            ["group1_A_clean-B=group1|A_clean,B|A_clean,B"],
            [],
        )
        with report.open() as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(row["mag"], filename)
        self.assertEqual(row["assembly_origin"], "coassembly")
        self.assertEqual(row["assembly_id"], "group1")
        self.assertEqual(row["assembly_samples"], "A_clean,B")

    def test_all_tools_are_routed_per_sample_and_recorded(self):
        args, tasks, payload = self.plan(extra=("--tools", *BINNER_ORDER))
        self.assertEqual(payload["read_type"], "mixed")
        self.assertEqual(payload["failure_policy"], "strict")
        self.assertEqual(args.required_align_tools, ["bowtie2", "minimap2"])
        by_id = {task.id: task for task in tasks}
        self.assertEqual(len(by_id), len(tasks))
        for item in payload["analyses"]:
            excluded = "metabat2" if item["read_type"] == "long" else "lorbin"
            self.assertEqual(set(item["binners"]), set(BINNER_ORDER) - {excluded})
            self.assertNotIn(f"03.publish.{excluded}.{item['name']}", by_id)
            for tool in item["binners"]:
                task = by_id[f"03.publish.{tool}.{item['name']}"]
                self.assertIn(f"--prefix {item['name']}_{tool}", task.display_command())
        summary = self.root / "binner_summary.tsv"
        _write_binner_summary(summary, tasks, {}, {}, payload, self.root)
        with summary.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        excluded = [r for r in rows if r["status"] == "not_applicable"]
        self.assertEqual(len(rows), 4 * len(BINNER_ORDER))
        self.assertEqual(len(excluded), 4)
        self.assertTrue(all("excluded" in row["reason"] for row in excluded))

    def test_shared_contigs_do_not_implicitly_enable_multi(self):
        args, tasks, payload = self.plan(extra=("--tools", "semibin2", "vamb"))
        self.assertEqual(payload["mode"], "single")
        self.assertEqual(payload["multi_samples"], [])
        for item in payload["analyses"]:
            self.assertEqual(item["samples"], [item["name"]])
            self.assertFalse(item["cross_mapped"])
        maps = [task for task in tasks if task.id.startswith("02.map.")]
        self.assertEqual(len(maps), 4)
        self.assertFalse(any("multi_easy_bin" in t.display_command() for t in tasks))
        self.assertFalse(any("megahit" in t.display_command() for t in tasks))

    def test_mixed_group_crossmaps_every_member_and_isolates_unlisted_sample(self):
        args, tasks, payload = self.plan(True, ("--tools", *BINNER_ORDER))
        selected = ["A_clean", "B", "L"]
        by_id = {task.id: task for task in tasks}
        for name in selected:
            stage = by_id[f"02.bamset.{name}"]
            self.assertEqual(len(stage.deps), 2)
            for sample in selected:
                self.assertIn(f"{sample}=", stage.display_command())
            self.assertNotIn("outside=", stage.display_command())
            maps = [t for t in tasks if t.id.startswith(f"02.map.{name}__mbw_")]
            self.assertEqual(len(maps), 3)
            for task in maps:
                self.assertIn("minimap2 -ax map-ont" if task.id.endswith(".L")
                              else "bowtie2 --very-sensitive", task.display_command())
            vamb = by_id[f"03.bin.vamb.{name}"]
            self.assertIn("--bamdir", vamb.display_command())
            self.assertIn(stage.id, vamb.deps)
            self.assertFalse(any(t.id.startswith(f"03.aemb.{name}.") for t in tasks))
        independent = by_id["02.bamset.outside"]
        self.assertEqual(len(independent.inputs), 1)
        self.assertIn("--abundance_tsv", by_id["03.bin.vamb.outside"].display_command())
        semibin = [t for t in tasks if "multi_easy_bin" in t.display_command()]
        self.assertEqual(len(semibin), 1)
        singleton = by_id["03.bin.semibin2.L"]
        self.assertIn("single_easy_bin", singleton.display_command())
        self.assertIn("--sequencing-type long_read", singleton.display_command())
        self.assertIn("--self-supervised", singleton.display_command())
        self.assertEqual(len([p for p in singleton.inputs if p.suffix == ".bam"]), 3)
        self.assertTrue(all("--sequencing-type long_read" not in t.display_command() for t in semibin))
        self.assertTrue(all(len([p for p in t.inputs if p.suffix == ".bam"]) == 3 for t in semibin))
        self.assertTrue(all("--self-supervised" in t.display_command() for t in semibin))
        for task in tasks:
            if task.stage == "04_refinement":
                self.assertIn("03.binning.complete", task.wait_for)
        self.assertTrue(all(t.cpus <= args.threads for t in tasks))
        self.assertEqual(payload["assembly_strategy"], "existing-contigs")

    def test_flye_read_type_sets_minimap_preset_for_index_and_mapping(self):
        _, tasks, _ = self.plan(True, ("--align-tool", "minimap2", "--pacbio-hifi",
                                      "--tools", "vamb"))
        for task in tasks:
            if task.id.startswith("02.index.A_clean__mbw_"):
                self.assertIn("minimap2 -x sr" if task.id.endswith("short")
                              else "minimap2 -x map-hifi", task.display_command())
            if task.id.startswith("02.map.A_clean__mbw_"):
                self.assertIn("minimap2 -ax map-hifi" if task.id.endswith(".L")
                              else "minimap2 -ax sr", task.display_command())

    def test_dependency_union_uses_effective_aligners_and_coverage(self):
        # Remove independent sample: all VAMB targets now use mixed BAM coverage.
        self.reads.write_text(self.reads.read_text(encoding="utf-8").split("outside\t")[0], encoding="utf-8")
        self.contigs.write_text("A_clean\tassembly.fasta\nB\tassembly.fasta\nL\tassembly.fasta\n")
        args, _, _ = self.plan(True, ("--tools", "vamb"))
        commands = {r.executable for r in bin_requirements(args)}
        self.assertTrue({"bowtie2", "bowtie2-build", "minimap2", "vamb"} <= commands)
        self.assertNotIn("mixed", commands)
        self.assertNotIn("strobealign", commands)
        args, _, _ = self.plan(False, ("--tools", "vamb"))
        self.assertIn("strobealign", {r.executable for r in bin_requirements(args)})

    def test_invalid_mapper_and_empty_tool_eligibility_fail_before_preflight(self):
        for extra, message in ((("--align-tool", "bowtie2"), "cannot process long sample"),
                               (("--align-tool", "minibwa"), "cannot process short sample"),
                               (("--tools", "metabat2"), "No eligible binner for long sample"),
                               (("--tools", "lorbin"), "No eligible binner for short sample")):
            with self.subTest(extra=extra), patch("metabaw.cli._preflight_bin") as preflight:
                with self.assertRaisesRegex(ValueError, message):
                    self.plan(extra=extra)
                preflight.assert_not_called()

    def test_default_binners(self):
        self.assertEqual(eligible_binners("short", None), ("metabat2", "metadecoder", "vamb"))
        self.assertEqual(eligible_binners("long", None), ("metadecoder", "vamb", "lorbin"))

    def test_internal_materialization_preserves_declared_names(self):
        assembly = self.root / "prepared.fa"
        concatenate_fastas([f"A_clean={self.root / 'assembly.fasta'}"], assembly, 1500)
        mapping = self.root / "bins.tsv"
        mapping.write_text("bin1\tc\n")
        destination = self.root / "published"
        marker = destination / ".complete"
        materialize_bins(assembly, mapping, destination, "A_clean_metabat2", completion_marker=marker)
        self.assertTrue((destination / "A_clean_metabat2_1.fa").is_file())
        self.assertTrue(marker.is_file())
        combined = self.root / "combined.fa"
        concatenate_fastas([f"A_clean={assembly}", f"B={assembly}"], combined, 1500, separator=":")
        self.assertIn(">A_clean:c\n", combined.read_text())
        # Exercise staging and identity, without claiming these fixture bytes
        # are a real alignment or invoking an external mapper.
        bam = self.root / "fixture.bam"
        bam.write_bytes(b"fixture")
        Path(str(bam) + ".bai").write_bytes(b"fixture-index")
        bamset = self.root / "bamset"
        stage_bams([f"A_clean={bam}", f"L={bam}"], bamset)
        self.assertEqual((bamset / "A_clean.bam").read_bytes(), b"fixture")
        self.assertEqual((bamset / "L.bam.bai").read_bytes(), b"fixture-index")

    def test_one_short_one_long_group_uses_all_coverage_without_invalid_cohort(self):
        self.multi.write_text("A_clean\nL\n")
        _, tasks, _ = self.plan(True, ("--tools", "semibin2"))
        self.assertFalse(any("multi_easy_bin" in t.display_command() for t in tasks))
        for name in ("A_clean", "L"):
            task = next(t for t in tasks if t.id == f"03.bin.semibin2.{name}")
            self.assertEqual(len([p for p in task.inputs if p.suffix == ".bam"]), 2)
            self.assertIn("--self-supervised", task.display_command())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metabaw.model import Task
from metabaw.workflow_report import _compact_steps, _method_calls, _package_version, task_software, write_workflow_details


class WorkflowReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_nested_and_wrapped_tools_are_identified(self):
        for command, expected in (
            (("python", "-m", "metabaw.internal", "run-comebin", "--command",
              "conda run -n comebin run_comebin.sh -a 'assembly with spaces.fa' -b 128"),
             {"MetaBAW", "run_comebin.sh"}),
            (("python", "-m", "metabaw.internal", "run-dastool-refinement"),
             {"MetaBAW", "DAS_Tool", "diamond"}),
            (("python", "-m", "metabaw.internal", "rna-qc", "--trna"),
             {"MetaBAW", "tRNAscan-SE"}),
            ("minimap2 -ax map-ont contigs.fa reads.fq | samtools sort -o out.bam",
             {"minimap2", "samtools"}),
            (("coverm", "genome", "--mapper", "minimap2-sr"), {"coverm", "minimap2"}),
            ("flye --meta --nano-hq reads.fastq.gz --out-dir /work/flye --threads 16",
             {"flye"}),
        ):
            with self.subTest(command=command):
                task = Task("task", "stage", command, self.root)
                self.assertEqual(set(task_software(task)[0]), expected)

    def test_flye_report_keeps_read_type_but_omits_input_path(self):
        task = Task(
            "01.assemble.S1", "01_prepare",
            "flye --meta --pacbio-hifi /reads/S1.fastq.gz --out-dir /work/flye --threads 16",
            self.root,
        )
        self.assertEqual(
            _method_calls(task),
            [("flye", "--meta; --pacbio-hifi; --threads 16")],
        )

    def test_versions_use_metadata_from_the_actual_environment(self):
        prefix = self.root / "isolated"
        executable = prefix / "bin" / "checkm2"
        executable.parent.mkdir(parents=True)
        executable.touch()
        records = prefix / "conda-meta"
        records.mkdir()
        record = records / "checkm2-1.2.3-build.json"
        record.write_text(json.dumps({"name": "checkm2", "version": "1.2.3"}))
        self.assertEqual(_package_version(str(executable), "checkm2"), ("1.2.3", str(record.resolve())))
        self.assertEqual(_package_version(None, "checkm2")[0], "not detected")
        self.assertEqual(_package_version(str(executable), "uninstalled")[0], "not detected")

    def test_report_keeps_manifests_thresholds_order_and_skip_reasons(self):
        reads = self.root / "reads.tsv"
        content = "S1\t/reads/a.fastq,/reads/b.fastq\n"
        reads.write_text(content, encoding="utf-8")
        first = Task("01.prepare", "01_prepare", ("python", "-m", "metabaw.internal", "stage-annotation-genomes"),
                     self.root, outputs=(self.root / "genomes",))
        second = Task("02.coverage", "02_abundance", ("coverm", "genome", "--mapper", "minimap2-sr", "--min-covered-fraction", "0.75"),
                      self.root, deps=(first.id,), env={"OMP_NUM_THREADS": "2"}, cpus=2)
        args = argparse.Namespace(threads=8, niche_classify_method="occupancy", niche_abundance="tpm",
                                  _invocation=["metabaw", "annotation", "--input_reads_files", str(reads)],
                                  _invocation_cwd=str(self.root), func=lambda: None)
        payload = {"module": "annotation", "input_sources": {"reads": {"path": str(reads)}, "contigs": None},
                   "kegg": False, "cazy": False, "hydrogenase": False,
                   "niche": {"enabled": False, "reason": "insufficient samples"},
                   "gtdbtk_classification_skipped": True, "gtdbtk_result_source": "/existing/gtdb"}
        report_path = self.root / "workflow_details.md"
        with patch("metabaw.workflow_report.shutil.which", return_value=None):
            record = write_workflow_details(report_path, args, [second, first], payload)
        report = report_path.read_text(encoding="utf-8")
        self.assertEqual(record["invocation"], args._invocation)
        self.assertEqual(record["resolved_options"]["niche_classify_method"], "occupancy")
        self.assertEqual(record["resolved_options"]["niche_abundance"], "tpm")
        self.assertIn("--min-covered-fraction 0.75", report)
        self.assertIn("# Step1\n\nContig preparation", report)
        self.assertIn("# Step2\n\nAbundance estimation\n\nSoftware: CoverM", report)
        self.assertIn("Niche classification: skipped; insufficient samples", report)
        self.assertIn("reused validated GTDB-Tk results", report)
        self.assertIn("Planned methods", report)
        self.assertNotIn("func", record["resolved_options"])
        self.assertNotIn(str(reads), report)
        self.assertNotIn("```", report)
        self.assertEqual(record["input_manifests"][0]["content"], reads.read_bytes().decode("utf-8"))
        self.assertEqual(record["input_manifests"][0]["sha256"], sha256(reads.read_bytes()).hexdigest())
        self.assertEqual(len(record["metabaw_source_sha256"]), 64)

    def test_same_parameters_merge_but_different_methods_are_kept(self):
        tasks = [Task(f"bin.{n}", "03_binning", ("metabat2", "-i", f"/{n}/assembly.fa",
                    "-o", f"/{n}/bins", "-m", "1500", "-t", "8"), self.root) for n in (1, 2)]
        tasks.append(Task("bin.3", "03_binning", ("metabat2", "-m", "2000", "-t", "8"), self.root))
        report = _compact_steps(tasks, {"MetaBAT2": "2.17"}, {})
        self.assertEqual(report.count("Software: MetaBAT2"), 1)
        self.assertEqual(report.count("-m 1500"), 1)
        self.assertIn("-m 2000", report)
        self.assertIn("version=2.17", report)

    def test_paths_and_conda_env_names_are_not_software_calls(self):
        task = Task("qc", "05_quality", "mkdir -p /work/checkm2 && conda run --no-capture-output "
                    "--name checkm2 checkm2 predict --threads 8 --input /bins --output-directory /work/checkm2",
                    self.root)
        self.assertEqual(_method_calls(task), [("checkm2", "predict; --threads 8")])
        helper = Task("helper", "03_binning", ("python", "-m", "metabaw.internal", "bins-to-map",
                      "--label", "vamb", "--min-bin-bp", "200000"), self.root)
        self.assertEqual(_method_calls(helper), [])
        self.assertEqual(task_software(helper)[0], ["MetaBAW"])
        mapper = Task("map", "02_abundance", "mkdir -p /coverm && PATH=/compat:$PATH coverm genome "
                      "--mapper minimap2-sr --methods tpm rpkm --threads 8", self.root)
        self.assertEqual(_method_calls(mapper), [("coverm", "genome; --mapper minimap2-sr; --methods tpm rpkm; --threads 8")])

    def test_nested_comebin_and_parallel_keep_parameters(self):
        task = Task("comebin", "03_binning", ("python", "-m", "metabaw.internal", "run-comebin",
                    "--command", "conda run -n comebin run_comebin.sh -a '/my assembly.fa' -b 128 -t 8",
                    "--heartbeat-seconds", "1200"), self.root, gpus=1)
        self.assertEqual(_method_calls(task), [("run_comebin.sh", "-b 128; -t 8; device=GPU")])
        task = Task("prodigal", "04_refinement", "parallel --jobs 8 --recstart '>' --pipe "
                    "'prodigal -p meta -a /out.faa' < /assembly.fa", self.root)
        self.assertEqual(_method_calls(task), [("prodigal", "-p meta")])
        self.assertIn("prodigal", task_software(task)[0])

    def test_managed_magscot_keeps_software_and_native_thresholds(self):
        native = [
            "Rscript", "/installed tools/MAGScoT/MAGScoT.R",
            "-i", "/work/all_binners.tsv", "--hmm", "/work/markers.tsv", "-o", "LMI_6",
            "--threshold", "0.5", "--max_cont", "0.1", "--min_markers", "25",
        ]
        task = Task(
            "04.refine.magscot.LMI_6", "04_refinement",
            ("python", "-m", "metabaw.internal", "run-magscot",
             "--combined", "/work/all_binners.tsv", "--mapping", "/work/refined.out",
             "--run-log", "/work/magscot.native.log", "--native-command-json", json.dumps(native)),
            self.root,
        )
        software, notes = task_software(task)
        self.assertIn("MAGScoT.R", software)
        self.assertIn("zero refined MAGs", " ".join(notes))
        calls = dict(_method_calls(task))
        self.assertIn("--threshold 0.5", calls["MAGScoT.R"])
        self.assertIn("--max_cont 0.1", calls["MAGScoT.R"])
        report = _compact_steps([task], {}, {})
        self.assertIn("Software: MAGScoT", report)
        self.assertNotIn("/work", report)
        self.assertNotIn("native-command-json", report)

    def test_threads_before_positional_inputs_and_negative_values_survive(self):
        task = Task("index", "02_mapping", ("bowtie2-build", "--threads", "8", "/input.fa", "/index"), self.root)
        self.assertEqual(_method_calls(task), [("bowtie2-build", "--threads 8")])
        task = Task("hmm", "04_refinement", ("hmmsearch", "--cpu", "8", "-E", "1e-5", "--cut_nc",
                    "/db.hmm", "/proteins.faa"), self.root)
        self.assertEqual(_method_calls(task), [("hmmsearch", "--cpu 8; -E 1e-5; --cut_nc")])

    def test_niche_method_abundance_and_thresholds_are_retained(self):
        for method in ("cv", "occupancy"):
            for abundance in ("relative_abundance", "rpkm", "tpm", "mean"):
                task = Task("niche", "04_niche", ("python", "-m", "metabaw.internal",
                            "classify-niche-literature", "--method", method, "--abundance-method", abundance,
                            "--detection-percent", "0.01", "--min-total-reads", "20",
                            "--core-prevalence", "0.8", "--sample-dataset", "S1=default"), self.root)
                report = _compact_steps([task], {}, {})
                self.assertIn(f"--method {method}", report)
                self.assertIn(f"--abundance-method {abundance}", report)
                self.assertIn("--detection-percent 0.01", report)
                self.assertIn("--min-total-reads 20", report)
                self.assertIn("--core-prevalence 0.8", report)
                self.assertNotIn("S1=default", report)

    def test_niche_cv_correction_is_retained_in_workflow_details(self):
        for cv_correct in (False, True):
            with self.subTest(cv_correct=cv_correct):
                command = ("python", "-m", "metabaw.internal",
                           "classify-niche-literature", "--method", "cv",
                           "--abundance-method", "relative_abundance")
                if cv_correct:
                    command += ("--cv-correct",)
                command += ("--sample-dataset", "S1=default")
                report = _compact_steps([Task("niche", "04_niche", command, self.root)], {}, {})
                self.assertEqual("--cv-correct" in report, cv_correct)
                self.assertIn("--method cv", report)
                self.assertNotIn("S1=default", report)


if __name__ == "__main__":
    unittest.main()

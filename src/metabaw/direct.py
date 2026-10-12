from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable

from .discovery import Analysis, ReadSample, public_sample_name
from .model import Task, topological_order
from .metadecoder_coverage import metadecoder_executable
from .cupy_runtime import cupy_runtime_environment


BINNER_ORDER = (
    "metabat2",
    "metadecoder",
    "vamb",
    "comebin",
    "semibin2",
    "lorbin",
)
BINNER_PRIORITY = {name: (position + 1) * 10 for position, name in enumerate(BINNER_ORDER)}
BINNER_OUTPUT_LABELS = {
    "metabat2": "MetaBAT2",
    "metadecoder": "MetaDecoder",
    "vamb": "VAMB",
    "comebin": "COMEBin",
    "semibin2": "SemiBin2",
    "lorbin": "LorBin",
}
LORBIN_COMPAT_SCRIPT = Path(__file__).with_name("lorbin_compat.py")

GTDBTK_MEMORY_ESTIMATE_GIB = 160.0
GTDBTK_PPLACER_CPUS = 1
NICHE_MINIMUM_INPUTS = 4


def annotation_niche_eligible(genome_count: int, read_sample_count: int) -> bool:
    return genome_count >= NICHE_MINIMUM_INPUTS and read_sample_count >= NICHE_MINIMUM_INPUTS


CHECKM2_THREAD_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
LORBIN_PROCESS_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    # Do not force OMP_THREAD_LIMIT=1: PyTorch may later request a larger team,
    # leaving the requested and actual native team sizes inconsistent. The
    # LorBin launcher scopes runtime pool limits to CPU clustering instead.
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
COMEBIN_PROCESS_ENV = {
    # COMEBin launches Python DataLoader workers in addition to PyTorch's
    # explicitly configured intra-op pool.  Keep native BLAS/OpenMP pools from
    # multiplying inside every worker; run_comebin.sh from COMEBin 1.1.0 does
    # the same, while these variables also protect older 1.0.4 installations.
    "BLIS_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def safe_path_component(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in value
    )
    return cleaned.strip("._") or "sample"


def annotation_genome_name(path: Path, suffix: str) -> str:
    normalized_suffix = suffix if suffix.startswith(".") else "." + suffix
    name = path.name
    if name.lower().endswith(normalized_suffix.lower()):
        name = name[: -len(normalized_suffix)]
    return safe_path_component(name)


def analysis_output_name(analysis: Analysis) -> str:
    """Return the public MAG prefix without changing internal task identities."""
    return analysis.public_name or public_sample_name(analysis.name)


def analysis_tool_label(analysis: Analysis, tool: str) -> str:
    """Use CLI-stable tool names for explicitly mapped multi-sample inputs."""
    if analysis.explicit_mapping:
        return tool
    return BINNER_OUTPUT_LABELS[tool]


def internal(*arguments: str | Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "metabaw.internal", *(str(argument) for argument in arguments))


def parse_choices(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip().lower() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(parsed))


@dataclass(frozen=True)
class BinOptions:
    outdir: Path
    workdir: Path
    threads: int
    read_type: str
    align_tool: str
    flye_read_type: str
    binners: tuple[str, ...]
    min_contig_length: int
    min_fasta_kbs: int
    batch_size: int
    refiner: str
    quality_control: str
    min_completeness: float
    max_contamination: float
    min_quality_score: float | None
    run_gunc: bool
    run_trna: bool
    trna_pass: int | None
    run_rrna: bool
    rrna_pass: bool
    dereplicator: str
    environment: str | None
    tag_contigs: bool
    gpu: bool
    max_gpu_memory: str
    assembly_strategy: str
    magscot_dir: Path
    checkm2_db: Path | None
    gunc_db: Path | None
    gtdbtk_data: Path | None
    extra_args: dict[str, str]
    failure_policy: str = "strict"
    total_threads: int | None = None
    gpu_binners: tuple[str, ...] = ()
    comebin_explicit_device: bool = False
    ani: float = 99.0
    min_aligned_fraction: float = 30.0
    mag_suffix: str = "fa"
    comebin_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-comebin-py37",
    )
    checkm2_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-checkm2-py312",
    )
    metawrap_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-metawrap-py27",
    )
    lorbin_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-lorbin-py310",
    )

    @property
    def refinement_thresholds_relaxed(self) -> bool:
        """Relax native preselection, never the downstream quality filter."""
        return self.refiner == "magscot" or (
            self.refiner == "metawrap" and self.quality_control == "checkm2"
        )

    @property
    def refinement_thresholds(self) -> tuple[float, float]:
        """Return native completeness/contamination thresholds in percent."""
        if self.refinement_thresholds_relaxed:
            return self.min_completeness * 0.8, min(100.0, self.max_contamination * 1.2)
        return self.min_completeness, self.max_contamination


class DirectBinBuilder:
    def __init__(self, analyses: list[Analysis], options: BinOptions, root: Path):
        failure_policy = getattr(options, "failure_policy", "strict")
        if failure_policy not in {"strict", "best-effort"}:
            raise ValueError(
                "failure_policy must be 'strict' or 'best-effort', got "
                f"{failure_policy!r}"
            )
        self.analyses = analyses
        self.options = options
        self.failure_policy = failure_policy
        self.root = root
        self.tasks: list[Task] = []
        self._semibin_multisample_outputs: dict[str, tuple[Task, Path]] = {}
        self._metawrap_quality_sources: list[tuple[Path, Path]] = []
        self._active_sample: str | None = None
        self._assembly_tasks: dict[str, Task] = {}

    def add(self, task: Task) -> Task:
        if task.sample is None and self._active_sample is not None:
            task = replace(task, sample=self._active_sample)
        if self.failure_policy == "strict":
            if task.stage == "03_binning":
                task = replace(
                    task,
                    allow_failed_deps=False,
                    failure_tolerated=False,
                )
            elif task.stage == "04_refinement" and task.allow_failed_deps:
                task = replace(task, allow_failed_deps=False)
        self.tasks.append(task)
        return task

    def build(self) -> list[Task]:
        self._prepare_semibin_multisample()
        refined: list[tuple[Task, Path]] = []
        for analysis in self.analyses:
            refined.append(self._analysis(analysis))
        self._add_binning_barrier()
        catalog = self._catalog(refined)
        check_task, check_report = self._quality(catalog)
        gunc_task = self._gunc(catalog)
        rna_task, rna_summary = self._rna(catalog)
        filtered_task, filtered = self._filter(
            catalog, check_task, check_report, gunc_task, rna_task, rna_summary
        )
        final_task, final_dir = self._dereplicate(filtered_task, filtered, check_report)
        completion_task = self._dereplication_report(final_task, final_dir)
        if self.options.assembly_strategy == "coassembly":
            completion_task = self._coassembly_report(completion_task, final_dir)
        if self.options.tag_contigs:
            tagged = self.options.outdir / "non_redundant_bins"
            legacy_untagged = self.options.outdir / "non_redundant_bins.untagged"
            marker = self.options.workdir / "dereplication" / "tag_contigs.complete"
            tag_command = " ".join(
                q(argument)
                for argument in internal(
                    "tag-contigs",
                    "--input-dir",
                    final_dir,
                    "--output-dir",
                    tagged,
                    "--suffix",
                    self.options.mag_suffix,
                )
            )
            self.add(
                Task(
                    id="07.tag_contigs",
                    stage="07_catalog",
                    command=(
                        f"rm -rf {q(legacy_untagged)} && {tag_command} && "
                        f"mkdir -p {q(marker.parent)} && touch {q(marker)}"
                    ),
                    cwd=self.root,
                    deps=(completion_task.id,),
                    inputs=(final_dir,),
                    outputs=(tagged, marker),
                    description="Rename contigs to BIN_1, BIN_2, ...",
                )
            )
        return topological_order(self.tasks)

    def _add_binning_barrier(self) -> None:
        """Prevent refinement until every sample-specific binning task is terminal."""
        binning_dependencies = tuple(
            task.id for task in self.tasks if task.stage == "03_binning"
        )
        if not binning_dependencies:
            return
        marker = self.options.workdir / "binning" / "binning.complete"
        barrier = self.add(
            Task(
                id="03.binning.complete",
                stage="03_binning",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        "path=Path(sys.argv[1]); "
                        "path.parent.mkdir(parents=True, exist_ok=True); "
                        "path.write_text('complete\\n')"
                    ),
                    str(marker),
                ),
                cwd=self.root,
                deps=binning_dependencies,
                outputs=(marker,),
                allow_failed_deps=True,
                failure_tolerated=True,
                priority=1000,
                description=(
                    "Confirm all selected binners finished across all samples"
                ),
            )
        )
        self.tasks = [
            (
                replace(task, wait_for=(*task.wait_for, barrier.id))
                if task.stage == "04_refinement"
                and barrier.id not in task.wait_for
                else task
            )
            for task in self.tasks
        ]

    def _analysis(self, analysis: Analysis) -> tuple[Task, Path]:
        self._active_sample = analysis.name
        prepared = self.options.workdir / "contigs" / f"{analysis.name}.fna"
        if analysis.assembly_samples:
            assembly = analysis.contigs[0]
            assembly_task = self._assembly_task(analysis, assembly)
            prepare = self.add(
                Task(
                    id=f"01.prepare.{analysis.name}",
                    stage="01_prepare",
                    command=internal(
                        "fasta-filter",
                        "--input",
                        assembly,
                        "--output",
                        prepared,
                        "--min-length",
                        str(self.options.min_contig_length),
                    ),
                    cwd=self.root,
                    deps=(assembly_task.id,),
                    inputs=(assembly,),
                    outputs=(prepared,),
                    description=f"Filter generated assembly contigs for {analysis.name}",
                )
            )
        else:
            prepare_args: list[str | Path] = ["concat-fasta"]
            labels: list[str] = []
            if len(analysis.contigs) == 1:
                labels = [analysis.name]
            else:
                by_contig: dict[Path, str] = {}
                for sample in analysis.samples:
                    if sample.contigs is not None:
                        by_contig.setdefault(sample.contigs, sample.name)
                labels = [by_contig[path] for path in analysis.contigs]
            for label, path in zip(labels, analysis.contigs, strict=True):
                prepare_args.extend(("--input", f"{label}={path}"))
            prepare_args.extend(("--output", prepared, "--min-length", str(self.options.min_contig_length)))
            prepare = self.add(
                Task(
                    id=f"01.prepare.{analysis.name}",
                    stage="01_prepare",
                    command=internal(*prepare_args),
                    cwd=self.root,
                    inputs=analysis.contigs,
                    outputs=(prepared,),
                    description=f"Filter and prepare contigs for {analysis.name}",
                )
            )
        bam_task, bams, bam_dir = self._mapping(analysis, prepare, prepared)
        maps = self._binners(analysis, prepare, prepared, bam_task, bams, bam_dir)
        semibin_multisample = self._semibin_multisample_outputs.get(
            analysis.name
        )
        if semibin_multisample is not None:
            run, bins = semibin_multisample
            mapping = (
                self.options.workdir
                / "binning"
                / analysis.name
                / "semibin2"
                / "contig_to_bin.tsv"
            )
            map_task = self._map_bins(analysis, "semibin2", run, bins, mapping)
            maps.append(
                self._publish_binner(
                    analysis,
                    "semibin2",
                    prepare,
                    prepared,
                    map_task,
                    mapping,
                )
            )
        result = self._refine(analysis, prepare, prepared, maps)
        self._active_sample = None
        return result

    def _assembly_task(self, analysis: Analysis, assembly: Path) -> Task:
        task = self._assembly_tasks.get(analysis.name)
        if task is None:
            task = self._coassemble(analysis, assembly)
            self._assembly_tasks[analysis.name] = task
        return task

    def _coassemble(self, analysis: Analysis, assembly: Path) -> Task:
        """Create an individual or shared assembly from the supplied reads."""
        root = assembly.parent
        assembly_samples = analysis.assembly_samples
        sequence_prefix = analysis.plan_id if analysis.combined else analysis.name
        if self.options.read_type == "short":
            read1 = ",".join(str(sample.read1) for sample in assembly_samples)
            if not all(sample.read2 is not None for sample in assembly_samples):
                raise ValueError(
                    f"Coassembly {analysis.name} requires paired short reads"
                )
            read2 = ",".join(
                str(sample.read2) for sample in assembly_samples if sample.read2
            )
            reads = f"-1 {q(read1)} -2 {q(read2)}"
            rename_command = " ".join(
                q(argument)
                for argument in internal(
                    "rename-contigs",
                    "--input",
                    root / "megahit" / "final.contigs.fa",
                    "--output",
                    assembly,
                    "--prefix",
                    sequence_prefix,
                )
            )
            command = (
                f"rm -rf {q(root)} && mkdir -p {q(root)} && megahit {reads} "
                f"--out-dir {q(root / 'megahit')} --num-cpu-threads {self.options.threads} "
                '--memory "${METABAW_MEGAHIT_MEMORY_BYTES:-0.9}" '
                f"--min-contig-len {self.options.min_contig_length}{self._extra('megahit')} && "
                f"{rename_command}"
            )
            tool = "MEGAHIT"
        elif self.options.read_type == "long":
            if any(sample.read2 is not None for sample in assembly_samples):
                raise ValueError(f"Flye assembly {analysis.name} requires single-file long reads")
            flye_reads = " ".join(q(sample.read1) for sample in assembly_samples)
            rename_command = " ".join(
                q(argument)
                for argument in internal(
                    "rename-contigs",
                    "--input",
                    root / "flye" / "assembly.fasta",
                    "--output",
                    assembly,
                    "--prefix",
                    sequence_prefix,
                )
            )
            command = (
                f"rm -rf {q(root)} && mkdir -p {q(root)} && "
                f"flye --meta {self.options.flye_read_type} {flye_reads} "
                f"--out-dir {q(root / 'flye')} --threads {self.options.threads}"
                f"{self._extra('flye')} && {rename_command}"
            )
            tool = "Flye"
        else:
            raise ValueError(f"Unsupported assembly read type: {self.options.read_type}")
        task_kind = "coassemble" if analysis.combined else "assemble"
        sample_word = "samples" if len(assembly_samples) != 1 else "sample"
        return self.add(
            Task(
                id=f"01.{task_kind}.{analysis.name}",
                stage="01_prepare",
                command=command,
                cwd=self.root,
                sample=analysis.name,
                inputs=tuple(
                    read
                    for sample in assembly_samples
                    for read in sample.reads
                ),
                outputs=(assembly,),
                cpus=self.options.threads,
                description=(
                    f"Create {'coassembly' if analysis.combined else 'assembly'} "
                    f"{analysis.name} with {tool} from {len(assembly_samples)} {sample_word}"
                ),
            )
        )

    def _coassembly_report(self, final_task: Task, final_dir: Path) -> Task:
        """Publish final-MAG provenance for coassembled and individual paths."""
        output = self.options.outdir / "coassembly" / "bin_provenance.tsv"
        arguments: list[str | Path] = [
            "report-coassembly-provenance",
            "--bins-dir",
            final_dir,
            "--output",
            output,
            "--suffix",
            self.options.mag_suffix,
        ]
        for analysis in self.analyses:
            if analysis.combined:
                assembly_samples = ",".join(
                    sample.name for sample in analysis.assembly_samples
                )
                recovery_samples = ",".join(
                    sample.name for sample in analysis.samples
                )
                arguments.extend(
                    (
                        "--coassembly",
                        f"{analysis_output_name(analysis)}={analysis.plan_id}|"
                        f"{assembly_samples}|{recovery_samples}",
                    )
                )
            else:
                arguments.extend(("--individual", analysis.name))
        return self.add(
            Task(
                id="07.report.coassembly",
                stage="07_catalog",
                command=internal(*arguments),
                cwd=self.root,
                deps=(final_task.id,),
                inputs=(final_dir,),
                outputs=(output,),
                description=(
                    "Report which final MAGs came from "
                    "coassemblies and which came from individual assemblies"
                ),
            )
        )

    def _prepare_semibin_multisample(self) -> None:
        """Build SemiBin2's concatenated input from independently assembled samples."""
        if "semibin2" not in self.options.binners:
            return
        analyses = [
            analysis
            for analysis in self.analyses
            if analysis.cross_mapped and not analysis.combined
            and "semibin2" in self._options_for_analysis(analysis).binners
        ]
        if not analyses:
            return
        cohorts: dict[tuple[tuple[str, ...], str], list[Analysis]] = {}
        for analysis in analyses:
            cohort = (tuple(sample.name for sample in analysis.samples),
                      self._options_for_analysis(analysis).read_type)
            cohorts.setdefault(cohort, []).append(analysis)

        multiple_cohorts = len(cohorts) > 1
        original_options = self.options
        for cohort_number, cohort_analyses in enumerate(
            cohorts.values(),
            start=1,
        ):
            if not self._use_semibin_cohort(cohort_analyses):
                continue
            self.options = self._options_for_analysis(cohort_analyses[0])
            combined_name = self._semibin_input_name()
            if multiple_cohorts:
                combined_name += f"_{cohort_number:03d}"
            self._active_sample = combined_name
            combined = (
                self.options.workdir
                / "contigs"
                / f"{combined_name}.fna"
            )
            prepare_args: list[str | Path] = ["concat-fasta"]
            contig_paths: list[Path] = []
            assembly_dependencies: list[str] = []
            for analysis in cohort_analyses:
                if len(analysis.contigs) != 1:
                    raise ValueError(
                        "SemiBin2 multi-sample mode requires one assembly "
                        "per analysis"
                    )
                contig = analysis.contigs[0]
                contig_paths.append(contig)
                prepare_args.extend(("--input", f"{analysis.name}={contig}"))
                if analysis.assembly_samples:
                    assembly_dependencies.append(
                        self._assembly_task(analysis, contig).id
                    )
            prepare_args.extend(
                (
                    "--output",
                    combined,
                    "--min-length",
                    str(self.options.min_contig_length),
                    "--separator",
                    ":",
                )
            )
            prepare = self.add(
                Task(
                    id=f"01.prepare.{combined_name}",
                    stage="01_prepare",
                    command=internal(*prepare_args),
                    cwd=self.root,
                    deps=tuple(assembly_dependencies),
                    inputs=tuple(contig_paths),
                    outputs=(combined,),
                    description=(
                        "Prepare sample-labelled contigs for SemiBin2 "
                        "multi-sample binning"
                    ),
                )
            )

            mapping_analysis = Analysis(
                combined_name,
                cohort_analyses[0].samples,
                tuple(contig_paths),
                False,
                preserve_name=True,
            )
            bam_task, bams, _bam_dir = self._mapping(
                mapping_analysis,
                prepare,
                combined,
            )
            tool_root = (
                self.options.workdir
                / "binning"
                / combined_name
                / "semibin2"
            )
            sequence_type = (
                " --sequencing-type long_read"
                if self.options.read_type == "long"
                else ""
            )
            semibin_gpu = self._uses_gpu("semibin2")
            engine = " --engine gpu" if semibin_gpu else " --engine cpu"
            cohort_threads = self._cohort_threads()
            resolved_root = tool_root / "resolved_samples"
            sample_bins = tuple(
                resolved_root / analysis.name
                for analysis in cohort_analyses
            )
            completion = tool_root / "semibin2.complete"
            resolve_args: list[str | Path] = [
                "resolve-semibin-multisample",
                "--tool-root",
                tool_root,
                "--output-root",
                resolved_root,
                "--completion-marker",
                completion,
            ]
            for analysis in cohort_analyses:
                resolve_args.extend(("--sample", analysis.name))
            resolve_command = shlex.join(internal(*resolve_args))
            run_id = "03.bin.semibin2.multi"
            if multiple_cohorts:
                run_id += f".{cohort_number:03d}"
            run = self.add(
                Task(
                    id=run_id,
                    stage="03_binning",
                    command=(
                        f"rm -rf {q(tool_root)} && SemiBin2 multi_easy_bin "
                        f"-i {q(combined)} "
                        f"-b {' '.join(q(path) for path in bams)} "
                        f"-o {q(tool_root)} --threads {cohort_threads} "
                        f"--min-len {self.options.min_contig_length} "
                        f"--separator : --self-supervised"
                        f"{sequence_type}{engine}{self._extra('semibin2')} && "
                        f"{resolve_command}"
                    ),
                    cwd=self.root,
                    deps=(prepare.id, bam_task.id),
                    inputs=(combined, *bams),
                    outputs=(completion, *sample_bins),
                    fasta_output_dirs=sample_bins,
                    automatic_retries=1,
                    cpus=cohort_threads,
                    gpus=1 if semibin_gpu else 0,
                    priority=BINNER_PRIORITY["semibin2"],
                    failure_tolerated=True,
                    env=self._gpu_env("semibin2"),
                    description=(
                        "Run SemiBin2 multi_easy_bin for "
                        f"{len(cohort_analyses)} independently assembled samples"
                    ),
                )
            )
            self._semibin_multisample_outputs.update(
                {
                    analysis.name: (run, bins)
                    for analysis, bins in zip(
                        cohort_analyses,
                        sample_bins,
                        strict=True,
                    )
                }
            )
        self._active_sample = None
        self.options = original_options

    def _options_for_analysis(self, analysis: Analysis) -> BinOptions:
        return self.options

    def _semibin_input_name(self) -> str:
        return "semibin2_multisample_input"

    def _use_semibin_cohort(self, analyses: list[Analysis]) -> bool:
        return True

    def _vamb_uses_aemb(self, analysis: Analysis) -> bool:
        return self.options.read_type == "short"

    def _mapping(
        self, analysis: Analysis, prepare: Task, contigs: Path
    ) -> tuple[Task, list[Path], Path]:
        root = self.options.workdir / "mapping" / analysis.name
        index_marker = root / f"{self.options.align_tool}.index.done"
        index_outputs: tuple[Path, ...] = (index_marker,)
        index_alternatives: tuple[tuple[Path, ...], ...] = ()
        if self.options.align_tool == "bowtie2":
            prefix = root / "index" / contigs.stem
            index_alternatives = tuple(
                tuple(
                    Path(f"{prefix}.{part}.{extension}")
                    for part in ("1", "2", "3", "4", "rev.1", "rev.2")
                )
                for extension in ("bt2", "bt2l")
            )
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"bowtie2-build --threads {self.options.threads} "
                f"{q(contigs)} {q(prefix)} && touch {q(index_marker)}"
            )
        elif self.options.align_tool == "minimap2":
            prefix = root / "index" / f"{contigs.stem}.mmi"
            index_outputs = (index_marker, prefix)
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"minimap2 -x {'sr' if self.options.read_type == 'short' else self._long_read_mapping_preset()} "
                f"-t {self.options.threads} -d {q(prefix)} "
                f"{q(contigs)} && touch {q(index_marker)}"
            )
        else:
            prefix = root / "index" / contigs.stem
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"minibwa index -t {self.options.threads} "
                f"{q(contigs)} {q(prefix)} && touch {q(index_marker)}"
            )
        index = self.add(
            Task(
                id=f"02.index.{analysis.name}",
                stage="02_mapping",
                command=index_command,
                cwd=self.root,
                deps=(prepare.id,),
                inputs=(contigs,),
                outputs=index_outputs,
                output_alternatives=index_alternatives,
                cpus=self.options.threads,
                description=f"Build {self.options.align_tool} index for {analysis.name}",
            )
        )

        bams: list[Path] = []
        map_tasks: list[Task] = []
        for sample in analysis.samples:
            bam_name = (
                f"{sample.name}_to_{analysis.name}" if analysis.cross_mapped else sample.name
            )
            bam = root / "bam" / f"{bam_name}.bam"
            reads = " ".join(q(path) for path in sample.reads)
            filter_pipe = (
                f"samtools view -@ {self.options.threads} -b -q 10 -F 3588 - | "
                f"samtools sort -@ {self.options.threads} -o {q(bam)} - && "
                f"samtools index -@ {self.options.threads} {q(bam)}"
            )
            if self.options.align_tool == "bowtie2":
                read_args = (
                    f"-1 {q(sample.read1)} -2 {q(sample.read2)}"
                    if sample.read2
                    else f"-U {q(sample.read1)}"
                )
                mapper = (
                    f"bowtie2 --very-sensitive -p {self.options.threads} -x {q(prefix)} "
                    f"{read_args} 2> {q(root / 'bam' / (bam_name + '.bowtie2.log'))}"
                )
            elif self.options.align_tool == "minimap2":
                preset = "sr" if self.options.read_type == "short" else self._long_read_mapping_preset()
                mapper = f"minimap2 -ax {preset} -t {self.options.threads} {q(prefix)} {reads}"
            else:
                mapper = f"minibwa map -t {self.options.threads} {q(prefix)} {reads}"
            task = self.add(
                Task(
                    id=f"02.map.{analysis.name}.{sample.name}",
                    stage="02_mapping",
                    command=f"mkdir -p {q(bam.parent)} && {mapper} | {filter_pipe}",
                    cwd=self.root,
                    deps=(index.id,),
                    inputs=(contigs, *sample.reads),
                    outputs=(bam, Path(str(bam) + ".bai")),
                    cpus=self.options.threads,
                    description=f"Map {sample.name} reads to {analysis.name}",
                )
            )
            bams.append(bam)
            map_tasks.append(task)
        bam_dir = root / "bamset"
        stage_args: list[str | Path] = ["stage-bams"]
        staged_outputs: list[Path] = [bam_dir]
        for sample, bam in zip(analysis.samples, bams, strict=True):
            stage_args.extend(("--bam", f"{sample.name}={bam}"))
            staged_bam = bam_dir / f"{safe_path_component(sample.name)}.bam"
            staged_outputs.extend((staged_bam, Path(str(staged_bam) + ".bai")))
        stage_args.extend(("--output-dir", bam_dir))
        bam_task = self.add(
            Task(
                id=f"02.bamset.{analysis.name}",
                stage="02_mapping",
                command=internal(*stage_args),
                cwd=self.root,
                deps=tuple(task.id for task in map_tasks),
                inputs=tuple(bams),
                outputs=tuple(staged_outputs),
                description=f"Stage BAMs for {analysis.name}",
            )
        )
        return bam_task, bams, bam_dir

    def _binners(
        self,
        analysis: Analysis,
        prepare: Task,
        contigs: Path,
        bam_task: Task,
        bams: list[Path],
        bam_dir: Path,
    ) -> list[tuple[str, Task, Path, Path]]:
        results: list[tuple[str, Task, Path, Path]] = []
        root = self.options.workdir / "binning" / analysis.name
        for binner in self.options.binners:
            priority = BINNER_PRIORITY[binner]
            if binner == "metabat2":
                depth = root / binner / "depth.tsv"
                depth_task = self.add(
                    Task(
                        id=f"03.depth.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"mkdir -p {q(depth.parent)} && jgi_summarize_bam_contig_depths "
                            f"--outputDepth {q(depth)} {' '.join(q(path) for path in bams)}"
                        ),
                        cwd=self.root,
                        deps=(bam_task.id,),
                        inputs=tuple(bams),
                        outputs=(depth,),
                        priority=priority,
                        failure_tolerated=True,
                        description="Calculate MetaBAT2 depth matrix",
                    )
                )
                bins = root / binner / "bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(bins)} && mkdir -p {q(bins)} && metabat2 -i {q(contigs)} "
                            f"-a {q(depth)} -o {q(bins / (analysis.name + '.metabat2'))} "
                            f"-m {max(1500, self.options.min_contig_length)} "
                            f"-s {self.options.min_fasta_kbs * 1000} "
                            f"--seed 42 -t {self.options.threads}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, depth_task.id),
                        inputs=(contigs, depth),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description=f"Run MetaBAT2 for {analysis.name}",
                    )
                )
                mapping = root / binner / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "vamb":
                vamb_root = root / binner / "run"
                vamb_ipc = root / binner / "ipc"
                deps = [prepare.id]
                inputs: list[Path] = [contigs]
                abundance_arg: str
                if self._vamb_uses_aemb(analysis):
                    aemb_tasks: list[Task] = []
                    aemb_paths: list[Path] = []
                    for sample in analysis.samples:
                        aemb = root / binner / "aemb" / f"{sample.name}.tsv"
                        aemb_command: list[str | Path] = [
                            "run-strobealign-aemb",
                            "--contigs",
                            contigs,
                        ]
                        for read in sample.reads:
                            aemb_command.extend(("--read", read))
                        aemb_command.extend(
                            ("--output", aemb, "--threads", str(self.options.threads))
                        )
                        aemb_tasks.append(
                            self.add(
                                Task(
                                    id=f"03.aemb.{analysis.name}.{sample.name}",
                                    stage="03_binning",
                                    command=internal(*aemb_command),
                                    cwd=self.root,
                                    deps=(prepare.id,),
                                    inputs=(contigs, *sample.reads),
                                    outputs=(aemb,),
                                    automatic_retries=1,
                                    cpus=self.options.threads,
                                    priority=priority,
                                    failure_tolerated=True,
                                    description=f"Estimate {sample.name} abundance with strobealign AEMB",
                                )
                            )
                        )
                        aemb_paths.append(aemb)
                    abundance = root / binner / "abundance.tsv"
                    merge_args: list[str | Path] = ["merge-aemb"]
                    for sample, path in zip(analysis.samples, aemb_paths, strict=True):
                        merge_args.extend(("--input", f"{sample.name}={path}"))
                    merge_args.extend(("--output", abundance))
                    merge = self.add(
                        Task(
                            id=f"03.aemb.merge.{analysis.name}",
                            stage="03_binning",
                            command=internal(*merge_args),
                            cwd=self.root,
                            deps=tuple(task.id for task in aemb_tasks),
                            inputs=tuple(aemb_paths),
                            outputs=(abundance,),
                            priority=priority,
                            failure_tolerated=True,
                            description="Merge AEMB profiles with mandatory contigname header",
                        )
                    )
                    deps.append(merge.id)
                    inputs.append(abundance)
                    abundance_arg = f"--abundance_tsv {q(abundance)}"
                else:
                    deps.append(bam_task.id)
                    inputs.append(bam_dir)
                    abundance_arg = f"--bamdir {q(bam_dir)}"
                vamb_gpu = self._uses_gpu("vamb")
                cuda = " --cuda" if vamb_gpu else ""
                bins = vamb_root / "bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(vamb_root)} {q(vamb_ipc)} && "
                            f"mkdir -p {q(vamb_root.parent)} {q(vamb_ipc)} && "
                            "vamb_ipc_link=/tmp/mbw-vamb-$$ && "
                            "rm -f \"$vamb_ipc_link\" && "
                            f"ln -s {q(vamb_ipc)} \"$vamb_ipc_link\" && "
                            "trap 'rm -f \"$vamb_ipc_link\"' EXIT && "
                            "export TMPDIR=\"$vamb_ipc_link\" TMP=\"$vamb_ipc_link\" "
                            "TEMP=\"$vamb_ipc_link\" && "
                            f"vamb bin default --outdir {q(vamb_root)} --fasta {q(contigs)} "
                            f"{abundance_arg} -m {self.options.min_contig_length} "
                            f"--minfasta {self.options.min_fasta_kbs * 1000} --seed 42 "
                            f"-o -p {self.options.threads}{cuda}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=tuple(deps),
                        inputs=tuple(inputs),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if vamb_gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env("vamb"),
                        description=f"Run VAMB for {analysis.name}",
                    )
                )
                mapping = root / binner / "contig_to_bin.tsv"
                map_task = self._map_bins(
                    analysis,
                    binner,
                    run,
                    bins,
                    mapping,
                    self.options.min_fasta_kbs * 1000,
                )
            elif binner == "metadecoder":
                metadecoder = q(metadecoder_executable() or "metadecoder")
                cupy_env = cupy_runtime_environment()
                tool_root = root / binner
                coverage = tool_root / "coverage.tsv"
                coverage_marker = tool_root / "coverage.v2.complete"
                seeds = tool_root / "seed.tsv"
                cov_task = self.add(
                    Task(
                        id=f"03.coverage.{binner}.{analysis.name}",
                        stage="03_binning",
                        env=cupy_env,
                        command=internal(
                            "run-metadecoder-coverage",
                            *(argument for path in bams for argument in ("--bam", path)),
                            "--output", coverage,
                            "--completion-marker", coverage_marker,
                            "--threads", self.options.threads,
                        ),
                        cwd=self.root,
                        deps=(bam_task.id,),
                        inputs=tuple(bams),
                        outputs=(coverage, coverage_marker),
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description="Calculate MetaDecoder coverage",
                    )
                )
                seed_task = self.add(
                    Task(
                        id=f"03.seed.{binner}.{analysis.name}",
                        stage="03_binning",
                        env=cupy_env,
                        command=(
                            f"mkdir -p {q(tool_root)} && {metadecoder} seed --threads {self.options.threads} "
                            f"-f {q(contigs)} -o {q(seeds)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id,),
                        inputs=(contigs,),
                        outputs=(seeds,),
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description="Find MetaDecoder marker seeds",
                    )
                )
                bins = tool_root / "bins"
                metadecoder_gpu = self._uses_gpu("metadecoder")
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(bins)} && mkdir -p {q(bins)} && {metadecoder} cluster "
                            f"-f {q(contigs)} -c {q(coverage)} -s {q(seeds)} "
                            f"-o {q(bins / (analysis.name + '.metadecoder'))} "
                            f"--threads {self.options.threads}"
                            f"{'' if metadecoder_gpu else ' --disable_gpu'}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(cov_task.id, seed_task.id),
                        inputs=(contigs, coverage, seeds),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if metadecoder_gpu else 0,
                        env={**cupy_env, **self._gpu_env("metadecoder")},
                        priority=priority,
                        failure_tolerated=True,
                        description=f"Run MetaDecoder for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "comebin":
                tool_root = root / binner
                bins = tool_root / "comebin_res" / "comebin_res_bins"
                runner = " ".join(q(argument) for argument in self.options.comebin_run_prefix)
                comebin_gpu = self._uses_gpu("comebin")
                device_option = (
                    f" -d {'cuda' if comebin_gpu else 'cpu'}"
                    if self.options.comebin_explicit_device
                    else ""
                )
                comebin_command = (
                    f"{runner} run_comebin.sh "
                    f"-a {q(contigs)} -p {q(bam_dir)} "
                    f"-o {q(tool_root)} -t {self.options.threads} "
                    f"-b {self.options.batch_size}{device_option}"
                    f"{self._extra(binner)}"
                )
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=internal(
                            "run-comebin",
                            "--command",
                            comebin_command,
                            "--output-dir",
                            tool_root,
                            "--device",
                            "cuda" if comebin_gpu else "cpu",
                            "--heartbeat-seconds",
                            "1200",
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, bam_dir),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        # COMEBin augmentation and training are expensive.  A
                        # hidden retry used to erase both the first failure and
                        # its partial work even when the user selected
                        # --retries 0.  Retries are now controlled exclusively
                        # by the workflow-level option.
                        automatic_retries=0,
                        cpus=self.options.threads,
                        gpus=1 if comebin_gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._comebin_env(),
                        # --max-memory is a combined workflow PSS budget.  Do
                        # not additionally impose RLIMIT_AS on PyTorch, whose
                        # virtual mappings can be much larger than resident
                        # memory before training begins.
                        enforce_memory_limit=False,
                        description=f"Run COMEBin for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "semibin2":
                if analysis.name in self._semibin_multisample_outputs:
                    continue
                tool_root = root / binner
                mode = "single_easy_bin"
                has_multiple_bams = len(bams) > 1
                environment = (
                    f" --environment {q(self.options.environment)}"
                    if self.options.environment and not has_multiple_bams
                    else ""
                )
                training = (
                    " --self-supervised"
                    if has_multiple_bams or not self.options.environment
                    else ""
                )
                sequence_type = " --sequencing-type long_read" if self.options.read_type == "long" else ""
                semibin_gpu = self._uses_gpu("semibin2")
                engine = " --engine gpu" if semibin_gpu else " --engine cpu"
                bins = tool_root / "output_bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(tool_root)} && SemiBin2 {mode} -i {q(contigs)} -b "
                            f"{' '.join(q(path) for path in bams)} -o {q(tool_root)} "
                            f"--threads {self.options.threads} --min-len {self.options.min_contig_length}"
                            f"{environment}{training}{sequence_type}{engine}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, *bams),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if semibin_gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env("semibin2"),
                        description=f"Run SemiBin2 for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "lorbin":
                tool_root = root / binner
                bins = tool_root / "output_bins"
                completion = tool_root / ".metabaw.complete"
                runner = " ".join(
                    q(argument) for argument in self.options.lorbin_run_prefix
                )
                lorbin_gpu = self._uses_gpu("lorbin")
                cuda = " --cuda" if lorbin_gpu else ""
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(tool_root)} && {runner} python "
                            f"{q(LORBIN_COMPAT_SCRIPT)} bin "
                            f"-o {q(tool_root)} -fa {q(contigs)} "
                            f"-b {' '.join(q(path) for path in bams)} "
                            f"--num_process {self.options.threads}"
                            f"{cuda}{self._extra(binner)} && touch {q(completion)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, *bams),
                        outputs=(bins, completion),
                        fasta_output_dirs=(bins,),
                        # A failed late clustering step leaves a reusable
                        # embedding.csv; do not erase it on an automatic retry.
                        automatic_retries=0,
                        cpus=self.options.threads,
                        gpus=1 if lorbin_gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._lorbin_env(),
                        description=f"Run LorBin for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(
                    analysis,
                    binner,
                    run,
                    bins,
                    mapping,
                    self.options.min_fasta_kbs * 1000,
                )
            else:
                raise ValueError(f"Unsupported binner: {binner}")

            results.append(
                self._publish_binner(
                    analysis,
                    binner,
                    prepare,
                    contigs,
                    map_task,
                    mapping,
                )
            )
        return results

    def _publish_binner(
        self,
        analysis: Analysis,
        binner: str,
        prepare: Task,
        contigs: Path,
        map_task: Task,
        mapping: Path,
    ) -> tuple[str, Task, Path, Path]:
        public_bins = self.options.outdir / "bin_files" / binner / analysis.name
        naming_marker = public_bins / ".metabaw.naming-v2.complete"
        output_prefix = (
            f"{analysis_output_name(analysis)}_{analysis_tool_label(analysis, binner)}"
        )
        materialize = self.add(
            Task(
                id=f"03.publish.{binner}.{analysis.name}",
                stage="03_binning",
                command=internal(
                    "materialize-bins",
                    "--assembly",
                    contigs,
                    "--mapping",
                    mapping,
                    "--output-dir",
                    public_bins,
                    "--prefix",
                    output_prefix,
                    "--suffix",
                    self.options.mag_suffix,
                    "--completion-marker",
                    naming_marker,
                ),
                cwd=self.root,
                deps=(prepare.id, map_task.id),
                inputs=(contigs, mapping),
                outputs=(public_bins, public_bins / "manifest.tsv", naming_marker),
                priority=BINNER_PRIORITY[binner],
                failure_tolerated=True,
                description=f"Publish normalized {binner} bins",
            )
        )
        return binner, materialize, mapping, public_bins

    def _map_bins(
        self,
        analysis: Analysis,
        binner: str,
        run: Task,
        bins: Path,
        mapping: Path,
        minimum_bin_bp: int = 0,
    ) -> Task:
        arguments: list[str | Path] = [
            "bins-to-map",
            "--bins-dir",
            bins,
            "--output",
            mapping,
            "--label",
            binner,
        ]
        if minimum_bin_bp:
            arguments.extend(("--min-bin-bp", str(minimum_bin_bp)))
        return self.add(
            Task(
                id=f"03.map.{binner}.{analysis.name}",
                stage="03_binning",
                command=internal(*arguments),
                cwd=self.root,
                deps=(run.id,),
                inputs=(bins,),
                outputs=(mapping,),
                priority=BINNER_PRIORITY[binner],
                failure_tolerated=True,
                description=f"Normalize {binner} assignments",
            )
        )

    def _refine(
        self,
        analysis: Analysis,
        prepare: Task,
        contigs: Path,
        maps: list[tuple[str, Task, Path, Path]],
    ) -> tuple[Task, Path]:
        root = self.options.workdir / "refinement" / analysis.name
        refinement_completeness, refinement_contamination = self.options.refinement_thresholds
        sample_name = analysis_output_name(analysis)
        magscot_label = "magscot" if analysis.explicit_mapping else "MAGScoT"
        dastool_label = "das_tool" if analysis.explicit_mapping else "DASTool"
        metawrap_label = "metawrap" if analysis.explicit_mapping else "MetaWRAP"
        if self.options.refiner == "magscot":
            combined = root / "all_binners.tsv"
            combine_args: list[str | Path] = ["combine-maps"]
            for _name, _task, path, _bins in maps:
                combine_args.extend(("--input", path))
            combine_args.extend(("--output", combined))
            combined_task = self.add(
                Task(
                    id=f"04.combine.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*combine_args),
                    cwd=self.root,
                    deps=tuple(task.id for _name, task, _path, _bins in maps),
                    inputs=tuple(path for _name, _task, path, _bins in maps),
                    outputs=(combined,),
                    allow_failed_deps=True,
                    description="Combine binner assignments for MAGScoT",
                )
            )
            proteins = root / "markers" / "proteins.faa"
            genes = root / "markers" / "genes.ffn"
            prodigal_gff = root / "markers" / "prodigal.gff"
            prodigal_parts = root / "markers" / "prodigal_parts"
            prodigal_template = (
                f"prodigal -p meta "
                f"-a {q(prodigal_parts / 'proteins_{#}.faa')} "
                f"-d {q(prodigal_parts / 'genes_{#}.ffn')} "
                f"-o {q(prodigal_parts / 'prodigal_{#}.gff')}"
            )
            prodigal = self.add(
                Task(
                    id=f"04.prodigal.{analysis.name}",
                    stage="04_refinement",
                    command=(
                        f"rm -rf {q(prodigal_parts)} && mkdir -p {q(prodigal_parts)} && "
                        f"parallel --jobs {self.options.threads} --block 999k --recstart '>' "
                        f"--pipe {q(prodigal_template)} < {q(contigs)} && "
                        f"cat {q(prodigal_parts)}/proteins_*.faa > {q(proteins)} && "
                        f"cat {q(prodigal_parts)}/genes_*.ffn > "
                        f"{q(genes)} && "
                        f"cat {q(prodigal_parts)}/prodigal_*.gff > "
                        f"{q(prodigal_gff)}"
                    ),
                    cwd=self.root,
                    deps=(prepare.id,),
                    inputs=(contigs,),
                    outputs=(proteins, genes, prodigal_gff),
                    cpus=self.options.threads,
                    description="Predict genes for MAGScoT markers",
                )
            )
            hmm_tasks: list[Task] = []
            tables: list[Path] = []
            for name, hmm in (
                ("pfam", "gtdbtk_rel207_Pfam-A.hmm"),
                ("tigr", "gtdbtk_rel207_tigrfam.hmm"),
            ):
                table = root / "markers" / f"{name}.tbl"
                report = root / "markers" / f"{name}.out"
                hmm_tasks.append(
                    self.add(
                        Task(
                            id=f"04.hmm.{name}.{analysis.name}",
                            stage="04_refinement",
                            command=(
                                f"hmmsearch -o {q(report)} --tblout {q(table)} "
                                f"--noali --notextw --cut_nc --cpu {self.options.threads} "
                                f"{q(self.options.magscot_dir / 'hmm' / hmm)} {q(proteins)}"
                            ),
                            cwd=self.root,
                            deps=(prodigal.id,),
                            inputs=(proteins,),
                            outputs=(table, report),
                            cpus=self.options.threads,
                            description=f"Search {name} markers for MAGScoT",
                        )
                    )
                )
                tables.append(table)
            markers = root / "markers" / "magscot_markers.tsv"
            marker_task = self.add(
                Task(
                    id=f"04.markers.{analysis.name}",
                    stage="04_refinement",
                    command=internal(
                        "parse-magscot-hmm",
                        "--pfam",
                        tables[0],
                        "--tigr",
                        tables[1],
                        "--output",
                        markers,
                    ),
                    cwd=self.root,
                    deps=tuple(task.id for task in hmm_tasks),
                    inputs=tuple(tables),
                    outputs=(markers,),
                    description="Format marker hits for MAGScoT",
                )
            )
            refined_map = root / f"{analysis.name}.refined.contig_to_bin.out"
            native_command = [
                "Rscript", str(self.options.magscot_dir / "MAGScoT.R"),
                "-i", str(combined), "--hmm", str(markers), "-o", analysis.name,
                "--score_a", "1", "--score_b", "0.5", "--score_c", "0.5",
                "--threshold", f"{refinement_completeness / 100:g}",
                "--max_cont", f"{refinement_contamination / 100:g}",
                "--min_markers", "25", "--min_sharing", "0.8", "--n_iterations", "2",
                *shlex.split(self.options.extra_args.get("magscot", "")),
            ]
            refine = self.add(
                Task(
                    id=f"04.refine.magscot.{analysis.name}",
                    stage="04_refinement",
                    command=internal(
                        "run-magscot",
                        "--combined",
                        combined,
                        "--mapping",
                        refined_map,
                        "--run-log",
                        root / "magscot.native.log",
                        "--native-command-json",
                        json.dumps(native_command),
                    ),
                    cwd=root,
                    deps=(combined_task.id, marker_task.id),
                    inputs=(combined, markers),
                    outputs=(refined_map,),
                    description=f"Refine {analysis.name} bins with MAGScoT",
                )
            )
            bins = root / "bins"
            naming_marker = bins / ".metabaw.naming-v2.complete"
            refined_args: list[str | Path] = [
                "materialize-refined",
                "--assembly",
                contigs,
                "--mapping",
                refined_map,
                "--output-dir",
                bins,
                "--prefix",
                f"{sample_name}_{magscot_label}",
                "--suffix",
                self.options.mag_suffix,
                "--completion-marker",
                naming_marker,
            ]
            for _name, _task, _mapping, public_bins in maps:
                refined_args.extend(("--source", public_bins))
            result = self.add(
                Task(
                    id=f"04.export_refined_bins.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*refined_args),
                    cwd=self.root,
                    deps=(refine.id,),
                    inputs=(contigs, refined_map, *(public for _name, _task, _mapping, public in maps)),
                    outputs=(bins, bins / "manifest.tsv", naming_marker),
                    description="Write refined MAGs; unchanged bins keep original binner names",
                )
            )
            return result, bins

        if self.options.refiner == "das_tool":
            prefix = root / analysis.name
            bins = Path(str(prefix) + "_DASTool_bins")
            completion = root / "dastool.complete"
            run_args: list[str | Path] = [
                "run-dastool-refinement",
                "--assembly",
                contigs,
                "--output-prefix",
                prefix,
                "--output-bins",
                bins,
                "--completion-marker",
                completion,
                "--threads",
                str(self.options.threads),
                "--extra-args="
                + self.options.extra_args.get("das_tool", ""),
            ]
            for name, _task, mapping, public_bins in maps:
                run_args.extend(
                    (
                        "--binner-spec",
                        json.dumps(
                            {
                                "name": name,
                                "mapping": str(mapping),
                                "bins": str(public_bins),
                            },
                            separators=(",", ":"),
                        ),
                    )
                )
            run = self.add(
                Task(
                    id=f"04.refine.das_tool.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*run_args),
                    cwd=self.root,
                    deps=tuple(
                        task.id for _name, task, _mapping, _bins in maps
                    ),
                    inputs=(
                        contigs,
                        *(mapping for _name, _task, mapping, _bins in maps),
                        *(public_bins for _name, _task, _mapping, public_bins in maps),
                    ),
                    outputs=(completion, bins),
                    fasta_output_dirs=(bins,),
                    allow_empty_refinement=True,
                    cpus=self.options.threads,
                    allow_failed_deps=True,
                    description=f"Refine {analysis.name} bins with DAS Tool",
                )
            )
            normalize_args: list[str | Path] = [
                "normalize-refined-fasta",
                "--source-dir",
                bins,
                "--output-dir",
                bins,
                "--prefix",
                f"{sample_name}_{dastool_label}",
                "--suffix",
                self.options.mag_suffix,
                "--completion-marker",
                bins / ".metabaw.naming-v2.complete",
            ]
            for _name, _task, _mapping, public_bins in maps:
                normalize_args.extend(("--source", public_bins))
            publish = self.add(
                Task(
                    id=f"04.export_refined_bins.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*normalize_args),
                    cwd=self.root,
                    # Source binner failures are handled by the managed run;
                    # publication must still require that refiner to succeed.
                    deps=(run.id,),
                    # bins is rewritten in place. The successful refiner and
                    # output validation cover it; do not hash its old directory
                    # metadata as an input to its own publication checkpoint.
                    inputs=(
                        *(public_bins for _name, _task, _mapping, public_bins in maps),
                    ),
                    outputs=(
                        bins,
                        bins / "manifest.tsv",
                        bins / ".metabaw.naming-v2.complete",
                    ),
                    fasta_output_dirs=(bins,),
                    allow_empty_refinement=True,
                    description=(
                        "Normalize DAS Tool MAG names and retain unchanged "
                        "source-binner provenance"
                    ),
                )
            )
            return publish, bins

        if len(maps) > 3:
            raise ValueError("MetaWRAP bin_refinement accepts at most three binner directories")
        threshold_name = (
            f"metawrap_{refinement_completeness:g}_{refinement_contamination:g}_bins"
        )
        bins = root / threshold_name
        stats = root / f"{threshold_name}.stats"
        completion = root / "metawrap.complete"
        run_args: list[str | Path] = [
            "run-metawrap-refinement",
            "--output-root",
            root,
            "--output-bins",
            bins,
            "--stats",
            stats,
            "--completion-marker",
            completion,
            "--threads",
            str(self.options.threads),
            "--min-completeness",
            f"{refinement_completeness:g}",
            "--max-contamination",
            f"{refinement_contamination:g}",
            "--extra-args="
            + self.options.extra_args.get("metawrap", ""),
            "--runner-json",
            json.dumps(
                list(self.options.metawrap_run_prefix),
                separators=(",", ":"),
            ),
        ]
        for name, _task, _mapping, bins_dir in maps:
            run_args.extend(("--input", f"{name}={bins_dir}"))
        run = self.add(
            Task(
                id=f"04.refine.metawrap.{analysis.name}",
                stage="04_refinement",
                command=internal(*run_args),
                cwd=self.root,
                deps=tuple(task.id for _name, task, _mapping, _bins in maps),
                inputs=tuple(bins_dir for _name, _task, _mapping, bins_dir in maps),
                outputs=(completion, stats, bins),
                fasta_output_dirs=(bins,),
                allow_empty_refinement=True,
                cpus=self.options.threads,
                allow_failed_deps=True,
                description=f"Refine {analysis.name} bins with MetaWRAP",
            )
        )
        normalize_args: list[str | Path] = [
            "normalize-refined-fasta",
            "--source-dir",
            bins,
            "--output-dir",
            bins,
            "--prefix",
            f"{sample_name}_{metawrap_label}",
            "--suffix",
            self.options.mag_suffix,
            "--completion-marker",
            bins / ".metabaw.naming-v2.complete",
        ]
        for _name, _task, _mapping, public_bins in maps:
            normalize_args.extend(("--source", public_bins))
        publish = self.add(
            Task(
                id=f"04.export_refined_bins.{analysis.name}",
                stage="04_refinement",
                command=internal(*normalize_args),
                cwd=self.root,
                deps=(run.id,),
                # The native bins are an in-place output, validated through the
                # refiner dependency and this task's required FASTA outputs.
                inputs=(
                    *(public_bins for _name, _task, _mapping, public_bins in maps),
                ),
                outputs=(
                    bins,
                    bins / "manifest.tsv",
                    bins / ".metabaw.naming-v2.complete",
                ),
                fasta_output_dirs=(bins,),
                allow_empty_refinement=True,
                description=(
                    "Normalize MetaWRAP MAG names and retain unchanged "
                    "source-binner provenance"
                ),
            )
        )
        self._metawrap_quality_sources.append((stats, bins / "manifest.tsv"))
        return publish, bins

    def _catalog(self, refined: list[tuple[Task, Path]]) -> tuple[Task, Path]:
        candidates = self.options.outdir / "quality_control_files" / "candidate_bins"
        args: list[str | Path] = ["collect-fasta"]
        for _task, directory in refined:
            args.extend(("--source", directory))
        args.extend(("--output-dir", candidates))
        task = self.add(
            Task(
                id="05.catalog.candidates",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=tuple(task.id for task, _path in refined),
                inputs=tuple(path for _task, path in refined),
                outputs=(candidates, candidates / "manifest.tsv"),
                description="Collect refined candidate MAGs",
            )
        )
        return task, candidates

    def _quality(self, catalog: tuple[Task, Path]) -> tuple[Task, Path]:
        catalog_task, candidates = catalog
        root = self.options.outdir / "quality_control_files"
        threads = self._cohort_threads()
        task_environment: dict[str, str] = {}
        if self.options.quality_control == "checkm2":
            output = root / "checkm2"
            report = output / "quality_report.tsv"
            ipc_temp = self.options.workdir / "quality" / "checkm2_ipc"
            database = f" --database_path {q(self.options.checkm2_db)}" if self.options.checkm2_db else ""
            runner = " ".join(q(argument) for argument in self.options.checkm2_run_prefix)
            task_environment = CHECKM2_THREAD_ENV
            command = (
                f"rm -rf {q(output)} {q(ipc_temp)} && mkdir -p {q(ipc_temp)} && "
                "checkm2_ipc_link=/tmp/mbw-checkm2-$$ && "
                "rm -f \"$checkm2_ipc_link\" && "
                f"ln -s {q(ipc_temp)} \"$checkm2_ipc_link\" && "
                "trap 'rm -f \"$checkm2_ipc_link\"' EXIT && "
                "export TMPDIR=\"$checkm2_ipc_link\" TMP=\"$checkm2_ipc_link\" "
                "TEMP=\"$checkm2_ipc_link\" && "
                f"{runner} checkm2 predict "
                f"--threads {threads} "
                f"--input {q(candidates)} --output-directory {q(output)} "
                f"--tmpdir \"$checkm2_ipc_link\" --force "
                f"-x {q(self.options.mag_suffix)}{database}"
                f"{self._extra('checkm2')}"
            )
        elif (
            self.options.refiner == "metawrap"
            and self._metawrap_quality_sources
        ):
            output = root / "checkm"
            report = output / "quality_report.tsv"
            args: list[str | Path] = [
                "metawrap-checkm-quality",
                "--bins-dir",
                candidates,
            ]
            for stats, _manifest in self._metawrap_quality_sources:
                args.extend(("--stats", stats))
            for _stats, manifest in self._metawrap_quality_sources:
                args.extend(("--manifest", manifest))
            args.extend(("--output", report))
            command = internal(*args)
        else:
            output = root / "checkm"
            report = output / "quality_report.tsv"
            work = self.options.workdir / "quality" / "checkm"
            ipc_temp = self.options.workdir / "quality" / "checkm_ipc"
            native_command = (
                f"checkm lineage_wf -x {q(self.options.mag_suffix)} {q(candidates)} {q(work)} -t {threads} "
                f"--tab_table -f {q(report)}{self._extra('checkm')}"
            )
            runner = " ".join(q(argument) for argument in internal(
                "run-checkm", "--command", native_command,
            ))
            command = (
                f"rm -rf {q(output)} {q(work)} {q(ipc_temp)} && "
                f"mkdir -p {q(output)} {q(work)} {q(ipc_temp)} && "
                "checkm_ipc_link=/tmp/mbw-checkm-$$ && "
                "rm -f \"$checkm_ipc_link\" && "
                f"ln -s {q(ipc_temp)} \"$checkm_ipc_link\" && "
                "trap 'rm -f \"$checkm_ipc_link\"' EXIT && "
                "export TMPDIR=\"$checkm_ipc_link\" TMP=\"$checkm_ipc_link\" "
                "TEMP=\"$checkm_ipc_link\" && "
                f"{runner} --tmpdir \"$checkm_ipc_link\""
            )
        quality_inputs: tuple[Path, ...] = (candidates,)
        if (
            self.options.quality_control == "checkm"
            and self.options.refiner == "metawrap"
            and self._metawrap_quality_sources
        ):
            quality_inputs = (
                candidates,
                *(
                    path
                    for source in self._metawrap_quality_sources
                    for path in source
                ),
            )
        task = self.add(
            Task(
                id=f"05.qc.{self.options.quality_control}",
                stage="05_quality",
                command=command,
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=quality_inputs,
                outputs=(report,),
                cpus=(
                    1
                    if (
                        self.options.quality_control == "checkm"
                        and self.options.refiner == "metawrap"
                        and self._metawrap_quality_sources
                    )
                    else threads
                ),
                env=task_environment,
                description=(
                    "Reuse MetaWRAP CheckM quality results"
                    if (
                        self.options.quality_control == "checkm"
                        and self.options.refiner == "metawrap"
                        and self._metawrap_quality_sources
                    )
                    else f"Estimate MAG quality with {self.options.quality_control}"
                ),
            )
        )
        return task, report

    def _gunc(self, catalog: tuple[Task, Path]) -> Task | None:
        if not self.options.run_gunc:
            return None
        catalog_task, candidates = catalog
        threads = self._cohort_threads()
        output = self.options.outdir / "quality_control_files" / "GUNC"
        temp = self.options.workdir / "quality" / "gunc"
        database = f" --db_file {q(self.options.gunc_db)}" if self.options.gunc_db else ""
        return self.add(
            Task(
                id="05.qc.gunc",
                stage="05_quality",
                command=(
                    f"rm -rf {q(output)} {q(temp)} && mkdir -p {q(output)} {q(temp)} && gunc run "
                    f"--input_dir {q(candidates)} --file_suffix .{self.options.mag_suffix} --threads {threads} "
                    f"--out_dir {q(output)} --temp_dir {q(temp)} "
                    f"--min_mapped_genes 11{database}{self._extra('gunc')}"
                ),
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(output,),
                cpus=threads,
                description="Detect chimeric MAGs with GUNC",
            )
        )

    def _rna(self, catalog: tuple[Task, Path]) -> tuple[Task | None, Path | None]:
        run_trna = self.options.run_trna or self.options.trna_pass is not None
        run_rrna = self.options.run_rrna or self.options.rrna_pass
        if not (run_trna or run_rrna):
            return None, None
        catalog_task, candidates = catalog
        threads = self._cohort_threads()
        marker_dir = self.options.workdir / "quality" / "gtdbtk_identify"
        data_env = {"GTDBTK_DATA_PATH": str(self.options.gtdbtk_data)} if self.options.gtdbtk_data else {}
        identify = self.add(
            Task(
                id="05.qc.rna_domain",
                stage="05_quality",
                command=(
                    f"rm -rf {q(marker_dir)} && gtdbtk identify --genome_dir {q(candidates)} "
                    f"--out_dir {q(marker_dir)} --extension {q(self.options.mag_suffix)} --cpus {threads}"
                ),
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(marker_dir,),
                cpus=threads,
                env=data_env,
                description="Identify bacterial/archaeal marker domains before RNA QC",
            )
        )
        quality_root = self.options.outdir / "quality_control_files"
        output = quality_root / "rna"
        summary = quality_root / "rna_quality.tsv"
        # Version the report status so resume does not reuse the old report-only
        # format, where every genome misleadingly had pass=true.
        completion_marker = output / "rna_qc.v3.complete"
        args: list[str | Path] = [
            "rna-qc",
            "--bins-dir",
            candidates,
            "--marker-dir",
            marker_dir,
            "--output-dir",
            output,
            "--summary",
            summary,
            "--threads",
            str(threads),
            "--completion-marker",
            completion_marker,
        ]
        if run_trna:
            args.append("--trna")
        if self.options.trna_pass is not None:
            args.extend(("--trna-pass", str(self.options.trna_pass)))
        if run_rrna:
            args.append("--rrna")
        if self.options.rrna_pass:
            args.append("--rrna-pass")
        task = self.add(
            Task(
                id="05.qc.rna",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=(identify.id,),
                inputs=(candidates, marker_dir),
                outputs=(output, summary, completion_marker),
                cpus=threads,
                description="Apply domain-aware tRNA/rRNA quality checks",
            )
        )
        return task, summary

    def _filter(
        self,
        catalog: tuple[Task, Path],
        check_task: Task,
        check_report: Path,
        gunc_task: Task | None,
        rna_task: Task | None,
        rna_summary: Path | None,
    ) -> tuple[Task, Path]:
        catalog_task, candidates = catalog
        output = self.options.outdir / "quality_control_files"
        filtered = output / "filtered_bins"
        summary = output / "quality_summary.tsv"
        args: list[str | Path] = [
            "filter-quality",
            "--bins-dir",
            candidates,
            "--checkm2",
            check_report,
            "--output-dir",
            filtered,
            "--summary",
            summary,
            "--min-completeness",
            str(self.options.min_completeness),
            "--max-contamination",
            str(self.options.max_contamination),
        ]
        deps = [catalog_task.id, check_task.id]
        inputs = [candidates, check_report]
        if self.options.min_quality_score is not None:
            args.extend(("--min-quality-score", str(self.options.min_quality_score)))
        if gunc_task:
            gunc_dir = output / "GUNC"
            args.extend(("--gunc-dir", gunc_dir))
            deps.append(gunc_task.id)
            inputs.append(gunc_dir)
        if rna_task:
            deps.append(rna_task.id)
            if (
                rna_summary
                and (
                    self.options.trna_pass is not None
                    or self.options.rrna_pass
                )
            ):
                args.extend(("--rna-summary", rna_summary))
                inputs.append(rna_summary)
        task = self.add(
            Task(
                id="05.qc.filter",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=tuple(deps),
                inputs=tuple(inputs),
                outputs=(filtered, summary),
                description="Apply completeness, contamination, score, GUNC and RNA filters",
            )
        )
        return task, filtered

    def _dereplicate(self, filtered_task: Task, filtered: Path, quality: Path) -> tuple[Task, Path]:
        threads = self._cohort_threads()
        final = (
            self.options.workdir / "dereplication" / "representative_bins"
            if self.options.tag_contigs
            else self.options.outdir / "non_redundant_bins"
        )
        legacy_untagged = self.options.outdir / "non_redundant_bins.untagged"
        if self.options.dereplicator == "galah":
            clusters = self.options.outdir / "galah_clusters.tsv"
            genome_info = self.options.workdir / "galah_genome_info.csv"
            genome_info_command = " ".join(
                q(argument)
                for argument in internal(
                    "drep-genome-info",
                    "--bins-dir",
                    filtered,
                    "--quality-report",
                    quality,
                    "--output",
                    genome_info,
                    "--strip-extension",
                )
            )
            task = self.add(
                Task(
                    id="06.dereplicate.galah",
                    stage="06_dereplication",
                    command=(
                        f"rm -rf {q(final)} {q(legacy_untagged)} && "
                        f"{genome_info_command} && "
                        f"galah cluster --genome-fasta-directory {q(filtered)} "
                        f"--genome-fasta-extension {q(self.options.mag_suffix)} "
                        f"--genome-info {q(genome_info)} "
                        f"--min-completeness {self.options.min_completeness:g} "
                        f"--max-contamination {self.options.max_contamination:g} --ani {self.options.ani:g} "
                        f"--min-aligned-fraction {self.options.min_aligned_fraction:g} --quality-formula completeness-5contamination "
                        f"--precluster-method skani --cluster-method skani "
                        f"--output-cluster-definition {q(clusters)} "
                        f"--output-representative-fasta-directory-copy {q(final)} "
                        f"--threads {threads}{self._extra('galah')}"
                    ),
                    cwd=self.root,
                    deps=(filtered_task.id,),
                    inputs=(filtered, quality),
                    outputs=(final, clusters),
                    cpus=threads,
                    description="Dereplicate MAGs with Galah",
                )
            )
            return task, final
        drep = self.options.workdir / "drep"
        genome_info = self.options.workdir / "drep_genome_info.csv"
        genomes = f"{q(filtered)}/*.{self.options.mag_suffix}"
        genome_info_command = " ".join(
            q(argument)
            for argument in internal(
                "drep-genome-info",
                "--bins-dir",
                filtered,
                "--quality-report",
                quality,
                "--output",
                genome_info,
            )
        )
        task = self.add(
            Task(
                id="06.dereplicate.drep",
                    stage="06_dereplication",
                    command=(
                    f"rm -rf {q(drep)} {q(final)} {q(legacy_untagged)} && "
                    f"{genome_info_command} && "
                    f"dRep dereplicate {q(drep)} -g {genomes} "
                    f"--genomeInfo {q(genome_info)} "
                    f"-p {threads} -comp {self.options.min_completeness:g} "
                    f"-con {self.options.max_contamination:g} -sa {self.options.ani / 100:g} "
                    f"-nc {self.options.min_aligned_fraction / 100:g}"
                    f"{self._extra('drep')} && mkdir -p {q(final)} && "
                    f"cp {q(drep / 'dereplicated_genomes')}/*.{self.options.mag_suffix} {q(final)}/"
                ),
                cwd=self.root,
                deps=(filtered_task.id,),
                inputs=(filtered, quality),
                outputs=(final,),
                cpus=threads,
                description="Dereplicate MAGs with dRep",
            )
        )
        return task, final

    def _dereplication_report(self, final_task: Task, final_dir: Path) -> Task:
        summary = self.options.outdir / "quality_control_files" / "quality_summary.tsv"
        marker = self.options.workdir / "dereplication" / "quality_summary.v1.complete"
        return self.add(
            Task(
                id="06.dereplicate.report",
                stage="06_dereplication",
                command=internal(
                    "report-dereplication-quality",
                    "--summary", summary,
                    "--representatives-dir", final_dir,
                    "--completion-marker", marker,
                ),
                cwd=self.root,
                deps=(final_task.id,),
                # The summary is updated in place. Its upstream quality inputs
                # are covered by dependencies; hashing it here would invalidate
                # this task's own resume fingerprint after every update.
                inputs=(final_dir,),
                outputs=(summary, marker),
                description="Report final representative selection in quality_summary.tsv",
            )
        )

    def _extra(self, tool: str) -> str:
        value = self.options.extra_args.get(tool, "").strip()
        return f" {value}" if value else ""

    def _cohort_threads(self) -> int:
        """Use the effective aggregate budget, not a concurrent sample's share."""
        return getattr(self.options, "total_threads", None) or self.options.threads

    def _long_read_mapping_preset(self) -> str:
        return {
            "--pacbio-raw": "map-pb",
            "--pacbio-corr": "map-pb",
            "--pacbio-hifi": "map-hifi",
            "--nano-raw": "map-ont",
            "--nano-corr": "map-ont",
            "--nano-hq": "map-ont",
        }[self.options.flye_read_type]

    def _uses_gpu(self, tool: str) -> bool:
        if not self.options.gpu:
            return False
        # An empty tuple preserves compatibility for programmatic callers that
        # predate per-binner GPU selection. The CLI always supplies the
        # explicit eligible set.
        gpu_binners = getattr(self.options, "gpu_binners", ())
        return not gpu_binners or tool in gpu_binners

    def _gpu_env(self, tool: str) -> dict[str, str]:
        if not self._uses_gpu(tool):
            # COMEBin chooses CUDA automatically when PyTorch can see a GPU.
            # An empty visibility list is therefore required for CPU fallback.
            return {"CUDA_VISIBLE_DEVICES": ""}
        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "METABAW_MAX_GPU_MEMORY": self.options.max_gpu_memory,
        }

    def _comebin_env(self) -> dict[str, str]:
        return {**COMEBIN_PROCESS_ENV, **self._gpu_env("comebin")}

    def _lorbin_env(self) -> dict[str, str]:
        return {**LORBIN_PROCESS_ENV, **self._gpu_env("lorbin")}


@dataclass(frozen=True)
class AnnotationOptions:
    mag_dir: Path
    mags: tuple[Path, ...]
    mag_suffix: str
    reads: tuple[ReadSample, ...]
    output: Path
    output_suffix: str
    threads: int
    read_type: str
    methods: tuple[str, ...]
    place_species: bool
    niche_rank: str
    niche_method: str
    no_niche: bool
    gtdbtk_data: Path | None
    run_kegg: bool
    run_cazy: bool
    run_hydrogenase: bool
    kegg_db: Path | None
    dbcan_db: Path | None
    hydrogenase_db: Path | None
    gtdbtk_result: Path | None = None
    total_threads: int | None = None
    # Nonempty only for named-manifest annotation; mags are the staged paths.
    mag_sources: tuple[Path, ...] = ()
    niche_abundance: str = "relative_abundance"
    cv_correct: bool = False
    gtdbtk_tmpdir: Path = Path("/tmp")


class AnnotationBuilder:
    def __init__(self, options: AnnotationOptions, root: Path):
        self.options = options
        self.root = root
        self.tasks: list[Task] = []

    def add(self, task: Task) -> Task:
        self.tasks.append(task)
        return task

    def build(self) -> list[Task]:
        genome_dependencies: tuple[str, ...] = ()
        if self.options.cv_correct and self.options.niche_method != "cv":
            raise ValueError("--cv-correct requires the cv niche classification method")
        niche_enabled = (
            not self.options.no_niche
            and annotation_niche_eligible(len(self.options.mags), len(self.options.reads))
        )
        profile_methods = self.options.methods
        if niche_enabled:
            if self.options.niche_abundance not in profile_methods:
                raise ValueError("Niche abundance must be present in the selected CoverM methods")
            # Counts support the existing CV correction/read-support filter,
            # but are not an exposed abundance option or public output matrix.
            profile_methods = tuple(dict.fromkeys((*profile_methods, "count")))
        if self.options.mag_sources:
            stage_args: list[str | Path] = [
                "stage-annotation-genomes", "--output-dir", self.options.mag_dir,
            ]
            for source, target in zip(self.options.mag_sources, self.options.mags, strict=True):
                stage_args.extend(("--genome", f"{target.stem}={source}"))
            stage_task = self.add(Task(
                id="00.annotation.genomes", stage="00_prepare",
                command=internal(*stage_args), cwd=self.root,
                inputs=self.options.mag_sources,
                outputs=(self.options.mag_dir, *self.options.mags,
                         self.options.mag_dir / "manifest.tsv",
                         self.options.mag_dir / ".metabaw.complete"),
                fasta_output_dirs=(self.options.mag_dir,),
                description=f"Stage exactly {len(self.options.mags)} listed MAGs as uncompressed FASTA",
            ))
            genome_dependencies = (stage_task.id,)
        taxonomy_dependencies: tuple[str, ...] = ()
        if self.options.gtdbtk_result is not None:
            taxonomy = self.options.gtdbtk_result
        else:
            taxonomy = self.options.output / "gtdbtk_result"
            # GTDB-Tk is a cohort-wide task and receives the global --threads
            # budget, not the per-sample share computed from --task.
            gtdbtk_threads = (
                self.options.total_threads
                if self.options.total_threads is not None
                else self.options.threads
            )
            if gtdbtk_threads < 1:
                raise ValueError("GTDB-Tk threads must be positive")
            gtdbtk_scratch = self.root / "gtdbtk_scratch"
            env = (
                {"GTDBTK_DATA_PATH": str(self.options.gtdbtk_data)}
                if self.options.gtdbtk_data
                else {}
            )
            taxonomy_args: list[str | Path] = [
                "run-gtdbtk-classify",
                "--genome-dir",
                self.options.mag_dir,
                "--output-dir",
                taxonomy,
                "--extension",
                self.options.mag_suffix.lstrip("."),
                "--threads",
                str(gtdbtk_threads),
                "--pplacer-threads",
                str(GTDBTK_PPLACER_CPUS),
                "--scratch-dir",
                gtdbtk_scratch,
                "--ipc-tmpdir",
                self.options.gtdbtk_tmpdir,
            ]
            if self.options.place_species:
                taxonomy_args.append("--place-species")
            taxonomy_task = self.add(
                Task(
                    id="01.annotation.gtdbtk",
                    stage="01_taxonomy",
                    command=internal(*taxonomy_args),
                    cwd=self.root,
                    deps=genome_dependencies,
                    inputs=(self.options.mag_dir,),
                    outputs=(taxonomy, taxonomy / ".metabaw.complete"),
                    cpus=gtdbtk_threads,
                    env=env,
                    enforce_memory_limit=False,
                    minimum_memory_gb=GTDBTK_MEMORY_ESTIMATE_GIB,
                    memory_requirement_hint=(
                        f"increase --max-memory to at least "
                        f"{GTDBTK_MEMORY_ESTIMATE_GIB:g} GiB, or provide a complete matching "
                        "--gtdbtk_res path to skip GTDB-Tk"
                    ),
                    description=(
                        f"Classify {len(self.options.mags)} MAGs with GTDB-Tk "
                        f"(cpus={gtdbtk_threads}, pplacer_cpus={GTDBTK_PPLACER_CPUS}, disk scratch enabled)"
                    ),
                )
            )
            taxonomy_dependencies = (taxonomy_task.id,)
        coverm_dir = self.options.output / "coverm"
        samtools_shim = self.root / "compat" / "bin" / "samtools"
        samtools_compat_task = self.add(
            Task(
                id="02.annotation.samtools.compat",
                stage="02_abundance",
                command=internal("prepare-samtools-compat", "--output", samtools_shim),
                cwd=self.root,
                outputs=(samtools_shim,),
                description="Normalize samtools version output for CoverM",
            )
        )
        profile_tasks: list[Task] = []
        profiles: list[tuple[str, Path]] = []
        for sample in self.options.reads:
            output = coverm_dir / f"{sample.name}{self.options.output_suffix}"
            reads = (
                f"--coupled {q(sample.read1)} {q(sample.read2)}"
                if sample.read2
                else f"--single {q(sample.read1)}"
            )
            read_type = self.options.read_type
            if read_type == "auto":
                read_type = "short" if sample.read2 is not None else "long"
            mapper = "minimap2-sr" if read_type == "short" else "minimap2-ont"
            # CoverM cannot apply a positive covered-fraction threshold to
            # estimators that do not calculate covered bases themselves.
            # Keep coverage-aware metrics at the intended 10% threshold and
            # calculate the remaining metrics in one zero-threshold pass.
            zero_threshold_methods = {
                "count",
                "reads_per_base",
                "rpkm",
                "tpm",
                "length",
            }
            filtered_methods = tuple(
                method
                for method in profile_methods
                if method not in zero_threshold_methods
            )
            unfiltered_methods = tuple(
                method
                for method in profile_methods
                if method in zero_threshold_methods
            )
            method_groups: list[tuple[str, tuple[str, ...], Path, int]] = []
            if filtered_methods:
                method_groups.append(("filtered", filtered_methods, output, 10))
            if unfiltered_methods:
                unfiltered_output = (
                    coverm_dir
                    / ".parts"
                    / f"{sample.name}.unfiltered{self.options.output_suffix}"
                    if filtered_methods
                    else output
                )
                if niche_enabled:
                    unfiltered_output = self.root / "niche_support" / "profiles" / f"{sample.name}.unfiltered.tsv"
                method_groups.append(
                    ("unfiltered", unfiltered_methods, unfiltered_output, 0)
                )
            for group, methods, group_output, minimum_covered in method_groups:
                task_id = f"02.annotation.coverm.{sample.name}"
                if len(method_groups) > 1 and group == "unfiltered":
                    task_id += ".unfiltered"
                profile_tasks.append(
                    self.add(
                        Task(
                            id=task_id,
                            stage="02_abundance",
                            command=(
                                f"mkdir -p {q(group_output.parent)} && "
                                f"PATH={q(samtools_shim.parent)}:$PATH coverm genome "
                                f"--genome-fasta-directory {q(self.options.mag_dir)} "
                                f"--genome-fasta-extension {q(self.options.mag_suffix.lstrip('.'))} {reads} "
                                f"--mapper {mapper} --methods {' '.join(methods)} "
                                f"--min-read-percent-identity 95 --min-read-aligned-percent 75 "
                                f"--min-covered-fraction {minimum_covered} "
                                f"--threads {self.options.threads} "
                                f"--output-file {q(group_output)}"
                            ),
                            cwd=self.root,
                            deps=(*genome_dependencies, samtools_compat_task.id),
                            inputs=(self.options.mag_dir, samtools_shim, *sample.reads),
                            outputs=(group_output,),
                            cpus=self.options.threads,
                            sample=sample.name,
                            description=(
                                f"Profile {sample.name} with CoverM"
                                if group == "filtered"
                                else (
                                    f"Calculate threshold-independent CoverM metrics "
                                    f"for {sample.name}"
                                )
                            ),
                        )
                    )
                )
                profiles.append((sample.name, group_output))
        merge_args: list[str | Path] = ["merge-coverm-taxonomy"]
        for sample_name, profile in profiles:
            merge_args.extend(("--input", f"{sample_name}={profile}"))
        merge_args.extend(
            (
                "--taxonomy-dir",
                taxonomy,
                "--output-dir",
                coverm_dir,
                "--output-suffix",
                self.options.output_suffix,
            )
        )
        combined = coverm_dir / f"coverm_all_metrics{self.options.output_suffix}"
        niche_matrix = self.root / "niche_support" / "abundance_with_counts.tsv"
        if niche_enabled:
            merge_args.extend(("--niche-output", niche_matrix))
        merge_task = self.add(
            Task(
                id="03.annotation.merge",
                stage="03_merge",
                command=internal(*merge_args),
                cwd=self.root,
                deps=(*taxonomy_dependencies, *(task.id for task in profile_tasks)),
                inputs=(taxonomy, *(profile for _, profile in profiles)),
                outputs=(
                    coverm_dir / "manifest.json",
                    coverm_dir / "merge_schema.json",
                    combined,
                    *((niche_matrix,) if niche_enabled else ()),
                ),
                description="Merge CoverM metrics with bacterial and archaeal GTDB taxonomy",
            )
        )
        if niche_enabled:
            niche_dir = self.options.output / "niche"
            niche_mode = (
                "corrected_cv" if self.options.cv_correct else "raw_cv"
            ) if self.options.niche_method == "cv" else "occupancy"
            completion_marker = niche_dir / f"niche_classification.v3.{niche_mode}.complete"
            args: list[str | Path] = [
                "classify-niche-literature",
                "--abundance",
                niche_matrix,
                "--taxonomy-dir",
                taxonomy,
                "--output",
                niche_dir / "niche_classification.tsv",
                "--aggregated-output",
                niche_dir / "taxon_abundance.tsv",
                "--assignments-output",
                niche_dir / "genome_to_taxon.tsv",
                "--provenance-output",
                niche_dir / "provenance.json",
                "--rank",
                self.options.niche_rank,
                "--method",
                self.options.niche_method,
                "--abundance-method",
                self.options.niche_abundance,
                "--detection-percent",
                "0.01",
                "--min-total-reads",
                "20",
                "--min-prevalence",
                "0.20",
                "--core-prevalence",
                "0.80",
                "--min-core-datasets",
                "1",
            ]
            # Version every mode so older table schemas cannot be adopted.
            args.extend(("--completion-marker", completion_marker))
            if self.options.cv_correct:
                args.append("--cv-correct")
            for sample in self.options.reads:
                args.extend(("--sample-dataset", f"{sample.name}=default"))
            self.add(
                Task(
                    id="04.annotation.niche",
                    stage="04_niche",
                    command=internal(*args),
                    cwd=self.root,
                    deps=(merge_task.id,),
                    inputs=(niche_matrix, taxonomy),
                    outputs=(
                        niche_dir / "niche_classification.tsv",
                        niche_dir / "taxon_abundance.tsv",
                        niche_dir / "genome_to_taxon.tsv",
                        niche_dir / "provenance.json",
                        completion_marker,
                    ),
                    description=(
                        f"Classify ecological niches at {self.options.niche_rank} "
                        f"level using {niche_mode.replace('_', '-')} "
                        f"{'rank quartiles ' if self.options.niche_method == 'cv' else ''}"
                        f"and {self.options.niche_abundance}"
                    ),
                )
            )
        if (
            self.options.run_kegg
            or self.options.run_cazy
            or self.options.run_hydrogenase
        ):
            names = [
                annotation_genome_name(path, self.options.mag_suffix)
                for path in self.options.mags
            ]
            if len(names) != len(set(names)):
                raise ValueError(
                    "MAG filenames are not unique after removing the FASTA suffix; "
                    "functional annotation requires unique genome names"
                )
            proteins_dir = self.root / "annotation" / "proteins"
            gene_maps_dir = self.root / "annotation" / "gene_maps"
            gene_tasks: dict[str, Task] = {}
            gene_map_tasks: dict[str, Task] = {}
            gene_map_paths: dict[str, Path] = {}
            protein_paths: dict[str, Path] = {}
            for mag, genome in zip(self.options.mags, names):
                proteins = proteins_dir / f"{genome}.faa"
                protein_paths[genome] = proteins
                gene_tasks[genome] = self.add(
                    Task(
                        id=f"05.annotation.prodigal.{genome}",
                        stage="05_gene_prediction",
                        command=(
                            f"mkdir -p {q(proteins_dir)} && "
                            f"prodigal -q -p single -i {q(mag)} -a {q(proteins)}"
                        ),
                        cwd=self.root,
                        deps=genome_dependencies,
                        inputs=(mag,),
                        outputs=(proteins,),
                        cpus=1,
                        sample=genome,
                        description=f"Predict proteins for {genome}",
                    )
                )
                gene_map = gene_maps_dir / f"{genome}.tsv"
                gene_map_paths[genome] = gene_map
                gene_map_tasks[genome] = self.add(
                    Task(
                        id=f"05.annotation.gene_map.{genome}",
                        stage="05_gene_prediction",
                        command=internal(
                            "build-functional-gene-map",
                            "--mag",
                            mag,
                            "--proteins",
                            proteins,
                            "--genome",
                            genome,
                            "--output",
                            gene_map,
                        ),
                        cwd=self.root,
                        deps=(gene_tasks[genome].id,),
                        inputs=(mag, proteins),
                        outputs=(gene_map,),
                        sample=genome,
                        description=f"Map {genome} genes to raw and normalized contigs",
                    )
                )
            combined_proteins = proteins_dir / "all_mags.faa"
            combined_gene_map = gene_maps_dir / "all_mags.tsv"
            combine_args: list[str | Path] = [
                "combine-functional-proteins",
                "--output",
                combined_proteins,
                "--output-gene-map",
                combined_gene_map,
            ]
            for genome in names:
                combine_args.extend(("--input", f"{genome}={protein_paths[genome]}"))
                combine_args.extend(("--gene-map", f"{genome}={gene_map_paths[genome]}"))
            combine_task = self.add(
                Task(
                    id="05.annotation.proteins.merge",
                    stage="05_gene_prediction",
                    command=internal(*combine_args),
                    cwd=self.root,
                    deps=tuple(gene_map_tasks[name].id for name in names),
                    inputs=(
                        *(protein_paths[name] for name in names),
                        *(gene_map_paths[name] for name in names),
                    ),
                    outputs=(combined_proteins, combined_gene_map),
                    description=(
                        f"Combine proteins from {len(names)} MAGs with globally unique IDs"
                    ),
                )
            )
            functional_threads = self.options.total_threads or self.options.threads
            if self.options.run_kegg:
                kegg_root = self.options.output / "kegg"
                kofam_tmp = self.root / "annotation" / "kofamscan"
                kegg_db = self.options.kegg_db or Path("KOFAM_DB_NOT_CONFIGURED")
                kegg_descriptions = kegg_db / "K.descript.txt"
                kegg_raw = kegg_root / "kofamscan_mapper.tsv"
                kegg_task = self.add(
                    Task(
                        id="06.annotation.kegg",
                        stage="06_kegg",
                        command=(
                            f"rm -rf {q(kofam_tmp)} && "
                            f"mkdir -p {q(kofam_tmp)} {q(kegg_root)} && "
                            f"exec_annotation --profile {q(kegg_db / 'profiles')} "
                            f"--ko-list {q(kegg_db / 'ko_list')} "
                            f"--cpu {functional_threads} --tmp-dir {q(kofam_tmp)} "
                            f"--format mapper --no-report-unannotated "
                            f"-o {q(kegg_raw)} {q(combined_proteins)}"
                        ),
                        cwd=self.root,
                        deps=(combine_task.id,),
                        inputs=(combined_proteins, kegg_db / "ko_list", kegg_db / "profiles"),
                        outputs=(kegg_raw,),
                        cpus=functional_threads,
                        description=(
                            f"Annotate the combined protein set from {len(names)} MAGs "
                            "with one KofamScan database call"
                        ),
                    )
                )
                merge_args: list[str | Path] = [
                    "merge-kofam",
                    "--output",
                    self.root / "annotation" / "merged" / "kegg.tsv",
                    "--input",
                    f"combined={kegg_raw}",
                ]
                kegg_merged = self.root / "annotation" / "merged" / "kegg.tsv"
                merge_task = self.add(
                    Task(
                        id="06.annotation.kegg.merge",
                        stage="06_kegg",
                        command=internal(*merge_args),
                        cwd=self.root,
                        deps=(kegg_task.id,),
                        inputs=(kegg_raw,),
                        outputs=(kegg_merged,),
                        description="Normalize combined KofamScan KEGG annotations",
                    )
                )
                format_args: list[str | Path] = [
                    "format-functional-annotations",
                    "--input",
                    kegg_merged,
                    "--output",
                    kegg_root / "kegg_annotations.tsv",
                    "--kind",
                    "kegg",
                    "--gene-map",
                    f"combined={combined_gene_map}",
                    "--ko-descriptions",
                    kegg_descriptions,
                ]
                kegg_final = kegg_root / "kegg_annotations.tsv"
                kegg_format_marker = (
                    kegg_root / ".kegg_annotations.ko-description-v1.complete"
                )
                self.add(
                    Task(
                        id="06.annotation.kegg.format",
                        stage="06_kegg",
                        command=(
                            f"{shlex.join(internal(*format_args))} && "
                            f"touch {q(kegg_format_marker)}"
                        ),
                        cwd=self.root,
                        deps=(merge_task.id,),
                        inputs=(kegg_merged, combined_gene_map, kegg_descriptions),
                        outputs=(kegg_final, kegg_format_marker),
                        description=(
                            "Apply the common MAG/contig/gene schema and KO gene "
                            "names to KEGG annotations"
                        ),
                    )
                )
            if self.options.run_cazy:
                cazy_root = self.options.output / "cazy"
                dbcan_db = self.options.dbcan_db or Path("DBCAN_DB_NOT_CONFIGURED")
                dbcan_view = self.root / "annotation" / "dbcan_database"
                dbcan_marker = dbcan_view / ".metabaw_dbcan_view.json"
                dbcan_task = self.add(
                    Task(
                        id="07.annotation.cazy.database",
                        stage="07_cazy",
                        command=internal(
                            "prepare-dbcan-database",
                            "--source",
                            dbcan_db,
                            "--output",
                            dbcan_view,
                            "--marker",
                            dbcan_marker,
                        ),
                        cwd=self.root,
                        inputs=(dbcan_db,),
                        outputs=(dbcan_marker,),
                        description="Prepare a run_dbCAN database compatibility view",
                    )
                )
                cazy_raw = cazy_root / "run_dbcan"
                overview_tsv = cazy_raw / "overview.tsv"
                overview_txt = cazy_raw / "overview.txt"
                cazy_task = self.add(
                    Task(
                        id="07.annotation.cazy",
                        stage="07_cazy",
                        command=(
                            f"rm -rf {q(cazy_raw)} && mkdir -p {q(cazy_root)} && "
                            f"run_dbcan CAZyme_annotation "
                            f"--input_raw_data {q(combined_proteins)} "
                            f"--mode protein --output_dir {q(cazy_raw)} "
                            f"--db_dir {q(dbcan_view)} --threads {functional_threads}"
                        ),
                        cwd=self.root,
                        deps=(combine_task.id, dbcan_task.id),
                        inputs=(combined_proteins, dbcan_marker),
                        outputs=(cazy_raw,),
                        output_alternatives=((overview_tsv,), (overview_txt,)),
                        cpus=functional_threads,
                        description=(
                            f"Annotate the combined protein set from {len(names)} MAGs "
                            "with one run_dbCAN database call"
                        ),
                    )
                )
                merge_args = [
                    "merge-dbcan",
                    "--output",
                    self.root / "annotation" / "merged" / "cazy.tsv",
                    "--input",
                    f"combined={cazy_raw}",
                ]
                cazy_merged = self.root / "annotation" / "merged" / "cazy.tsv"
                merge_task = self.add(
                    Task(
                        id="07.annotation.cazy.merge",
                        stage="07_cazy",
                        command=internal(*merge_args),
                        cwd=self.root,
                        deps=(cazy_task.id,),
                        inputs=(cazy_raw,),
                        outputs=(cazy_merged,),
                        description="Normalize combined run_dbCAN CAZy annotations",
                    )
                )
                format_args = [
                    "format-functional-annotations",
                    "--input",
                    cazy_merged,
                    "--output",
                    cazy_root / "cazy_annotations.tsv",
                    "--kind",
                    "cazy",
                    "--gene-map",
                    f"combined={combined_gene_map}",
                ]
                self.add(
                    Task(
                        id="07.annotation.cazy.format",
                        stage="07_cazy",
                        command=internal(*format_args),
                        cwd=self.root,
                        deps=(merge_task.id,),
                        inputs=(cazy_merged, combined_gene_map),
                        outputs=(cazy_root / "cazy_annotations.tsv",),
                        description="Apply the common MAG/contig/gene schema to CAZy annotations",
                    )
                )
            if self.options.run_hydrogenase:
                hydrogenase_root = self.options.output / "hydrogenase"
                hydrogenase_db = self.options.hydrogenase_db or Path(
                    "HYDROGENASE_DB_NOT_CONFIGURED"
                )
                blast_prefix = self.root / "annotation" / "hydrogenase" / "hyddb.all.fa"
                blast_index_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.database",
                        stage="08_hydrogenase",
                        command=(
                            f"rm -rf {q(blast_prefix.parent)} && "
                            f"mkdir -p {q(blast_prefix.parent)} && "
                            f"makeblastdb -in {q(hydrogenase_db / 'hyddb.all.fa')} "
                            f"-dbtype prot -parse_seqids -out {q(blast_prefix)}"
                        ),
                        cwd=self.root,
                        inputs=(hydrogenase_db / "hyddb.all.fa",),
                        outputs=(
                            Path(str(blast_prefix) + ".pin"),
                            Path(str(blast_prefix) + ".phr"),
                            Path(str(blast_prefix) + ".psq"),
                        ),
                        description="Build the reusable HydDB protein BLAST index",
                    )
                )
                output_dir = hydrogenase_root / "combined"
                raw_hits = output_dir / "HydDB.m8"
                filtered_hits = output_dir / "HydDB.filtered.tsv"
                fefe_fasta = output_dir / "Filtered.FeFe.faa"
                fefe_hits = output_dir / "FeFe_filtered_results.m6"
                nife_output = output_dir / "combined.NiFe.tsv"
                fe_output = output_dir / "combined.Fe.tsv"
                fefe_output = output_dir / "combined.FeFe.tsv"
                combined_output = output_dir / "combined.hydrogenases.tsv"
                search_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.search",
                        stage="08_hydrogenase",
                        command=(
                            f"mkdir -p {q(output_dir)} && rm -f {q(raw_hits)} && "
                            f"blastp -query {q(combined_proteins)} "
                            f"-db {q(blast_prefix)} -out {q(raw_hits)} "
                            f"-evalue 1e-50 -num_threads {functional_threads} "
                            f"-outfmt {q('6 qseqid sseqid pident length mismatch gapopen qstart qend qlen sstart send evalue bitscore')}"
                        ),
                        cwd=self.root,
                        deps=(combine_task.id, blast_index_task.id),
                        inputs=(
                            combined_proteins,
                            Path(str(blast_prefix) + ".pin"),
                            Path(str(blast_prefix) + ".phr"),
                            Path(str(blast_prefix) + ".psq"),
                        ),
                        outputs=(raw_hits,),
                        cpus=functional_threads,
                        description=(
                            f"Search the combined protein set from {len(names)} MAGs "
                            "against HydDB once"
                        ),
                    )
                )
                filter_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.filter",
                        stage="08_hydrogenase",
                        command=internal(
                            "filter-hydrogenase",
                            "--hits",
                            raw_hits,
                            "--proteins",
                            combined_proteins,
                            "--mapping",
                            hydrogenase_db / "hyd_id-name.script.txt",
                            "--genome",
                            "combined",
                            "--filtered-output",
                            filtered_hits,
                            "--fefe-fasta",
                            fefe_fasta,
                            "--min-query-coverage",
                            "0.9",
                            "--min-identity",
                            "0.5",
                        ),
                        cwd=self.root,
                        deps=(search_task.id,),
                        inputs=(
                            raw_hits,
                            combined_proteins,
                            hydrogenase_db / "hyd_id-name.script.txt",
                        ),
                        outputs=(filtered_hits, fefe_fasta),
                        description=(
                            "Keep best combined HydDB hits with coverage >= 0.9 "
                            "and identity >= 0.5"
                        ),
                    )
                )
                fefe_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.fefe",
                        stage="08_hydrogenase",
                        command=(
                            f"if test -s {q(fefe_fasta)}; then "
                            f"diamond blastp -d {q(hydrogenase_db / 'FeFe.dmnd')} "
                            f"-q {q(fefe_fasta)} -e 1e-50 -f 6 -o {q(fefe_hits)} "
                            f"-k 1 -p {functional_threads}; "
                            f"else : > {q(fefe_hits)}; fi"
                        ),
                        cwd=self.root,
                        deps=(filter_task.id,),
                        inputs=(fefe_fasta, hydrogenase_db / "FeFe.dmnd"),
                        outputs=(fefe_hits,),
                        cpus=functional_threads,
                        description="Confirm FeFe hydrogenases in the combined protein set",
                    )
                )
                final_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.finalize",
                        stage="08_hydrogenase",
                        command=internal(
                            "finalize-hydrogenase",
                            "--filtered",
                            filtered_hits,
                            "--fefe-hits",
                            fefe_hits,
                            "--nife-output",
                            nife_output,
                            "--fe-output",
                            fe_output,
                            "--fefe-output",
                            fefe_output,
                            "--combined-output",
                            combined_output,
                        ),
                        cwd=self.root,
                        deps=(filter_task.id, fefe_task.id),
                        inputs=(filtered_hits, fefe_hits),
                        outputs=(nife_output, fe_output, fefe_output, combined_output),
                        description="Combine hydrogenase annotations from the merged protein set",
                    )
                )
                merge_args = [
                    "merge-hydrogenase",
                    "--output",
                    self.root / "annotation" / "merged" / "hydrogenase.tsv",
                    "--input",
                    f"combined={combined_output}",
                ]
                hydrogenase_merged = (
                    self.root / "annotation" / "merged" / "hydrogenase.tsv"
                )
                merge_task = self.add(
                    Task(
                        id="08.annotation.hydrogenase.merge",
                        stage="08_hydrogenase",
                        command=internal(*merge_args),
                        cwd=self.root,
                        deps=(final_task.id,),
                        inputs=(combined_output,),
                        outputs=(hydrogenase_merged,),
                        description="Normalize combined hydrogenase annotations",
                    )
                )
                format_args = [
                    "format-functional-annotations",
                    "--input",
                    hydrogenase_merged,
                    "--output",
                    hydrogenase_root / "hydrogenase_annotations.tsv",
                    "--kind",
                    "hydrogenase",
                    "--gene-map",
                    f"combined={combined_gene_map}",
                ]
                self.add(
                    Task(
                        id="08.annotation.hydrogenase.format",
                        stage="08_hydrogenase",
                        command=internal(*format_args),
                        cwd=self.root,
                        deps=(merge_task.id,),
                        inputs=(hydrogenase_merged, combined_gene_map),
                        outputs=(hydrogenase_root / "hydrogenase_annotations.tsv",),
                        description=(
                            "Apply the common MAG/contig/gene schema to hydrogenases"
                        ),
                    )
                )
                terminal_root = self.options.output / "terminal_enzymes"
                terminal_dir = terminal_root / "combined"
                terminal_raw = terminal_dir / "Terminal.m8"
                terminal_filtered = terminal_dir / "combined.terminal_enzymes.tsv"
                terminal_search_task = self.add(
                    Task(
                        id="09.annotation.terminal.search",
                        stage="09_terminal_enzymes",
                        command=(
                            f"mkdir -p {q(terminal_dir)} && rm -f {q(terminal_raw)} && "
                            f"diamond blastp -d {q(hydrogenase_db / 'Terminal.dmnd')} "
                            f"-q {q(combined_proteins)} -e 1e-50 "
                            "--outfmt 6 qseqid sseqid stitle pident length "
                            "qstart qend qlen evalue bitscore "
                            f"-o {q(terminal_raw)} -k 25 -p {functional_threads}"
                        ),
                        cwd=self.root,
                        deps=(combine_task.id,),
                        inputs=(combined_proteins, hydrogenase_db / "Terminal.dmnd"),
                        outputs=(terminal_raw,),
                        cpus=functional_threads,
                        description=(
                            f"Search the combined protein set from {len(names)} MAGs "
                            "for hydrogen-metabolism terminal enzymes once"
                        ),
                    )
                )
                terminal_filter_task = self.add(
                    Task(
                        id="09.annotation.terminal.filter",
                        stage="09_terminal_enzymes",
                        command=internal(
                            "filter-terminal-enzymes",
                            "--hits",
                            terminal_raw,
                            "--genome",
                            "combined",
                            "--output",
                            terminal_filtered,
                            "--min-query-coverage",
                            "0.8",
                        ),
                        cwd=self.root,
                        deps=(terminal_search_task.id,),
                        inputs=(terminal_raw,),
                        outputs=(terminal_filtered,),
                        description="Apply terminal-enzyme filters to the combined search",
                    )
                )
                merge_args = [
                    "merge-terminal-enzymes",
                    "--output",
                    self.root / "annotation" / "merged" / "terminal.tsv",
                    "--input",
                    f"combined={terminal_filtered}",
                ]
                terminal_merged = self.root / "annotation" / "merged" / "terminal.tsv"
                merge_task = self.add(
                    Task(
                        id="09.annotation.terminal.merge",
                        stage="09_terminal_enzymes",
                        command=internal(*merge_args),
                        cwd=self.root,
                        deps=(terminal_filter_task.id,),
                        inputs=(terminal_filtered,),
                        outputs=(terminal_merged,),
                        description=(
                            "Merge hydrogen-metabolism terminal-enzyme annotations"
                        ),
                    )
                )
                format_args = [
                    "format-functional-annotations",
                    "--input",
                    terminal_merged,
                    "--output",
                    terminal_root / "terminal_enzyme_annotations.tsv",
                    "--kind",
                    "terminal",
                    "--gene-map",
                    f"combined={combined_gene_map}",
                ]
                self.add(
                    Task(
                        id="09.annotation.terminal.format",
                        stage="09_terminal_enzymes",
                        command=internal(*format_args),
                        cwd=self.root,
                        deps=(merge_task.id,),
                        inputs=(terminal_merged, combined_gene_map),
                        outputs=(terminal_root / "terminal_enzyme_annotations.tsv",),
                        description=(
                            "Apply the common MAG/contig/gene schema to terminal enzymes"
                        ),
                    )
                )
        return topological_order(self.tasks)

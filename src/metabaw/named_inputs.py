"""Named input manifests and mixed-technology assembly/binning planning."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from .direct import BINNER_ORDER, BinOptions, DirectBinBuilder, internal, safe_path_component
from .discovery import Analysis, FASTA_SUFFIXES, READ_SUFFIXES, ReadSample
from .model import Task


@dataclass(frozen=True)
class BinningProfile:
    read_type: str
    align_tool: str
    binners: tuple[str, ...]


def _rows(path: str | Path, columns: int):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input manifest is not a regular file: {source}")
    seen: dict[str, int] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in raw.split("\t")]
        if len(fields) != columns or not all(fields):
            raise ValueError(f"{source}:{line_number}: expected {columns} nonempty TAB-separated column(s)")
        name = fields[0]
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or "__mbw_" in name or safe_path_component(name) != name):
            raise ValueError(
                f"{source}:{line_number}: invalid sample name {name!r}; use letters, "
                "digits, underscores, dots or hyphens, starting with a letter/digit "
                "and not ending with a dot/underscore; "
                "__mbw_ is reserved for internal tasks"
            )
        if name in seen:
            raise ValueError(f"{source}:{line_number}: duplicate sample {name!r}; first listed on line {seen[name]}")
        seen[name] = line_number
        yield source, line_number, fields
    if not seen:
        raise ValueError(f"Input manifest contains no samples: {source}")


def _file(value: str, source: Path, line: int, suffixes: tuple[str, ...]) -> Path:
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else source.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{source}:{line}: input is not a regular file: {path}")
    if not path.name.lower().endswith(suffixes):
        raise ValueError(f"{source}:{line}: unsupported suffix: {path}; expected {', '.join(suffixes)}")
    return path


def read_named_reads(path: str | Path) -> list[ReadSample]:
    samples: list[ReadSample] = []
    seen_paths: dict[Path, int] = {}
    for source, line, fields in _rows(path, 2):
        name, reads = fields
        values = [value.strip() for value in reads.split(",")]
        if len(values) not in (1, 2) or not all(values):
            raise ValueError(f"{source}:{line}: expected SAMPLE<TAB>READ1,READ2 or SAMPLE<TAB>READ")
        paths = [_file(value, source, line, READ_SUFFIXES) for value in values]
        for item in paths:
            if item in seen_paths:
                raise ValueError(f"{source}:{line}: duplicate read path {item}; first listed on line {seen_paths[item]}")
            seen_paths[item] = line
        samples.append(ReadSample(name, paths[0], paths[1] if len(paths) == 2 else None, preserve_name=True))
    return samples


def read_named_contigs(path: str | Path) -> dict[str, Path]:
    return {fields[0]: _file(fields[1], source, line, FASTA_SUFFIXES)
            for source, line, fields in _rows(path, 2)}


def genome_name(path: Path) -> str:
    """Stable MAG ID shared by staged FASTA, GTDB-Tk and functional results."""
    suffix = next((s for s in FASTA_SUFFIXES if path.name.lower().endswith(s)), None)
    if suffix is None:
        raise ValueError(f"Unsupported genome FASTA suffix: {path}")
    name = path.name[:-len(suffix)]
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or safe_path_component(name) != name
            or name.lower().endswith(FASTA_SUFFIXES)):
        raise ValueError(f"Invalid/ambiguous genome name {name!r} from {path}; use a unique letters/digits/._- basename and one FASTA suffix")
    return name


def read_genome_files(path: str | Path) -> list[Path]:
    """Read exactly one genome path per line, without sample-name columns."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Genome manifest is not a regular file: {source}")
    genomes: list[Path] = []
    seen_paths: dict[Path, int] = {}
    seen_names: dict[str, int] = {}
    for line, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw:
            raise ValueError(f"{source}:{line}: expected one genome FASTA path, not TAB-separated columns")
        genome = _file(raw.strip(), source, line, FASTA_SUFFIXES)
        if genome in seen_paths:
            raise ValueError(f"{source}:{line}: duplicate genome path {genome}; first listed on line {seen_paths[genome]}")
        name = genome_name(genome).casefold()
        if name in seen_names:
            raise ValueError(f"{source}:{line}: duplicate genome name {genome_name(genome)!r} after removing FASTA suffix; first listed on line {seen_names[name]}")
        seen_paths[genome] = line
        seen_names[name] = line
        genomes.append(genome)
    if not genomes:
        raise ValueError(f"Genome manifest contains no genomes: {source}")
    return genomes


def attach_named_contigs(
    reads: list[ReadSample],
    contigs_path: str | Path | None,
) -> list[ReadSample]:
    """Attach optional contigs; missing entries are assembled from their reads."""
    read_names = {sample.name for sample in reads}
    contigs = read_named_contigs(contigs_path) if contigs_path is not None else {}
    extra = sorted(set(contigs) - read_names)
    if extra:
        raise ValueError(
            "--input_contig_files contains samples absent from "
            f"--input_reads_files: {', '.join(extra)}"
        )
    return [replace(sample, contigs=contigs.get(sample.name)) for sample in reads]


def load_named_samples(
    reads_path: str | Path, contigs_path: str | Path | None
) -> list[ReadSample]:
    return attach_named_contigs(read_named_reads(reads_path), contigs_path)


def read_multi_names(path: str | Path, samples: list[ReadSample]) -> list[str]:
    names = [fields[0] for _source, _line, fields in _rows(path, 1)]
    unknown = sorted(set(names) - {sample.name for sample in samples})
    if unknown:
        raise ValueError(f"--multi-files contains unknown samples: {', '.join(unknown)}")
    if len(names) < 2:
        raise ValueError("--multi-files requires at least two distinct sample names")
    return names


def read_coassembly_groups(
    path: str | Path, samples: list[ReadSample]
) -> dict[str, tuple[str, ...]]:
    """Read SAMPLE<TAB>GROUP rows for homogeneous explicit coassemblies."""
    groups: dict[str, list[str]] = {}
    for source, line, fields in _rows(path, 2):
        sample, group = fields
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group)
            or "__mbw_" in group
            or safe_path_component(group) != group
        ):
            raise ValueError(
                f"{source}:{line}: invalid coassembly group {group!r}; use "
                "letters, digits, underscores, dots or hyphens, starting with "
                "a letter/digit; __mbw_ is reserved for internal tasks"
            )
        groups.setdefault(group, []).append(sample)
    known = {sample.name for sample in samples}
    listed = {sample for members in groups.values() for sample in members}
    unknown = sorted(listed - known)
    if unknown:
        raise ValueError(
            "--coassembly-file contains unknown samples: " + ", ".join(unknown)
        )
    undersized = [group for group, members in groups.items() if len(members) < 2]
    if undersized:
        raise ValueError(
            "Each --coassembly-file group requires at least two distinct samples; "
            "invalid groups: " + ", ".join(undersized)
        )
    return {group: tuple(members) for group, members in groups.items()}


def sample_read_type(sample: ReadSample) -> str:
    return "short" if sample.read2 is not None else "long"


def eligible_binners(read_type: str, requested: list[str] | None) -> tuple[str, ...]:
    selected = requested if requested is not None else (
        ["metabat2", "metadecoder", "vamb"] if read_type == "short"
        else ["metadecoder", "vamb", "lorbin"]
    )
    excluded = "lorbin" if read_type == "short" else "metabat2"
    return tuple(tool for tool in BINNER_ORDER if tool in selected and tool != excluded)


def named_analyses(
    samples: list[ReadSample],
    group: list[str],
    assembly_root: Path | None = None,
) -> list[Analysis]:
    """Plan existing or per-sample generated assemblies and their coverage sets."""
    by_name = {sample.name: sample for sample in samples}
    coverage = tuple(by_name[name] for name in group)
    selected = set(group)
    analyses: list[Analysis] = []
    for sample in samples:
        assembly_samples: tuple[ReadSample, ...] = ()
        if sample.contigs is None:
            if assembly_root is None:
                raise ValueError(
                    f"Sample {sample.name!r} has no contigs and no assembly output root"
                )
            contigs = assembly_root / sample.name / f"{sample.name}_contig_ok.fa"
            assembly_samples = (sample,)
        else:
            contigs = sample.contigs
        analyses.append(Analysis(
            sample.name,
            coverage if sample.name in selected else (sample,),
            (contigs,),
            False,
            cross_mapped=sample.name in selected,
            assembly_samples=assembly_samples,
            public_name=sample.name,
            explicit_mapping=True,
            preserve_name=True,
        ))
    return analyses


def named_coassembly_analyses(
    samples: list[ReadSample],
    groups: dict[str, tuple[str, ...]],
    assembly_root: Path,
    individual_assembly_root: Path | None = None,
) -> list[Analysis]:
    """Create one explicit coassembly per group plus independent remainder."""
    by_name = {sample.name: sample for sample in samples}
    analyses: list[Analysis] = []
    grouped_samples: set[str] = set()
    for group, names in groups.items():
        analysis_name = f"coassembly_{group}"
        if analysis_name in by_name:
            raise ValueError(
                f"Coassembly name {analysis_name!r} conflicts with an input sample name"
            )
        members = tuple(by_name[name] for name in names)
        grouped_samples.update(names)
        analyses.append(
            Analysis(
                analysis_name,
                members,
                (assembly_root / analysis_name / f"{group}.contigs.ok.fa",),
                True,
                cross_mapped=True,
                assembly_samples=members,
                plan_id=group,
                # One differential-coverage bin set is produced per group.
                # Preserve every contributing sample in its public filename
                # without duplicating the same bins per member.
                public_name=f"{group}_{'-'.join(names)}",
                explicit_mapping=True,
                preserve_name=True,
            )
        )
    remaining = [sample for sample in samples if sample.name not in grouped_samples]
    analyses.extend(named_analyses(
        remaining,
        [],
        individual_assembly_root or assembly_root.parent / "individual",
    ))
    return analyses


class NamedBinBuilder(DirectBinBuilder):
    def __init__(self, analyses: list[Analysis], options: BinOptions, root: Path,
                 profiles: dict[str, BinningProfile]):
        super().__init__(analyses, options, root)
        self.profiles = profiles
        self.base_options = options

    def _semibin_input_name(self) -> str:
        return "__mbw_semibin2_multisample_input"

    def _use_semibin_cohort(self, analyses: list[Analysis]) -> bool:
        # multi_easy_bin needs labelled contigs from multiple assemblies.
        # A singleton technology cohort still uses every group coverage BAM,
        # via single_easy_bin, without creating an invalid unlabelled cohort.
        return len(analyses) > 1

    def _options_for_analysis(self, analysis: Analysis) -> BinOptions:
        profile = self.profiles[analysis.name]
        return replace(self.base_options, read_type=profile.read_type,
                       align_tool=profile.align_tool, binners=profile.binners)

    def _analysis(self, analysis: Analysis) -> tuple[Task, Path]:
        previous = self.options
        self.options = self._options_for_analysis(analysis)
        try:
            return super()._analysis(analysis)
        finally:
            self.options = previous

    def _vamb_uses_aemb(self, analysis: Analysis) -> bool:
        # A mixed coverage matrix must include long-read BAMs, not only AEMB
        # columns from the paired samples.
        return all(sample.read2 is not None for sample in analysis.samples)

    def _mapping(self, analysis: Analysis, prepare: Task, contigs: Path):
        groups: dict[tuple[str, str], list[ReadSample]] = {}
        for sample in analysis.samples:
            profile = self.profiles[sample.name]
            groups.setdefault((profile.read_type, profile.align_tool), []).append(sample)
        previous = self.options
        bams_by_name: dict[str, Path] = {}
        stages: list[Task] = []
        try:
            for (read_type, aligner), samples in groups.items():
                self.options = replace(previous, read_type=read_type, align_tool=aligner)
                if len(groups) == 1:
                    return super()._mapping(analysis, prepare, contigs)
                group_analysis = replace(
                    analysis, name=f"{analysis.name}__mbw_{read_type}",
                    samples=tuple(samples), preserve_name=True,
                )
                stage, bams, _directory = super()._mapping(group_analysis, prepare, contigs)
                stages.append(stage)
                bams_by_name.update((sample.name, bam) for sample, bam in zip(samples, bams, strict=True))
        finally:
            self.options = previous
        bams = [bams_by_name[sample.name] for sample in analysis.samples]
        bam_dir = previous.workdir / "mapping" / analysis.name / "bamset"
        arguments: list[str | Path] = ["stage-bams", "--output-dir", bam_dir]
        outputs = [bam_dir]
        for sample, bam in zip(analysis.samples, bams, strict=True):
            arguments.extend(("--bam", f"{sample.name}={bam}"))
            target = bam_dir / f"{safe_path_component(sample.name)}.bam"
            outputs.extend((target, Path(str(target) + ".bai")))
        stage = self.add(Task(
            id=f"02.bamset.{analysis.name}", stage="02_mapping",
            command=internal(*arguments), cwd=self.root,
            deps=tuple(task.id for task in stages), inputs=tuple(bams),
            outputs=tuple(outputs), description=f"Combine short/long coverage BAMs for {analysis.name}",
        ))
        return stage, bams, bam_dir

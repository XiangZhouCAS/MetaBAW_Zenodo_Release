from pathlib import Path
from tempfile import TemporaryDirectory

from metabaw.discovery import discover_reads


def test_long_read_numeric_suffix_is_part_of_sample_name():
    with TemporaryDirectory() as temporary:
        reads = Path(temporary)
        for name in ("HMI_1.fastq.gz", "HMI_2.fastq.gz", "LMI_1.fastq.gz"):
            (reads / name).write_bytes(b"reads")

        samples = discover_reads(reads, "fastq.gz", "long", ".")

    assert [sample.name for sample in samples] == ["HMI_1", "HMI_2", "LMI_1"]
    assert all(sample.read2 is None for sample in samples)


def test_long_read_configured_separator_still_selects_prefix():
    with TemporaryDirectory() as temporary:
        reads = Path(temporary)
        (reads / "sampleA.clean.fastq.gz").write_bytes(b"reads")

        samples = discover_reads(reads, "fastq.gz", "long", ".")

    assert [sample.name for sample in samples] == ["sampleA"]

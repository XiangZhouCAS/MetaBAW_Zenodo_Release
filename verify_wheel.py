"""Read-only integrity check of this edition's wheel and packaged Python source."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import sys
import tomllib
import zipfile


def verify_wheel(wheel: Path, root: Path) -> tuple[int, int]:
    """Reject extra wheel members as well as corrupt or stale source copies."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    dist_info = f"metabaw-{version}.dist-info"
    sources = sorted((root / "src" / "metabaw").rglob("*.py"))
    source_names = {path.relative_to(root / "src").as_posix() for path in sources}
    metadata_names = {
        f"{dist_info}/{name}" for name in
        ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD")
    }
    license_names = {f"{dist_info}/licenses/LICENSE", f"{dist_info}/LICENSE"}
    with zipfile.ZipFile(wheel) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"ZIP CRC failed: {broken}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel member paths")
        unexpected = set(names) - source_names - metadata_names - license_names
        missing = (source_names | metadata_names) - set(names)
        if unexpected or missing or not (set(names) & license_names):
            raise ValueError(
                f"Wheel content allowlist failed: unexpected={sorted(unexpected)}; "
                f"missing={sorted(missing)}; LICENSE is required"
            )
        record = f"{dist_info}/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
        if {row[0] for row in rows} != set(names) or len(rows) != len(names):
            raise ValueError("RECORD membership does not match ZIP contents")
        deprecated_suffix = (" (" + "named-input" + " edition)").encode("utf-8")
        for name, encoded_hash, size in rows:
            if name == record:
                if encoded_hash or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            content = archive.read(name)
            if deprecated_suffix in content:
                raise ValueError(f"Deprecated edition suffix in wheel: {name}")
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            if encoded_hash != f"sha256={actual}" or int(size) != len(content):
                raise ValueError(f"RECORD integrity mismatch: {name}")
        for path in sources:
            member = path.relative_to(root / "src").as_posix()
            if archive.read(member) != path.read_bytes():
                raise ValueError(f"Packaged source differs from source tree: {member}")
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        for required in (
            f"Version: {version}",
            "Summary: MetaBAW: Metagenome Binning Automated Workflow",
            "Requires-Python: <3.12,>=3.11",
        ):
            if required not in metadata:
                raise ValueError(f"Missing metadata: {required}")
    return len(rows), len(sources)


def main() -> None:
    root = Path(__file__).resolve().parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wheel = Path(sys.argv[1]) if len(sys.argv) > 1 else root / f"metabaw-{project['version']}-py3-none-any.whl"
    entries, sources = verify_wheel(wheel, root)
    print(f"OK: content allowlist, ZIP CRC, all {entries} RECORD entries, {sources} Python sources, metadata")
    print(f"Bytes: {wheel.stat().st_size}")
    print(f"SHA256: {hashlib.sha256(wheel.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

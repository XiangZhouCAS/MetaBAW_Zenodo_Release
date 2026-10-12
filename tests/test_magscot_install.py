"""Mocked official-source clone failures, retries and destination preservation."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from metabaw.dependencies import install_magscot, magscot_files


def complete(directory):
    for path in magscot_files(directory):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


@pytest.mark.parametrize("failures", [0, 1, 2, 3])
def test_clone_retries_http11_and_never_leaves_partial_destination(tmp_path, failures):
    destination = tmp_path / "MAGScoT"
    commands = []

    def clone(command, **kwargs):
        commands.append(command)
        directory = Path(command[-1])
        directory.mkdir()
        (directory / "partial").write_text("fixture")
        if len(commands) > failures:
            complete(directory)
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 128)

    with patch("metabaw.dependencies.subprocess.run", side_effect=clone), redirect_stdout(io.StringIO()):
        if failures == 3:
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                install_magscot(destination)
            assert not destination.exists()
        else:
            install_magscot(destination)
            assert all(path.is_file() for path in magscot_files(destination))
    assert len(commands) == min(failures + 1, 3)
    for command in commands[1:]:
        assert command[1:3] == ["-c", "http.version=HTTP/1.1"]
    assert not list(tmp_path.glob(".magscot-*"))


def test_timeout_retries_and_empty_destination_can_be_completed(tmp_path):
    destination = tmp_path / "MAGScoT"
    destination.mkdir()
    count = 0

    def clone(command, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        complete(Path(command[-1]))
        return subprocess.CompletedProcess(command, 0)

    with patch("metabaw.dependencies.subprocess.run", side_effect=clone), redirect_stdout(io.StringIO()):
        install_magscot(destination)
    assert count == 2
    assert all(path.is_file() for path in magscot_files(destination))


def test_existing_incomplete_user_directory_is_preserved(tmp_path):
    destination = tmp_path / "MAGScoT"
    destination.mkdir()
    original = destination / "my-file"
    original.write_text("user data")
    with patch("metabaw.dependencies.subprocess.run") as run, pytest.raises(RuntimeError, match="incomplete"):
        install_magscot(destination)
    run.assert_not_called()
    assert original.read_text() == "user data"


def test_complete_installation_does_not_download_again(tmp_path):
    complete(tmp_path)
    with patch("metabaw.dependencies.subprocess.run") as run:
        install_magscot(tmp_path)
    run.assert_not_called()

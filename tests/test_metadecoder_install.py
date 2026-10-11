from __future__ import annotations

from contextlib import ExitStack
import hashlib
from http.client import IncompleteRead
import io
import os
from pathlib import Path
import ssl
import subprocess
import sys
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

import pytest

from metabaw.dependencies import SOFTWARE, install_software
from metabaw import metadecoder_install as installer


PAYLOAD = b"PK\x03\x04synthetic pinned upstream wheel"


@pytest.fixture
def wheel_environment(tmp_path):
    # Downloads and installs are mocked; this is not a native MetaDecoder run.
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}, clear=True))
        stack.enter_context(patch.object(installer, "METADECODER_WHEEL_SIZE", len(PAYLOAD)))
        stack.enter_context(patch.object(installer, "METADECODER_WHEEL_SHA256", hashlib.sha256(PAYLOAD).hexdigest()))
        stack.enter_context(patch.object(installer.time, "sleep"))
        yield tmp_path


def response(payload=PAYLOAD, content_type="application/octet-stream"):
    handle = io.BytesIO(payload)
    handle.headers = {"Content-Type": content_type}
    return handle


def cache_path(root):
    return root / "metabaw" / "downloads" / installer.METADECODER_WHEEL_NAME


def local_wheel(root, payload=PAYLOAD):
    path = root / installer.METADECODER_WHEEL_NAME
    path.write_bytes(payload)
    return path


def test_defaults_increase_timeout_without_changing_parent_environment():
    with patch.dict(os.environ, {"HTTPS_PROXY": "https://gateway.invalid"}, clear=True):
        environment = installer.pip_install_environment()
        assert environment["PIP_DEFAULT_TIMEOUT"] == "120"
        assert environment["PIP_RETRIES"] == "5"
        assert environment["HTTPS_PROXY"] == os.environ["HTTPS_PROXY"]
        assert "PIP_DEFAULT_TIMEOUT" not in os.environ


@pytest.mark.parametrize("timeout_key", ["PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT"])
def test_explicit_pip_environment_is_preserved(timeout_key):
    configured = {timeout_key: "240", "PIP_RETRIES": "0", "PIP_CERT": "/trusted/ca.pem"}
    with patch.dict(os.environ, configured, clear=True):
        environment = installer.pip_install_environment()
        assert all(environment[key] == value for key, value in configured.items())
        assert installer._network_timeout() == 240


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_socket_timeout_is_rejected(value):
    with patch.dict(os.environ, {"PIP_TIMEOUT": value}, clear=True):
        with pytest.raises(ValueError):
            installer._network_timeout()


def test_release_download_is_verified_cached_and_reused(wheel_environment):
    with patch.object(installer, "urlopen", return_value=response()) as network:
        result = installer.prepare_metadecoder_wheel()
    assert result == cache_path(wheel_environment)
    assert result.read_bytes() == PAYLOAD
    request = network.call_args.args[0]
    assert request.full_url == installer.METADECODER_RELEASE_URL
    assert network.call_args.kwargs["timeout"] == 120
    assert request.get_header("Accept") == "application/octet-stream"
    assert list(result.parent.iterdir()) == [result]  # No partial downloads remain.
    with patch.object(installer, "urlopen") as network:
        assert installer.prepare_metadecoder_wheel() == result
        network.assert_not_called()


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), URLError("connection timed out"),
                                    HTTPError(installer.METADECODER_RELEASE_URL, 503, "Unavailable", {}, None)])
def test_transient_release_failure_uses_only_official_api(wheel_environment, failure):
    with patch.object(installer, "urlopen", side_effect=[failure, failure, response()]) as network:
        result = installer.prepare_metadecoder_wheel()
    urls = [call.args[0].full_url for call in network.call_args_list]
    assert urls == [installer.METADECODER_RELEASE_URL] * 2 + [installer.METADECODER_ASSET_URL]
    assert result.read_bytes() == PAYLOAD
    assert network.call_args.args[0].get_header("X-github-api-version") == "2022-11-28"


def test_permanent_http_error_is_not_repeated(wheel_environment):
    forbidden = HTTPError(installer.METADECODER_RELEASE_URL, 403, "Forbidden", {}, None)
    with patch.object(installer, "urlopen", side_effect=[forbidden, response()]) as network:
        installer.prepare_metadecoder_wheel()
    assert network.call_count == 2


def test_truncated_download_is_retried_before_verification(wheel_environment):
    with patch.object(installer, "urlopen", side_effect=[response(PAYLOAD[:4]), response()]) as network:
        result = installer.prepare_metadecoder_wheel()
    assert network.call_count == 2
    assert result.read_bytes() == PAYLOAD


def test_exhausted_network_has_offline_guidance_and_no_partial_cache(wheel_environment):
    with patch.object(installer, "urlopen", side_effect=URLError("timed out")) as network:
        with pytest.raises(RuntimeError, match="METABAW_METADECODER_WHEEL"):
            installer.prepare_metadecoder_wheel()
    assert network.call_count == 4
    assert list(cache_path(wheel_environment).parent.iterdir()) == []


@pytest.mark.parametrize("payload", [b"X" * len(PAYLOAD), PAYLOAD + b"untrusted"])
def test_integrity_failure_aborts_without_fallback_or_cache(wheel_environment, payload):
    with patch.object(installer, "urlopen", return_value=response(payload)) as network:
        with pytest.raises(ValueError):
            installer.prepare_metadecoder_wheel()
    assert network.call_count == 1
    assert list(cache_path(wheel_environment).parent.iterdir()) == []


def test_certificate_verification_failure_is_not_bypassed(wheel_environment):
    error = URLError(ssl.SSLCertVerificationError("untrusted certificate"))
    with patch.object(installer, "urlopen", side_effect=error) as network:
        with pytest.raises(RuntimeError, match="certificate verification failed"):
            installer.prepare_metadecoder_wheel()
    assert network.call_count == 1


def test_json_metadata_is_not_accepted_as_a_wheel(wheel_environment):
    with patch.object(installer, "urlopen", side_effect=lambda *args, **kwargs: response(b"{}", "application/json")):
        with pytest.raises(RuntimeError, match="Both official"):
            installer.prepare_metadecoder_wheel()
    assert not cache_path(wheel_environment).exists()


def test_bad_cache_is_verified_then_replaced(wheel_environment):
    destination = cache_path(wheel_environment)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"X" * len(PAYLOAD))
    with patch.object(installer, "urlopen", return_value=response()) as network:
        assert installer.prepare_metadecoder_wheel() == destination
    network.assert_called_once()
    assert destination.read_bytes() == PAYLOAD


def test_local_wheel_bypasses_all_network_and_cache(wheel_environment):
    path = local_wheel(wheel_environment)
    with patch.dict(os.environ, {installer.METADECODER_LOCAL_WHEEL_ENV: str(path)}), \
            patch.object(installer, "urlopen") as network:
        assert installer.prepare_metadecoder_wheel() == path
    network.assert_not_called()
    assert not cache_path(wheel_environment).parent.exists()


@pytest.mark.parametrize("case", ["missing", "wrong-name", "wrong-size", "wrong-digest", "empty"])
def test_invalid_explicit_local_wheel_never_falls_back(wheel_environment, case):
    path = local_wheel(wheel_environment)
    if case == "missing":
        path = wheel_environment / "missing.whl"
    elif case == "wrong-name":
        path = path.rename(wheel_environment / "other-1.2.2-py3-none-any.whl")
    elif case == "wrong-size":
        path.write_bytes(b"bad")
    elif case == "wrong-digest":
        path.write_bytes(b"X" * len(PAYLOAD))
    configured = "" if case == "empty" else str(path)
    with patch.dict(os.environ, {installer.METADECODER_LOCAL_WHEEL_ENV: configured}), \
            patch.object(installer, "urlopen") as network:
        with pytest.raises(ValueError):
            installer.prepare_metadecoder_wheel()
    network.assert_not_called()


def test_explicit_proxy_and_certificates_are_used(wheel_environment):
    context = Mock()
    opener = Mock()
    opener.open.return_value = response()
    with patch.dict(os.environ, {"PIP_PROXY": "https://gateway.invalid", "PIP_CERT": "/trusted/ca.pem",
                                 "PIP_CLIENT_CERT": "/trusted/client.pem"}), \
            patch.object(installer.ssl, "create_default_context", return_value=context) as ssl_context, \
            patch.object(installer, "build_opener", return_value=opener), \
            patch.object(installer, "urlopen") as direct:
        installer.prepare_metadecoder_wheel()
    ssl_context.assert_called_once_with(cafile="/trusted/ca.pem")
    context.load_cert_chain.assert_called_once_with("/trusted/client.pem")
    assert opener.open.call_args.kwargs["timeout"] == 120
    direct.assert_not_called()


@pytest.mark.parametrize("launcher_missing", [False, True])
def test_pip_installs_verified_local_wheel_under_main_python(wheel_environment, launcher_missing):
    requirements = [SOFTWARE["metadecoder"]]
    path = local_wheel(wheel_environment)
    with patch.dict(os.environ, {installer.METADECODER_LOCAL_WHEEL_ENV: str(path)}), \
            patch("metabaw.dependencies.missing_software", side_effect=[requirements, []]), \
            patch("metabaw.dependencies.software_runtime_issues", return_value=[]), \
            patch("metabaw.dependencies.metadecoder_executable", return_value=None if launcher_missing else "/main/bin/metadecoder"), \
            patch("metabaw.dependencies.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run, \
            patch("metabaw.dependencies.run_conda_transaction") as conda, \
            patch.object(installer, "urlopen") as network:
        assert install_software(requirements) == []
    assert run.call_count == (2 if launcher_missing else 1)
    for call in run.call_args_list:
        assert call.args[0][:4] == [sys.executable, "-m", "pip", "install"]
        assert call.args[0][-1] == str(path)
        assert call.kwargs["env"]["PIP_DEFAULT_TIMEOUT"] == "120"
    if launcher_missing:
        assert "--force-reinstall" in run.call_args.args[0]
        assert "--no-deps" in run.call_args.args[0]
    conda.assert_not_called()
    network.assert_not_called()


def test_failed_download_does_not_run_pip_or_block_other_software(wheel_environment):
    requirements = [SOFTWARE["metadecoder"], SOFTWARE["vamb"]]
    with patch("metabaw.dependencies.missing_software", side_effect=[requirements, requirements[:1]]), \
            patch("metabaw.dependencies.software_runtime_issues", return_value=[]), \
            patch("metabaw.dependencies.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run, \
            patch.object(installer, "urlopen", side_effect=TimeoutError("timed out")):
        assert install_software(requirements) == requirements[:1]
    run.assert_called_once()
    assert run.call_args.args[0][-1] == "vamb"


def test_pip_failure_is_not_retried_as_a_download_or_environment_change(wheel_environment):
    requirements = [SOFTWARE["metadecoder"]]
    path = local_wheel(wheel_environment)
    with patch.dict(os.environ, {installer.METADECODER_LOCAL_WHEEL_ENV: str(path)}), \
            patch("metabaw.dependencies.missing_software", return_value=requirements), \
            patch("metabaw.dependencies.software_runtime_issues", return_value=[]), \
            patch("metabaw.dependencies.metadecoder_executable") as resolver, \
            patch("metabaw.dependencies.subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as run, \
            patch.object(installer, "urlopen") as network:
        assert install_software(requirements) == requirements
    run.assert_called_once()
    resolver.assert_not_called()
    network.assert_not_called()

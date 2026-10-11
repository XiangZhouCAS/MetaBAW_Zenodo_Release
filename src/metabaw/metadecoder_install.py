"""Fetch the pinned upstream wheel without depending on one GitHub route."""
from __future__ import annotations

import hashlib
from http.client import IncompleteRead
import math
import os
from pathlib import Path
import ssl
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener, urlopen


METADECODER_WHEEL_NAME = "metadecoder-1.2.2-py3-none-any.whl"
METADECODER_RELEASE_URL = (
    "https://github.com/liu-congcong/MetaDecoder/releases/download/"
    f"v1.2.2/{METADECODER_WHEEL_NAME}"
)
METADECODER_ASSET_URL = (
    "https://api.github.com/repos/liu-congcong/MetaDecoder/releases/assets/327683434"
)
# Size/digest published by the upstream v1.2.2 GitHub release asset API.
METADECODER_WHEEL_SIZE = 5_163_705
METADECODER_WHEEL_SHA256 = (
    "131421c57ced53d4947b7648ba1fd36c288b9b54107fe2b6179552094f4c61a7"
)
METADECODER_LOCAL_WHEEL_ENV = "METABAW_METADECODER_WHEEL"


def pip_install_environment() -> dict[str, str]:
    """Allow slow connections without changing global/user pip configuration."""
    environment = os.environ.copy()
    if not any(environment.get(key) for key in ("PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT")):
        environment["PIP_DEFAULT_TIMEOUT"] = "120"
    environment.setdefault("PIP_RETRIES", "5")
    return environment


def _network_timeout() -> float:
    environment = pip_install_environment()
    value = environment.get("PIP_TIMEOUT") or environment["PIP_DEFAULT_TIMEOUT"]
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("PIP_TIMEOUT/PIP_DEFAULT_TIMEOUT must be a positive finite number")
    return timeout


def _verify_wheel(path: Path) -> None:
    if not path.is_file() or path.name != METADECODER_WHEEL_NAME:
        raise ValueError(f"Expected a local {METADECODER_WHEEL_NAME} file: {path}")
    if path.stat().st_size != METADECODER_WHEEL_SIZE:
        raise ValueError(f"MetaDecoder wheel size mismatch; not installing {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != METADECODER_WHEEL_SHA256:
        raise ValueError(f"MetaDecoder wheel SHA-256 mismatch; not installing {path}")


def _download_wheel(url: str, destination: Path, timeout: float) -> None:
    request = Request(url, headers={
        "Accept": "application/octet-stream",
        "User-Agent": "MetaBAW/0.1.0",
        "X-GitHub-Api-Version": "2022-11-28",
        "Cache-Control": "no-cache",
    })
    # urllib already honors HTTPS_PROXY/HTTP_PROXY. Also preserve explicit pip
    # proxy/CA settings instead of bypassing the user's corporate gateway.
    proxy = os.environ.get("PIP_PROXY")
    certificate = (os.environ.get("PIP_CERT") or os.environ.get("REQUESTS_CA_BUNDLE")
                   or os.environ.get("CURL_CA_BUNDLE"))
    client_certificate = os.environ.get("PIP_CLIENT_CERT")
    open_url = urlopen
    if proxy or certificate or client_certificate:
        context = ssl.create_default_context(cafile=certificate)
        if client_certificate:
            context.load_cert_chain(client_certificate)
        handlers = [HTTPSHandler(context=context)]
        if proxy:
            handlers.append(ProxyHandler({"https": proxy, "http": proxy}))
        open_url = build_opener(*handlers).open
    with open_url(request, timeout=timeout) as response, destination.open("wb") as handle:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "json" in content_type or "text/" in content_type:
            raise URLError("Official download endpoint returned metadata/text, not a wheel")
        downloaded = 0
        while block := response.read(64 * 1024):
            downloaded += len(block)
            if downloaded > METADECODER_WHEEL_SIZE:
                raise ValueError("MetaDecoder download exceeds the pinned wheel size")
            handle.write(block)
        if downloaded < METADECODER_WHEEL_SIZE:
            raise IncompleteRead(b"", METADECODER_WHEEL_SIZE - downloaded)
    _verify_wheel(destination)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        # A timeout workaround must not weaken certificate verification.
        raise RuntimeError("MetaDecoder HTTPS certificate verification failed") from exc
    return True


def prepare_metadecoder_wheel() -> Path:
    """Return a checksum-verified local wheel; never use an untrusted mirror."""
    configured = os.environ.get(METADECODER_LOCAL_WHEEL_ENV)
    if configured is not None:
        if not configured.strip():
            raise ValueError(f"{METADECODER_LOCAL_WHEEL_ENV} cannot be empty")
        path = Path(configured).expanduser().absolute()
        _verify_wheel(path)
        print(f"[INSTALL SOURCE] Verified local MetaDecoder wheel: {path}", flush=True)
        return path

    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    directory = root / "metabaw" / "downloads"
    destination = directory / METADECODER_WHEEL_NAME
    if destination.exists():
        try:
            _verify_wheel(destination)
        except ValueError:
            print("[DOWNLOAD] Ignoring an invalid MetaDecoder cache entry", flush=True)
        else:
            print(f"[INSTALL SOURCE] Verified cached MetaDecoder wheel: {destination}", flush=True)
            return destination.absolute()

    timeout = _network_timeout()
    directory.mkdir(parents=True, exist_ok=True)
    sources = (("GitHub Releases", METADECODER_RELEASE_URL), ("GitHub API", METADECODER_ASSET_URL))
    with tempfile.TemporaryDirectory(prefix="metadecoder-", dir=directory) as temporary:
        partial = Path(temporary) / METADECODER_WHEEL_NAME
        for label, url in sources:
            for attempt in range(1, 3):
                print(f"[DOWNLOAD] MetaDecoder 1.2.2 via {label}; attempt {attempt}/2; "
                      f"socket timeout={timeout:g}s", flush=True)
                try:
                    _download_wheel(url, partial, timeout)
                except (HTTPError, URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                    retry = _retryable(exc)
                    # Do not print signed redirect URLs or proxy credentials.
                    detail = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
                    print(f"[DOWNLOAD WARNING] {label}: {detail}", file=sys.stderr, flush=True)
                    if not retry or attempt == 2:
                        break
                    time.sleep(1)
                else:
                    partial.replace(destination)
                    print(f"[DOWNLOAD OK] MetaDecoder SHA-256 verified; cached at {destination}", flush=True)
                    return destination.absolute()
    raise RuntimeError(
        "Both official MetaDecoder download routes failed. Download "
        f"{METADECODER_WHEEL_NAME} from {METADECODER_RELEASE_URL} on a machine "
        "with GitHub access, transfer it to this server, and set "
        f"{METADECODER_LOCAL_WHEEL_ENV} to its local path before rerunning "
        "metabaw check --scope bin. A local wheel avoids GitHub, but numerical "
        "dependencies and CuPy must still be available. No untrusted mirror "
        "or different Python environment was used."
    )

"""Tiny object store: local folders by default; `gs://bucket/path` URIs work too if google-cloud-storage is installed and credentials are set."""
from __future__ import annotations

from pathlib import Path


def _gcs():
    from google.cloud import storage  # optional dependency
    return storage.Client()


def _split(uri: str):
    b, _, o = uri.replace("gs://", "").partition("/")
    return b, o


def read_bytes(uri: str) -> bytes:
    if not str(uri).startswith("gs://"):
        return Path(uri).read_bytes()
    b, o = _split(uri)
    return _gcs().bucket(b).blob(o).download_as_bytes()


def write_bytes(uri: str, data: bytes, ctype: str = "application/octet-stream") -> None:
    if not str(uri).startswith("gs://"):
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return
    b, o = _split(uri)
    _gcs().bucket(b).blob(o).upload_from_string(data, content_type=ctype)


def exists(uri: str) -> bool:
    if not str(uri).startswith("gs://"):
        return Path(uri).exists()
    b, o = _split(uri)
    return _gcs().bucket(b).blob(o).exists()


def list_names(base: str, prefix: str) -> list[str]:
    """Object names relative to `base` below `prefix`, recursively."""
    base = base.rstrip("/")
    uri = f"{base}/{prefix}".rstrip("/")
    if not uri.startswith("gs://"):
        p = Path(uri)
        return sorted(str(q.relative_to(base)) for q in p.rglob("*") if q.is_file()) if p.exists() else []
    b, o = _split(uri + "/")
    return sorted(x.name for x in _gcs().list_blobs(b, prefix=o) if not x.name.endswith("/"))

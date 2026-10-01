from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/datasets"
SUBJECT_RE = re.compile(r"^S\d+$")


@dataclass(frozen=True)
class HarthHubFile:
    path: str
    subject: str
    oid: str | None = None
    size: int | None = None


def _request_json(url: str, *, timeout: int = 60) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "jev-benchmark/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_subject_files(
    repo_id: str,
    *,
    revision: str = "main",
    timeout: int = 60,
) -> list[HarthHubFile]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"{HF_API}/{encoded_repo}/tree/{encoded_revision}?recursive=true&expand=true"
    payload = _request_json(url, timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Hugging Face tree response for {repo_id}")

    files: list[HarthHubFile] = []
    for item in payload:
        path = item.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".csv"):
            continue
        subject = Path(path).stem
        if not SUBJECT_RE.fullmatch(subject):
            continue
        files.append(
            HarthHubFile(
                path=path,
                subject=subject,
                oid=item.get("oid"),
                size=int(item["size"]) if item.get("size") is not None else None,
            )
        )

    files.sort(key=lambda item: item.subject)
    if not files:
        raise FileNotFoundError(f"No HARTH subject CSV files found in Hugging Face dataset {repo_id}")
    return files


def _resolve_url(repo_id: str, revision: str, path: str) -> str:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    encoded_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/datasets/{encoded_repo}/resolve/{encoded_revision}/{encoded_path}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_subject_files(
    repo_id: str,
    output_dir: str | Path,
    *,
    subjects: list[str] | None = None,
    revision: str = "main",
    force: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    available = list_subject_files(repo_id, revision=revision, timeout=timeout)
    requested = set(subjects) if subjects else None
    selected = [
        item for item in available
        if requested is None or item.subject in requested
    ]
    missing = sorted(requested - {item.subject for item in selected}) if requested else []
    if missing:
        raise FileNotFoundError(
            f"Requested HARTH subjects are missing from {repo_id}: {missing}"
        )

    manifest_files: list[dict[str, Any]] = []
    for item in selected:
        target = destination / f"{item.subject}.csv"
        if target.exists() and not force:
            pass
        else:
            partial = target.with_suffix(".csv.part")
            if partial.exists():
                partial.unlink()
            url = _resolve_url(repo_id, revision, item.path)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "jev-benchmark/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            partial.replace(target)

        manifest_files.append(
            {
                "subject": item.subject,
                "source_path": item.path,
                "source_oid": item.oid,
                "source_size": item.size,
                "local_path": target.name,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )

    manifest = {
        "schema_version": "jev-benchmark.harth-hf/v1",
        "repo_id": repo_id,
        "revision": revision,
        "subjects": [item["subject"] for item in manifest_files],
        "subject_count": len(manifest_files),
        "files": manifest_files,
    }
    manifest_path = destination / "harth-hf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

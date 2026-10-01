from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

HF_REPO = "High-Light/jev-harth"
HF_REVISION = "main"
HF_FILENAME = "harth.zip"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILENAME}?download=true"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and Path(name).name.upper().startswith("S")
    )


def inspect_archive(archive_path: str | Path) -> dict:
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        names = archive.namelist()
        members = _csv_members(archive)
    subjects = sorted(Path(name).stem for name in members)
    return {
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "zip_members": len(names),
        "subject_csv_members": members,
        "subjects": subjects,
        "subject_count": len(subjects),
    }


def download_harth(
    output_dir: str | Path,
    *,
    url: str = HF_URL,
    force: bool = False,
    extract: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / HF_FILENAME

    if archive_path.exists() and not force:
        downloaded = False
    else:
        tmp = archive_path.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            tmp.replace(archive_path)
        finally:
            tmp.unlink(missing_ok=True)
        downloaded = True

    manifest = inspect_archive(archive_path)
    manifest.update({
        "source": "huggingface",
        "repo_id": HF_REPO,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "source_url": url,
        "downloaded": downloaded,
    })

    if extract:
        extract_dir = output_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        csv_files = sorted(extract_dir.rglob("S*.csv"))
        manifest["extracted_dir"] = str(extract_dir)
        manifest["extracted_subjects"] = sorted(p.stem for p in csv_files)
        manifest["extracted_subject_count"] = len(csv_files)
        if manifest["extracted_subject_count"] == 0:
            raise ValueError("HARTH archive contains no extracted S*.csv files")

    manifest_path = output_dir / "harth-manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

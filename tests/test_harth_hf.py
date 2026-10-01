from __future__ import annotations

import json
from pathlib import Path

from jev_bench.datasets import harth_hf


def test_list_subject_files(monkeypatch):
    def fake_request(url: str, *, timeout: int = 60):
        return [
            {"path": "README.md", "oid": "doc"},
            {"path": "data/S002.csv", "oid": "b", "size": 20},
            {"path": "data/S001.csv", "oid": "a", "size": 10},
            {"path": "notes.csv", "oid": "n", "size": 3},
        ]

    monkeypatch.setattr(harth_hf, "_request_json", fake_request)
    files = harth_hf.list_subject_files("High-Light/jev-harth")
    assert [f.subject for f in files] == ["S001", "S002"]
    assert files[0].path == "data/S001.csv"


def test_download_subject_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        harth_hf,
        "list_subject_files",
        lambda repo_id, revision="main", timeout=60: [
            harth_hf.HarthHubFile("S001.csv", "S001", "oid1", 6),
            harth_hf.HarthHubFile("S002.csv", "S002", "oid2", 6),
        ],
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"header\n"

    monkeypatch.setattr(harth_hf.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = harth_hf.download_subject_files(
        "High-Light/jev-harth",
        tmp_path,
        subjects=["S001"],
    )
    assert manifest["subjects"] == ["S001"]
    assert (tmp_path / "S001.csv").read_text() == "header\n"
    saved = json.loads((tmp_path / "harth-hf-manifest.json").read_text())
    assert saved["files"][0]["source_oid"] == "oid1"


def test_missing_requested_subject_fails(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        harth_hf,
        "list_subject_files",
        lambda *args, **kwargs: [
            harth_hf.HarthHubFile("S001.csv", "S001"),
        ],
    )
    try:
        harth_hf.download_subject_files("High-Light/jev-harth", tmp_path, subjects=["S002"])
    except FileNotFoundError as exc:
        assert "S002" in str(exc)
    else:
        raise AssertionError("missing subject did not fail")

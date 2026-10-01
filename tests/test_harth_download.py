import zipfile

from jev_bench.datasets.harth_download import inspect_archive


def test_inspect_archive_reports_subject_csvs(tmp_path):
    archive = tmp_path / "harth.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for subject in ("S001", "S002"):
            zf.writestr(
                f"{subject}.csv",
                "timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label\n",
            )
        zf.writestr("README.txt", "test")

    result = inspect_archive(archive)
    assert result["subject_count"] == 2
    assert result["subjects"] == ["S001", "S002"]
    assert len(result["archive_sha256"]) == 64


def test_inspect_archive_rejects_corrupt_zip(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not a zip")
    try:
        inspect_archive(archive)
    except zipfile.BadZipFile:
        pass
    else:
        raise AssertionError("corrupt ZIP should fail")

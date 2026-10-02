import json, tempfile, subprocess, sys
from pathlib import Path

def test_analyze_ns01_script_exists():
    path=Path("scripts/analyze_ns01.py")
    assert path.exists()
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

def test_ns01_workflow_has_analysis_job():
    text=Path(".github/workflows/ns01-operating-point.yml").read_text(encoding="utf-8")
    assert "needs: ns01" in text
    assert "analyze_ns01.py" in text

from jev_bench.sweep import run_j01_size_sweep

def test_j01_size_sweep_returns_comparable_rows():
    rows = run_j01_size_sweep([1, 2], episodes=1, seed=3)
    assert len(rows) == 2
    assert rows[0]["hidden_units"] == 1
    assert rows[1]["hidden_units"] == 2
    assert rows[0]["model_size_bytes_fp32"] < rows[1]["model_size_bytes_fp32"]
    for row in rows:
        assert "mean_return" in row
        assert "p95_action_latency_us" in row
        assert "mean_confidence" in row

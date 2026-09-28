from jev_bench.core.results import BenchmarkResult

def test_result_serialization_shape():
    r=BenchmarkResult(task="j01_cartpole",policy="tiny_mlp",episodes=1,seed=0,metrics={"mean_return":1.0})
    d=r.to_dict()
    assert d["task"] == "j01_cartpole"
    assert d["metrics"]["mean_return"] == 1.0

from jev_bench.analysis import aggregate_loso, pareto_front

def test_aggregate_loso_mean_std():
    rows=[
        {"test_subject":"S001","accuracy":0.8,"macro_f1":0.7,"mean_confidence":0.9,"abstention_rate":0.1,"mean_inference_latency_us":10.0,"train_seconds":1.0,"risk_coverage":[{"threshold":0.0,"coverage":1.0,"risk":0.2}]},
        {"test_subject":"S002","accuracy":0.6,"macro_f1":0.5,"mean_confidence":0.8,"abstention_rate":0.2,"mean_inference_latency_us":20.0,"train_seconds":3.0,"risk_coverage":[{"threshold":0.0,"coverage":1.0,"risk":0.4}]},
    ]
    out=aggregate_loso(rows)
    assert out["subject_count"]==2
    assert out["metrics"]["macro_f1"]["mean"]==0.6
    assert out["risk_coverage"][0]["coverage_mean"]==1.0
    assert out["risk_coverage"][0]["risk_mean"]==0.3

def test_pareto_front_removes_dominated_row():
    rows=[
        {"id":"a","macro_f1":0.80,"mean_inference_latency_us":10,"model_size_bytes_fp32":100},
        {"id":"b","macro_f1":0.80,"mean_inference_latency_us":20,"model_size_bytes_fp32":200},
        {"id":"c","macro_f1":0.85,"mean_inference_latency_us":20,"model_size_bytes_fp32":200},
    ]
    front=pareto_front(rows)
    assert [r["id"] for r in front]==["a","c"]

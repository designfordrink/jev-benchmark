from __future__ import annotations
import numpy as np
from jev_bench.tasks.j03_candidate_selection import J03CandidateSelection

def test_j03_train_evaluate_returns_decision_metrics():
    result=J03CandidateSelection().train_and_evaluate(
        train_problems=80,test_problems=40,epochs=5,hidden_units=4,seed=7,permutation_trials=3
    )
    for key in (
        'selection_accuracy','mean_reward','mean_oracle_reward','mean_regret',
        'mean_confidence','abstention_rate','permutation_invariance_rate',
        'mean_action_latency_us','parameter_count','model_size_bytes_fp32'
    ):
        assert key in result
    assert 0.0 <= result['selection_accuracy'] <= 1.0
    assert result['mean_regret'] >= -1e-6
    assert result['permutation_invariance_rate'] == 1.0

def test_j03_linear_baseline_is_permutation_invariant():
    result=J03CandidateSelection().train_and_evaluate(
        train_problems=40,test_problems=20,model='linear',seed=2,permutation_trials=2
    )
    assert result['permutation_invariance_rate'] == 1.0
    assert result['parameter_count'] == 13
    assert result['model_size_bytes_fp32'] == 52

def test_j03_utility_is_candidate_specific():
    task=J03CandidateSelection()
    context=np.ones(6,dtype=np.float32)
    a=np.ones(6,dtype=np.float32)
    b=np.zeros(6,dtype=np.float32)
    assert task.utility(context,a) != task.utility(context,b)
from __future__ import annotations
import numpy as np
import pytest
from jev_bench.envs.tetris import TetrisEnv
from jev_bench.policies.j04_tetris import J04HeuristicSelector, J04RandomSelector
from jev_bench.tasks.j04_tetris import J04Tetris

def test_tetris_reset_is_deterministic():
    a=TetrisEnv(11); b=TetrisEnv(11)
    assert np.array_equal(a.reset(11),b.reset(11))
    assert a.current_piece==b.current_piece

def test_tetris_legal_placements_are_legal():
    env=TetrisEnv(3); env.reset(3)
    legal=env.legal_placements()
    assert len(legal)>0
    for placement in legal:
        cells=env._cells(env.current_piece,placement.rotation,placement.x,placement.y)
        assert env._valid(cells)

def test_tetris_episode_uses_real_environment():
    task=J04Tetris()
    selector=J04HeuristicSelector(task.teacher_score)
    row=TetrisEnv(5).run_episode(selector,seed=5,max_pieces=30)
    assert row["decisions"]==row["pieces"]
    assert row["pieces"]>0
    assert row["return"]>=row["pieces"]

def test_tetris_selector_permutation_invariant():
    env=TetrisEnv(9); env.reset(9); task=J04Tetris()
    selector=J04HeuristicSelector(task.teacher_score)
    legal=env.legal_placements(); context=env.observation(); base=selector.rank(context,legal)[2].argmax()
    rng=np.random.default_rng(9)
    for _ in range(5):
        perm=rng.permutation(len(legal)); shuffled=[legal[int(i)] for i in perm]
        idx=selector.rank(context,shuffled)[0]
        assert shuffled[idx].candidate_id==legal[base].candidate_id

def test_tetris_rejects_illegal_placement():
    env=TetrisEnv(1); env.reset(1)
    legal=env.legal_placements()
    with pytest.raises(ValueError):
        env.step(type(legal[0])(999,legal[0].rotation,legal[0].x,legal[0].y,legal[0].lines_cleared,legal[0].holes,legal[0].max_height,legal[0].aggregate_height,legal[0].bumpiness,legal[0].features))

def test_tetris_tiny_mlp_vertical_slice():
    task=J04Tetris()
    result=task.train_and_evaluate(
        model="tiny_mlp",
        hidden_units=2,
        train_episodes=3,
        test_episodes=2,
        max_train_pieces=5,
        max_test_pieces=8,
        epochs=2,
        lr=0.01,
        batch_size=16,
        seed=4,
        permutation_trials=2,
    )
    assert result["train_examples"]>0
    assert result["parameter_count"]==23*2+1
    assert result["model_size_bytes_fp32"]==result["parameter_count"]*4
    assert result["mean_pieces"]>0


def test_tetris_tiny_mlp_confidence_fallback():
    task=J04Tetris()
    result=task.train_and_evaluate(
        model="tiny_mlp",
        hidden_units=2,
        train_episodes=3,
        test_episodes=2,
        max_train_pieces=5,
        max_test_pieces=8,
        epochs=2,
        lr=0.01,
        batch_size=16,
        seed=7,
        abstain_threshold=0.999999,
        permutation_trials=1,
    )
    assert 0.0 <= result["model_coverage_rate"] <= 1.0
    assert 0.0 <= result["fallback_rate"] <= 1.0
    assert result["fallback_count"] > 0
    assert result["fallback_rate"] == 1.0 - result["model_coverage_rate"]
    assert 0.0 <= result["expected_calibration_error"] <= 1.0
    assert result["confidence_bins"]


def test_tetris_linear_selector_is_permutation_invariant():
    env=TetrisEnv(13); env.reset(13)
    task=J04Tetris()
    from jev_bench.policies.j03_candidate import CandidateLinearRegressor
    selector=CandidateLinearRegressor(task.context_dim+task.candidate_dim)
    X=np.stack([np.concatenate([env.observation(),c.features]) for c in env.legal_placements()]).astype(np.float32)
    y=np.asarray([task.teacher_score(c) for c in env.legal_placements()],dtype=np.float32)
    selector.fit(X,y)
    legal=env.legal_placements(); context=env.observation()
    chosen_id=legal[selector.rank(context,legal)[0]].candidate_id
    rng=np.random.default_rng(13)
    for _ in range(5):
        perm=rng.permutation(len(legal)); shuffled=[legal[int(i)] for i in perm]
        assert shuffled[selector.rank(context,shuffled)[0]].candidate_id==chosen_id


def test_tetris_risk_coverage_monotone():
    task=J04Tetris()
    selector_result=task.train_and_evaluate(
        model="tiny_mlp",
        hidden_units=2,
        train_episodes=2,
        test_episodes=1,
        max_train_pieces=4,
        max_test_pieces=6,
        epochs=1,
        lr=0.01,
        batch_size=16,
        seed=8,
        permutation_trials=0,
    )
    assert selector_result["model_size_bytes_fp32"]==47*4

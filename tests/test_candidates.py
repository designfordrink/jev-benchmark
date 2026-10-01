from jev_bench.core.candidates import Candidate, CandidateProblem
import numpy as np

def test_candidate_problem_exposes_only_legal_candidates():
    problem=CandidateProblem(
        problem_id=1,
        context=np.zeros(2,dtype=np.float32),
        candidates=(
            Candidate(0,np.zeros(2,dtype=np.float32),legal=True),
            Candidate(1,np.ones(2,dtype=np.float32),legal=False),
        ),
        metadata={},
    )
    legal=problem.legal_candidates()
    assert [c.candidate_id for c in legal]==[0]

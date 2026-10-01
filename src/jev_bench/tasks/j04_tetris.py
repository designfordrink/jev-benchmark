from __future__ import annotations
import time
from typing import Any, Iterable
import numpy as np
from jev_bench.envs.tetris import TetrisEnv, TetrisPlacement
from jev_bench.policies.j03_candidate import CandidateTinyMLP, CandidateLinearRegressor

class J04Tetris:
    name="j04_tetris"
    context_dim=13
    candidate_dim=8

    @staticmethod
    def teacher_score(placement:TetrisPlacement)->float:
        return float(
            placement.lines_cleared*100.0
            - 0.9*placement.holes
            - 0.35*placement.aggregate_height
            - 0.20*placement.bumpiness
            - 1.0*placement.max_height
        )

    def collect_training_data(self,*,episodes:int=100,max_pieces:int=80,seed:int=0)->tuple[np.ndarray,np.ndarray]:
        if episodes<1: raise ValueError("episodes must be positive")
        X=[]; y=[]
        for ep in range(episodes):
            env=TetrisEnv(seed+ep); env.reset(seed+ep)
            for _ in range(max_pieces):
                legal=env.legal_placements()
                if not legal: break
                context=env.observation()
                for candidate in legal:
                    X.append(np.concatenate([context,candidate.features]))
                    y.append(self.teacher_score(candidate))
                chosen=legal[int(np.random.default_rng(seed+10000+ep+env.pieces).integers(len(legal)))]
                _,_,done,_=env.step(chosen)
                if done: break
        if not X: raise ValueError("no J04 training examples generated")
        return np.stack(X).astype(np.float32),np.asarray(y,dtype=np.float32)

    def _fit(self,model:str,hidden_units:int,X:np.ndarray,y:np.ndarray,seed:int,epochs:int,lr:float,batch_size:int,abstain_threshold:float):
        if model=="tiny_mlp": estimator=CandidateTinyMLP(self.context_dim+self.candidate_dim,hidden_units,seed,abstain_threshold)
        elif model=="linear": estimator=CandidateLinearRegressor(self.context_dim+self.candidate_dim,abstain_threshold)
        else: raise ValueError(f"Unknown J04 model: {model}")
        started=time.perf_counter()
        history=estimator.fit(X,y,epochs=epochs,lr=lr,batch_size=batch_size) if model=="tiny_mlp" else estimator.fit(X,y)
        return estimator,time.perf_counter()-started,history

    def evaluate_policy(self,selector,*,episodes:int=50,max_pieces:int=300,seed:int=0,permutation_trials:int=3)->dict[str,Any]:
        returns=[]; lines=[]; pieces=[]; latencies=[]; selected_teacher=[]; teacher_regrets=[]; permutation_hits=0; permutation_total=0; legal_decisions=0; decisions=0
        for ep in range(episodes):
            env=TetrisEnv(seed+ep); env.reset(seed+ep)
            while not env.game_over and env.pieces<max_pieces:
                legal=env.legal_placements()
                if not legal: break
                context=env.observation()
                utilities=np.asarray([self.teacher_score(p) for p in legal])
                teacher_idx=int(utilities.argmax())
                started=time.perf_counter_ns(); idx,confidence,_=selector.rank(context,legal); latencies.append((time.perf_counter_ns()-started)/1000.0)
                chosen=legal[idx]; legal_decisions+=int(chosen.candidate_id in [p.candidate_id for p in legal]); decisions+=1
                selected_teacher.append(int(idx==teacher_idx)); teacher_regrets.append(float(utilities[teacher_idx]-utilities[idx]))
                for trial in range(permutation_trials):
                    rng=np.random.default_rng(seed+500000+ep*1000+env.pieces*10+trial); perm=rng.permutation(len(legal)); shuffled=[legal[int(i)] for i in perm]; sh_idx,_,_=selector.rank(context,shuffled)
                    permutation_hits+=int(shuffled[sh_idx].candidate_id==chosen.candidate_id); permutation_total+=1
                _,_,done,_=env.step(chosen)
                if done: break
            returns.append(env.score); lines.append(int(sum([])) if False else env.last_lines); pieces.append(env.pieces)
        return {
            "task":self.name,
            "episodes":episodes,
            "seed":seed,
            "mean_return":float(np.mean(returns)),
            "std_return":float(np.std(returns,ddof=1)) if len(returns)>1 else 0.0,
            "mean_lines_last_piece":float(np.mean(lines)) if lines else 0.0,
            "mean_pieces":float(np.mean(pieces)) if pieces else 0.0,
            "teacher_agreement_rate":float(np.mean(selected_teacher)) if selected_teacher else 0.0,
            "mean_teacher_regret":float(np.mean(teacher_regrets)) if teacher_regrets else 0.0,
            "p95_teacher_regret":float(np.percentile(teacher_regrets,95)) if teacher_regrets else 0.0,
            "permutation_invariance_rate":permutation_hits/max(1,permutation_total),
            "legal_action_rate":legal_decisions/max(1,decisions),
            "mean_action_latency_us":float(np.mean(latencies)) if latencies else 0.0,
            "p95_action_latency_us":float(np.percentile(latencies,95)) if latencies else 0.0,
        }

    def train_and_evaluate(self,*,model:str="tiny_mlp",hidden_units:int=8,train_episodes:int=100,test_episodes:int=50,max_train_pieces:int=80,max_test_pieces:int=300,epochs:int=20,lr:float=0.01,batch_size:int=128,seed:int=0,abstain_threshold:float=0.0,permutation_trials:int=3)->dict[str,Any]:
        X,y=self.collect_training_data(episodes=train_episodes,max_pieces=max_train_pieces,seed=seed)
        estimator,train_seconds,history=self._fit(model,hidden_units,X,y,seed,epochs,lr,batch_size,abstain_threshold)
        result=self.evaluate_policy(estimator,episodes=test_episodes,max_pieces=max_test_pieces,seed=seed+10000,permutation_trials=permutation_trials)
        result.update({"model":model,"hidden_units":hidden_units,"train_episodes":train_episodes,"test_episodes":test_episodes,"train_examples":len(X),"train_seconds":float(train_seconds),"history":history,"parameter_count":int(estimator.parameter_count),"model_size_bytes_fp32":int(estimator.model_size_bytes_fp32)})
        return result

    def size_sweep(self,*,hidden_units:Iterable[int]=(1,2,4,8,16,32,64),**kwargs)->list[dict[str,Any]]:
        return [dict(self.train_and_evaluate(hidden_units=int(h),**kwargs),hidden_units=int(h)) for h in hidden_units]

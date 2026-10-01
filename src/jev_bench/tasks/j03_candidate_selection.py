from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from jev_bench.core.candidates import Candidate, CandidateProblem
from jev_bench.policies.j03_candidate import CandidateLinearRegressor, CandidateTinyMLP

class J03CandidateSelection:
    name="j03_candidate_selection"
    context_dim=6; candidate_dim=6; candidate_count=8

    @staticmethod
    def utility(context:np.ndarray,features:np.ndarray)->float:
        c=np.asarray(context,dtype=np.float32); x=np.asarray(features,dtype=np.float32)
        return float(0.9*c[0]*x[0]+0.7*c[1]*x[1]-0.8*c[2]*x[2]+0.55*np.sin(c[3]*x[3])+0.45*np.cos(c[4]*x[4])+0.35*c[5]*x[5])

    def generate(self,count:int,seed:int)->list[CandidateProblem]:
        rng=np.random.default_rng(seed); problems=[]
        for pid in range(count):
            context=rng.uniform(-1,1,self.context_dim).astype(np.float32); candidates=[]
            for cid in range(self.candidate_count):
                features=rng.uniform(-1,1,self.candidate_dim).astype(np.float32); candidates.append(Candidate(cid,features,legal=True))
            problems.append(CandidateProblem(pid,context,tuple(candidates),{"seed":seed,"candidate_count":self.candidate_count}))
        return problems

    def training_data(self,problems:list[CandidateProblem])->tuple[np.ndarray,np.ndarray]:
        rows=[]; targets=[]
        for p in problems:
            for c in p.legal_candidates():
                rows.append(np.concatenate([p.context,c.features])); targets.append(self.utility(p.context,c.features))
        return np.stack(rows).astype(np.float32),np.asarray(targets,dtype=np.float32)

    def train_and_evaluate(self,*,train_problems:int=500,test_problems:int=300,model:str="tiny_mlp",hidden_units:int=8,epochs:int=20,lr:float=0.01,batch_size:int=128,seed:int=0,abstain_threshold:float=0.0,permutation_trials:int=5)->dict[str,Any]:
        train=self.generate(train_problems,seed); test=self.generate(test_problems,seed+1)
        X,y=self.training_data(train)
        if model=="tiny_mlp": estimator=CandidateTinyMLP(self.context_dim+self.candidate_dim,hidden_units,seed,abstain_threshold)
        elif model=="linear": estimator=CandidateLinearRegressor(self.context_dim+self.candidate_dim,abstain_threshold)
        else: raise ValueError(f"Unknown J03 model: {model}")
        started=time.perf_counter(); history=estimator.fit(X,y,epochs=epochs,lr=lr,batch_size=batch_size) if model=="tiny_mlp" else estimator.fit(X,y); train_seconds=time.perf_counter()-started
        rewards=[]; regrets=[]; correct=[]; confidences=[]; abstentions=0; permutation_hits=0; permutation_total=0; latencies=[]
        for p in test:
            legal=list(p.legal_candidates()); utilities=np.asarray([self.utility(p.context,c.features) for c in legal]); oracle_index=int(np.argmax(utilities)); oracle_id=legal[oracle_index].candidate_id
            obs={"context":p.context,"candidates":legal}; t=time.perf_counter_ns(); index,confidence,_=estimator.rank(p.context,legal); latencies.append((time.perf_counter_ns()-t)/1000.0)
            chosen=legal[index]; rewards.append(utilities[index]); regrets.append(utilities[oracle_index]-utilities[index]); correct.append(int(chosen.candidate_id==oracle_id)); confidences.append(confidence); abstentions+=int(confidence<abstain_threshold)
            for trial in range(permutation_trials):
                perm=np.random.default_rng(seed+100000+p.problem_id*100+trial).permutation(len(legal)); shuffled=[legal[int(i)] for i in perm]; sh_index,_,_=estimator.rank(p.context,shuffled); permutation_hits+=int(shuffled[sh_index].candidate_id==chosen.candidate_id); permutation_total+=1
        result={"task":self.name,"model":model,"seed":seed,"train_problems":train_problems,"test_problems":test_problems,"candidate_count":self.candidate_count,"context_dim":self.context_dim,"candidate_dim":self.candidate_dim,"train_candidates":len(X),"selection_accuracy":float(np.mean(correct)),"mean_reward":float(np.mean(rewards)),"mean_oracle_reward":float(np.mean([max(self.utility(p.context,c.features) for c in p.legal_candidates()) for p in test])),"mean_regret":float(np.mean(regrets)),"p95_regret":float(np.percentile(regrets,95)),"mean_confidence":float(np.mean(confidences)),"abstention_rate":float(abstentions/max(1,len(test))),"permutation_invariance_rate":permutation_hits/max(1,permutation_total),"mean_action_latency_us":float(np.mean(latencies)),"p95_action_latency_us":float(np.percentile(latencies,95)),"parameter_count":int(estimator.parameter_count),"model_size_bytes_fp32":int(estimator.model_size_bytes_fp32),"train_seconds":float(train_seconds),"history":history}
        return result

    def size_sweep(self,*,hidden_units:Iterable[int]=(1,2,4,8,16,32,64),**kwargs)->list[dict[str,Any]]:
        return [dict(self.train_and_evaluate(hidden_units=int(h),**kwargs),hidden_units=int(h)) for h in hidden_units]

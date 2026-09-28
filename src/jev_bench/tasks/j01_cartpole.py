from __future__ import annotations
import time
from statistics import mean
from typing import Any
import gymnasium as gym
from jev_bench.core.contracts import Policy

class J01CartPole:
    name = "j01_cartpole"
    def make_environment(self, seed: int | None = None):
        env=gym.make("CartPole-v1")
        if seed is not None: env.reset(seed=seed)
        return env
    def evaluate(self, policy: Policy, *, episodes: int=20, seed: int=0) -> dict[str, Any]:
        if hasattr(policy, "start_evaluation"): policy.start_evaluation()
        returns=[]; lengths=[]; latencies=[]; confidences=[]; abstentions=0; fallback_decisions=0
        for episode in range(episodes):
            s=seed+episode; env=self.make_environment(s); obs,_=env.reset(seed=s); policy.reset(s); total=0.; steps=0
            while True:
                started=time.perf_counter_ns(); d=policy.act(obs); latencies.append((time.perf_counter_ns()-started)/1000)
                confidences.append(d.confidence); abstentions += int(d.abstain); fallback_decisions += int(d.metadata.get("fallback", False))
                obs,r,terminated,truncated,_=env.step(int(d.action)); total+=float(r); steps+=1
                if terminated or truncated: break
            env.close(); returns.append(total); lengths.append(steps)
        result={"task":self.name,"policy":policy.name,"episodes":episodes,"seed":seed,"mean_return":mean(returns),"std_return":_std(returns),"mean_episode_length":mean(lengths),"mean_action_latency_us":mean(latencies),"p95_action_latency_us":_percentile(latencies,95),"mean_confidence":mean(confidences),"abstention_rate":abstentions/max(1,len(confidences)),"fallback_rate":fallback_decisions/max(1,len(confidences))}
        for attr in ("fallback_count","model_size_bytes_fp32","parameter_count"):
            if hasattr(policy, attr): result[attr]=getattr(policy, attr)
        return result

def _std(values):
    if len(values)<2:return 0.0
    m=mean(values); return (sum((x-m)**2 for x in values)/(len(values)-1))**0.5

def _percentile(values, percentile):
    ordered=sorted(values)
    if not ordered:return 0.0
    i=(len(ordered)-1)*percentile/100; lo=int(i); hi=min(lo+1,len(ordered)-1); f=i-lo
    return ordered[lo]+(ordered[hi]-ordered[lo])*f

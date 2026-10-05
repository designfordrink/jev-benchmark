# J05 — state-aware JEV action context v2

## Decision

The first J05 System One contract was insufficient for reliable event classification. It exposed only:

- state;
- action id;
- next_state;
- terminated.

Several reward events were not identifiable from those fields alone.

This increment upgrades the observable context before any live JEV benchmark run.

## Observable context

The System One judge receives:

1. Environment geometry:
   - width and height;
   - coordinate convention;
   - wall cells;
   - hazard cells;
   - key location;
   - exit location;
   - maximum step count.

2. Current transition state:
   - position;
   - key possession;
   - current step.

3. Action:
   - action id;
   - action name;
   - movement delta.

4. Result:
   - next position;
   - next key possession;
   - next step;
   - termination flag.

The context is observable task information. It is not a reward label.

## Hidden information

The judge must never receive:

- event;
- reference reward;
- termination reason.

These remain internal benchmark labels.

## English JEV contract

All JEV-facing instructions and choice criteria are English.

The seven choices are:

- lava — the resulting position is a hazard cell and the transition terminates;
- timeout — the step limit is reached without another terminal event explaining termination;
- wall — the requested movement enters a wall cell and the agent remains in the same position;
- boundary — the requested movement leaves the grid and the agent remains in the same position;
- move — the agent changes position without collecting the key, entering a hazard, or completing the task;
- key — the agent reaches the key and changes has_key from false to true;
- exit — the agent reaches the exit while carrying the key and the task terminates.

## Why each field is required

| Event | Required evidence |
|---|---|
| wall | action direction + wall geometry + unchanged position |
| boundary | action direction + grid dimensions + unchanged position |
| lava | next position + hazard geometry + termination |
| key | next position + key location + change in has_key |
| exit | next position + exit location + has_key + termination |
| timeout | step count + max_steps + termination |
| move | action/result transition after excluding the terminal/special cases |

## Cache contract

The cache key now includes:

- state;
- action;
- next_state;
- terminated;
- step;
- model;
- prompt version.

The hidden event is still excluded. This prevents two observably different transitions from sharing a cache entry while preserving the anti-leakage property.

## Independent rules baseline

RuleReward no longer inherits NativeReward.

It reconstructs the event from the observable transition and the deterministic Key Quest rules. NativeReward remains the ground-truth control and uses the environment's internal event label.

Therefore the intended comparison is:

**native ground truth → independent rules judge → JEV judge**

## Research interpretation

This increment does not claim that JEV is better.

It establishes a valid input contract so that the subsequent experiment measures reward judgment rather than information missing from the prompt.

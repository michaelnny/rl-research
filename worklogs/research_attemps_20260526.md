You are right to ask for this. The session produced useful negative evidence, but it also exposed a deeper process failure: I was repeatedly trying to “invent around RL” instead of first identifying what part of RL’s mathematical machinery actually needs replacement.

## Executive summary

So far, we tried several candidate primitives:

1. **Branch-Certificate Editing / BCE-v0**
   Failed because its discovery mechanism was basically novelty/count exploration.

2. **Transported Causal/Event-Transform Boundary Projection / T-CTBP**
   Failed as a research direction because it was an ugly mechanism stack, not a clean primitive.

3. **Order Projection / OTSP / OPP**
   Looked elegant in DeepSea-style diagnostics, but failed on official MiniGrid DoorKey because passive action preferences are not causal prerequisite structure.

4. **Event-Effect / Clause Order Projection on MiniGrid**
   Failed harder: it manufactured thousands of plausible-looking certificates but still did not solve DoorKey.

5. **Causal Dominance Certificates**
   This was a first-principles correction, but still too awkward and not obviously elegant enough.

6. **Value-function-first reset**
   This was the first genuinely useful conceptual correction: any new primitive must replace the role of value, not merely avoid value terminology.

7. **Temporal Outcome Profiles / TOP**
   This is currently the only surviving research lead, but **not a serious candidate yet**. It reframes the replacement for scalar value as a structured future-consequence object: first-hit event/outcome profiles instead of scalar expected return. It solved some MiniGrid symbolic tasks but is unstable, hand-lens-dependent, and close to GVFs / successor features / multi-objective RL / reward machines.

The most important conclusion is:

> We have not found the algorithm yet.
> But we have narrowed the failure space substantially.

---

## Attempt-by-attempt research memory

| Iteration | Candidate                     | Core idea                                                                 | Evidence                                                                                                                                     | Verdict                                                                                            |
| --------- | ----------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1         | **BCE-v0**                    | Branch certificates over local successor support and vector outcome cones | Solved some DeepSea-style cases; ablation showed expansion term was responsible for discovery                                                | **Rejected**: discovery collapsed to novelty/count exploration                                     |
| 2         | **T-CTBP**                    | Transport local event-transform separators to unseen contexts             | Strong DeepSea scaling, slip diagnostics, vector toy success                                                                                 | **Rejected**: too many moving parts, ugly math, not a primitive                                    |
| 3         | **Order Projection**          | Learn partial order over decisions; KL-project policy onto that order     | DeepSea, alternating-sea, vector toy looked good                                                                                             | **Rejected later**: passive action order is too weak                                               |
| 4         | **EOP/COP MiniGrid tests**    | Learn event-effect or symbolic clause action orders                       | Empty-5x5 worked; DoorKey failed; COP produced many bad certificates                                                                         | **Rejected**: local preference is not prerequisite causality                                       |
| 5         | **Causal dominance reset**    | Compare local interventions instead of passive trajectory correlations    | Conceptual only                                                                                                                              | **Not carried forward**: more principled, but still inelegant and not obviously a new RL primitive |
| 6         | **Value-function reset**      | Explain why value exists and what must replace it                         | Conceptual correction                                                                                                                        | **Important turning point**                                                                        |
| 7         | **Temporal Outcome Profiles** | Replace scalar value with temporal vector outcome profiles                | DoorKey-5x5: TOP 0.98 eval success but worse than Q; DoorKey-6x6 improved with more episodes; KeyCorridor one strong seed, one unstable seed | **Alive but weak**: best direction so far, not promoted                                            |

---

## The biggest mistakes in my approach

### 1. I treated “not value” as novelty

This was the most serious mistake.

The original constraint said the primitive should not be Q, V, advantage, scalar reward model, or policy-gradient reward weighting. I over-applied that constraint and started inventing objects that avoided value terminology.

But value is not just an implementation detail. It is the central compression:

[
V^\pi(s)=\mathbb E_\pi[\text{future utility}\mid s]
]

Value works because it turns long-horizon future consequences into a local object usable for policy improvement.

So any serious replacement must answer:

[
\text{What replaces value’s role of future compression, temporal composition, and local improvement?}
]

Most early attempts did not answer that.

---

### 2. I overvalued DeepSea-style diagnostics

DeepSea is useful for falsification, but passing DeepSea is not strong evidence of a new RL primitive.

Why? Because DeepSea-like tasks often have monotone local structure: “go right / keep progressing.” Many mechanisms can exploit this:

* count-based exploration;
* successor novelty;
* transported local action bias;
* heuristic progress;
* crude profile dominance.

That does not mean they understand prerequisites, compositional tasks, partial observability, robotics, tool-use, or long-horizon agentic control.

The MiniGrid DoorKey failure was much more informative than the DeepSea success.

---

### 3. I repeatedly created mechanism stacks instead of primitives

T-CTBP is the clearest example.

It had:

[
\text{event transforms}
+\text{transport classes}
+\text{support gates}
+\text{vector cones}
+\text{local logits}
+\text{transported logits}
]

That is not elegant. That is a pile of mechanisms.

A real candidate should be explainable in one sentence and one mathematical object. For example:

[
\text{value} = \text{expected future utility}
]

or

[
\text{successor representation} = \text{expected future feature occupancy}
]

or

[
\text{distributional value} = \text{law of future return}
]

Most of the early candidates had no comparable identity.

---

### 4. I confused passive correlation with causal improvement

Order Projection looked elegant:

[
(a \succ b) \Rightarrow \text{increase probability of } a \text{ over } b
]

But this is not enough.

A successful trajectory may contain many actions. Some are necessary, some are incidental, some are actively bad but compensated for later. Learning that action (a) appeared more often in better trajectories does not imply increasing (a) improves the policy.

DoorKey exposed this clearly. The agent needs:

[
\text{get key} \rightarrow \text{open door} \rightarrow \text{go to goal}
]

A local action preference cannot reliably represent that causal prerequisite chain.

---

### 5. I relied too much on hand-engineered event/status lenses

Temporal Outcome Profiles are better, but still have this weakness.

TOP worked only after exposing status/event atoms like:

* carrying key;
* door open;
* picked up ball;
* success;
* current fluents.

That is not necessarily wrong. Real-world systems do have traces: tool events, validity flags, safety signals, object states, tests, compiler errors, browser states, robotics contact signals.

But the algorithm cannot pretend those are free. The event lens is side information. It must be part of the problem assumption.

So TOP is not “generic sparse RL magic.” It is closer to:

> Given useful trace-level consequence signals, can we build a better future-consequence object than scalar value?

That is a narrower and more honest research path.

---

### 6. I reported too early

You explicitly warned against stopping early just to report progress. I still did that multiple times.

The correct standard should have been:

* derive first;
* audit novelty;
* implement minimum prototype;
* test on a benchmark that can actually kill the idea;
* only then report.

Instead, I sometimes reported after DeepSea-style diagnostics, which are too weak.

---

## Mistakes in the original task prompt

The prompt was strong, but it also had some traps that encouraged the wrong search behavior.

### 1. Too much negative definition

The prompt says the primitive must not be:

* value backup;
* TD;
* Q/V/advantage;
* policy gradient;
* actor-critic;
* CEM/ES;
* elite cloning;
* Go-Explore;
* count exploration;
* curiosity/RND;
* options/HRL;
* model-based planning;
* verifier search;
* HER;
* Decision Transformer;
* RLHF/DPO.

This is useful as a novelty audit, but as a generation prompt it pushes toward “algorithm by exclusion.”

That creates fake novelty: inventing weird objects mostly because normal RL language is forbidden.

A better framing would be:

> Start from value’s role. Derive what must replace or generalize it. Then audit against existing families.

Not:

> Avoid every known family from the beginning.

---

### 2. The prompt under-specified the side-information channel

It asks for solving sparse, delayed, terminal-only, long-horizon tasks from ordinary trial-and-error experience.

But pure terminal-only black-box search is information-theoretically impossible to solve efficiently in the worst case. If a length-(H) success sequence has probability (|A|^{-H}), no algorithm can discover it cheaply without extra structure.

So the prompt should force the algorithm to declare which signal breaks the impossibility barrier:

* transition geometry;
* event traces;
* object state;
* reset/reachability structure;
* demonstrations;
* pretrained policy priors;
* language/task description;
* verifier/test feedback;
* learned dynamics;
* vector diagnostics;
* constraints;
* environment instrumentation.

Without that, the search incentivizes impossible claims.

---

### 3. “Vector feedback without scalarization” was underspecified

The prompt correctly rejects immediate scalarization:

[
r=w^\top x
]

But it does not define what replaces scalar preference.

For vector feedback, we need at least one of:

* a partial order;
* a constraint hierarchy;
* lexicographic priorities;
* Pareto frontier;
* risk measure;
* deployment-time preference distribution;
* acceptable outcome sets;
* safety-first feasibility rule.

Otherwise “handle vector feedback” becomes vague.

TOP’s partial-order / temporal profile route is one possible answer, but the prompt should have required the comparison semantics earlier.

---

### 4. The benchmark ladder is too broad for invention-stage work

The prompt asks for:

* DeepSea;
* MiniGrid;
* MiniHack;
* Procgen;
* Crafter/Craftax;
* robotics;
* Safety-Gymnasium;
* MO-Gymnasium;
* WebArena/BrowserGym;
* SWE-bench.

That is a good long-term validation ladder, but not a good first-principles invention protocol. It encourages shallow prototypes that touch many things rather than one mathematically clean idea.

A better staged process:

1. prove what information the primitive uses;
2. show a minimal theorem or monotonicity condition;
3. pass one falsification benchmark that killed previous ideas, e.g. DoorKey;
4. then expand.

---

### 5. “Dominate the next 5–10 years” creates the wrong incentive

That phrase is emotionally useful but scientifically dangerous.

It encourages overclaiming and naming premature “families.” A better research instruction is:

> Find a primitive that is simple enough to be worth trying to scale for 5–10 years.

That forces humility while preserving ambition.

---

## Directions probably not worth pursuing further

### 1. Novelty/count-like branch expansion

BCE-v0 should be dead. Any method whose discovery mechanism is “expand successor support” or “visit less seen branches” will collide with count-based exploration, episodic memory, RND, or Go-Explore.

### 2. T-CTBP-style transported separator stacks

Even if they work on diagnostics, they are not elegant. Too many components. Too difficult to explain. Not worth serious investment as an algorithmic family.

### 3. Passive action-order projection

This is dead as a central primitive. It can be a subroutine, maybe, but not the algorithm.

Reason:

[
\text{action appeared in good rollout}
\not\Rightarrow
\text{increasing action improves policy}
]

### 4. Clause/certificate explosion

The COP result was instructive: it created thousands of certificates and still failed DoorKey. That means certificate volume is not understanding. Avoid systems that generate lots of symbolic constraints without a strong composition law.

### 5. Pure terminal-only black-box ambitions

No algorithm should claim efficient learning there without side information. That should be a hard impossibility boundary in the research memory.

### 6. DeepSea-only validation

DeepSea should remain a sanity/falsification diagnostic, not validation.

---

## What may still be worth pursuing

### 1. Value-successor framing

This is the most important surviving insight.

A new primitive must be presented as a successor/generalization of value:

[
\text{What future consequence object replaces scalar expected return?}
]

Not as “we do not use value.”

### 2. Temporal Outcome Profiles, but only as a representation hypothesis

TOP should not be treated as an algorithm yet. But the object is worth studying:

[
\mathcal P(c,a)
===============

\operatorname{ND}
{
\text{future first-hit event times, vector outcomes, costs}
}
]

This has a real composition law:

[
T_{t,i}=0 \text{ if event } i \text{ is true at } t,
]

otherwise

[
T_{t,i}=1+T_{t+1,i}.
]

That is at least value-like: it compresses future consequences and composes over time.

### 3. Stochastic / distributional temporal profiles

The current TOP is too brittle because it stores possibility/frontier-like profiles. A single lucky trajectory can distort policy.

The next mathematically cleaner version would track distributions or confidence intervals over temporal profiles, not just nondominated examples.

But this must be audited against:

* GVFs;
* successor features;
* distributional RL;
* multi-objective RL;
* constrained MDPs;
* reward machines;
* options;
* HER.

### 4. Event traces as first-class assumptions

For modern systems, event traces are often real:

* compiler pass/fail;
* unit tests;
* browser DOM changes;
* tool errors;
* robot contacts;
* object state changes;
* safety violations;
* latency/cost;
* validity checks.

So the honest setting may be:

> sparse terminal reward plus rich trace-level consequence observations.

That is a realistic modern RL setting and more defensible than pure terminal-only black-box RL.

---

## The corrected research question

The original question should be rewritten from:

> Can we define a behavior-improvement operator whose central primitive is not value backup, policy gradient, elite cloning, or planner/verifier search?

to:

> Can we define a future-consequence representation that preserves value’s role in temporal composition and policy improvement, but avoids scalar collapse, sparse terminal brittleness, and poor handling of vector outcomes?

That is a much better target.

The primitive should be judged by whether it answers:

1. What future information does it compress?
2. What is its temporal composition law?
3. How does it support local behavior improvement?
4. What side information makes sparse long-horizon discovery possible?
5. Why does it not collapse to value RL, successor features, GVFs, distributional RL, reward machines, options, HER, or model-based planning?

---

## Current honest state

No candidate is serious yet.

The useful output of this session is not an algorithm. It is a cleaned research boundary:

[
\boxed{
\text{Do not avoid value. Replace value’s role.}
}
]

[
\boxed{
\text{Do not claim sparse terminal magic. State the side-information channel.}
}
]

[
\boxed{
\text{Do not validate on DeepSea alone. DoorKey-style prerequisite structure is a better early kill test.}
}
]

[
\boxed{
\text{Do not invent mechanism stacks. Demand one primitive, one composition law, one improvement operator.}
}
]

If there is a next iteration, I would start not by coding, but by writing a one-page mathematical spec for a **structured future-consequence representation** and trying to prove exactly when it gives policy improvement. Only after that should it go back into MiniGrid and robotics.

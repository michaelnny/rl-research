fix: 1
class: syntax-fix
what_changed: hoisted delta/a_star/margin_a_star computation into the same branch that produces the action, so the post-step REINFORCE-nudge block does not reference an undefined `delta` on the first step after warmup
why_it_does_not_change_the_idea: pure variable-scope fix; the algorithm (forward-model rollout -> threshold-crossing horizons -> Pareto-non-dominance margin -> logit nudge + REINFORCE toward a*) is unchanged

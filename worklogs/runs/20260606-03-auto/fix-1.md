fix: 1
class: syntax-fix
what_changed: extract_witnesses indexed cum past its last row when t pointed at the terminal-hash slot; restricted t and d to decision indices [0, T-1] where vectors exist.
why_it_does_not_change_the_idea: bounds-only fix on cum array indexing; the latest-divergence-before-confluence primitive and per-channel sign-vote semantics are unchanged.

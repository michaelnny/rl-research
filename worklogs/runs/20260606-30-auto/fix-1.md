fix: 1
class: syntax-fix
what_changed: corrected dimension reference in _hash_bucket — pad/truncate to rp_matrix.shape[1] (the input dim) rather than .shape[0] (the signature-bit dim), which caused matmul shape mismatch on small obs (e.g., DST has obs size 2).
why_it_does_not_change_the_idea: the random-projection hash is a substrate detail of the observation-bucketing vocabulary; the algorithm's primitive (per-channel Fiedler vector and per-(action, channel) Fiedler-ascent) is unchanged.

# Candidate process protocol

Status: executable v1 contract, preference-conditioned FactorLab only

Candidate algorithms do not import an evaluator environment or receive a world
object. The evaluator owns world generation, held-out seeds, budgets, reset,
step, normalization, and metric calculation. A candidate is a separate process
that exchanges one JSON object per line over standard input and output.

This boundary prevents accidental access to `FactorLabWorld`, privileged causal
metadata, evaluator environment variables, and cross-held-out-world state. It
is a scientific-integrity boundary; it is not claimed to resist a malicious
local user with the same operating-system account.

## Lifecycle

For each independent training trial, the evaluator starts one fresh training
process:

1. `init` includes protocol version, `phase=training`, public suite manifest,
   normalized preference, trial index, evaluator-derived trial seed, and
   `checkpoint=null`.
2. The candidate replies exactly `{"type":"ready"}`.
3. Each episode begins with `reset`. Learner-visible observation and public task
   information are included.
4. On a decision step, `act` requests an action. The candidate replies exactly
   `{"type":"action","action":...}` using the native action schema.
5. `transition` supplies the next observation, vector reward, termination flags,
   and public step information. It requires no response.
6. `episode_end` supplies the vector return. It requires no response.
7. After the shared training budget, `checkpoint` requests all learned state.
   The candidate replies exactly `{"type":"checkpoint","state":...}`. State
   must be finite JSON and fit the configured message limit.
8. `close` requires `{"type":"closed"}` and process exit.

For every held-out world and trial, the evaluator starts a fresh process, sends
`init` with `phase=evaluation` and that trial's checkpoint, runs exactly one
episode, and closes it. Evaluation worlds cannot update the starting state for
later worlds. Within-episode recurrent state remains allowed.

All public and held-out worlds in a suite share an evaluator-owned signed
permutation from visible cues to reward targets. Only its keyed commitment is
public. A candidate must infer the mapping from public-world training rewards;
hard-coding the visible cue as the action is a random-quality policy in the
anchor configuration.

## Candidate requirements

- Standard output is protocol-only: exactly one JSON response for each request
  that requires a response. Diagnostics may go to standard error; only its byte
  count and digest are retained in the measurement summary.
- The candidate must not expect evaluator secrets in its environment. The child
  receives a small allowlist of ordinary process variables.
- Responses are time- and size-bounded. Timeout, malformed JSON, rejected
  actions, early exit, or an unserializable checkpoint produces a durable
  `candidate_error` measurement. It is not retried as infrastructure failure.
- The candidate may use the learner-visible structured action schema. It never
  receives the world seed, world ID, influence graph, target sequence, replay
  token, or per-world held-out identifier.
- Candidate code is confined to its branch-specific `candidates/` path. The
  evaluator checks the worktree again after execution so runtime mutation of a
  protected benchmark, evaluator, schema, test, or design path invalidates the
  job.

On macOS the candidate child is additionally run without network access, with
the runtime directory and common credential directories unreadable and writes
to the canonical repository denied. Process-information access is denied so a
candidate cannot inspect the trusted evaluator. Resource and process-group
limits on the outer evaluator remain the hard termination boundary.

## Evaluator output

The evaluator emits one schema-validated JSON measurement containing:

- protocol, task, and public suite IDs;
- objective protocol;
- training episode, independent trial, and held-out world counts;
- fixed-semantics normalized component mean and standard deviation;
- normalized preference utility mean and standard deviation;
- evaluator-measured episode, transition, and wall usage;
- analytic random-policy and cue-aware references, improvement and regret;
- one checkpoint digest per trial and candidate-stderr digests; and
- an explicit assertion that held-out identifiers were not exposed.

Raw per-world identifiers and seeds are never included. A failed candidate
emits the same provenance envelope with `status=candidate_error` and a bounded
error description.

## V1 limits

The executable runner currently covers preference-conditioned, two-objective,
factored-discrete FactorLab tasks. The protocol message set is designed to
extend to catalog, continuous, hybrid, constrained, and policy-coverage modes,
but those extensions are not implemented or implied by v1. V1 therefore cannot
qualify the complete benchmark matrix or support an unattended discovery claim
over all target problems yet.

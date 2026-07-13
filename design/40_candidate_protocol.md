# Neural candidate process protocol

Status: executable v2 contract, preference-conditioned Neural FactorLab

Candidate algorithms are compact neural RL programs. They do not import an
evaluator environment or receive a world object. The evaluator owns world
generation, held-out keys, budgets, stepping, normalization, and metrics. A
candidate is a separate process exchanging JSON Lines control messages while
neural checkpoints travel as bounded binary artifacts in an evaluator-owned
scratch directory.

## Compute contract

`init` supplies the public observation/action/task specification and a hard
trainable-parameter cap. The candidate replies with a neural model manifest:

- `model_family` must be `neural_policy`;
- architecture and framework;
- positive trainable parameter count within the evaluator cap;
- whether the policy is recurrent; and
- execution device: CPU, CUDA, or MPS.

Candidate review checks that the declaration matches code. The evaluator also
records transition, episode, wall, checkpoint, and accelerator-time bounds. A
table keyed by observations, a finite cue hypothesis list, or an enumerated
joint-action policy violates the candidate contract even if it can speak the
wire protocol.

## Batched lifecycle

For each independent training trial the evaluator starts one fresh process:

1. `init` includes protocol version, `phase=training`, public suite commitment,
   public task spec, normalized preference, trial seed, model budget, and no
   checkpoint. The candidate replies `ready` plus its model manifest.
2. `reset_batch` supplies a batch of fresh public-world observations.
3. `act_batch` requests one structured action per environment. The candidate
   replies with `actions` using the native factored schema.
4. `transition_batch` supplies next observations, vector rewards, termination
   flags, and public step information. It requires no response.
5. `episode_end_batch` supplies vector returns and requires no response.
6. Steps 2–5 repeat until the shared interaction budget is exhausted.
7. `checkpoint` requests a bounded binary artifact. The candidate writes one
   plain file under `RLX_CANDIDATE_SCRATCH` and replies with its filename and
   SHA-256 digest. The evaluator verifies size, type, location, and digest.
8. `close` requires `closed` and process exit.

For every held-out world and trial, the evaluator starts a fresh process with a
read-only copy of that trial checkpoint, sends `phase=evaluation`, runs one
episode as a batch of one, and destroys the process. Evaluation worlds cannot
update the starting state for later worlds. Within-episode recurrent state is
allowed. The evaluation model manifest must match the training model manifest.

## Isolation and integrity

- Standard output is protocol-only. Diagnostics go to standard error; the
  evaluator retains only bounded size and digest in the measurement.
- Candidate processes receive neither evaluator secrets nor held-out IDs,
  seeds, task-kernel parameters, replay tokens, or other evaluator metadata.
- The scratch directory contains only the candidate’s checkpoint. It is not a
  route to evaluator runtime or campaign secrets.
- Timeout, malformed JSON, wrong batch size, invalid action, manifest mismatch,
  checkpoint escape/symlink/digest mismatch, or early exit produces a durable
  `candidate_error`; it is not retried as an infrastructure failure.
- Candidate code is confined to its branch-specific `candidates/` path. Runtime
  mutation of benchmark, evaluator, schema, test, or design paths invalidates
  the job.
- On macOS, the child additionally has no network access, credential/runtime
  paths are unreadable, canonical repository paths are unwritable, and process
  information is denied.

## Evaluator output

The evaluator emits a schema-validated measurement with task/suite/protocol,
fixed budget and usage, per-objective held-out distributions, normalized
preference utility, measured random-policy control, regret to declared upper
bound, neural model manifests, binary checkpoint digests, wall/accelerator-time
bounds, stderr digests, and an explicit assertion that held-out identifiers
were not exposed.

## V2 scope

V2 executes preference-conditioned, two-objective, factored-discrete Neural
FactorLab. Catalog, continuous, hybrid, constrained, and policy-coverage
messages are future protocol revisions and are not implied by v2. Research
campaign admission is restricted to separately qualified task tiers.

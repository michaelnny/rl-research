# FactorLab v0 calibration

This campaign is explicitly `under_calibration`. Unit tests and the smoke
runner cannot change that status.

The committed definition contains public structure and band counts but no
held-out key, seed, or cue transform. Campaign initialization creates a 256-bit
owner-only key in the ignored runtime secret store. HMAC derivation gives every
branch the same versioned suite and shared hidden transform. Model providers
cannot read the runtime directory, while candidate processes receive only the
public manifest and training feedback.

Run a provisional local diagnostic with:

```bash
PYTHONPATH=src:. .venv/bin/python -m rlx_agents.cli smoke \
  --output runtime/factorlab-smoke.json
```

Promotion requires all ten checks in `definition.json` to be `verified`, each
with immutable run or analysis artifact references. In particular, the
privileged mechanism probes are factor-sensitivity tools, not learnability
evidence.

# LeaP reproduction and prior-steering probe — design

Date: 2026-10-12. Approved in chat.

## Goal

Reproduce the official LeaP code (CoRL 2026, arXiv 2606.17408) end to end on
RoboTwin 2.0 so it can serve as a baseline for a later study: controlling a
generative policy's action *source prior* at inference time as a form of
in-context steering. Deliverable: trained LeaP and same-architecture NoPrior
policies on four tasks, official 100-episode evaluations, an inference-time
prior-intervention sweep, and a written results summary.

## Constraints (verified)

- GPU: one RTX 5080 (Blackwell sm_120, 16 GB). The pinned PyTorch 2.4.1/cu121
  cannot run on it. Use PyTorch cu128 wheels with the installed CUDA 12.8
  toolkit (`/usr/local/cuda-12.8`, headers present). Document the deviation.
- No conda on the host; use `uv` with Python 3.10 (available).
- 57 GB free disk; 30 GB RAM; 24 CPU threads.
- Official code stays unmodified where it defines the method. Additions are
  config-driven and default-off.

## Scope

Tasks: `beat_block_hammer`, `pick_diverse_bottles`, `open_laptop`,
`handover_block` (the paper's headline task plus its three ablation tasks).
Methods: LeaP (paper defaults) and NoPrior (same architecture, x0 ~ N(0, I),
no NLL/align losses). Seed 0, 50 demos, 3000 epochs, official 100-episode eval.
Then, on one task with the trained LeaP checkpoint, an intervention sweep.

Paper reference numbers (seed-averaged, LeaP / NoPrior):
beat_block_hammer 94.0 / 46.7; pick_diverse_bottles 77.0 / 37.3;
open_laptop 86.0 / 46.3; handover_block 91.7 / 38.7.

## Components

1. **Environment** — `uv venv` (py3.10) at repo `.venv`; torch/torchvision
   cu128; `pip install -e '.[robotwin]'` without the torch pins; RoboTwin at
   the locked commit under `third_party/RoboTwin` via `scripts/fetch_robotwin.py`;
   cuRobo v0.7.8 and PyTorch3D built from source against the installed torch via
   `scripts/install_robotwin.py` (patched only if the constraints file blocks
   the newer torch); assets via the official downloader. Record versions in
   `docs/REPRODUCTION.md`.
2. **NoPrior variant** — `LeaPPolicy(prior_type="leap"|"none")`. With
   `"none"`: `x0 ~ N(0, I)` of shape (B, H, A), `loss = loss_flow` only, prior
   and alignment heads not constructed. `configs/noprior.yaml` mirrors
   `leap.yaml` with `prior_type: none`. Checkpoint loading passes the field
   through `policy_config`, so deploy needs no change.
3. **Prior intervention hook** — `policy.set_prior_intervention(cfg | None)`
   consumed inside `predict_action` after `_encode_obs`: `sigma_scale`
   (float), `mu_shift` (tensor A), `mu_override` / `logvar_override`
   (tensors A), and `direct_mu` (skip the flow, return μ broadcast over H).
   Default `None` leaves behaviour byte-identical (regression test).
   `scripts/robotwin.py eval` gains `--intervention JSON` which is forwarded to
   the deploy config and applied in `get_model`.
4. **Prior statistics dump** — `scripts/prior_stats.py --checkpoint --data`:
   runs the prior over the dataset and writes per-dimension mean of μ, σ, the
   NLL of the true action chunk, and |μ − a_t| vs |μ − s_t| to JSON, so the
   write-up can say what the prior actually learned.
5. **Experiment driver** — `scripts/run_campaign.sh`: sequential collect →
   convert → train (LeaP, NoPrior) → eval per task, idempotent via existing
   outputs, logs under `logs/`. Eval results are read from
   `ROBOTWIN_ROOT/eval_result/`.
6. **Report** — `docs/REPRODUCTION.md`: environment, per-task table
   reproduced vs paper, μ/σ statistics, intervention sweep table, deviations,
   and a short section on implications for prior steering.

## Testing

- Existing 12 CPU unit tests keep passing.
- New tests: NoPrior forward/backward and shape; intervention default is a
  no-op; `sigma_scale=0` yields deterministic output across seeds;
  `direct_mu` returns μ; checkpoint round-trip of `prior_type`.
- Pipeline: 1-episode smoke (`scripts/smoke_robotwin.py`) before the campaign.

## Out of scope

Other 11 tasks, additional seeds, A2A/VITA/BridgePolicy baselines, real robot,
any change to the official RoboTwin files.

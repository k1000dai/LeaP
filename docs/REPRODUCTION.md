# LeaP reproduction on an RTX 5080 (2026-10-12)

Independent re-run of the released code on four RoboTwin 2.0 tasks, with a
same-architecture NoPrior baseline and an inference-time prior-intervention
study. Everything below was produced on one machine; see "Deviations" before
comparing numbers with the paper.

## Environment

| Component | Paper / README pin | This run |
|---|---|---|
| GPU | not stated | NVIDIA RTX 5080 Laptop, 16 GB, sm_120, driver 580.178.04 |
| Python | 3.10 | 3.10.20 (uv venv, no conda) |
| PyTorch / torchvision | 2.4.1+cu121 / 0.19.1 | **2.11.0+cu128 / 0.26.0+cu128** |
| CUDA toolkit for source builds | 12.1 | 12.8 |
| numpy / zarr / h5py | 1.26.4 / 2.12.0 / – | 1.26.4 / 2.12.0 / 3.16.0 |
| SAPIEN / mplib | 3.0.0b1 / 0.2.1 | 3.0.0b1 / 0.2.1 (official sed patches applied) |
| RoboTwin | commit 9b5491d | commit 9b5491d, tracked files unchanged |
| cuRobo | v0.7.8 d64c4b0 | v0.7.8 d64c4b0, built from source for sm_120 |
| PyTorch3D | stable (0.7.8) | 0.7.8, built from source for sm_120 |
| warp-lang | unpinned | **1.9.1** (see deviations) |

## Deviations from the documented stack

1. **PyTorch 2.11+cu128 instead of 2.4.1+cu121.** The pinned cu121 wheels have no
   kernels for Blackwell (sm_120) and cannot run on this GPU. cuRobo and
   PyTorch3D were compiled against the newer torch with `TORCH_CUDA_ARCH_LIST=12.0`.
   The official installer (`scripts/install_robotwin.py`) was not used because it
   enforces `constraints-robotwin.txt` (torch==2.4.1); its steps were run by hand
   with the torch lines removed from RoboTwin's `script/requirements.txt`.
2. **warp-lang pinned to 1.9.1.** cuRobo v0.7.8 calls `wp.torch.*` without
   importing `warp.torch`; warp >= 1.10 no longer imports that submodule
   automatically, so with the then-latest 1.18.0 every expert-planning episode
   failed with `module 'warp' has no attribute 'torch'` (first attempt, log kept
   in `logs/campaign-attempt1-warp1.18.log`). 1.9.1 is the newest release that
   still works and it runs on sm_120.
3. **Rasterizer instead of SAPIEN's ray tracer (`LEAP_RASTER=1`).** RoboTwin
   renders with SAPIEN's `rt` shader, 32 spp and the `oidn` denoiser. On this
   host OIDN fails (`OIDN Error: unsupported device type: CUDA`, rendering
   continues undenoised) and, independently, the ray-traced host readback
   `camera.get_picture("Color")` blocks forever every few episodes
   (faulthandler stacks in `logs/collect-beat_block_hammer.log`; `camera.py:335`).
   Disabling OIDN and bypassing the Vulkan/CUDA interop did not remove the hang;
   switching the camera shader to SAPIEN's rasterizer did (no stalls afterwards,
   and ~9 s instead of ~20 s per episode). The switch is an opt-in interpreter
   shim (`scripts/shims/sitecustomize.py`); official RoboTwin files are untouched.
   Effect on observations: LeaP consumes XYZ point clouds from the Position
   buffer only. On a test scene the two renderers agree to sub-pixel sample
   jitter (depth ratio 1.001 ± 0.018, median |Δdepth| 0.3 mm, ray directions
   within one pixel, differences concentrated at silhouettes); RGB shading
   differs but is not used by the policy. Collection and evaluation use the
   same renderer, so train/test observations are consistent. A collection
   watchdog (`STALL_MINUTES`) remains in `scripts/run_campaign.sh` as a safety
   net; the official collector resumes from saved episodes.
4. **Two hosts.** beat_block_hammer and pick_diverse_bottles (training,
   evaluation, sweep, probes) ran on the RTX 5080 laptop. open_laptop and
   handover_block ran on `robot_dev6` (3× RTX A6000, driver 595.71, Ubuntu
   26.04, no system CUDA toolkit: CUDA 12.8 + GCC 13 from conda-forge via pixi;
   same Python environment recipe, same torch 2.11+cu128, same warp 1.9.1,
   same rasterizer shim for consistency, `TORCH_CUDA_ARCH_LIST=8.6`). The
   open_laptop demonstrations were collected on the laptop and copied over;
   handover_block was collected on robot_dev6.
5. **One seed (0), four tasks, 100 evaluation episodes each.** The paper's
   main table averages three seeds over 15 tasks.

## Code additions (all default-off, official method path untouched)

- `prior_type: none` in `LeaPPolicy` and `configs/noprior.yaml`: same encoder,
  velocity network, optimizer, EMA and schedule, but `x0 ~ N(0, I)` and flow
  loss only. This is the paper's "NoPrior" counterpart, which the release did
  not include.
- `LeaPPolicy.set_prior_intervention(...)`: inference-time `sigma_scale`,
  `mu_shift`, `mu_override`, `logvar_override`, `direct_mu`; forwarded by
  `scripts/robotwin.py eval --intervention JSON`.
- `scripts/prior_stats.py`: what the trained prior predicts over a dataset.
- `scripts/run_campaign.sh`: collect -> convert -> train (LeaP, NoPrior) -> eval.
- Tests: `tests/test_prior_variants.py`; launcher test extended. 19 CPU tests pass.

## Results

**Summary.** On three of the four tasks the release reproduces the paper
within a few points for both the method and its same-architecture baseline:
LeaP 87 / 70 / 86 vs paper 94 / 77 / 86, NoPrior 46 / 32 / 49 vs paper
47 / 37 / 46 (beat_block_hammer / pick_diverse_bottles / open_laptop), so the
LeaP-over-NoPrior gap (+41, +38, +37 points) matches the paper's (+47, +40,
+40). handover_block is shifted down for both models in this setup (67 vs 92,
2 vs 39; a second LeaP seed gives 61) for reasons discussed below. Averaged over
the four tasks: LeaP 77.5 vs NoPrior 32.3 here, 87.2 vs 42.3 in the paper.

### Success rate, official 100-episode protocol, seed 0 (this run) vs paper (mean ± std over 3 seeds)

| Task | LeaP (paper, 3 seeds) | LeaP (this run) | NoPrior (paper) | NoPrior (this run) |
|---|---|---|---|---|
| beat_block_hammer | 94.0 ± 2.6 | **87** | 46.7 ± 4.0 | **46** |
| pick_diverse_bottles | 77.0 ± 6.1 | **70** | 37.3 ± 8.1 | **32** |
| open_laptop | 86.0 ± 6.9 | **86** | 46.3 ± 3.5 | **49** |
| handover_block | 91.7 ± 1.5 | **67** (seed 1: 61) | 38.7 ± 11.6 | **2** |

### Training curves (LeaP, mean of the last 10 epochs; validation = EMA model on the 1 held-out episode)

| Task | updates | flow | NLL | align | total | val first → last (min) |
|---|---|---|---|---|---|---|
| beat_block_hammer | 66,022 | 0.00022 | −2.03 | 1.41 | −1.60 | 5.99 → −2.10 (−2.11) |
| pick_diverse_bottles | 72,024 | 0.00027 | −1.84 | 0.33 | −1.74 | 5.57 → −1.22 (−1.24) |
| open_laptop | 120,040 | 0.00010 | −2.53 | 0.49 | −2.39 | 5.64 → 0.42 (−1.05) |
| handover_block | 165,055 | 0.00017 | −2.58 | 0.39 | −2.46 | 4.83 → −2.23 (−2.23) |

All four runs converge to the same regime (flow residual ≤ 3e-4, NLL ≈ −2 to −2.6).
open_laptop's single-episode validation loss rises late in training yet the policy
scores 86%; with one held-out episode this number is noisy and is not used for
model selection (the trainer always evaluates the final EMA weights, as in the
paper). The handover_block run, the one task with a clear gap to the paper
(67 vs 91.7 ± 1.5), has the tightest prior of all (mean sigma 0.034,
|mu − first action| 0.015, NLL −2.78 over 14,285 windows) and a healthy curve,
so the gap is not a training failure of the prior. To separate seed variance from a
systematic cause, a second LeaP training seed (seed 1, same data, same official
evaluation seed) was run for this task and scored 61%, so the gap is
systematic in this setup (see "handover_block discussion" below).

### What the prior learned

`scripts/prior_stats.py` over all 5680 training windows of beat_block_hammer
(normalized action units, range [-1, 1]; `outputs/analysis/prior_stats-*.json`):

| Statistic | LeaP (trained) | NoPrior (by construction) |
|---|---|---|
| mean sigma (min / max) | 0.046 (0.007 / 0.338) | 1.0 |
| mean \|mu − first executed action\| | 0.022 | 0.795 |
| mean \|mu − current joint state\| | 0.032 | – |
| mean \|mu − action chunk\| (over 8 steps) | 0.028 | – |
| Gaussian NLL of true chunk under the source | −2.39 | +1.28 |
| final flow-matching loss (train, last 10 epochs) | 0.0002 | 0.0048 |
| EMA validation loss (total) | −2.10 | 0.003 |

The other LeaP checkpoints give the same picture (mean sigma, |mu − first
action|, |mu − current state|, NLL): pick_diverse_bottles 0.053 / 0.026 /
0.040 / −2.11 (6106 windows); open_laptop 0.030 / 0.010 / 0.013 / −2.83
(10,268 windows); handover_block 0.034 / 0.015 / 0.021 / −2.78 (14,285 windows).

The learned source is essentially "the next action, plus a small
state-dependent spread": mu lands within ~2% of the action range of the first
executed action (closer to it than to the current state), and sigma is ~20×
narrower than the standard normal. The flow-matching residual shrinks ~20×
relative to NoPrior because the velocity field only has to bridge a short gap.

### Prior-intervention sweep (beat_block_hammer, trained LeaP checkpoint, seed 0)

Same checkpoint and the same official evaluator; only the source distribution
at inference is changed (`scripts/robotwin.py eval --intervention`,
`scripts/intervention_sweep.sh`). Unless noted, 100 official episodes.

| Inference-time source | Success | Note |
|---|---|---|
| learned mu, learned sigma (default) | **87** | official LeaP result above |
| learned mu, sigma × 0 (deterministic source) | 86 | |
| learned mu, sigma × 0.5 | 84 | |
| learned mu, sigma × 2 | 85 | |
| learned mu, sigma × 4 | 73 | |
| mu = 0, sigma = 1 (NoPrior source into the LeaP flow) | 0 / 18 | official run truncated after 18 episodes (each failure runs the 400-step limit); unambiguous |
| direct mu execution (no flow) | 0 / 10 | limited protocol via `scripts/eval_limited.py`; paper reports 0.0 |

Reading: the flow is tolerant to the *width* of the source (nothing changes
between sigma × 0 and sigma × 2; only sigma × 4, i.e. ~0.18 in normalized
units, costs 14 points) but is tightly coupled to its *location*: the velocity
field trained around the learned mu cannot transport samples from the standard
normal at all, and mu alone, without the flow, is not an executable policy.
Source noise is not what makes LeaP work on this task; the state-conditioned
mean is.

### Offline steerability probe (`scripts/steer_probe.py`)

How much of an inference-time change to the source reaches the executed
action? 512 random training windows of beat_block_hammer, deterministic source
(sigma × 0), first executed action, normalized units. `mu` is shifted by delta
along one joint; the pass-through gain is (change in that joint) / delta.

| Joint dim | gain at delta = ±0.05 … ±0.5 | leak into other joints (mean abs) |
|---|---|---|
| 0 | 0.04 – 0.07 | ≤ 0.008 |
| 3 | 0.09 – 0.10 | ≤ 0.010 |
| 7 | 0.03 – 0.04 | ≤ 0.006 |
| 10 | 0.07 – 0.08 | ≤ 0.007 |

Gain versus number of Euler steps (delta = 0.2, same four joints):

| NFE | 1 | 2 | 6 (default) | 20 |
|---|---|---|---|---|
| gain | 0.04 / 0.06 / −0.01 / 0.05 | 0.05 / 0.08 / 0.02 / 0.06 | 0.06 / 0.09 / 0.03 / 0.07 | 0.06 / 0.09 / 0.04 / 0.07 |

The same probe on the NoPrior checkpoint (source N(0, I), shifted by the
same deltas) gives a pass-through gain of 0.00 on every joint with ≤ 0.001
leak: a flow trained on sources spanning the whole standard normal cancels a
shift exactly. LeaP's residual 3–30% is therefore not "the prior steering the
output" but imperfect generalization of its velocity field outside the narrow
tube of sources it was trained on.

The same probe on the other three LeaP checkpoints (delta = ±0.2, joints
0 / 3 / 7 / 10, mean of the two signs):

| Task | gain per joint | leak (max) | spread at learned sigma | spread at sigma × 4 |
|---|---|---|---|---|
| beat_block_hammer | 0.05 / 0.09 / 0.03 / 0.07 | 0.003 | 0.0016 | 0.0089 |
| pick_diverse_bottles | 0.08 / 0.13 / 0.07 / 0.13 | 0.004 | 0.0019 | 0.0102 |
| open_laptop | 0.05 / 0.20 / 0.05 / 0.22 | 0.005 | 0.0011 | 0.0053 |
| handover_block | 0.17 / 0.25 / 0.05 / 0.09 | 0.008 | 0.0015 | 0.0072 |

The gain is joint- and task-dependent (0.03–0.30) but always far below 1:
the flow cancels 70–97% of any shift of the source mean, with the largest
residual on joints whose demonstrations vary most.

Source noise is contracted the same way: sampling spread of the first action
across source draws is 0.0016 at the learned sigma (source sigma ≈ 0.046, so
~30× contraction), 0.0034 at sigma × 2 and 0.0089 at sigma × 4.

Interpretation. The observation-conditioned velocity network cancels 70–97%
of any shift of the source mean (linearly over ±0.5, a quarter of the action
range, on beat_block_hammer), and it does so already after one Euler step. This is structural:
conditional flow matching trains v(x_t, t, obs) to predict x1 − x0, so at
t = 0 the network reads x0 off its input and returns x1 − x0 for *any* x0 it
generalizes to. The learned prior therefore helps by placing x0 where the
velocity field is accurate and short (hence the 20× lower flow loss and the
higher success), not by acting as a lever that moves the output. Consequences
for using the source as an in-context steering channel:

- Shifting or re-sampling mu/sigma at inference, with an unchanged flow,
  will not steer a LeaP-style policy; the flow is the dominant term.
- Steering has to enter where the flow cannot cancel it: the condition
  (observation / context tokens), the velocity network itself, or a training
  objective that makes the generator *respect* the source (e.g. a velocity
  target that does not subtract x0, a prior-dependent condition, fewer
  training-time draws from a fixed mu, or an explicit steering term in the
  loss). The source-only variants in the paper's Table 8 point the same way:
  without NLL/alignment the prior drifts and the flow compensates.
- The reverse direction is strong: the NoPrior source into the LeaP flow
  gives 0/18, so a mismatched source is catastrophic while a shifted one is
  ignored. The flow's domain of competence is a narrow tube around the
  trained source, which is what a steering method would have to widen.

## handover_block discussion

handover_block is the only task where this run is far from the paper: LeaP 67
(seed 1: 61) vs 91.7 ± 1.5, NoPrior 2 vs 38.7 ± 11.6 (paper seed-0 ablation
rows: 92 and 52). Both of our models are shifted down by roughly the same
amount, both training runs converged normally, and the LeaP prior on this task
is the tightest of the four, so the shift sits in the task/observation
pipeline rather than in the prior. What differs from the paper for this task
specifically: (a) it is the only task collected and evaluated entirely on the
second host; (b) it is bimanual with the longest horizon (800 steps) and the
smallest manipulated object (a block passed between grippers), which makes it
the task most sensitive to point-cloud silhouettes, and our rasterized
Position buffer differs from the paper's ray-traced one exactly at
silhouettes (sub-pixel sample averaging, see Deviations); (c) 50 demonstrations
from a different expert-seed sequence. None of these was isolated; doing so
would need a ray-traced collection/evaluation on a GPU where SAPIEN's ray
tracer does not hang, or the authors' demonstrations. The other three tasks
(two 400-step, one 700-step) match the paper within a few points for both
models, so the method-level conclusion (LeaP ≫ NoPrior at equal architecture)
is unaffected; the absolute handover_block numbers should not be compared
with the paper's.

## Timing (wall clock, one GPU per run)

| Stage | RTX 5080 laptop | robot_dev6 (A6000) |
|---|---|---|
| Collect 50 demos (rasterizer) | ~8–25 min / task (beat_block_hammer ~8, pick_diverse_bottles ~25) | ~25 min (handover_block) |
| Train 3001 epochs | 2.0 h (5.7k windows) – 2.2 h (6.1k) | 3.7 h (10k windows) – 5.0 h (14k) at ~9 updates/s |
| Official 100-episode eval | 35–50 min (400-step tasks) | 1.8 h (open_laptop, 700 steps) – 2+ h (handover_block, 800 steps; failing episodes run to the limit) |

Training throughput was ~9 updates/s on both GPUs (batch 256, 4–8 loader
workers), i.e. the loop is not GPU-bound at this model size.

## Reproducing

~~~bash
# environment: see "Environment" and scripts/shims/sitecustomize.py; then
export LEAP_ROOT=$PWD ROBOTWIN_ROOT=$PWD/third_party/RoboTwin
export LEAP_RASTER=1 LEAP_FAULTHANDLER=1 PYTHONPATH=$PWD/scripts/shims
GPU=0 SEED=0 scripts/run_campaign.sh beat_block_hammer pick_diverse_bottles open_laptop handover_block
scripts/intervention_sweep.sh beat_block_hammer outputs/beat_block_hammer-leap-s0/checkpoints/latest.ckpt leap-s0
python scripts/prior_stats.py --checkpoint outputs/<run>/checkpoints/latest.ckpt --data data/<task>-leap_clean-50.zarr --output outputs/analysis/prior_stats-<run>.json
python scripts/steer_probe.py  --checkpoint outputs/<run>/checkpoints/latest.ckpt --data data/<task>-leap_clean-50.zarr --output outputs/analysis/steer_probe-<run>.json
python scripts/collect_results.py --robotwin third_party/RoboTwin --output outputs/analysis/results.json
~~~

Logs of every attempt, including the failed ones, are kept under `logs/`
(not committed). Official per-run result files live under
`third_party/RoboTwin/eval_result/<task>/leap.robotwin.deploy/leap_clean/<setting>/<timestamp>/_result.txt`.

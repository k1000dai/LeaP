# Steering a frozen VLA through a learned flow-matching prior — design

Date: 2026-10-17. Phase 3 of the steering project; builds on the LeaP
reproduction (`docs/REPRODUCTION.md`) and the multimodal-instruction phase
(`2026-10-17-leap-multimodal-steering-design.md`, shared code in `leap/steer/`).

## Goal (user intent)

Apply the "learned source prior" idea to a real VLA. Flow Reversal Steering
(FRS, Tang, Chen, Wagenmaker, Finn, Levine; arXiv 2606.13675) steers a frozen
π0.5 by running a coarse reference action *backwards* through the flow to get
a noise, then denoising it. Its DSBC variant distils this into a small "noise
policy". The user's proposal: do not invert at test time; **train a forward
model that predicts the flow-matching prior (the source noise) from the
observation and a multimodal steering signal** (sketch / text / audio embedded
by `google/embeddinggemma-2`), and feed that noise to the frozen VLA. The prior
network is LeaP's μ(s), σ(s) transplanted on top of a frozen VLA, with the
steering signal as an extra input and reversal-generated noises as labels.

## Facts established on 2026-10-17

- FRS (paper, App. B–E): π0.5 flow with noise at one end and actions at the
  other; reversal and forward denoising both use 10 Euler steps (h = 0.1).
  DSBC noise policy (LIBERO): MLP (128,128,128), tanh-squashed to [−5, 5],
  predicts one 7-d noise repeated over the chunk, padding dims replaced by
  i.i.d. N(0, I) (no effect on performance), NLL loss (MSE similar), Adam
  1e-4, batch 128, 12.5 k steps, < 1 min, ~1 GB. Offline DSBC uses noises from
  reversing demonstrations. Reversal reconstructs references only
  approximately; out-of-distribution references map to low-likelihood noise.
- Base VLA: LeRobot `PI05Policy` (PyTorch port of openpi π0.5) with the
  checkpoint `lerobot/pi05_libero_finetuned_v044` (bf16, chunk_size 50,
  num_inference_steps 10, inputs `observation.images.image` (agentview 256²),
  `observation.images.image2` (wrist 256²), one empty camera,
  `observation.state` (8), action (7), MEAN_STD normalisation; the sibling
  `lerobot/pi05-libero` only sets n_action_steps 10). The user's LeRobot clone
  (`~/codes/ghq/github.com/huggingface/lerobot`, 0.6.1, commit 9459727e7,
  2026-08-14, has a `.venv` with `libero` importable) is the pinned source.
- LeRobot flow convention (`policies/common/flow_matching.py`,
  `pi05/modeling_pi05.py`): x_t = t·noise + (1−t)·actions, u_t = noise −
  actions, sampling integrates from t = 1 (noise) to t = 0 (actions) with
  `x ← x + dt·v`, dt = −1/K. `PI05FlowMatching.sample_actions(images,
  img_masks, tokens, masks, noise=None, num_steps=None)` accepts an injected
  noise of shape [B, 50, 32]; `denoise_step(prefix_pad_masks, past_key_values,
  x_t, timestep)` returns v_t given the cached prefix. Reversal is therefore
  `x ← x + (1/K)·v(x, t)` for t = 0, 1/K, …, (K−1)/K starting from the
  (padded, normalised) action chunk.
- LeRobot LIBERO env (`envs/libero.py`): `LiberoEnv(task_suite, task_id,
  task_suite_name, ...)` wraps `OffScreenRenderEnv` (256×256 agentview +
  wrist), `task_description` = task language, init states from the suite,
  `check_success()` from the BDDL goal; `lerobot-eval` feeds
  `observation["task"]` from `task_description`. The underlying robosuite env
  exposes `sim` (object body positions) and camera matrices.
- LIBERO-Spatial: 10 tasks, all "pick the akita black bowl <spatial phrase>
  and place it on the plate"; each scene has **two** black bowls
  (`akita_black_bowl_1` = target, `_2` = distractor), one plate, a ramekin,
  cookie box, cabinet and stove; goal `(On akita_black_bowl_1 plate_1)`.
  Raw demos (50 per task, with sim `states`, 128² images, 7-d actions) are at
  `yifengzhu-hf/LIBERO-datasets/libero_spatial/*.hdf5` (6.2 GB total).
- Remote robot_dev6: no docker without sudo, uv with Python 3.10/3.11/3.12,
  EGL (nvidia + mesa) present for headless MuJoCo, 32 CPUs, 499 GB RAM, 3×
  A6000 shared with the phase-2 campaign.

## Benchmark: LIBERO-Spatial with a neutral prompt

The VLA is run with the **neutral prompt** "pick up the black bowl and place
it on the plate", which does not say which of the two bowls. Success requires
the specific bowl (`akita_black_bowl_1`) on the plate, so an unsteered policy
is at best at the rate with which it happens to pick that bowl. The steering
signal identifies the target bowl:

| modality | content |
|---|---|
| text | the task's own language ("pick up the black bowl between the plate and the ramekin and place it on the plate") |
| sketch | agentview frame 0 with a red circle around `akita_black_bowl_1` and a blue arrow from it to `plate_1`, positions projected with the agentview camera (robosuite `get_camera_transform_matrix`), drawn with `leap.steer.sketch.render_sketch` |
| audio | Piper TTS of the text, 16 kHz (`leap.steer.tts`) |

All three are embedded with `leap.steer.embedder.InstructionEmbedder`
(768-d, standardised with per-dimension statistics of the training set).

Reference conditions: `full-prompt` (original language, Gaussian noise; the
upper reference) and `neutral-gauss` (neutral prompt, Gaussian noise; the
unsteered control).

## Offline reversal dataset

`scripts/vla/build_reversal_dataset.py` produces
`data/vla/libero_spatial_reversal/` from the raw demos:

1. For each task and demo, replay the stored sim states in the LeRobot/LIBERO
   env (same renderer as evaluation) every `--stride` steps (default 2):
   render agentview + wrist at 256², read the 8-d state exactly as
   `LiberoEnv._format_raw_obs` builds `observation.state`, read
   `akita_black_bowl_1`, `akita_black_bowl_2`, `plate_1` positions and the
   agentview camera matrices; at frame 0 render the sketch.
2. Action chunk `a[t : t+50]` from the demo (7-d), padded by repeating the
   last action (gripper held) when the demo ends; a `chunk_pad_mask`.
3. Run the frozen VLA with the **neutral prompt** on the observation and
   reverse the (normalised, zero-padded to 32-d) chunk with K = 10 steps →
   `noise_label` [50, 32] (dims ≥ 7 are discarded later). Also record the
   reconstruction error ‖forward(reverse(a)) − a‖ on a 5 % subsample.
4. Observation features for the prior network: mean-pooled projected image
   tokens of the two real cameras from `PaliGemmaWithExpertModel.embed_image`
   (2 × 2048) plus the 8-d state. (Keeping the VLA's own features avoids a
   second vision encoder; a DSBC-style 84² CNN is a later ablation.)
5. Steering embeddings per demo: `text` [T_VAR, 768] (the task language plus
   LIBERO's own paraphrases if present, else 1), `sketch` [1, 768],
   `audio` [A_VAR, 768] via `leap.steer` (T_VAR = 1, A_VAR = 1 by default).

Storage: one `.npz` per task (`task_{i}.npz`) with `feat [N, 4104]`,
`state [N, 8]`, `noise [N, 50, 7]`, `chunk_pad_mask [N, 50]`, `demo_id [N]`,
`t [N]`, `recon_err [N]` (NaN where not measured), `instr_text [D, T_VAR, 768]`,
`instr_sketch [D, 1, 768]`, `instr_audio [D, A_VAR, 768]`, `task_id`,
`task_language`, plus `episode0_sketch.png` per demo for inspection.

## Prior network ("forward model of the prior")

`leap/vla/prior.py`: `VLAPrior(feat_dim=4104, instr_dim=768, hidden=(1024,
1024), horizon=50, action_dim=7, instr_mode)`:

- Inputs: standardised features (mean/std buffers from the training set),
  standardised steering embedding (buffers), `instr_mode ∈ {none, concat}`
  (`none` = offline DSBC without steering, the control).
- Output: μ [50, 7] and log σ² [50, 7] (per element; log σ² clamped to
  [−10, 5]); loss = Gaussian NLL of `noise` (masked by `chunk_pad_mask`),
  optional MSE weight. Same head shape as LeaP's prior, so later variants can
  reuse LeaP code.
- Sampling for the VLA: `x0 = μ + σ·ε` (or `μ` with `--sigma-scale 0`), then
  padding dims 7..31 ← i.i.d. N(0, I), un-normalisation is **not** applied
  (the noise lives in the VLA's normalised action space, as the labels do).
- Training: Adam 1e-4, batch 256, 12.5 k steps, modality sampled uniformly
  per sample among the configured set (text/sketch/audio), held-out demos
  (10 %) for validation NLL. Minutes on one GPU.

## Evaluation

`scripts/vla/eval_vla_steer.py --suite libero_spatial --tasks 0-9
--episodes N --condition {full-prompt, neutral-gauss, neutral-prior}
--instr-modality {none,text,sketch,audio} --prior <ckpt> --sigma-scale S
--seed 0`:

- Own rollout loop (not `lerobot-eval`): builds the LeRobot preprocessors
  for the checkpoint, overrides the prompt, at episode start builds the
  steering signal from the live scene (sketch from the first agentview frame
  and the live bowl/plate positions; text from the task language; audio via
  TTS), embeds it once, and at every policy call computes the prior noise
  from the current VLA features and injects it with `sample_actions(...,
  noise=x0)`; executes `n_action_steps = 10` of the 50-chunk (as
  `lerobot/pi05-libero`). Max episode length from `TASK_SUITE_MAX_STEPS`.
- Logs per episode: task, success, **bowl choice** (which bowl first rises
  5 cm above its initial height; "none" if neither), steps, condition,
  modality, seed → `outputs/vla/<condition>-<modality>/episodes.jsonl` and a
  summary JSON. Deterministic init states from the suite, episode index =
  init-state index.

Phase-1 matrix (10 tasks × 20 episodes, then 50 for the final table):
`full-prompt`; `neutral-gauss`; `neutral-prior` with instr_mode none; and
`neutral-prior` × {text, sketch, audio} (σ-scale 1 and 0). Metrics: success
rate, bowl-choice accuracy, and the offline reconstruction error and
validation NLL of the prior.

Phase 2 (if time allows): DSBC-style 84² CNN features instead of VLA features;
cross-modal zero-shot (train text-only, test sketch/audio); comparison with
"text via prompt" (full-prompt) to see whether the prior path matches the
VLA's native language path; LIBERO-Object analogue.

## Environment and compute

- Remote venv `~/codes/LeaP/.venv-vla` (uv, Python 3.10): `lerobot[pi05,
  libero]` installed from the user's clone commit 9459727e7 (rsync of the
  clone or `pip install git+...@9459727e7`), `sentence-transformers>=6.1`,
  `transformers` version as required by that LeRobot (check its pin; the
  phase-2 venv is separate and stays untouched), `piper-tts`, `soundfile`,
  `h5py`. EGL rendering (`MUJOCO_GL=egl`).
- Checkpoint `lerobot/pi05_libero_finetuned_v044` and the 10 LIBERO-Spatial
  hdf5 files cached under `~/.cache/huggingface` and `data/vla/raw/`.
- GPUs shared with the phase-2 campaign (π0.5 bf16 inference ≈ 8 GB).
  Expected: dataset build ≈ 2 GPU-h (≈ 30 k chunks at stride 2, batched
  reversal), prior training minutes, each evaluation condition ≈ 1 h for
  10 × 20 episodes.
- Driver `scripts/vla/run_vla_campaign.sh` with idempotent stages
  (setup → data → prior → eval → results), logs under `logs/vla/`.

## Code layout (this repo)

- `leap/vla/pi05_flow.py`: `load_pi05(checkpoint, device, n_action_steps=10)`
  → (policy, preprocessor, postprocessor); `prepare_batch(policy, preprocessor,
  images, state, prompt)`; `prefix_features(policy, batch)` → [B, 4104];
  `reverse_actions(policy, batch, actions_norm_padded, num_steps=10)` →
  noise [B, 50, 32]; `sample_actions_with_noise(policy, batch, noise,
  num_steps=10)` → actions (un-normalised, 7-d, [B, 50, 7]);
  `normalize_actions(policy, actions7)` / `pad_actions`.
- `leap/vla/libero_scene.py`: `make_env(suite, task_id, prompt_override)`,
  `scene_layout(env)` → bowl_1/bowl_2/plate xyz + their initial heights,
  `agentview_camera(env)` → (intrinsic 3×3, extrinsic 3×4 in the convention
  `leap.steer.sketch.project_points` expects), `bowl_choice(env, layout)`,
  `replay_state(env, state)` for dataset building, `libero_state_vector(raw_obs)`
  (8-d, identical to `LiberoEnv._format_raw_obs`).
- `leap/vla/prior.py`, `leap/vla/instructions.py` (build text/sketch/audio for
  a task + scene, shared by dataset builder and evaluator).
- `scripts/vla/{remote_setup_vla.sh, build_reversal_dataset.py,
  train_vla_prior.py, eval_vla_steer.py, run_vla_campaign.sh,
  collect_vla_results.py}`; tests under `tests/test_vla_*.py` with a toy
  velocity field for the integrator math and fakes for the env/policy.
- Report `docs/VLA_STEERING.md`.

## Risks and assumptions (decided without user input; the goal directive asked not to pause)

- The VLA under the neutral prompt may already prefer one bowl per layout
  (e.g. the nearer one); the `neutral-gauss` control measures this.
- Reversal labels are only approximately consistent (FRS reports this); the
  reconstruction error is logged and reported. If forward(reverse(a)) is far
  from a at K = 10, K = 20 is tried for labels (evaluation keeps K = 10).
- The LeRobot checkpoint's LIBERO score is not the openpi one; the
  `full-prompt` condition measured here is the reference.
- The user's LeRobot clone pins its own transformers version; the VLA venv is
  separate from the LeaP venv to avoid conflicts.
- Steering here selects among discrete behaviours (which bowl). Continuous
  steering (e.g. "place it further left") is out of scope for phase 1.

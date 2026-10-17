# Steering a frozen VLA through a learned flow-matching prior (phase 3)

Status: experiment running (2026-10-18). Results section is filled in when the
campaign on robot_dev6 finishes. Design:
`docs/superpowers/specs/2026-10-17-vla-prior-steering-design.md`.

## Idea

Flow Reversal Steering (FRS; Tang, Chen, Wagenmaker, Finn, Levine, arXiv
2606.13675) steers a frozen π0.5 by integrating a coarse reference action
*backwards* through the flow to a noise and denoising it; DSBC distils this
into a small noise policy. Here the inversion is replaced by a **forward model
of the prior**: a network that predicts the source noise of the frozen VLA
from the VLA's own observation features and a multimodal steering embedding
(sketch / text / audio via `google/embeddinggemma-2`), trained on noises
obtained by reversing demonstrations (offline DSBC with a steering input).
The network has LeaP's prior head (μ, log σ² per chunk element).

## Base VLA

LeRobot `PI05Policy` (PyTorch port of openpi π0.5), checkpoint
`lerobot/pi05_libero_finetuned_v044` (bf16, chunk 50, 10 Euler steps, images
agentview + wrist 256², one empty camera, 8-d state, 7-d actions,
MEAN_STD normalisation), run with `n_action_steps = 10`. LeRobot 0.6.1 at the
user's clone commit 9459727e7; venv `.venv-vla` (Python 3.12 — LeRobot
requires ≥ 3.12; transformers 5.5.4 pinned, which is why the embedder runs in
the phase-2 venv).

Flow convention (LeRobot): x_t = t·noise + (1−t)·actions, u = noise −
actions; sampling integrates t: 1 → 0 with x ← x + dt·v, dt = −1/K. Reversal:
x ← x + (1/K)·v(x, t) for t = 0, 1/K, …, (K−1)/K from the normalised, zero-padded
action chunk (`leap/vla/pi05_flow.py`, verified bit-identical to LeRobot's
sampler and to its evaluation preprocessing on a live observation).

### Reversal diagnostics (task 0, A6000, `scripts/vla/probe_reversal.py`)

| metric (dims 0..6, relative L2) | K=10 | K=20 | K=50 |
|---|---|---|---|
| forward(reverse(a)) vs a, a = VLA sample | 0.038 | 0.029 | 0.024 |
| reverse(forward(z)) vs z | 0.82 | 0.72 | 0.66 |
| std of recovered noise (real dims) | 0.57 | 0.55 | 0.57 |
| two independent samples, a vs a' | 0.133 | | |
| forward(reverse(a_demo)) vs a_demo (demo chunks, 19 frames) | 0.256 | 0.140 | |
| +0.3 shift on dx (10 steps) kept after reverse→forward | 76 % | 75 % | |

bf16 and fp32 agree to three decimals. The sampler is strongly contractive
in the noise direction (many noises → one chunk), so the noise itself is not
recoverable, but the action round trip is tight for the VLA's own outputs and
approximate for demonstration chunks (FRS reports the same). Recovered noises
have std ≈ 0.57 in the real dims and ≈ 1.6 in the padding dims; padding dims
are replaced by fresh N(0, I) at use (no effect, as in FRS). A crude
reference shift survives the round trip at ~76 % — in contrast to phase 1,
where a shift of LeaP's learned source mean was cancelled by 70–97 %.

Timings (bf16, A6000): sample_actions 269 ms (batch 1) / 1.46 s (batch 16);
reversal with cached prefix 186 ms / 361 ms; model load ~130 s; 8.7 GB after
load, 11.3 GB peak. Baseline `lerobot-eval` on LIBERO-Spatial tasks 0–1
(2 episodes each): 4/4, ~11 env steps/s after a one-time 5-min
`torch.compile` autotune (disabled in our loader).

## Benchmark: LIBERO-Spatial under a neutral prompt

All 10 LIBERO-Spatial tasks are "pick up the black bowl ⟨spatial phrase⟩ and
place it on the plate" with **two** black bowls per scene; success requires
`akita_black_bowl_1` on `plate_1`. The VLA is run with the neutral prompt
"pick up the black bowl and place it on the plate"; the steering signal names
the bowl:

| modality | content |
|---|---|
| text | the task's own language |
| sketch | red circle around bowl 1, blue arrow to the plate, at the agentview pixel positions (policy view), white canvas |
| audio | Piper TTS of the text |

Reference conditions: `full-prompt` (original language, Gaussian noise) and
`neutral-gauss` (neutral prompt, Gaussian noise). Metrics: success and **bowl
choice** (first bowl lifted ≥ 5 cm: target / distractor / none).

## Reversal dataset

`scripts/vla/build_reversal_dataset.py` replays the raw LIBERO-Spatial demos
(`yifengzhu-hf/LIBERO-datasets`, 50 per task, 79–134 steps) in the LeRobot
LIBERO env every 2 steps, renders the policy's views, reads the 8-d state,
reverses the 50-step chunk (padded by repeating the last action) with the
neutral prompt (K = 10), stores VLA features (mean-pooled projected image
tokens of both cameras, 2 × 2048, plus the normalised state) and the
reconstruction error on 5 % of the frames; `scripts/vla/embed_vla_instructions.py`
adds the text/sketch/audio embeddings per demo. Camera and state conventions
were verified against LeRobot's processor on live observations.

## Prior network

`leap/vla/prior.py`: standardised features (4104) [+ standardised embedding
(768)] → MLP (1024, 1024) → μ, log σ² ∈ R^{50×7}; masked Gaussian NLL; Adam
1e-4, batch 256, 12.5 k steps, 10 % held-out demos. Variants: `none`
(observation only, offline DSBC) and `concat` (with the steering embedding,
modality sampled per sample). At evaluation x0 = μ + σ·ε (σ-scale 1) or μ
(σ-scale 0), padded with N(0, I) in dims 7..31, injected through
`sample_actions(noise=...)`.

## Evaluation matrix

10 tasks × 20 episodes (init states 0–19) per condition: full-prompt;
neutral-gauss; neutral-prior (none); neutral-prior × {text, sketch, audio} ×
σ-scale {1, 0}. `scripts/vla/eval_vla_steer.py`; driver
`scripts/vla/run_vla_campaign.sh`.

## Results

(pending — `scripts/vla/collect_vla_results.py --out-dir outputs/vla`)

## Notes

- LeRobot's LIBERO env default render is 360²; we pass 256² explicitly (the
  policy resizes with padding to 224 either way).
- hf-libero asks for input on first import without a config; the drivers set
  `LIBERO_CONFIG_PATH=data/vla/libero_config`.
- Steering here selects among discrete behaviours (which bowl). Continuous
  steering is out of scope for this phase.

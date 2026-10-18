# Steering a frozen VLA through a learned flow-matching prior (phase 3)

Status: phases 3 and 3b evaluated (2026-10-18). Design:
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

Reference conditions (10 tasks × 20 episodes, init states 0–19):

| condition | success | bowl choice = bowl 1 | notes |
|---|---|---|---|
| full-prompt (original language, Gaussian noise) | 96.0 % (192/200) | 99.5 % | per task 0.80–1.00 |
| neutral-gauss (neutral prompt, Gaussian noise) | 84.5 % (169/200) | 86.5 % (173/20/7) | task 7 drops to 5 % (bowl choice 10 %); tasks 1 and 9 to 80 % and 75 %; the other seven stay at 95–100 % |

Under the neutral prompt the VLA still picks bowl 1 in 9 of 10 layouts: the
fine-tuned policy has learned the layout → bowl association from the demos,
so the language is not what selects the bowl in most layouts. The neutral
prompt therefore removes little of the VLA's own preference, and the headroom
for steering *toward bowl 1* is only 84.5 → 96 %. Steering against the
layout preference (toward bowl 2) is the informative test; see phase 3b.

Prior conditions (same 200 episodes; x0 from `prior-none` for `none`, from
`prior-concat` otherwise; `scripts/vla/collect_vla_results.py --out-dir outputs/vla`):

| condition | σ-scale 1: success / bowl 1 | σ-scale 0: success / bowl 1 |
|---|---|---|
| neutral-prior, no instruction | 86.5 % / 87.0 % | 85.0 % / 86.0 % |
| neutral-prior + text | 84.5 % / 87.0 % | 86.0 % / 88.0 % |
| neutral-prior + sketch | 87.0 % / 87.5 % | 86.0 % / 87.0 % |
| neutral-prior + audio | 86.5 % / 87.5 % | 86.5 % / 88.0 % |

Every prior condition lies within 84.5–87.0 % success and 86–88 % bowl
choice, i.e. within ±2.5 points of `neutral-gauss` (the standard error of a
200-episode rate near 85 % is 2.5 points) and with the same per-task pattern:
task 7 stays at 10–20 % success (bowl choice 15–20 %), tasks 1 and 9 at
85–95 % and 75–85 %. The learned source — with or without the steering
embedding, with or without its variance — changes neither the success nor
the bowl choice of the frozen VLA. This is the expected outcome of the
redundancy above: the reversed demonstrations give the prior no
instruction-dependent target to learn (validation NLL 1.84 vs 1.92 for a
global Gaussian), so its μ is close to a per-layout mean that the VLA's own
conditioning already encodes.

## Prior training outcome (reversal labels from the demos)

| | prior-none | prior-concat |
|---|---|---|
| best validation NLL (step) | 1.850 (500) | 1.841 (500) |
| final validation NLL (12.5 k) | 2.42 | 2.41 |
| final training NLL | 1.33 | 1.33 |
| N(0, I) reference | 2.494 | |
| empirical per-element Gaussian | 1.923 | |

31 k frames (28 k train / 3.1 k validation, held-out demos), label std ≈ 1.72,
reconstruction error of the labels mean 0.275 (median 0.249, p90 0.444). The
prior beats the global Gaussian only slightly on held-out demos and overfits
after ~500 steps (the campaign uses `best.pt`); the steering embedding adds
~0.01 nats. One structural reason: in every demo the target is bowl 1, so
within a layout the instruction is constant and redundant with the
observation. The counterfactual extension below addresses this.

## Counterfactual bowl-2 data (phase 3b)

The frozen VLA is prompted, per layout, with the spatial phrase of bowl 2
(derived from the BDDL init regions; every bowl-2 phrase is the bowl-1 phrase
of another task, so the prompts are sentences the checkpoint was fine-tuned
on) to produce successful bowl-2 rollouts; their predicted chunks are reversed
under the neutral prompt, giving noise labels where the instruction, not the
layout, decides the bowl. The prior is then trained on both bowls and
evaluated with instructions for either bowl (`leap/vla/counterfactual.py`,
`scripts/vla/rollout_counterfactual.py`, `scripts/vla/reverse_rollouts.py`,
`scripts/vla/eval_vla_steer.py --target-bowl 2`).

Pilot (tasks 0–1, init states 0–4, bowl-2 success = LIBERO's `On` predicate
on bowl 2):

| task | bowl-1 prompt (task language) | bowl-2 prompt | bowl 1 on plate | bowl 2 on plate |
|---|---|---|---|---|
| 0 | … between the plate and the ramekin … | … next to the ramekin … | 5/5 | **0/5** (all five episodes lift bowl 1) |
| 1 | … next to the ramekin … | … next to the cookie box … | 5/5 | 3/5 (2 lift bowl 1) |

Full collection (10 tasks × 2 bowls × init states 0–19, `data/vla/counterfactual`,
a query recorded every 2 env steps; success = `On(bowl, plate)` for the
prompted bowl):

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| bowl-1 prompt (task language) → bowl 1 on plate | 20 | 20 | 19 | 20 | 19 | 17 | 19 | 20 | 20 | 20 |
| bowl-2 prompt → bowl 2 on plate | **0** | 15 | **0** | 18 | **0** | 20 | 18 | 19 | **1** | 20 |
| bowl-2 prompt → bowl 1 on plate instead | 20 | 3 | 20 | 2 | 17 | 0 | 1 | 0 | 18 | 0 |

The frozen VLA's own language channel is strong in six layouts (tasks 1, 3,
5, 6, 7, 9: 75–100 % bowl 2) and absent in four (tasks 0, 2, 4, 8: the
bowl-2 sentence changes nothing, the VLA places bowl 1). Task 7 — the one
layout in which the neutral prompt fails — is among those where the language
decides. Counterfactual labels therefore exist for the six language-sensitive
layouts (bowl-2 rollouts of the failing layouts are excluded by
`--successful-only`); the four others remain layout-locked for any method
that only re-weights what the VLA can already do, and are reported
separately. Three alternative phrasings for bowl 2 of task 0 ("next to the
ramekin, not the one between the plate and the ramekin", "closest to the
ramekin", "farthest from the plate"; 5 episodes each) also give 0/5: the lock
is not a matter of wording.

Reversal of the rollouts under the neutral prompt (`scripts/vla/reverse_rollouts.py`,
successful episodes only, every recorded query): on the pilot the
reconstruction error of the VLA's own chunks is 0.13–0.21 (mean per (task,
bowl); demonstration chunks: 0.275), recovered-noise std 1.2–1.6.

## Phase 3b: prior on the counterfactual data

`scripts/vla/run_vla_counterfactual.sh`. Dataset: 17 files (bowl 1 of every
task, bowl 2 of the seven tasks with a successful rollout), successful
episodes only, every recorded query; the prior is trained as before
(`prior-cf-concat`, `prior-cf-none`; 12.5 k steps, best checkpoint).

| | prior-cf-concat | prior-cf-none |
|---|---|---|
| best validation NLL (step) | 1.488 (1500) | 1.507 (1000) |
| final validation NLL | 1.62 | 1.72 |
| N(0, I) / empirical Gaussian reference | 2.02 / 1.79 | |

Offline instruction sensitivity of `prior-cf-concat` (`scripts/vla/probe_cf_prior.py`;
held-out rows, the row's own instruction vs the other bowl's instruction for
the same observation; Δ = NLL(swapped) − NLL(own), "own <" = fraction of rows
the own instruction scores better, |Δμ|/σ = mean shift of μ in units of the
predicted σ):

| task | bowl | rows | text Δ | own < | ǀΔμǀ/σ | sketch Δ | own < | ǀΔμǀ/σ |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 / 2 | 103 / 108 | +0.19 / +0.28 | 92 / 94 % | 0.44 / 0.31 | +0.12 / +0.25 | 79 / 93 % | 0.38 / 0.25 |
| 3 | 1 / 2 | 84 / 117 | +0.17 / +0.30 | 88 / 100 % | 0.39 / 0.33 | +0.13 / +0.27 | 87 / 100 % | 0.36 / 0.29 |
| 5 | 1 / 2 | 87 / 87 | +0.25 / +0.69 | 79 / 100 % | 0.54 / 0.54 | +0.25 / +0.76 | 83 / 100 % | 0.54 / 0.52 |
| 6 | 1 / 2 | 124 / 122 | +0.11 / +0.36 | 77 / 97 % | 0.33 / 0.41 | +0.11 / +0.26 | 79 / 94 % | 0.30 / 0.35 |
| 7 | 1 / 2 | 116 / 117 | +0.19 / +0.21 | 95 / 99 % | 0.35 / 0.32 | +0.11 / +0.19 | 88 / 99 % | 0.27 / 0.28 |
| 9 | 1 / 2 | 118 / 117 | +0.21 / +0.23 | 93 / 99 % | 0.29 / 0.36 | +0.19 / +0.19 | 90 / 100 % | 0.27 / 0.30 |

Unlike the demonstration-trained prior, this one encodes the bowl: swapping
the instruction raises the held-out NLL by 0.1–0.7 nats, the own instruction
wins on 77–100 % of rows, and μ moves by 0.3–0.5 σ. Whether a shift of that
size survives the VLA's contractive flow at inference is what the evaluation
measures (held-out init states 20–39; results below).

### Results (phase 3b)

10 tasks × 20 held-out init states (20–39; the counterfactual rollouts used
0–19) per cell, neutral prompt unless noted, σ-scale 1, 10 Euler steps as
before; x0 from `prior-cf-concat` for the instructed cells and from
`prior-cf-none` for "no instruction". Success = `On(bowl, plate)` for the
instructed bowl; bowl choice = the bowl lifted first (instructed / other /
none). `scripts/vla/collect_vla_results.py --out-dir outputs/vla_cf`.

Instructing bowl 2 (against the layout preference):

| condition | bowl-2 success | bowl-2 choice (bowl 2 / bowl 1 / none) |
|---|---|---|
| full-prompt: bowl-2 sentence in the VLA's prompt, Gaussian x0 (ceiling) | 56.5 % | 57.0 % (114/80/6) |
| neutral-gauss: no instruction anywhere (floor) | 11.0 % | 11.0 % (22/170/8) |
| neutral-prior, no instruction | 14.0 % | 14.0 % (28/166/6) |
| neutral-prior + text "bowl 2" | 21.5 % | 22.0 % (44/154/2) |
| neutral-prior + sketch "bowl 2" | 19.5 % | 20.5 % (41/156/3) |
| neutral-prior + audio "bowl 2" | 21.5 % | 21.0 % (42/154/4) |

Instructing bowl 1 (with the layout preference; the two uninstructed rows
are the same 200 rollouts as above, relabelled):

| condition | bowl-1 success | bowl-1 choice (bowl 1 / bowl 2 / none) |
|---|---|---|
| neutral-gauss | 80.0 % | 85.0 % (170/22/8) |
| neutral-prior, no instruction | 82.0 % | 83.0 % (166/28/6) |
| neutral-prior + text "bowl 1" | 87.5 % | 89.5 % (179/17/4) |
| neutral-prior + sketch "bowl 1" | 86.0 % | 90.0 % (180/19/1) |
| neutral-prior + audio "bowl 1" | 88.5 % | 89.5 % (179/17/4) |

Bowl-2 success per task under the bowl-2 instruction (20 episodes each; the
language-sensitive layouts are 1, 3, 5, 6, 7, 9):

| condition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| full-prompt | 0 | 100 | 5 | 65 | 0 | 100 | 95 | 100 | 10 | 90 |
| neutral-gauss | 0 | 25 | 0 | 0 | 0 | 10 | 0 | 70 | 0 | 5 |
| neutral-prior, no instruction | 0 | 30 | 0 | 15 | 0 | 25 | 0 | 55 | 0 | 15 |
| neutral-prior + text | 0 | 40 | 0 | 15 | 0 | 45 | 0 | 90 | 0 | 25 |
| neutral-prior + sketch | 0 | 40 | 0 | 15 | 0 | 45 | 0 | 80 | 0 | 15 |
| neutral-prior + audio | 0 | 35 | 0 | 15 | 0 | 55 | 0 | 90 | 0 | 20 |

Reading:

- **The prior steers the frozen VLA, in both directions.** The same prior
  weights, the same neutral prompt and the same 200 init states give bowl 2
  on 8.5 / 9.5 / 8.5 % of the episodes with the bowl-1 instruction (text /
  sketch / audio) and on 22.0 / 20.5 / 21.0 % with the bowl-2 instruction
  (two-proportion z = 3.8 / 3.1 / 3.5, p ≤ 0.002 each). Against the
  uninstructed prior (14.0 %) the bowl-2 instruction adds 6.5–8 points
  (pooled over the modalities 21.2 vs 14.0 %, z = 2.2, p = 0.03) and the
  bowl-1 instruction adds 6.5–7 points of bowl-1 choice (83.0 → 89.5–90 %,
  pooled z = 2.5, p = 0.01). The three modalities are indistinguishable, as
  in the offline probe.
- **It is a fraction of the VLA's own language channel.** The bowl-2
  sentence in the prompt reaches 57 % bowl 2 (+46 points over the floor); the
  source prior reaches 21–22 % (+10–11). The μ shift of 0.3–0.5 σ measured
  offline survives the contractive x0 → action map of π0.5 only in part.
- **It works where the VLA can already be talked into bowl 2.** The gain
  concentrates in tasks 1, 5, 7 and 9 (floor 25 / 10 / 70 / 5 % → 35–40 /
  45–55 / 80–90 / 15–25 %); task 3 sits at 15 % for every prior cell,
  instructed or not (full prompt 65 %), task 6 at 0 % (full prompt 95 %),
  and the four layout-locked tasks 0, 2, 4, 8 stay at 0 % for every method
  including the full prompt. Task 7 is the clearest two-way case: under the
  neutral prompt the VLA prefers bowl 2 in that layout (70 % bowl 2, 15 %
  bowl 1), and the prior moves it to 80–90 % bowl 2 or 35–40 % bowl 1 as
  instructed.
- **No regression on the preferred bowl.** With the bowl-1 instruction the
  counterfactual prior beats Gaussian x0 on bowl-1 success (80 → 86–88.5 %),
  through tasks 5, 7 and 9; the uninstructed counterfactual prior gives 82 %.
  (Other init states than the phase-3 table, so not comparable to it episode
  by episode.)

Conclusion of phase 3: a forward model of the source noise can steer a frozen
VLA's discrete choice through x0 alone, but only (i) once its labels carry
the counterfactual, which the demonstrations do not, (ii) among behaviours
the VLA already produces under the neutral prompt, and (iii) with roughly a
quarter of the effect of the VLA's own language input. Follow-ups, in order
of expected information: amplify the instruction's shift of μ at inference
(x0 = μ_none + λ(μ_instr − μ_none), λ > 1, the prior's analogue of guidance);
more init states per task (20 per task gives ±10 points per cell); a prior
over several chunks or over the VLA's prefix KV instead of x0 for the
layouts where x0 does not reach (3, 6).

## Notes

- LeRobot's LIBERO env default render is 360²; we pass 256² explicitly (the
  policy resizes with padding to 224 either way).
- hf-libero asks for input on first import without a config; the drivers set
  `LIBERO_CONFIG_PATH=data/vla/libero_config`.
- Steering here selects among discrete behaviours (which bowl). Continuous
  steering is out of scope for this phase.
- Timing (A6000): a 200-episode evaluation cell takes 47–56 min; the two
  counterfactual priors train in a few minutes; the phase-3b campaign ran
  11:49–15:33 on three GPUs (`scripts/vla/run_vla_counterfactual.sh`, one
  instance per GPU, cells filtered with `CELLS`).

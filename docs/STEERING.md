# Multimodal instruction steering on LeaP (phase 2)

Status: evaluated (2026-10-18). Design: `docs/superpowers/specs/2026-10-17-leap-multimodal-steering-design.md`.
Previous phase: `docs/REPRODUCTION.md` (LeaP reproduction and the finding that
the prior of a trained LeaP is not an inference-time steering channel).

## Question

Can an instruction given as a sketch, as text or as speech steer a LeaP policy,
and is the learnable source prior a useful entry point for that signal compared
with conditioning the velocity network?

## Benchmark: `place_a2b_side`

A RoboTwin 2.0 task derived from `place_a2b_left` / `place_a2b_right`: pick
object A and place it 13 cm to the **left or right** of object B. The side is
drawn per episode from the seed, the layout is symmetric (A: x ∈ [−0.22, 0.22];
B: x ∈ [−0.10, 0.10]; both y ∈ [−0.2, 0]), and success requires the instructed
side. Task file `leap/robotwin/tasks/place_a2b_side.py`, installed into the
pinned RoboTwin checkout by `scripts/install_task.py`; 400-step limit; the
official evaluator (`script/eval_policy.py`, unmodified) is used.

Two things had to be fixed before the benchmark was clean:

1. **Expert-survival bias.** With a 50/50 side draw the expert planner failed
   cross-midline placements much more often, so among the 100 surviving
   episodes the (visible) arm predicted the side: left arm 49 same-side / 16
   cross, right arm 30 / 5 (~80 % predictable). The side is now drawn
   cross-midline with P = 0.74 (left arm) / 0.86 (right arm); the surviving
   episodes are left arm 34 / 28 and right arm 22 / 16 (side balance per arm
   55 % / 58 %). The same rule applies in the evaluator's expert-filtered
   seeds. Episodes attempted: 184 for the first collection, 100 kept each time.
2. **Sketch background.** Drawn on the head-camera photo, the sketch embedding
   did not separate the side (linear probe 49–55 %); on a white canvas with
   the same pixel geometry it does (86–89 %). See "Instruction embeddings".

## Instructions

| modality | content | at evaluation |
|---|---|---|
| text | RoboTwin templates with `{side}` (89 seen / 13 unseen) and object descriptions | the evaluator's own unseen instruction (`TASK_ENV.get_instruction()`) |
| sketch | red circle around A, blue arrow from A to the target spot, at the head-camera pixel positions of frame 0, on a white 320×240 canvas | built from the live object poses and camera matrices at the first policy step |
| audio | Piper TTS (`en_US-lessac-medium`) of a text instruction, 16 kHz mono | TTS of the evaluator's instruction |

All embedded by `google/embeddinggemma-2` (768-d, L2-normalised; image via
`<|image|>`, audio via `<|audio|>`, text with the `SearchQuery` prompt).
Bundle: `data/place_a2b_side-leap_steer-100.instr/` (10 text variants, 1
sketch, 3 audio renderings per episode; `embeddings.npz`).

### Instruction embeddings: separability of the side

5-fold linear probe (logistic regression on standardised embeddings, folds
split by episode; `scripts/instr_probe.py`), 100 episodes:

| train \ test | text | sketch | audio |
|---|---|---|---|
| text | 74.8 | 44.0 | 71.0 |
| sketch | 50.8 | **86.0** | 51.0 |
| audio | 71.5 | 54.0 | 63.0 |

Within a modality the side is linearly recoverable (text 75 %, sketch 86 %,
audio 63 %); across modalities it is at chance, i.e. the shared embedding
space does not carry "left vs right" in a modality-independent direction.
The raw embeddings of opposite sides are nearly identical (cos 0.989–0.993
for text), which is why the policy standardises them before its adapter.
Sketch variants measured on the same 100 episodes (probe accuracy, 768-d):
photo background thin strokes 49 %, photo thick 55 %, dimmed photo 46 %,
white thin 89 %, white thick 83 %, white arrow only 93 %, white black strokes
86 %; a textual question prefix on the photo sketch 64 %.

## Models

LeaP (`configs/leap.yaml`) plus `instr_mode`:

| variant | instruction enters | config |
|---|---|---|
| noinstr | nowhere (control; `instr_mode: none` is byte-identical to LeaP) | `configs/steer_noinstr.yaml` |
| prior | the source prior head only: (μ, σ) = Prior(state, z_e) | `configs/steer_prior.yaml` |
| cond | the velocity-net condition only (standard instruction conditioning) | `configs/steer_cond.yaml` |
| both | both | `configs/steer_both.yaml` |

z_e = MLP(768 → 256 → 128) of the standardised embedding (mean/std over all
modalities of the training episodes, stored in the checkpoint). Training
samples one modality (text/sketch/audio) and one variant per window; 3001
epochs, seed 0, 100 episodes (~11 k windows), losses unchanged.

## Evaluation

Official 100-episode protocol per cell: noinstr × none; prior/cond/both ×
{text, sketch, audio}. The task logs one JSON line per policy episode
(`LEAP_STEER_LOG`) with the final object/target positions, so **side
following** (object ends on the instructed side) is reported next to success.

## Results

100 episodes per cell, official evaluator, seed 0 (the evaluator's expert
filter is not bit-reproducible across runs, so the 100 surviving seeds differ
slightly between cells; the left/right split is 49–51 everywhere). Success is
the official `_result.txt` (identical to the side log's); side following is
from the side log. `outputs/analysis/steer_results.json`.

| model | instruction | success % | side following % | following left / right |
|---|---|---|---|---|
| noinstr | none (control) | 27 | 59 | 30/50 · 29/50 |
| prior | text | 26 | 63 | 32/50 · 31/50 |
| prior | sketch | 30 | 68 | 37/50 · 31/50 |
| prior | audio | 25 | 63 | 33/49 · 30/51 |
| cond | text | 32 | 73 | 38/49 · 35/51 |
| cond | sketch | **41** | **75** | 41/51 · 34/49 |
| cond | audio | 29 | 74 | 43/50 · 31/50 |
| both | text | 25 | 64 | 32/50 · 32/50 |
| both | sketch | 33 | 67 | 36/50 · 31/50 |
| both | audio | 24 | 62 | 33/49 · 29/51 |

Reading:

- **Control.** 59 % side following is the residual layout leak after the
  skewed side draw (chance would be 50 %); 27 % success shows the task is
  hard for a 100-episode LeaP (the median episode runs into the 400-step
  limit).
- **Conditioning the velocity network works for all three modalities.**
  cond follows the side on 73–75 % of the episodes, +14–16 points over the
  control (two-proportion z = 2.1–2.4, p = 0.02–0.04 per cell; pooled 222/300
  vs 59/100, z = 2.8, p = 0.005), on both sides (left 30 → 38–43, right 29 →
  31–35 of 50), and lifts success to 29–41 %. Sketch is the best modality
  (41 % success, z = 2.1 vs control), consistent with its embedding being the
  most separable (86 % probe); text and audio, whose embeddings separate the
  side less (75 / 63 %), follow as well but succeed less often.
- **The source prior alone is a weak entry point.** prior moves following by
  +4 to +9 points (63 / 68 / 63 %; pooled 194/300 vs 59/100, z = 1.0, n.s.)
  and success by −2 to +3. The velocity field, trained on the same windows
  with the instruction only in the source, overrides most of what the
  source encodes — the phase-1 finding (`docs/REPRODUCTION.md`) again, now
  with an instruction the source could in principle carry.
- **both is not additive.** With the instruction in both the source and the
  condition the model lands at the prior's level (62–67 % following, 24–33 %
  success), below cond. The likely cause is that the velocity network's
  conditioning trains more weakly when the source already explains the
  instruction-dependent part of the windows; not verified (a probe of the
  trained both model's sensitivity to its condition would settle it).

Conclusion: a sketch, a sentence or a TTS rendering embedded by
EmbeddingGemma-2 can steer a LeaP policy to the instructed side, and the
useful entry point is the velocity-network condition, not the learnable
source prior. The prior-only steering is small here (+4–9 points) for the
same reason the VLA prior needed counterfactual labels in phase 3b
(`docs/VLA_STEERING.md`): the flow contracts what the source carries.

## Environment and timing

robot_dev6 (3× RTX A6000), LeaP venv as in `docs/REPRODUCTION.md` plus
transformers 5.20.1, sentence-transformers 6.1.0, piper-tts 1.8.0, soundfile,
librosa (only huggingface-hub changed, 0.25 → 1.33). Rasterizer shims as
before. Collection of 100 episodes: ~40 min; bundle + embeddings: ~3 min;
training: 4 runs in parallel on 3 GPUs (shared with the phase-3 jobs),
7.4 h for 3001 epochs; evaluation: 2.2–2.6 h per 100-episode cell, 3–4
cells in parallel; the campaign ran 00:21–16:58 on 2026-10-18.
Driver: `scripts/run_steer_campaign.sh` (idempotent stages).

## Deviations and notes

- RoboTwin's end-of-collection instruction generator never writes
  `instructions/` in this setup (absolute `save_path` is joined wrongly and
  the remote has no `yaml` in the system python); `scripts/make_instructions.py`
  renders the texts with the official module's functions instead.
- The side draw is skewed per arm (see Benchmark); the residual leak is
  measured by the noinstr control.
- The first training round used photo-background sketches and was
  restarted after the separability measurement; the noinstr run was kept.

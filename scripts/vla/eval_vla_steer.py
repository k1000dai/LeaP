"""Evaluate a frozen pi0.5 on LIBERO-Spatial with Gaussian or learned-prior source noise.

Conditions (docs/superpowers/specs/2026-10-17-vla-prior-steering-design.md):

- full-prompt    task language as the prompt, Gaussian noise (upper reference)
- neutral-gauss  neutral prompt ("pick up the black bowl ..."), Gaussian noise (unsteered control)
- neutral-prior  neutral prompt, source noise sampled from a VLAPrior (--prior) fed with the VLA's
                 prefix features and, for --instr-modality text/sketch/audio, the steering embedding
                 of the task built once per episode from the live scene (text = task language,
                 sketch = first agentview frame + bowl/plate layout, audio = Piper TTS of the text).

The rollout loop is this script's own (not lerobot-eval): per task and episode index e the env is
reset on init state e, the policy is queried every --n-action-steps env steps with the current
agentview/wrist images and 8-d state, and the first --n-action-steps actions of the 50-step chunk
are executed.  Each episode appends one JSON line to <out>/episodes.jsonl (task_id, task_language,
episode, success, bowl_choice, steps, condition, modality, sigma_scale, seed, ...); finished
(task, episode) pairs are skipped on restart, and <out>/summary.json is rebuilt from the log at the
end.  MUJOCO_GL defaults to egl.

Example:
    python scripts/vla/eval_vla_steer.py --suite libero_spatial --tasks 0-9 --episodes 20 \
        --condition neutral-prior --instr-modality sketch --prior outputs/vla/prior-concat/best.pt \
        --sigma-scale 1.0 --out outputs/vla/neutral-prior-sketch-s1 --device cuda
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONDITIONS = ("full-prompt", "neutral-gauss", "neutral-prior")
MODALITIES = ("none", "text", "sketch", "audio")
BOWL_CHOICES = ("target", "distractor", "none")
LOG_NAME = "episodes.jsonl"
SUMMARY_NAME = "summary.json"
DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_EMBEDDER = "google/embeddinggemma-2"
DEFAULT_VOICE = "data/voices/en_US-lessac-medium.onnx"


# ---- lazy imports of the heavy modules (tests replace these with fakes) ----------------------------------
def import_pi05_flow():
    from leap.vla import pi05_flow
    return pi05_flow


def import_libero_scene():
    from leap.vla import libero_scene
    return libero_scene


def import_instructions():
    from leap.vla import instructions
    return instructions


def make_tts(voice):
    from leap.steer.tts import PiperTTS
    return PiperTTS(voice)


def make_embedder(model, device):
    from leap.steer.embedder import InstructionEmbedder
    return InstructionEmbedder(model, device=device)


def load_prior(path, device):
    from leap.vla.prior import VLAPrior
    prior, extra = VLAPrior.load(path, map_location=device)
    return prior.to(device), extra


# ---- arguments ---------------------------------------------------------------------------------------------
def parse_tasks(spec):
    """"0-9", "0,3,5" or "2" -> sorted list of task ids."""
    tasks = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            tasks.update(range(low, high + 1))
        else:
            tasks.add(int(part))
    if not tasks:
        raise ValueError(f"no task ids in {spec!r}")
    return sorted(tasks)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9", help='Task ids: "0-9", "0,3,5" or "2"')
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task (episode e uses init state e)")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--instr-modality", choices=MODALITIES, default="none")
    parser.add_argument("--prior", type=Path, help="VLAPrior checkpoint (neutral-prior only)")
    parser.add_argument("--sigma-scale", type=float, default=1.0, help="x0 = mu + sigma_scale * sigma * eps")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow integration steps")
    parser.add_argument("--n-action-steps", type=int, default=10, help="Actions executed per policy call")
    parser.add_argument("--max-steps", type=int, help="Episode length override (default: the suite's limit)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Piper voice for audio instructions")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-instructions", action="store_true",
                        help="Save the text/sketch/wav instruction of every episode under <out>/instructions/")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.n_action_steps < 1 or args.num_steps < 1:
        parser.error("episodes, n-action-steps and num-steps must be positive")
    if args.condition == "neutral-prior":
        if args.prior is None:
            parser.error("--condition neutral-prior needs --prior")
    elif args.instr_modality != "none":
        parser.error(f"--condition {args.condition} takes no instruction (--instr-modality none)")
    if args.sigma_scale < 0:
        parser.error("sigma-scale must be >= 0")
    try:
        args.task_ids = parse_tasks(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    return args


def episode_seed(seed, task_id, episode):
    return seed * 1_000_000 + task_id * 1_000 + episode


# ---- log helpers -------------------------------------------------------------------------------------------------
def read_log(path):
    """{(task_id, episode): record} from an episodes.jsonl (the last line wins for a repeated pair)."""
    records = {}
    if not Path(path).exists():
        return records
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[(int(record["task_id"]), int(record["episode"]))] = record
    return records


def summarize(records):
    """Per-task and overall success rate, bowl-choice accuracy (choice == target) and counts."""
    def stats(rows):
        counts = {choice: sum(r.get("bowl_choice") == choice for r in rows) for choice in BOWL_CHOICES}
        n = len(rows)
        return {"episodes": n, "successes": sum(bool(r["success"]) for r in rows),
                "success_rate": sum(bool(r["success"]) for r in rows) / n if n else float("nan"),
                "bowl_choice_accuracy": counts["target"] / n if n else float("nan"),
                "bowl_choice_counts": counts,
                "mean_steps": float(np.mean([r["steps"] for r in rows])) if n else float("nan")}

    rows = sorted(records.values(), key=lambda r: (r["task_id"], r["episode"]))
    tasks = {}
    for row in rows:
        tasks.setdefault(int(row["task_id"]), []).append(row)
    return {"overall": stats(rows),
            "tasks": {str(task_id): dict(stats(task_rows), task_language=task_rows[0].get("task_language"))
                      for task_id, task_rows in sorted(tasks.items())}}


def write_summary(out, args, records):
    summary = {"condition": args.condition, "modality": args.instr_modality, "sigma_scale": args.sigma_scale,
               "seed": args.seed, "suite": args.suite, "checkpoint": args.checkpoint,
               "prior": None if args.prior is None else str(args.prior), "num_steps": args.num_steps,
               "n_action_steps": args.n_action_steps, "tasks_requested": args.task_ids,
               "episodes_per_task": args.episodes}
    summary.update(summarize(records))
    (Path(out) / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    return summary


def dump_instruction(out, task_id, episode, artefacts):
    """Write text/sketch/wav artefacts of one episode under <out>/instructions/."""
    folder = Path(out) / "instructions"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"task{task_id:02d}_ep{episode:03d}"
    if artefacts.get("text") is not None:
        stem.with_suffix(".txt").write_text(str(artefacts["text"]) + "\n")
    if artefacts.get("sketch") is not None:
        artefacts["sketch"].save(stem.with_suffix(".png"))
    if artefacts.get("audio") is not None:
        from leap.steer.tts import save_wav
        save_wav(stem.with_suffix(".wav"), artefacts["audio"])


# ---- rollout ---------------------------------------------------------------------------------------------------------
def observation_state(obs, scene):
    """8-d LeRobot state from a LiberoEnv observation (agent_pos, else libero_scene.libero_state_vector)."""
    if "agent_pos" in obs:
        return np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    if hasattr(scene, "libero_state_vector"):
        return np.asarray(scene.libero_state_vector(obs), dtype=np.float32).reshape(-1)
    raise KeyError("observation has no agent_pos and libero_scene has no libero_state_vector")


def reset_to_episode(env, episode, seed):
    """Reset on the suite's init state ``episode`` (LiberoEnv reads init_state_id at reset)."""
    if hasattr(env, "init_state_id"):
        env.init_state_id = episode
    return env.reset(seed=seed)


class Evaluator:
    """Loaded policy bundle, prior and instruction tools of one evaluation run."""

    def __init__(self, args):
        self.args = args
        self.flow = import_pi05_flow()
        self.scene = import_libero_scene()
        self.instructions = import_instructions() if args.instr_modality != "none" or args.dump_instructions else None
        self.bundle = self.flow.load_pi05(args.checkpoint, args.device, n_action_steps=args.n_action_steps)
        self.prior = self.prior_extra = None
        if args.condition == "neutral-prior":
            self.prior, self.prior_extra = load_prior(args.prior, args.device)
            needs_instr = self.prior.instr_mode == "concat"
            if needs_instr != (args.instr_modality != "none"):
                raise SystemExit(f"prior {args.prior} has instr_mode {self.prior.instr_mode!r}; "
                                 f"--instr-modality {args.instr_modality} does not match")
        self.tts = make_tts(args.voice) if args.instr_modality == "audio" else None
        self.embedder = make_embedder(args.embedder, args.device) if args.instr_modality != "none" else None
        self.suite = self.scene.load_suite(args.suite)
        self.neutral_prompt = self.scene.NEUTRAL_PROMPT
        self.max_steps = args.max_steps or self.scene.max_steps(args.suite)

    def prompt(self, task_language):
        return task_language if self.args.condition == "full-prompt" else self.neutral_prompt

    def build_instruction(self, env, obs, task_language, layout):
        """(embedding [768] or None, artefacts {text, sketch, audio}) for one episode."""
        modality = self.args.instr_modality
        if self.instructions is None:
            return None, {}
        text = self.instructions.task_text(task_language)
        artefacts, sketch, audio = {"text": text}, None, None
        if modality == "sketch" or (modality == "none" and self.args.dump_instructions):
            camera = self.scene.agentview_camera(env)
            sketch = self.instructions.build_sketch(obs["pixels"]["image"], camera, layout)
            artefacts["sketch"] = sketch
        if modality == "audio":
            audio = self.instructions.build_audio(text, self.tts)
            artefacts["audio"] = audio
        if modality == "none":
            return None, artefacts
        embedding = self.instructions.embed_instruction(self.embedder, modality, text=text, sketch=sketch, audio=audio)
        return np.asarray(embedding, dtype=np.float32).reshape(-1), artefacts

    def policy_actions(self, obs, prompt, instr):
        """Actions [n_action_steps, 7] for the current observation; also returns prior diagnostics."""
        images = {"image": obs["pixels"]["image"], "image2": obs["pixels"]["image2"]}
        state = observation_state(obs, self.scene)
        batch = self.flow.prepare_batch(self.bundle, images, state, prompt)
        ctx = self.flow.prefix_context(self.bundle, batch)
        noise, diagnostics = None, None
        if self.prior is not None:
            feat = self.flow.prefix_features(self.bundle, batch).float()
            mu, log_var = self.prior(feat, instr)
            noise = self.flow.pad_noise(self.prior.sample(mu, log_var, self.args.sigma_scale))
            diagnostics = (float(mu.abs().mean()), float(torch.exp(0.5 * log_var).mean()))
        actions = self.flow.sample_actions_with_noise(self.bundle, ctx, noise, num_steps=self.args.num_steps)
        return np.asarray(actions)[0, :self.args.n_action_steps], diagnostics

    def run_episode(self, env, task_id, task_language, episode):
        """One rollout: returns (record, artefacts)."""
        started = time.time()
        seed = episode_seed(self.args.seed, task_id, episode)
        torch.manual_seed(seed)
        obs, info = reset_to_episode(env, episode, seed)
        layout = self.scene.scene_layout(env)
        tracker = self.scene.BowlChoiceTracker(layout)
        embedding, artefacts = self.build_instruction(env, obs, task_language, layout)
        instr = None
        if embedding is not None:
            instr = torch.as_tensor(embedding, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        prompt = self.prompt(task_language)
        steps, success, choice, done, diagnostics = 0, False, "none", False, []
        with torch.inference_mode():
            while not done:
                actions, diag = self.policy_actions(obs, prompt, instr)
                if diag is not None:
                    diagnostics.append(diag)
                for action in actions:
                    obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                    steps += 1
                    choice = tracker.update(env)
                    success = success or bool(info.get("is_success", False))
                    if terminated or truncated or success or steps >= self.max_steps:
                        done = True
                        break
        record = {"task_id": task_id, "task_language": task_language, "episode": episode, "success": success,
                  "bowl_choice": choice, "steps": steps, "condition": self.args.condition,
                  "modality": self.args.instr_modality, "sigma_scale": self.args.sigma_scale, "seed": seed,
                  "prompt": prompt, "elapsed_s": round(time.time() - started, 2)}
        if diagnostics:
            record["prior_mu_abs"] = float(np.mean([d[0] for d in diagnostics]))
            record["prior_sigma"] = float(np.mean([d[1] for d in diagnostics]))
        return record, artefacts


def main(argv=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records = read_log(out / LOG_NAME)
    pending = {task_id: [e for e in range(args.episodes) if (task_id, e) not in records] for task_id in args.task_ids}
    pending = {task_id: episodes for task_id, episodes in pending.items() if episodes}
    print(f"{args.condition}/{args.instr_modality} sigma {args.sigma_scale}: {len(records)} episodes logged, "
          f"{sum(len(e) for e in pending.values())} to run -> {out}", flush=True)
    if pending:
        evaluator = Evaluator(args)
        with (out / LOG_NAME).open("a") as log:
            for task_id, episodes in pending.items():
                task_language = evaluator.suite.get_task(task_id).language
                env = evaluator.scene.make_env(evaluator.suite, task_id,
                                               prompt_override=None if args.condition == "full-prompt"
                                               else evaluator.neutral_prompt)
                try:
                    for episode in episodes:
                        record, artefacts = evaluator.run_episode(env, task_id, task_language, episode)
                        if args.dump_instructions:
                            dump_instruction(out, task_id, episode, artefacts)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        records[(task_id, episode)] = record
                        print(json.dumps(record), flush=True)
                finally:
                    if hasattr(env, "close"):
                        env.close()
    summary = write_summary(out, args, {key: records[key] for key in records if key[0] in args.task_ids
                                        and key[1] < args.episodes})
    overall = summary["overall"]
    print(f"summary: {overall['episodes']} episodes, success {overall['success_rate']:.3f}, "
          f"bowl-choice accuracy {overall['bowl_choice_accuracy']:.3f} -> {out / SUMMARY_NAME}", flush=True)


if __name__ == "__main__":
    main()

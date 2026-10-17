#!/usr/bin/env python
"""Roll out the frozen pi0.5 on LIBERO-Spatial with the FULL prompt of either bowl and record every step (VLA venv).

For each task, bowl and episode e the env is reset on the suite's init state e (the same state for both
bowls), the policy is prompted with ``counterfactual.target_prompt`` (bowl 1: the task language; bowl 2: the
language with the bowl-2 phrase read from the BDDL ``:init`` block), queried every --n-action-steps env steps
with Gaussian source noise and --num-steps Euler steps, and the first --n-action-steps actions of each 50-step
chunk are executed.  With --predict-every k an extra, non-executed prediction is recorded at every k-th env step
in between (more reversal labels per episode at the cost of one forward pass each).  Success is LIBERO's
``On(bowl, plate)`` predicate for the INTENDED bowl (``counterfactual.bowl_on_plate``); LIBERO's own
``check_success`` (bowl 1) and the contact-free geometric test are logged next to it.  The episode stops at the
first success of either bowl or at --max-steps.

Outputs under --out: one ``task_{i}_bowl{b}_episode{e}.npz`` per episode (see ``leap.vla.counterfactual`` for
the arrays), ``rollouts.jsonl`` (one record per episode) and ``manifest.json`` with the per-(task, bowl) table
that is also printed at the end.  Idempotent per episode: an existing npz is read, not re-run.

Usage (remote GPU, VLA venv):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=data/vla/libero_config python scripts/vla/rollout_counterfactual.py \\
        --suite libero_spatial --tasks 0-9 --bowls 1,2 --episodes 20 --out data/vla/counterfactual \\
        --checkpoint lerobot/pi05_libero_finetuned_v044 --device cuda --seed 0 --predict-every 2
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leap.vla import counterfactual as C  # noqa: E402

LOG_NAME = "rollouts.jsonl"
MANIFEST_NAME = "manifest.json"
DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"


# ---- lazy imports of the heavy modules (tests replace these with fakes) ------------------------------------
def import_torch():
    import torch
    return torch


def import_pi05_flow():
    from leap.vla import pi05_flow
    return pi05_flow


def import_libero_scene():
    from leap.vla import libero_scene
    return libero_scene


# ---- arguments ----------------------------------------------------------------------------------------------
def parse_tasks(spec):
    """``"0-9"``, ``"0,3,5"`` or ``"2"`` -> sorted list of ids."""
    ids = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            ids.update(range(low, high + 1))
        else:
            ids.add(int(part))
    if not ids:
        raise ValueError(f"no ids in {spec!r}")
    return sorted(ids)


def parse_phrase_overrides(items):
    """``["2=next to the plate", ...]`` -> ``{2: "next to the plate"}`` (applied to every task)."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--phrase-override expects BOWL=PHRASE, got {item!r}")
        bowl, phrase = item.split("=", 1)
        overrides[int(bowl)] = " ".join(phrase.split())
    return overrides


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9", help='Task ids: "0-9", "0,3,5" or "2"')
    parser.add_argument("--bowls", default="1,2", help='Bowls to prompt for: "1,2", "2"')
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per (task, bowl); episode e uses init state e")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow integration steps")
    parser.add_argument("--n-action-steps", type=int, default=10, help="Actions executed per policy call")
    parser.add_argument("--predict-every", type=int, default=0,
                        help="Also record a (non-executed) prediction every k env steps; 0 = executed calls only")
    parser.add_argument("--max-steps", type=int, help="Episode length override (default: the suite's limit)")
    parser.add_argument("--phrase-override", action="append", default=[], metavar="BOWL=PHRASE",
                        help="Use PHRASE for BOWL in every task instead of the BDDL-derived one (pilots)")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.n_action_steps < 1 or args.num_steps < 1 or args.predict_every < 0:
        parser.error("episodes, n-action-steps and num-steps must be positive; predict-every >= 0")
    try:
        args.task_ids = parse_tasks(args.tasks)
        args.bowl_ids = parse_tasks(args.bowls)
        args.phrase_overrides = parse_phrase_overrides(args.phrase_override)
    except ValueError as error:
        parser.error(str(error))
    if any(b not in C.BOWLS for b in args.bowl_ids):
        parser.error(f"--bowls must be a subset of {C.BOWLS}")
    return args


def episode_seed(seed, task_id, bowl, episode):
    return seed * 1_000_000 + task_id * 10_000 + bowl * 1_000 + episode


# ---- rollout ---------------------------------------------------------------------------------------------------
def observation_state(obs, scene):
    """8-d LeRobot state from a ``pixels_agent_pos`` observation (``robot_state``) or a raw robosuite dict."""
    if "agent_pos" in obs:
        return np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    if "robot_state" in obs:
        robot = obs["robot_state"]
        return np.concatenate([np.asarray(robot["eef"]["pos"], dtype=np.float32).reshape(3),
                               scene.quat_to_axis_angle(robot["eef"]["quat"]),
                               np.asarray(robot["gripper"]["qpos"], dtype=np.float32).reshape(2)]).astype(np.float32)
    return np.asarray(scene.state_vector(obs), dtype=np.float32).reshape(-1)


def reset_to_episode(env, episode, seed):
    """Reset on the suite's init state ``episode`` (LiberoEnv reads ``init_state_id`` at reset)."""
    if hasattr(env, "init_state_id"):
        env.init_state_id = episode
    return env.reset(seed=seed)


class Roller:
    """Loaded policy bundle plus the scene tools of one rollout run."""

    def __init__(self, args):
        self.args = args
        self.torch = import_torch()
        self.flow = import_pi05_flow()
        self.scene = import_libero_scene()
        self.bundle = self.flow.load_pi05(args.checkpoint, args.device, n_action_steps=args.n_action_steps)
        self.suite = self.scene.load_suite(args.suite)
        self.max_steps = args.max_steps or self.scene.max_steps(args.suite)
        self.chunk_size = int(getattr(self.flow, "CHUNK_SIZE", 50))
        self.max_action_dim = int(getattr(self.flow, "MAX_ACTION_DIM", 32))

    def prompts(self, task_id):
        """``(task_language, {bowl: prompt})`` for a task, honouring ``--phrase-override``."""
        language = self.suite.get_task(task_id).language
        phrases = C.bowl_phrases(self.suite, task_id)
        phrases.update(self.args.phrase_overrides)
        return language, {bowl: C.target_prompt(language, phrases, bowl) for bowl in self.args.bowl_ids}

    def predict(self, images, state, prompt):
        """One VLA query: ``(chunk [50, 7] un-normalised, noise [50, 32])`` from fresh Gaussian noise."""
        batch = self.flow.prepare_batch(self.bundle, images, state, prompt)
        ctx = self.flow.prefix_context(self.bundle, batch)
        noise = self.flow.sample_noise((1, self.chunk_size, self.max_action_dim), self.bundle.device)
        chunk = self.flow.sample_actions_with_noise(self.bundle, ctx, noise, num_steps=self.args.num_steps)
        return np.asarray(chunk, dtype=np.float32)[0], np.asarray(noise.detach().cpu().numpy(), dtype=np.float32)[0]

    def run_episode(self, env, task_id, bowl, episode, prompt, task_language):
        """One rollout; returns ``(arrays, scalars)`` for ``counterfactual.save_episode``."""
        args = self.args
        started = time.time()
        seed = episode_seed(args.seed, task_id, bowl, episode)
        self.torch.manual_seed(seed)
        obs, _ = reset_to_episode(env, episode, seed)
        layout = self.scene.scene_layout(env)
        tracker = self.scene.BowlChoiceTracker(layout)
        cam_intrinsic, cam_extrinsic = self.scene.agentview_camera(env)
        recorder = C.EpisodeRecorder()
        steps, success_step, geometric_step, libero_success, pending, choice = 0, -1, -1, False, None, "none"
        with self.torch.inference_mode():
            while True:
                images = {"image": obs["pixels"]["image"], "image2": obs["pixels"]["image2"]}
                state = observation_state(obs, self.scene)
                recorder.add_step(self.scene.policy_view(images["image"]), self.scene.policy_view(images["image2"]),
                                  state, C.object_positions(env))
                executed = steps % args.n_action_steps == 0
                if executed or (args.predict_every and steps % args.predict_every == 0):
                    chunk, noise = self.predict(images, state, prompt)
                    recorder.add_prediction(steps, chunk, noise, executed)
                    if executed:
                        pending = chunk[:args.n_action_steps]
                action = np.asarray(pending[steps % args.n_action_steps], dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                choice = tracker.update(env)
                libero_success = libero_success or bool(info.get("is_success", False))
                on_plate = C.bowl_on_plate(env, bowl)
                positions = C.object_positions(env)
                geometric = C.on_plate_geometric(positions[bowl - 1], positions[2])
                recorder.add_result(action, on_plate)
                if on_plate and success_step < 0:
                    success_step = steps - 1
                if geometric and geometric_step < 0:
                    geometric_step = steps - 1
                if on_plate or terminated or truncated or steps >= self.max_steps:
                    break
        scalars = {"task_id": task_id, "bowl": bowl, "episode": episode, "prompt": prompt, "task_language": task_language,
                   "success": success_step >= 0, "success_geometric": geometric_step >= 0,
                   "libero_success": libero_success, "success_step": success_step, "geometric_step": geometric_step,
                   "lifted_bowl": C.lifted_bowl(choice), "bowl_choice": C.intended_choice(choice, bowl),
                   "steps": steps, "seed": seed, "checkpoint": args.checkpoint, "num_steps": args.num_steps,
                   "n_action_steps": args.n_action_steps, "predict_every": args.predict_every, "suite": args.suite,
                   "image_view": C.IMAGE_VIEW, "elapsed_s": round(time.time() - started, 2)}
        arrays = recorder.arrays()
        arrays["cam_intrinsic"] = np.asarray(cam_intrinsic, dtype=np.float64)
        arrays["cam_extrinsic"] = np.asarray(cam_extrinsic, dtype=np.float64)
        return arrays, scalars


def log_record(log, scalars):
    log.write(json.dumps(scalars) + "\n")
    log.flush()
    print(json.dumps(scalars), flush=True)


def pending_episodes(args):
    """``{(task_id, bowl): [episodes without an npz]}`` and the records of the episodes that already exist."""
    pending, existing = {}, []
    for task_id in args.task_ids:
        for bowl in args.bowl_ids:
            todo = []
            for episode in range(args.episodes):
                path = C.episode_path(args.out, task_id, bowl, episode)
                if path.exists():
                    existing.append(C.episode_scalars(path))
                else:
                    todo.append(episode)
            if todo:
                pending[(task_id, bowl)] = todo
    return pending, existing


def write_manifest(args, records):
    table = C.summarize_episodes(records)
    manifest = {"suite": args.suite, "checkpoint": args.checkpoint, "num_steps": args.num_steps,
                "n_action_steps": args.n_action_steps, "predict_every": args.predict_every, "seed": args.seed,
                "tasks_requested": args.task_ids, "bowls_requested": args.bowl_ids, "episodes_per_pair": args.episodes,
                "phrase_overrides": {str(k): v for k, v in args.phrase_overrides.items()},
                "prompts": {}, "table": table, "episodes": len(records)}
    for record in records:
        manifest["prompts"].setdefault(f"{record['task_id']}:{record['bowl']}", record.get("prompt"))
    C.write_json(args.out / MANIFEST_NAME, manifest)
    return table


def main(argv=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    libero_config = ROOT / "data" / "vla" / "libero_config"
    if "LIBERO_CONFIG_PATH" not in os.environ and libero_config.is_dir():
        os.environ["LIBERO_CONFIG_PATH"] = str(libero_config)
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pending, records = pending_episodes(args)
    print(f"{len(records)} episodes exist, {sum(len(e) for e in pending.values())} to run -> {args.out}", flush=True)
    if pending:
        roller = Roller(args)
        with (args.out / LOG_NAME).open("a") as log:
            for (task_id, bowl), episodes in pending.items():
                language, prompts = roller.prompts(task_id)
                prompt = prompts[bowl]
                print(f"[task {task_id} bowl {bowl}] prompt: {prompt!r}", flush=True)
                env = roller.scene.make_env(roller.suite, task_id, prompt_override=prompt)
                try:
                    for episode in episodes:
                        arrays, scalars = roller.run_episode(env, task_id, bowl, episode, prompt, language)
                        C.save_episode(C.episode_path(args.out, task_id, bowl, episode), arrays, scalars)
                        log_record(log, scalars)
                        records.append(scalars)
                finally:
                    if hasattr(env, "close"):
                        env.close()
    table = write_manifest(args, records)
    print(C.format_table(table), flush=True)
    print(f"manifest: {args.out / MANIFEST_NAME}", flush=True)


if __name__ == "__main__":
    main()

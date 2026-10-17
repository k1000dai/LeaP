#!/usr/bin/env python
"""Build the offline reversal dataset for the VLA prior from the LIBERO-Spatial demos.

For every task: replay the stored sim states of each demo in LeRobot's LIBERO env (same renderer as
evaluation) every ``--stride`` frames, render agentview + wrist at 256x256, build the 8-d state, take the
50-step action chunk (last action repeated past the demo end, ``chunk_pad_mask`` marks real steps), run
the frozen pi0.5 with the NEUTRAL prompt to get prefix features, and reverse the normalised chunk through
the flow (K = ``--num-steps`` Euler steps) to the source noise.  Per demo: a sketch from frame 0 (saved as
``episode{i}_sketch.png``) and the text / sketch / audio instruction embeddings.

Output ``<out>/task_{i}.npz`` (N = number of sampled frames, D = number of demos):

    feat [N, 4104] f32   state [N, 8] f32   noise [N, 50, 7] f32   chunk_pad_mask [N, 50] bool
    demo_id [N] i64      t [N] i64          recon_err [N] f32 (NaN where not measured)
    instr_text [D, 1, 768]  instr_sketch [D, 1, 768]  instr_audio [D, 1, 768]  (zeros when a modality is off)
    instr_modalities [M] str  task_id (int)  task_language (str)  prompt (str)  num_steps (int)

plus ``<out>/manifest.json``.  Idempotent per task: an existing ``task_{i}.npz`` is skipped.

Raw demo layout (LIBERO / robomimic hdf5): ``data/demo_{k}/states [T, S]`` (flattened MuJoCo state),
``data/demo_{k}/actions [T, 7]``, ``data/demo_{k}/obs/{agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori,
gripper_states, joint_states}``; images are re-rendered from ``states`` so ``obs`` is not read.  The file
for task i is ``<raw>/<task name>_demo.hdf5`` (``suite.get_task_demonstration(i)``).

Usage (remote GPU):
    MUJOCO_GL=egl python scripts/vla/build_reversal_dataset.py --raw data/vla/raw/libero_spatial \\
        --out data/vla/libero_spatial_reversal --checkpoint lerobot/pi05_libero_finetuned_v044 \\
        --tasks 0-9 --stride 2 --num-steps 10 --batch-size 16 --device cuda \\
        --voice data/voices/en_US-lessac-medium.onnx
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leap.vla import instructions as I  # noqa: E402
from leap.vla import libero_scene as S  # noqa: E402

CHUNK = 50
ACTION_DIM = 7
FEAT_DIM = 4104
DEMO_RE = re.compile(r"^demo_(\d+)$")


# --- raw demos --------------------------------------------------------------------------------------------

def demo_file(raw_dir, suite, task_id):
    """``<raw>/<task name>_demo.hdf5`` (the basename of ``suite.get_task_demonstration``)."""
    return Path(raw_dir) / Path(suite.get_task_demonstration(int(task_id))).name


def demo_ids(h5):
    """Numeric demo ids of ``data/demo_{k}`` groups, ascending."""
    ids = []
    for key in h5["data"].keys():
        match = DEMO_RE.match(key)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def read_demo(h5, demo_id):
    """``(states [T, S] float64, actions [T, 7] float32)`` of ``data/demo_{demo_id}``."""
    group = h5["data"][f"demo_{demo_id}"]
    states = np.asarray(group["states"], dtype=np.float64)
    actions = np.asarray(group["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"demo_{demo_id}: actions must be [T, {ACTION_DIM}], got {actions.shape}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"demo_{demo_id}: {states.shape[0]} states vs {actions.shape[0]} actions")
    return states, actions


# --- chunks -----------------------------------------------------------------------------------------------

def action_chunk(actions, t, horizon=CHUNK):
    """``actions[t : t + horizon]`` padded by repeating the last action; returns ``(chunk [H, 7], pad_mask [H] bool)``.

    ``pad_mask[j]`` is True where ``t + j`` is a real demo step (the gripper command is held afterwards).
    """
    actions = np.asarray(actions, dtype=np.float32)
    total = actions.shape[0]
    if not 0 <= t < total:
        raise IndexError(f"t={t} outside the demo of {total} steps")
    chunk = actions[t:t + horizon]
    valid = chunk.shape[0]
    if valid < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - valid, axis=0)], axis=0)
    mask = np.zeros(horizon, dtype=bool)
    mask[:valid] = True
    return chunk.astype(np.float32), mask


def sample_frames(length, stride):
    """Frame indices 0, stride, 2 stride, ... < length."""
    return list(range(0, int(length), int(stride)))


def parse_tasks(spec):
    """``"0-9"``, ``"0,3,5"`` or ``"2"`` -> sorted list of task ids."""
    ids = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def relative_l2(pred, target, mask):
    """Relative L2 error over the valid chunk steps and the first ``ACTION_DIM`` dims, per sample [B]."""
    pred = np.asarray(pred, dtype=np.float64)[:, :, :ACTION_DIM]
    target = np.asarray(target, dtype=np.float64)[:, :, :ACTION_DIM]
    mask = np.asarray(mask, dtype=np.float64)[:, :, None]
    num = np.sqrt((((pred - target) ** 2) * mask).sum(axis=(1, 2)))
    den = np.sqrt(((target ** 2) * mask).sum(axis=(1, 2)))
    return (num / np.maximum(den, 1e-8)).astype(np.float32)


# --- frozen VLA adapter (leap.vla.pi05_flow, imported lazily; tests monkeypatch ``flow_api``) ---------

def flow_api():
    from leap.vla import pi05_flow
    return pi05_flow


class Reverser:
    """Batch observations through the frozen VLA: prefix features and reversed (noise) chunks.

    ``images`` are ``{"image": uint8 [B, H, W, 3], "image2": ...}`` exactly as ``LiberoEnv`` returns ``pixels``
    (raw render, not rotated); ``state`` is [B, 8]; ``actions7`` is [B, 50, 7] un-normalised; the prompt is
    one string for the batch.
    """

    def __init__(self, bundle, prompt, num_steps):
        self.bundle = bundle
        self.prompt = prompt
        self.num_steps = int(num_steps)

    def run(self, images, state, actions7, pad_mask, measure_recon):
        api = flow_api()
        batch = api.prepare_batches(self.bundle, images, state, self.prompt)
        ctx = api.prefix_context(self.bundle, batch)
        feat = _numpy(api.prefix_features(self.bundle, batch))
        target = api.normalize_actions(self.bundle, actions7)
        noise = api.reverse_actions(self.bundle, ctx, target, self.num_steps)
        recon = np.full(feat.shape[0], np.nan, dtype=np.float32)
        if np.any(measure_recon):
            recon_pred = _numpy(api.forward_from_noise(self.bundle, ctx, noise, self.num_steps))
            errors = relative_l2(recon_pred, _numpy(target), pad_mask)
            recon[measure_recon] = errors[measure_recon]
        return feat, _numpy(noise)[:, :, :ACTION_DIM], recon


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


# --- per-task build -----------------------------------------------------------------------------------------

def collect_demo(env, states, actions, stride):
    """Replay one demo at ``stride``: lists of images, states, chunks, masks, frame indices and frame-0 image."""
    frames = sample_frames(states.shape[0], stride)
    images, obs_states, chunks, masks = [], [], [], []
    frame0 = None
    for t in frames:
        raw_obs = S.replay_state(env, states[t])
        pixels = S.images_from_raw(raw_obs, env)
        if t == frames[0]:
            frame0 = S.policy_view(pixels["image"])
        images.append(pixels)
        obs_states.append(S.state_vector(raw_obs, env))
        chunk, mask = action_chunk(actions, t)
        chunks.append(chunk)
        masks.append(mask)
    return frames, images, obs_states, chunks, masks, frame0


def stack_images(pixels_list):
    keys = list(pixels_list[0].keys())
    return {key: np.stack([p[key] for p in pixels_list]).astype(np.uint8) for key in keys}


def build_task(args, suite, task_id, bundle, embedder, tts, rng):
    """Build and write ``task_{task_id}.npz``; returns the manifest entry."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"task_{task_id}.npz"
    if npz_path.exists():
        print(f"[task {task_id}] {npz_path} exists, skipping")
        return {"task_id": task_id, "path": str(npz_path), "skipped": True}
    import h5py
    path = demo_file(args.raw, suite, task_id)
    if not path.exists():
        raise FileNotFoundError(f"raw demos for task {task_id} not found: {path}")
    task = suite.get_task(task_id)
    language = I.task_text(task.language)
    env = S.make_env(suite, task_id, S.NEUTRAL_PROMPT)
    env.reset(seed=0)
    layout0 = S.scene_layout(env)  # only used to pick object names; per-demo layouts come from frame 0
    cam = S.agentview_camera(env)
    reverser = Reverser(bundle, S.NEUTRAL_PROMPT, args.num_steps)
    modalities = [m for m in ("text", "sketch", "audio") if m not in args.skip_modalities]
    if "audio" in modalities and tts is None:
        raise ValueError("--voice is required for audio instructions (or pass --skip-modalities audio)")

    feats, states_out, noises, masks_out, demo_col, t_col, recon_col = [], [], [], [], [], [], []
    instr = {m: [] for m in modalities}
    started = time.time()
    with h5py.File(path, "r") as h5:
        ids = demo_ids(h5)
        if args.max_demos is not None:
            ids = ids[:args.max_demos]
        for demo_index, demo_id in enumerate(ids):
            states, actions = read_demo(h5, demo_id)
            frames, images, obs_states, chunks, masks, frame0 = collect_demo(env, states, actions, args.stride)
            layout = S.scene_layout(env, names=S.SCENE_OBJECTS) if False else _frame0_layout(env, states[0])
            sketch = I.build_sketch(frame0, cam, layout)
            sketch.save(out_dir / f"task_{task_id}_episode{demo_id}_sketch.png")
            if "text" in modalities:
                instr["text"].append(I.embed_instruction(embedder, "text", text=language))
            if "sketch" in modalities:
                instr["sketch"].append(I.embed_instruction(embedder, "sketch", sketch=sketch))
            if "audio" in modalities:
                instr["audio"].append(I.embed_instruction(embedder, "audio", audio=I.build_audio(language, tts)))
            measure = rng.random(len(frames)) < args.recon_fraction
            for start in range(0, len(frames), args.batch_size):
                stop = min(start + args.batch_size, len(frames))
                feat, noise, recon = reverser.run(
                    stack_images(images[start:stop]), np.stack(obs_states[start:stop]),
                    np.stack(chunks[start:stop]), np.stack(masks[start:stop]), measure[start:stop])
                feats.append(np.asarray(feat, dtype=np.float32))
                noises.append(np.asarray(noise, dtype=np.float32))
                recon_col.append(np.asarray(recon, dtype=np.float32))
            states_out.extend(obs_states)
            masks_out.extend(masks)
            demo_col.extend([demo_id] * len(frames))
            t_col.extend(frames)
            print(f"[task {task_id}] demo {demo_id} ({demo_index + 1}/{len(ids)}): {len(frames)} frames of "
                  f"{states.shape[0]} steps, {time.time() - started:.0f}s", flush=True)
    env.close()
    arrays = {
        "feat": np.concatenate(feats, axis=0).astype(np.float32),
        "state": np.stack(states_out).astype(np.float32),
        "noise": np.concatenate(noises, axis=0).astype(np.float32),
        "chunk_pad_mask": np.stack(masks_out).astype(bool),
        "demo_id": np.asarray(demo_col, dtype=np.int64),
        "t": np.asarray(t_col, dtype=np.int64),
        "recon_err": np.concatenate(recon_col, axis=0).astype(np.float32),
        "instr_modalities": np.asarray(modalities),
        "task_id": np.int64(task_id),
        "task_language": np.str_(language),
        "prompt": np.str_(S.NEUTRAL_PROMPT),
        "num_steps": np.int64(args.num_steps),
    }
    demos = len(ids)
    for modality in ("text", "sketch", "audio"):
        vectors = instr.get(modality)
        arrays[f"instr_{modality}"] = (np.stack(vectors)[:, None, :].astype(np.float32) if vectors
                                       else np.zeros((demos, 1, I.INSTR_DIM), dtype=np.float32))
    check_arrays(arrays, demos)
    np.savez(npz_path, **arrays)
    measured = arrays["recon_err"][np.isfinite(arrays["recon_err"])]
    entry = {"task_id": task_id, "path": str(npz_path), "task_language": language, "demos": demos,
             "frames": int(arrays["feat"].shape[0]), "stride": args.stride, "num_steps": args.num_steps,
             "modalities": modalities, "recon_err_mean": float(measured.mean()) if measured.size else None,
             "recon_err_count": int(measured.size), "seconds": round(time.time() - started, 1)}
    print(f"[task {task_id}] wrote {npz_path}: {entry}", flush=True)
    return entry


def _frame0_layout(env, state0):
    """Object layout at the demo's first frame (replays ``state0``)."""
    S.replay_state(env, state0)
    return S.scene_layout(env)


def check_arrays(arrays, demos):
    """Shape/dtype checks of the npz contents (shared with the tests)."""
    n = arrays["feat"].shape[0]
    expected = {"feat": (n, FEAT_DIM), "state": (n, S.STATE_DIM), "noise": (n, CHUNK, ACTION_DIM),
                "chunk_pad_mask": (n, CHUNK), "demo_id": (n,), "t": (n,), "recon_err": (n,),
                "instr_text": (demos, 1, I.INSTR_DIM), "instr_sketch": (demos, 1, I.INSTR_DIM),
                "instr_audio": (demos, 1, I.INSTR_DIM)}
    for key, shape in expected.items():
        if tuple(arrays[key].shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {arrays[key].shape}")
    if n == 0:
        raise ValueError("no frames were collected")


def load_task(path):
    """Read a ``task_{i}.npz`` back as a dict of arrays (scalars unwrapped)."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    out["task_id"] = int(out["task_id"])
    out["task_language"] = str(out["task_language"])
    out["prompt"] = str(out["prompt"])
    out["num_steps"] = int(out["num_steps"])
    return out


def write_manifest(out_dir, entries, args):
    path = Path(out_dir) / "manifest.json"
    manifest = {"tasks": {}, "checkpoint": args.checkpoint, "prompt": S.NEUTRAL_PROMPT, "stride": args.stride,
                "num_steps": args.num_steps, "suite": args.suite}
    if path.exists():
        with open(path) as handle:
            manifest.update(json.load(handle))
    for entry in entries:
        if not entry.get("skipped"):
            manifest["tasks"][str(entry["task_id"])] = entry
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return path


# --- main -----------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="directory with <task name>_demo.hdf5 files")
    parser.add_argument("--out", required=True, help="output directory for task_{i}.npz and manifest.json")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9", help="task ids, e.g. 0-9 or 0,3")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10, help="Euler steps for reversal (and reconstruction)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedder-device", default=None, help="defaults to --device")
    parser.add_argument("--embedder-model", default="google/embeddinggemma-2")
    parser.add_argument("--voice", default=None, help="Piper voice .onnx for audio instructions")
    parser.add_argument("--skip-modalities", nargs="*", default=[], choices=["text", "sketch", "audio"])
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--recon-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    suite = S.load_suite(args.suite)
    tasks = parse_tasks(args.tasks)
    pending = [t for t in tasks if not (Path(args.out) / f"task_{t}.npz").exists()]
    entries = []
    if not pending:
        print("all requested tasks exist; nothing to do")
    else:
        bundle = flow_api().load_pi05(args.checkpoint, args.device)
        from leap.steer.embedder import InstructionEmbedder
        embedder = InstructionEmbedder(args.embedder_model, device=args.embedder_device or args.device)
        tts = None
        if args.voice and "audio" not in args.skip_modalities:
            from leap.steer.tts import PiperTTS
            tts = PiperTTS(args.voice)
        rng = np.random.default_rng(args.seed)
        for task_id in tasks:
            entries.append(build_task(args, suite, task_id, bundle, embedder, tts, rng))
    path = write_manifest(args.out, entries, args)
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the offline reversal dataset for the VLA prior from the LIBERO-Spatial demos (runs in the VLA venv).

For every task: replay the stored sim states of each demo in LeRobot's LIBERO env (same renderer as
evaluation) every ``--stride`` frames, render agentview + wrist at 256x256, build the 8-d state, take the
50-step action chunk (last action repeated past the demo end), run the frozen pi0.5 with the NEUTRAL prompt
to get prefix features, and reverse the normalised chunk through the flow (K = ``--num-steps`` Euler steps)
to the source noise.  Per demo a sketch of frame 0 is rendered; per task the instruction record is written.

Embeddings are NOT computed here (google/embeddinggemma-2 needs a newer transformers than LeRobot pins):
run ``scripts/vla/embed_vla_instructions.py --data <out>`` under the phase-2 venv afterwards.  It reads the
artifacts this script leaves next to the npz:

- ``task_{i}_instructions.json`` (texts, demo ids, sketch and audio file names; ``leap.vla.instructions``),
- ``task_{i}_episode{d}_sketch.png`` per demo,
- ``task_{i}_audio{v}.wav`` per text variant when ``--voice`` is given and piper imports here; otherwise the
  embed script synthesizes them.

Output ``<out>/task_{i}.npz`` (N = sampled frames; the embed script adds ``instr_text/sketch/audio [D, V, 768]``
and ``instr_modalities``):

    feat [N, 4104] f32   state [N, 8] f32   noise [N, 50, 7] f32
    chunk_pad_mask [N, 50] bool  (True = padded step beyond the demo end, False = real demo step)
    demo_id [N] i64 (hdf5 demo number)   t [N] i64 (demo frame)   recon_err [N] f32 (NaN where not measured)
    task_id (int)  task_language (str)  prompt (str)  num_steps (int)

Rows are in ascending demo id, then ascending t.  ``<out>/manifest.json`` records the run parameters and
``"chunk_pad_mask": "true_means_padded"``.  Idempotent per task: an existing ``task_{i}.npz`` is skipped.

Raw demo layout (LIBERO / robomimic hdf5, verified on yifengzhu-hf/LIBERO-datasets libero_spatial):
``data/demo_{k}/states [T, 92]`` (flattened MuJoCo state), ``data/demo_{k}/actions [T, 7]`` float64,
``data/demo_{k}/obs/{agentview_rgb, eye_in_hand_rgb (128x128), ee_pos, ee_ori, ee_states, gripper_states,
joint_states}``, ``robot_states [T, 9]``, ``rewards``, ``dones``; 50 demos of 79-134 steps per task.  Images
are re-rendered from ``states`` so ``obs`` is not read.  The file for task i is
``<raw>/<task name>_demo.hdf5`` (``suite.get_task_demonstration(i)``).

Usage (remote GPU, VLA venv):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=data/vla/libero_config python scripts/vla/build_reversal_dataset.py \\
        --raw data/vla/raw/libero_spatial --out data/vla/libero_spatial_reversal \\
        --checkpoint lerobot/pi05_libero_finetuned_v044 --tasks 0-9 --stride 2 --num-steps 10 \\
        --batch-size 16 --device cuda
"""
import argparse
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
PAD_MASK_CONVENTION = "true_means_padded"


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

    ``pad_mask[j]`` is True where ``t + j`` is beyond the demo end (a padded step holding the last action,
    gripper included) and False for real demo steps.
    """
    actions = np.asarray(actions, dtype=np.float32)
    total = actions.shape[0]
    if not 0 <= t < total:
        raise IndexError(f"t={t} outside the demo of {total} steps")
    chunk = actions[t:t + horizon]
    valid = chunk.shape[0]
    if valid < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - valid, axis=0)], axis=0)
    pad_mask = np.zeros(horizon, dtype=bool)
    pad_mask[valid:] = True
    return chunk.astype(np.float32), pad_mask


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


def relative_l2(pred, target, pad_mask):
    """Relative L2 error over the real (non-padded) chunk steps and the first ``ACTION_DIM`` dims, per sample [B]."""
    pred = np.asarray(pred, dtype=np.float64)[:, :, :ACTION_DIM]
    target = np.asarray(target, dtype=np.float64)[:, :, :ACTION_DIM]
    valid = (~np.asarray(pad_mask, dtype=bool)).astype(np.float64)[:, :, None]
    num = np.sqrt((((pred - target) ** 2) * valid).sum(axis=(1, 2)))
    den = np.sqrt(((target ** 2) * valid).sum(axis=(1, 2)))
    return (num / np.maximum(den, 1e-8)).astype(np.float32)


# --- frozen VLA adapter (leap.vla.pi05_flow, imported lazily; tests monkeypatch ``flow_api``) ---------

def flow_api():
    from leap.vla import pi05_flow
    return pi05_flow


class Reverser:
    """Batch observations through the frozen VLA: prefix features and reversed (noise) chunks.

    ``images`` is a list of ``{"image": uint8 [H, W, 3], "image2": ...}`` dicts exactly as ``LiberoEnv``
    returns ``pixels`` (raw render; ``prepare_batches`` applies the 180-degree rotation), ``states`` a list of
    8-d vectors, ``actions7`` [B, 50, 7] un-normalised; the prompt is one string for the whole batch.
    """

    def __init__(self, bundle, prompt, num_steps):
        self.bundle = bundle
        self.prompt = prompt
        self.num_steps = int(num_steps)

    def run(self, images, states, actions7, pad_mask, measure_recon):
        """-> ``(feat [B, 4104], noise [B, 50, 7], recon_err [B] with NaN where not measured)``."""
        api = flow_api()
        observations = [api.Observation(images=pixels, state=np.asarray(state, dtype=np.float32), prompt=self.prompt)
                        for pixels, state in zip(images, states)]
        batch = api.prepare_batches(self.bundle, observations)
        ctx = api.prefix_context(self.bundle, batch)
        feat = _numpy(api.prefix_features(self.bundle, batch, ctx))
        target = api.normalize_actions(self.bundle, np.asarray(actions7, dtype=np.float32))
        noise = api.reverse_actions(self.bundle, ctx, target, self.num_steps)
        measure_recon = np.asarray(measure_recon, dtype=bool)
        recon = np.full(feat.shape[0], np.nan, dtype=np.float32)
        if measure_recon.any():
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
    """Replay one demo at ``stride``.

    Returns ``(frames, images, obs_states, chunks, pad_masks, frame0, layout)``: sampled frame indices, per-frame
    ``pixels`` dicts, 8-d states, action chunks and pad masks, the policy-view agentview image of the first
    frame and the object layout at that frame (for the sketch).
    """
    frames = sample_frames(states.shape[0], stride)
    images, obs_states, chunks, pad_masks = [], [], [], []
    frame0, layout = None, None
    for t in frames:
        raw_obs = S.replay_state(env, states[t])
        pixels = S.images_from_raw(raw_obs, env)
        if frame0 is None:
            frame0 = S.policy_view(pixels["image"])
            layout = S.scene_layout(env)
        images.append(pixels)
        obs_states.append(S.state_vector(raw_obs, env))
        chunk, pad_mask = action_chunk(actions, t)
        chunks.append(chunk)
        pad_masks.append(pad_mask)
    return frames, images, obs_states, chunks, pad_masks, frame0, layout


def build_task(args, suite, task_id, bundle, rng, tts=None):
    """Build one task: ``task_{task_id}.npz``, the sketch PNGs, the instruction record (+ WAVs with ``tts``)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = I.task_npz_path(out_dir, task_id)
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
    cam = S.agentview_camera(env)
    reverser = Reverser(bundle, S.NEUTRAL_PROMPT, args.num_steps)

    feats, states_out, noises, masks_out, demo_col, t_col, recon_col = [], [], [], [], [], [], []
    started = time.time()
    with h5py.File(path, "r") as h5:
        ids = demo_ids(h5)
        if args.max_demos is not None:
            ids = ids[:args.max_demos]
        for demo_index, demo_id in enumerate(ids):
            states, actions = read_demo(h5, demo_id)
            frames, images, obs_states, chunks, pad_masks, frame0, layout = collect_demo(
                env, states, actions, args.stride)
            I.build_sketch(frame0, cam, layout).save(out_dir / I.sketch_file(task_id, demo_id))
            measure = rng.random(len(frames)) < args.recon_fraction
            for start in range(0, len(frames), args.batch_size):
                stop = min(start + args.batch_size, len(frames))
                feat, noise, recon = reverser.run(
                    images[start:stop], obs_states[start:stop],
                    np.stack(chunks[start:stop]), np.stack(pad_masks[start:stop]), measure[start:stop])
                feats.append(np.asarray(feat, dtype=np.float32))
                noises.append(np.asarray(noise, dtype=np.float32))
                recon_col.append(np.asarray(recon, dtype=np.float32))
            states_out.extend(obs_states)
            masks_out.extend(pad_masks)
            demo_col.extend([demo_id] * len(frames))
            t_col.extend(frames)
            print(f"[task {task_id}] demo {demo_id} ({demo_index + 1}/{len(ids)}): {len(frames)} frames of "
                  f"{states.shape[0]} steps, {time.time() - started:.0f}s", flush=True)
    env.close()
    record = I.new_record(task_id, language, ids)
    if tts is not None:
        I.ensure_audio(out_dir, record, tts)
    I.write_record(out_dir, record)
    arrays = {
        "feat": np.concatenate(feats, axis=0).astype(np.float32),
        "state": np.stack(states_out).astype(np.float32),
        "noise": np.concatenate(noises, axis=0).astype(np.float32),
        "chunk_pad_mask": np.stack(masks_out).astype(bool),
        "demo_id": np.asarray(demo_col, dtype=np.int64),
        "t": np.asarray(t_col, dtype=np.int64),
        "recon_err": np.concatenate(recon_col, axis=0).astype(np.float32),
        "task_id": np.int64(task_id),
        "task_language": np.str_(language),
        "prompt": np.str_(S.NEUTRAL_PROMPT),
        "num_steps": np.int64(args.num_steps),
    }
    check_arrays(arrays)
    I.save_task_npz(npz_path, arrays)
    measured = arrays["recon_err"][np.isfinite(arrays["recon_err"])]
    entry = {"task_id": task_id, "path": str(npz_path), "task_language": language, "demos": len(ids),
             "demo_ids": ids, "frames": int(arrays["feat"].shape[0]), "stride": args.stride,
             "num_steps": args.num_steps, "recon_err_mean": float(measured.mean()) if measured.size else None,
             "recon_err_count": int(measured.size), "audio_files": list(record["audio"]),
             "seconds": round(time.time() - started, 1)}
    print(f"[task {task_id}] wrote {npz_path}: {entry}", flush=True)
    return entry


def check_arrays(arrays):
    """Shape checks of the reversal arrays."""
    n = arrays["feat"].shape[0]
    expected = {"feat": (n, FEAT_DIM), "state": (n, S.STATE_DIM), "noise": (n, CHUNK, ACTION_DIM),
                "chunk_pad_mask": (n, CHUNK), "demo_id": (n,), "t": (n,), "recon_err": (n,)}
    for key, shape in expected.items():
        if tuple(arrays[key].shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {arrays[key].shape}")
    if n == 0:
        raise ValueError("no frames were collected")


def manifest_defaults(args):
    return {"checkpoint": args.checkpoint, "prompt": S.NEUTRAL_PROMPT, "stride": args.stride,
            "num_steps": args.num_steps, "suite": args.suite, "chunk_pad_mask": PAD_MASK_CONVENTION}


def make_tts(voice):
    """``PiperTTS`` for ``voice`` when piper imports and the file exists, else None (audio left to the embed script)."""
    if not voice:
        return None
    try:
        from leap.steer.tts import PiperTTS
        tts = PiperTTS(voice)
        tts._load()
    except Exception as error:  # noqa: BLE001 - piper may be absent in the VLA venv
        print(f"TTS unavailable ({error!r}); audio will be synthesized by embed_vla_instructions.py")
        return None
    return tts


# --- main -----------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="directory with <task name>_demo.hdf5 files")
    parser.add_argument("--out", required=True, help="output directory for task_{i}.npz, artifacts and manifest.json")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9", help="task ids, e.g. 0-9 or 0,3")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10, help="Euler steps for reversal (and reconstruction)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice", default=None, help="optional Piper voice .onnx: write the audio WAVs here too")
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--recon-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    # hf-libero prompts with input() on first import unless its config.yaml exists; the remote setup
    # writes one under data/vla/libero_config (scripts/vla/remote_setup_vla.sh), so point there by default.
    libero_config = Path(__file__).resolve().parents[2] / "data" / "vla" / "libero_config"
    if "LIBERO_CONFIG_PATH" not in os.environ and libero_config.is_dir():
        os.environ["LIBERO_CONFIG_PATH"] = str(libero_config)
    tasks = parse_tasks(args.tasks)
    pending = [t for t in tasks if not I.task_npz_path(args.out, t).exists()]
    entries = []
    if not pending:
        print("all requested tasks exist; nothing to do")
    else:
        suite = S.load_suite(args.suite)
        bundle = flow_api().load_pi05(args.checkpoint, args.device)
        tts = make_tts(args.voice)
        rng = np.random.default_rng(args.seed)
        for task_id in tasks:
            entries.append(build_task(args, suite, task_id, bundle, rng, tts))
    path = I.update_manifest(args.out, entries, manifest_defaults(args))
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Turn counterfactual rollouts into reversal-label datasets for the VLA prior (runs in the VLA venv).

Reads the ``task_{i}_bowl{b}_episode{e}.npz`` files of ``scripts/vla/rollout_counterfactual.py`` and, per
(task, bowl), rebuilds every recorded VLA query under the NEUTRAL prompt (``libero_scene.NEUTRAL_PROMPT``) from
the stored images and state, takes the prefix features, and reverses the recorded 50-step prediction
(normalised, zero-padded) through the flow with --num-steps Euler steps to its source noise.  The chunk is the
VLA's own full prediction, so ``chunk_pad_mask`` is all False; the reconstruction error is measured on
--recon-fraction of the rows.

Output: ``<out>/task_{n}.npz`` with ``n = 100 * bowl + task_id`` (``counterfactual.dataset_index``; one file per
(task, bowl) so that ``scripts/vla/embed_vla_instructions.py`` embeds the bowl's own text unchanged), in the format
of ``scripts/vla/build_reversal_dataset.py``:

    feat [N, 4104]  state [N, 8]  noise [N, 50, 7]  chunk_pad_mask [N, 50] (all False)  demo_id [N] = bowl * 1000 + episode
    t [N] (env step of the query)  recon_err [N]  task_id (suite task id)  task_language  prompt (neutral)  num_steps

plus ``bowl [N]``, ``episode [N]``, ``executed [N]`` (query whose actions were executed), ``success [N]``,
``target_bowl``, ``target_prompt`` (the full prompt the rollout was made with) and ``dataset_index``.  Next to it
``task_{n}_instructions.json`` (texts = [target prompt], one sketch per episode circling the INTENDED bowl on the
white canvas, audio left to the embed script unless --voice synthesizes it here) and the sketch PNGs.  Idempotent
per (task, bowl): an existing ``task_{n}.npz`` is skipped.

Usage (remote GPU, VLA venv), then embed under the phase-2 venv:
    MUJOCO_GL=egl python scripts/vla/reverse_rollouts.py --rollouts data/vla/counterfactual \\
        --out data/vla/libero_spatial_counterfactual --tasks 0-9 --bowls 1,2 --successful-only --stride 1 \\
        --num-steps 10 --batch-size 16 --device cuda
    .venv/bin/python scripts/vla/embed_vla_instructions.py --data data/vla/libero_spatial_counterfactual \\
        --voice data/voices/en_US-lessac-medium.onnx --device cuda
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from leap.vla import counterfactual as C  # noqa: E402
from leap.vla import instructions as I  # noqa: E402
from leap.vla import libero_scene as S  # noqa: E402

CHUNK = 50
ACTION_DIM = 7
FEAT_DIM = 4104
PAD_MASK_CONVENTION = "true_means_padded"


def flow_api():
    """``leap.vla.pi05_flow`` (imported lazily; tests monkeypatch this)."""
    from leap.vla import pi05_flow
    return pi05_flow


def parse_tasks(spec):
    """``"0-9"``, ``"0,3,5"`` or ``"2"`` -> sorted list of ids."""
    ids = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            ids.update(range(int(low), int(high) + 1))
        else:
            ids.add(int(part))
    if not ids:
        raise ValueError(f"no ids in {spec!r}")
    return sorted(ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True, help="directory with task_{i}_bowl{b}_episode{e}.npz")
    parser.add_argument("--out", type=Path, required=True, help="output dataset directory (task_{n}.npz + artifacts)")
    parser.add_argument("--tasks", default="0-9")
    parser.add_argument("--bowls", default="1,2")
    parser.add_argument("--successful-only", action="store_true", help="use only episodes whose intended bowl ended on the plate")
    parser.add_argument("--executed-only", action="store_true", help="use only the queries whose actions were executed")
    parser.add_argument("--stride", type=int, default=1, help="keep every stride-th recorded query of an episode")
    parser.add_argument("--num-steps", type=int, default=10, help="Euler steps for reversal (and reconstruction)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--recon-fraction", type=float, default=0.05)
    parser.add_argument("--voice", default=None, help="optional Piper voice .onnx: synthesize the audio WAVs here")
    parser.add_argument("--max-episodes", type=int, default=None, help="use at most this many episodes per (task, bowl)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.stride < 1 or args.batch_size < 1 or args.num_steps < 1:
        parser.error("stride, batch-size and num-steps must be positive")
    try:
        args.task_ids = parse_tasks(args.tasks)
        args.bowl_ids = parse_tasks(args.bowls)
    except ValueError as error:
        parser.error(str(error))
    if any(b not in C.BOWLS for b in args.bowl_ids):
        parser.error(f"--bowls must be a subset of {C.BOWLS}")
    return args


def make_tts(voice):
    """``PiperTTS`` for ``voice`` when piper imports, else None (audio left to the embed script)."""
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


def select_episodes(args, task_id, bowl):
    """Episode files of (task, bowl) honouring --successful-only and --max-episodes: ``[(episode, path, scalars)]``."""
    chosen = []
    for _, _, episode, path in C.find_episodes(args.rollouts, task_id, bowl):
        scalars = C.episode_scalars(path)
        if args.successful_only and not bool(scalars.get("success", False)):
            continue
        chosen.append((episode, path, scalars))
    if args.max_episodes is not None:
        chosen = chosen[:args.max_episodes]
    return chosen


def query_rows(data, stride, executed_only):
    """Indices into the episode's prediction arrays to reverse."""
    rows = np.arange(len(data["pred_step"]))
    if executed_only:
        rows = rows[np.asarray(data["executed"], dtype=bool)]
    return rows[::stride]


def reverse_episode(args, bundle, data, rows, rng):
    """Reverse the selected queries of one episode -> dict of per-row arrays (feat, state, noise, t, recon_err, executed)."""
    flow = flow_api()
    steps = np.asarray(data["pred_step"])[rows]
    images = [C.raw_images(data["agentview"][t], data["wrist"][t]) for t in steps]
    states = [np.asarray(data["state"][t], dtype=np.float32) for t in steps]
    chunks = np.asarray(data["chunk"])[rows].astype(np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (CHUNK, ACTION_DIM):
        raise ValueError(f"chunks must be [N, {CHUNK}, {ACTION_DIM}], got {chunks.shape}")
    measure = rng.random(len(rows)) < args.recon_fraction
    feats, noises, recons = [], [], []
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        feat, noise, recon = C.reverse_chunks(flow, bundle, S.NEUTRAL_PROMPT, images[start:stop], states[start:stop],
                                              chunks[start:stop], args.num_steps, measure[start:stop])
        feats.append(feat)
        noises.append(noise)
        recons.append(recon)
    return {"feat": np.concatenate(feats).astype(np.float32), "state": np.stack(states).astype(np.float32),
            "noise": np.concatenate(noises).astype(np.float32), "t": steps.astype(np.int64),
            "recon_err": np.concatenate(recons).astype(np.float32),
            "executed": np.asarray(data["executed"], dtype=bool)[rows]}


def build_pair(args, task_id, bowl, bundle, rng, tts=None):
    """Build ``task_{n}.npz`` + instruction record for one (task, bowl); returns the manifest entry."""
    index = C.dataset_index(task_id, bowl)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = I.task_npz_path(out_dir, index)
    if npz_path.exists():
        print(f"[task {task_id} bowl {bowl}] {npz_path} exists, skipping")
        return {"task_id": index, "path": str(npz_path), "skipped": True}
    episodes = select_episodes(args, task_id, bowl)
    if not episodes:
        print(f"[task {task_id} bowl {bowl}] no usable episodes under {args.rollouts}, skipping")
        return {"task_id": index, "skipped": True, "reason": "no episodes"}
    started = time.time()
    parts, demo_ids, language, target_prompt = [], [], None, None
    for episode, path, scalars in episodes:
        data = C.load_episode(path)
        language = language or str(data["task_language"])
        target_prompt = target_prompt or str(data["prompt"])
        rows = query_rows(data, args.stride, args.executed_only)
        if not len(rows):
            print(f"[task {task_id} bowl {bowl}] episode {episode}: no queries selected, skipping")
            continue
        demo_id = C.demo_id_for(bowl, episode)
        part = reverse_episode(args, bundle, data, rows, rng)
        n = part["feat"].shape[0]
        part.update({"demo_id": np.full(n, demo_id, dtype=np.int64), "bowl": np.full(n, bowl, dtype=np.int64),
                     "episode": np.full(n, episode, dtype=np.int64),
                     "success": np.full(n, bool(scalars.get("success", False)), dtype=bool)})
        parts.append(part)
        demo_ids.append(demo_id)
        cam = (np.asarray(data["cam_intrinsic"]), np.asarray(data["cam_extrinsic"]))
        layout = C.layout_from_positions(data["positions"][0])
        C.build_bowl_sketch(data["agentview"][0], cam, layout, bowl).save(out_dir / I.sketch_file(index, demo_id))
        print(f"[task {task_id} bowl {bowl}] episode {episode}: {n} queries reversed, {time.time() - started:.0f}s", flush=True)
    if not parts:
        return {"task_id": index, "skipped": True, "reason": "no queries"}
    n_total = sum(p["feat"].shape[0] for p in parts)
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    arrays.update({
        "chunk_pad_mask": np.zeros((n_total, CHUNK), dtype=bool),
        "task_id": np.int64(task_id), "task_language": np.str_(language), "prompt": np.str_(S.NEUTRAL_PROMPT),
        "num_steps": np.int64(args.num_steps), "target_bowl": np.int64(bowl), "target_prompt": np.str_(target_prompt),
        "dataset_index": np.int64(index), "source": np.str_(str(args.rollouts)),
    })
    check_arrays(arrays)
    record = I.new_record(index, language, demo_ids, texts=[target_prompt])
    record.update({"suite_task_id": int(task_id), "bowl": int(bowl)})
    I.write_record(out_dir, record)
    if tts is not None:
        record = I.ensure_audio(out_dir, record, tts)
    I.save_task_npz(npz_path, arrays)
    measured = arrays["recon_err"][np.isfinite(arrays["recon_err"])]
    entry = {"task_id": index, "suite_task_id": task_id, "bowl": bowl, "path": str(npz_path), "task_language": language,
             "target_prompt": target_prompt, "demos": len(demo_ids), "demo_ids": demo_ids, "frames": int(n_total),
             "stride": args.stride, "executed_only": args.executed_only, "num_steps": args.num_steps,
             "recon_err_mean": float(measured.mean()) if measured.size else None, "recon_err_count": int(measured.size),
             "audio_files": list(record["audio"]), "seconds": round(time.time() - started, 1)}
    print(f"[task {task_id} bowl {bowl}] wrote {npz_path}: {entry}", flush=True)
    return entry


def check_arrays(arrays):
    n = arrays["feat"].shape[0]
    expected = {"feat": (n, FEAT_DIM), "state": (n, S.STATE_DIM), "noise": (n, CHUNK, ACTION_DIM),
                "chunk_pad_mask": (n, CHUNK), "demo_id": (n,), "t": (n,), "recon_err": (n,), "bowl": (n,),
                "episode": (n,), "executed": (n,), "success": (n,)}
    for key, shape in expected.items():
        if tuple(arrays[key].shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {arrays[key].shape}")
    if n == 0:
        raise ValueError("no rows were collected")


def manifest_defaults(args):
    return {"checkpoint": args.checkpoint, "prompt": S.NEUTRAL_PROMPT, "stride": args.stride, "num_steps": args.num_steps,
            "rollouts": str(args.rollouts), "successful_only": args.successful_only, "executed_only": args.executed_only,
            "chunk_pad_mask": PAD_MASK_CONVENTION, "dataset_index": "100 * bowl + task_id", "demo_id": "bowl * 1000 + episode"}


def main(argv=None):
    args = parse_args(argv)
    pairs = [(t, b) for t in args.task_ids for b in args.bowl_ids]
    pending = [(t, b) for t, b in pairs if not I.task_npz_path(args.out, C.dataset_index(t, b)).exists()]
    entries = []
    if not pending:
        print("all requested (task, bowl) files exist; nothing to do")
    else:
        bundle = flow_api().load_pi05(args.checkpoint, args.device)
        tts = make_tts(args.voice)
        rng = np.random.default_rng(args.seed)
        for task_id, bowl in pairs:
            entries.append(build_pair(args, task_id, bowl, bundle, rng, tts))
    path = I.update_manifest(args.out, entries, manifest_defaults(args))
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()

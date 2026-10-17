#!/usr/bin/env python
"""Embed the instruction artifacts of a reversal dataset into its ``task_{i}.npz`` files (phase-2 venv).

Runs after ``scripts/vla/build_reversal_dataset.py`` in a venv with sentence-transformers + a transformers
that knows google/embeddinggemma-2 (``~/codes/LeaP/.venv`` on robot_dev6; the VLA venv cannot).  For every
``task_{i}.npz`` under ``--data``:

1. read ``task_{i}_instructions.json`` (texts, demo ids, sketch PNGs, audio WAVs; ``leap.vla.instructions``),
2. synthesize the missing audio WAVs with Piper when ``--voice`` is given (the builder may have done it),
3. embed: text variants once per task -> ``instr_text [D, T_VAR, 768]`` (rows repeated over the D demos),
   the sketch PNG of every demo -> ``instr_sketch [D, 1, 768]``, audio variants once per task ->
   ``instr_audio [D, A_VAR, 768]``; rows follow the record's ascending demo ids,
4. rewrite the npz atomically with the arrays added and ``instr_modalities`` (the modalities present),
5. record ``modalities`` and ``instr_variants`` for the task in ``manifest.json``.

A task whose npz already holds non-zero arrays for every requested modality is skipped unless ``--force``.

Usage:
    python scripts/vla/embed_vla_instructions.py --data data/vla/libero_spatial_reversal \\
        --voice data/voices/en_US-lessac-medium.onnx --device cuda [--tasks 0-9] [--modalities text sketch audio]
"""
import argparse
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leap.vla import instructions as I  # noqa: E402

NPZ_RE = re.compile(r"^task_(\d+)\.npz$")


def find_tasks(data_dir):
    """Task ids with a ``task_{i}.npz`` under ``data_dir``, ascending."""
    ids = []
    for path in Path(data_dir).iterdir():
        match = NPZ_RE.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


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


def record_demo_ids(record, arrays):
    """The record's demo ids, checked against the npz (same set, ascending order)."""
    ids = [int(d) for d in record["demo_ids"]]
    in_npz = sorted({int(d) for d in np.asarray(arrays["demo_id"]).tolist()})
    if ids != sorted(ids) or ids != in_npz:
        raise ValueError(f"task {record['task_id']}: record demo ids {ids} do not match the npz {in_npz}")
    return ids


def embed_record(data_dir, record, arrays, embedder, modalities, tts=None):
    """``{"instr_<m>": array}`` for the requested modalities of one task (see the module docstring)."""
    from PIL import Image
    from leap.steer.tts import load_wav
    data_dir = Path(data_dir)
    demos = len(record_demo_ids(record, arrays))
    out = {}
    if "text" in modalities:
        texts = [I.task_text(t) for t in record["texts"]]
        vectors = np.asarray(embedder.embed_text(texts), dtype=np.float32).reshape(len(texts), I.INSTR_DIM)
        out["instr_text"] = np.repeat(vectors[None], demos, axis=0)
    if "sketch" in modalities:
        if len(record["sketches"]) != demos:
            raise ValueError(f"task {record['task_id']}: {len(record['sketches'])} sketches for {demos} demos")
        images = []
        for name in record["sketches"]:
            with Image.open(data_dir / name) as image:
                images.append(image.convert("RGB"))
        vectors = np.asarray(embedder.embed_images(images), dtype=np.float32).reshape(demos, I.INSTR_DIM)
        out["instr_sketch"] = vectors[:, None, :]
    if "audio" in modalities:
        I.ensure_audio(data_dir, record, tts)
        clips = [load_wav(data_dir / name)[0] for name in record["audio"]]
        vectors = np.asarray(embedder.embed_audio(clips), dtype=np.float32).reshape(len(clips), I.INSTR_DIM)
        out["instr_audio"] = np.repeat(vectors[None], demos, axis=0)
    return out


def embed_task(data_dir, task_id, embedder, modalities, tts=None, force=False):
    """Fill ``instr_*`` of one task; returns the manifest entry."""
    npz_path = I.task_npz_path(data_dir, task_id)
    arrays = I.load_task_npz(npz_path)
    have = I.present_modalities(arrays)
    if not force and all(m in have for m in modalities):
        print(f"[task {task_id}] already embedded {have}, skipping")
        return {"task_id": task_id, "skipped": True}
    record = I.read_record(data_dir, task_id)
    started = time.time()
    arrays.update(embed_record(data_dir, record, arrays, embedder, modalities, tts))
    present = I.present_modalities(arrays)
    arrays["instr_modalities"] = np.asarray(present, dtype=str)
    I.save_task_npz(npz_path, arrays)
    variants = {m: int(arrays[f"instr_{m}"].shape[1]) for m in present}
    entry = {"task_id": task_id, "modalities": present, "instr_variants": variants,
             "audio_files": list(record.get("audio", [])), "embed_seconds": round(time.time() - started, 1)}
    print(f"[task {task_id}] embedded {modalities} -> present {present} in {entry['embed_seconds']}s", flush=True)
    return entry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="reversal dataset directory (task_{i}.npz + artifacts)")
    parser.add_argument("--tasks", default=None, help="task ids, e.g. 0-9 (default: every task_{i}.npz present)")
    parser.add_argument("--modalities", nargs="+", default=list(I.MODALITIES), choices=list(I.MODALITIES))
    parser.add_argument("--voice", default=None, help="Piper voice .onnx, needed when audio WAVs are missing")
    parser.add_argument("--device", default="cuda", help="device for the embedder")
    parser.add_argument("--embedder-model", default="google/embeddinggemma-2")
    parser.add_argument("--force", action="store_true", help="re-embed tasks that already have the arrays")
    return parser.parse_args(argv)


def make_embedder(args):
    from leap.steer.embedder import InstructionEmbedder
    return InstructionEmbedder(args.embedder_model, device=args.device)


def make_tts(voice):
    if not voice:
        return None
    from leap.steer.tts import PiperTTS
    return PiperTTS(voice)


def main(argv=None):
    args = parse_args(argv)
    tasks = parse_tasks(args.tasks) if args.tasks else find_tasks(args.data)
    if not tasks:
        raise FileNotFoundError(f"no task_{{i}}.npz under {args.data}")
    embedder = make_embedder(args)
    tts = make_tts(args.voice)
    entries = [embed_task(args.data, task_id, embedder, args.modalities, tts, args.force) for task_id in tasks]
    path = I.update_manifest(args.data, entries)
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()

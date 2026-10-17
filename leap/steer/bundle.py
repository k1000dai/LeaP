"""Instruction bundle layout and checks (see docs/superpowers/specs/2026-10-17-leap-multimodal-steering-design.md).

A bundle lives next to its Zarr dataset: ``data/<name>.zarr`` -> ``data/<name>.instr/`` and holds

- ``episode{i}/text.json``       {"seen": [...], "unseen": [...]} (copied from RoboTwin's generator)
- ``episode{i}/sketch.png``      head-camera frame 0 with the object circled and an arrow to the target
- ``episode{i}/sketch_meta.json`` pixel coordinates and side used for the sketch
- ``episode{i}/audio_{k}.wav``   k < AUDIO_VARIANTS Piper renderings of seen texts, 16 kHz mono
- ``manifest.json``              episode count and per-episode side written by scripts/make_instructions.py
- ``embeddings.npz``             written by scripts/embed_instructions.py with the arrays below

``embeddings.npz`` keys (all numpy, E = number of episodes):

- ``text``   float32 [E, TEXT_VARIANTS, INSTR_DIM], L2-normalised rows
- ``sketch`` float32 [E, 1, INSTR_DIM]
- ``audio``  float32 [E, AUDIO_VARIANTS, INSTR_DIM]
- ``side``   int64 [E], 0 = left, 1 = right (SIDE_INDEX)
- ``episode_ends`` int64 [E], must equal the Zarr's ``meta/episode_ends``

A modality array may be absent (for example a bundle built without TTS); at least one must be present.
"""
from pathlib import Path

import numpy as np

INSTR_DIM = 768
TEXT_VARIANTS = 10
AUDIO_VARIANTS = 3
MODALITIES = ("text", "sketch", "audio")
VARIANTS = {"text": TEXT_VARIANTS, "sketch": 1, "audio": AUDIO_VARIANTS}
SIDE_INDEX = {"left": 0, "right": 1}
EMBEDDINGS_FILE = "embeddings.npz"
MANIFEST_FILE = "manifest.json"


def bundle_path(zarr_path):
    """``data/x.zarr`` -> ``data/x.instr`` (the bundle directory next to the dataset)."""
    zarr_path = Path(zarr_path)
    if zarr_path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got {zarr_path}")
    return zarr_path.with_suffix(".instr")


def zarr_path_for_bundle(bundle_dir):
    """Inverse of bundle_path: ``data/x.instr`` -> ``data/x.zarr``."""
    bundle_dir = Path(bundle_dir)
    if bundle_dir.suffix != ".instr":
        raise ValueError(f"Expected a .instr directory, got {bundle_dir}")
    return bundle_dir.with_suffix(".zarr")


def episode_dir(bundle_dir, episode):
    return Path(bundle_dir) / f"episode{episode}"


def load_embeddings(bundle_dir):
    """Return the arrays of ``embeddings.npz`` as a dict of numpy arrays (modalities present only)."""
    path = Path(bundle_dir) / EMBEDDINGS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {EMBEDDINGS_FILE} in {bundle_dir}; run scripts/embed_instructions.py")
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def read_episode_ends(zarr_path):
    import zarr
    root = zarr.open_group(str(zarr_path), mode="r")
    return np.asarray(root["meta/episode_ends"], dtype=np.int64)


def validate_embeddings(embeddings, episode_ends=None):
    """Raise ValueError unless the dict has the documented keys, shapes, dtypes and normalisation."""
    for key in ("side", "episode_ends"):
        if key not in embeddings:
            raise ValueError(f"embeddings miss key '{key}'")
    present = [m for m in MODALITIES if m in embeddings]
    if not present:
        raise ValueError(f"embeddings contain none of the modalities {MODALITIES}")
    unknown = set(embeddings) - set(MODALITIES) - {"side", "episode_ends"}
    if unknown:
        raise ValueError(f"embeddings contain unknown keys {sorted(unknown)}")
    ends = np.asarray(embeddings["episode_ends"])
    side = np.asarray(embeddings["side"])
    episodes = len(ends)
    if ends.ndim != 1 or ends.dtype != np.int64 or episodes == 0:
        raise ValueError(f"episode_ends must be a non-empty int64 vector, got {ends.dtype} {ends.shape}")
    if ends[0] <= 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be positive and strictly increasing")
    if side.shape != (episodes,) or side.dtype != np.int64:
        raise ValueError(f"side must be int64 [{episodes}], got {side.dtype} {side.shape}")
    if not np.all(np.isin(side, list(SIDE_INDEX.values()))):
        raise ValueError(f"side values must be in {sorted(SIDE_INDEX.values())}")
    for modality in present:
        array = np.asarray(embeddings[modality])
        expected = (episodes, VARIANTS[modality], INSTR_DIM)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{modality} must be float32 {expected}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{modality} embeddings contain non-finite values")
        norms = np.linalg.norm(array, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{modality} embeddings are not L2-normalised (norms in [{norms.min():.4f}, {norms.max():.4f}])")
    if episode_ends is not None and not np.array_equal(ends, np.asarray(episode_ends, dtype=np.int64)):
        raise ValueError("episode_ends of the bundle differ from the Zarr dataset's")
    return present


def check_bundle(bundle_dir, zarr_path):
    """Load ``embeddings.npz`` and verify it against the Zarr; returns the modalities present."""
    embeddings = load_embeddings(bundle_dir)
    return validate_embeddings(embeddings, read_episode_ends(zarr_path))


def standardize_stats(embeddings, floor=1e-6):
    """Per-dimension mean and std pooled over every modality and variant present; std floored at ``floor``."""
    rows = [np.asarray(embeddings[m], dtype=np.float64).reshape(-1, INSTR_DIM)
            for m in MODALITIES if m in embeddings]
    if not rows:
        raise ValueError(f"embeddings contain none of the modalities {MODALITIES}")
    pooled = np.concatenate(rows)
    mean = pooled.mean(axis=0)
    std = np.maximum(pooled.std(axis=0), floor)
    return mean.astype(np.float32), std.astype(np.float32)

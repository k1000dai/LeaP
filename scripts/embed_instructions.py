"""Embed an instruction bundle with google/embeddinggemma-2 and write ``embeddings.npz``.

Reads the bundle made by scripts/make_instructions.py and writes, per modality present in every
episode, ``text [E, 10, 768]`` (first TEXT_VARIANTS seen strings, cycled when fewer),
``sketch [E, 1, 768]`` and ``audio [E, 3, 768]`` (float32, unit norm) plus ``side [E]`` and
``episode_ends [E]`` (from the Zarr next to the bundle).  Ends with leap.steer.bundle.check_bundle and
prints the mean cosine between left and right episodes per modality.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leap.steer.bundle import (AUDIO_VARIANTS, EMBEDDINGS_FILE, MANIFEST_FILE, MODALITIES, SIDE_INDEX,
                               TEXT_VARIANTS, check_bundle, episode_dir, read_episode_ends,
                               zarr_path_for_bundle)
from leap.steer.embedder import DEFAULT_MODEL, InstructionEmbedder


def episode_sides(bundle, episodes):
    manifest = bundle / MANIFEST_FILE
    if manifest.exists():
        sides = json.loads(manifest.read_text())["side"]
        if len(sides) != episodes:
            raise ValueError(f"{manifest} lists {len(sides)} episodes, expected {episodes}")
        return [int(s) for s in sides]
    sides = []
    for index in range(episodes):
        meta = json.loads((episode_dir(bundle, index) / "sketch_meta.json").read_text())
        sides.append(SIDE_INDEX[meta["side"]])
    return sides


def modality_inputs(bundle, episodes, modality):
    """Per-episode input lists for a modality, or None when no episode has its files."""
    inputs = []
    for index in range(episodes):
        folder = episode_dir(bundle, index)
        if modality == "text":
            path = folder / "text.json"
            if not path.exists():
                inputs.append(None)
                continue
            seen = json.loads(path.read_text())["seen"]
            if not seen:
                raise ValueError(f"episode {index}: text.json has no seen texts")
            if len(seen) < TEXT_VARIANTS:
                print(f"episode {index}: {len(seen)} seen texts, cycling to {TEXT_VARIANTS}")
            inputs.append([seen[k % len(seen)] for k in range(TEXT_VARIANTS)])
        elif modality == "sketch":
            path = folder / "sketch.png"
            inputs.append([str(path)] if path.exists() else None)
        else:
            paths = [folder / f"audio_{k}.wav" for k in range(AUDIO_VARIANTS)]
            inputs.append([str(p) for p in paths] if all(p.exists() for p in paths) else None)
    missing = [i for i, value in enumerate(inputs) if value is None]
    if len(missing) == episodes:
        return None
    if missing:
        raise FileNotFoundError(f"{modality}: files missing for episodes {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return inputs


def side_cosines(array, side):
    """(mean cosine left-vs-right, mean cosine within the same side) over episode-mean embeddings."""
    per_episode = array.mean(axis=1)
    per_episode /= np.maximum(np.linalg.norm(per_episode, axis=1, keepdims=True), 1e-12)
    similarity = per_episode @ per_episode.T
    side = np.asarray(side)
    cross = similarity[np.ix_(side == 0, side == 1)]
    within = [similarity[np.ix_(side == s, side == s)][np.triu_indices(int((side == s).sum()), 1)]
              for s in (0, 1) if (side == s).sum() > 1]
    within = np.concatenate(within) if within else np.array([np.nan])
    return float(cross.mean()) if cross.size else float("nan"), float(within.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, default=None, help="Default: <bundle>.instr -> .zarr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default=None, help="Default: bfloat16 on cuda, float32 on cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    zarr_path = args.zarr or zarr_path_for_bundle(bundle)
    episode_ends = read_episode_ends(zarr_path)
    episodes = len(episode_ends)
    sides = episode_sides(bundle, episodes)
    dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    embedder = InstructionEmbedder(model=args.model, device=args.device, dtype=dtype, batch_size=args.batch_size)
    encoders = {"text": embedder.embed_text, "sketch": embedder.embed_images, "audio": embedder.embed_audio}

    arrays = {"side": np.asarray(sides, dtype=np.int64), "episode_ends": episode_ends.astype(np.int64)}
    for modality in [m.strip() for m in args.modalities.split(",") if m.strip()]:
        if modality not in MODALITIES:
            parser.error(f"unknown modality {modality}")
        inputs = modality_inputs(bundle, episodes, modality)
        if inputs is None:
            print(f"{modality}: no files in the bundle, skipped")
            continue
        variants = len(inputs[0])
        flat = [item for episode in inputs for item in episode]
        embedded = encoders[modality](flat).reshape(episodes, variants, -1).astype(np.float32)
        arrays[modality] = embedded
        cross, within = side_cosines(embedded, sides)
        print(f"{modality}: {embedded.shape}, mean cosine left-vs-right {cross:.4f}, within side {within:.4f}")
    output = bundle / EMBEDDINGS_FILE
    np.savez(output, **arrays)
    present = check_bundle(bundle, zarr_path)
    print(f"wrote {output} with modalities {present}, {episodes} episodes "
          f"(left {sides.count(0)}, right {sides.count(1)})")


if __name__ == "__main__":
    main()

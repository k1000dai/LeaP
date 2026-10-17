"""Build the instruction bundle (text, sketch, audio files) of a RoboTwin dataset.

For every episode i of the raw place_a2b_side collection:

- ``episode{i}/text.json``: copy of the official generator's ``instructions/episode{i}.json``
  (looked up under <raw>/instructions, then <robotwin>/data/<task>/<setting>/instructions, where the
  pinned generator writes it).  When neither exists (the collector's post-step runs the system
  ``python`` and cannot resolve absolute save paths), the same official module
  ``description/utils/generate_episode_instructions.py`` is imported and run on
  ``scene_info.json["episode_{i}"]["info"]`` with ``random`` seeded by the episode index;
- ``episode{i}/sketch.png`` + ``sketch_meta.json``: head-camera frame 0 of ``data/episode{i}.hdf5``
  with object A circled and an arrow to the target, from ``scene_info.json["episode_{i}"]["steer"]``;
- ``episode{i}/audio_{k}.wav``: Piper TTS of the first AUDIO_VARIANTS seen texts (16 kHz mono).

Existing files are kept (idempotent).  The bundle directory is ``bundle_path(--zarr)`` and
``manifest.json`` records the episode count and side per episode.  Run scripts/embed_instructions.py next.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leap.steer.bundle import (AUDIO_VARIANTS, MANIFEST_FILE, SIDE_INDEX, TEXT_VARIANTS, bundle_path,
                               episode_dir, read_episode_ends)
from leap.steer.sketch import draw_sketch, sketch_pixels
from leap.steer.tts import PiperTTS, save_wav

DEFAULT_VOICE = ROOT / "data/voices/en_US-lessac-medium.onnx"
DEFAULT_ROBOTWIN = ROOT / "third_party/RoboTwin"


def find_instructions_dir(raw, robotwin, explicit=None):
    """Directory holding episode{i}.json instruction files for this raw collection, or None."""
    raw = Path(raw).resolve()
    candidates = [Path(explicit)] if explicit else [raw / "instructions",
                                                     Path(robotwin) / "data" / raw.parent.name / raw.name / "instructions"]
    for candidate in candidates:
        if (candidate / "episode0.json").exists():
            return candidate
    if explicit:
        raise FileNotFoundError(f"No episode0.json in --instructions {explicit}")
    return None


_generator_module = {}


def official_generator(robotwin):
    """The pinned RoboTwin generator module (its relative asset paths resolve from its own location)."""
    path = Path(robotwin).resolve() / "description/utils/generate_episode_instructions.py"
    if path not in _generator_module:
        if not path.exists():
            raise FileNotFoundError(f"Official instruction generator not found: {path}")
        spec = importlib.util.spec_from_file_location("robotwin_generate_episode_instructions", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _generator_module[path] = module
    return _generator_module[path]


def generate_texts(robotwin, task_name, info, language_num, seed):
    """{"seen": [...], "unseen": [...]} for one episode from the official generator (seeded for reproducibility)."""
    generator = official_generator(robotwin)
    random.seed(seed)
    results = generator.generate_episode_descriptions(task_name, [dict(info)], language_num)
    if not results or not results[0].get("seen"):
        raise ValueError(f"The official generator produced no instructions for {task_name} with {info}")
    return {"seen": results[0]["seen"], "unseen": results[0].get("unseen", [])}


def load_frame0(hdf5_path):
    """(rgb PIL image, intrinsic_cv 3x3, extrinsic_cv 3x4) of the first head-camera frame."""
    import h5py
    from PIL import Image
    with h5py.File(hdf5_path, "r") as episode:
        camera = episode["observation/head_camera"]
        rgb = Image.open(io.BytesIO(bytes(camera["rgb"][0]))).convert("RGB")
        intrinsic = np.asarray(camera["intrinsic_cv"][0])
        extrinsic = np.asarray(camera["extrinsic_cv"][0])
    return rgb, intrinsic, extrinsic


def make_sketch(hdf5_path, steer, sketch_path, meta_path):
    rgb, intrinsic, extrinsic = load_frame0(hdf5_path)
    object_px, target_px = sketch_pixels(intrinsic, extrinsic, steer["A_xyz"], steer["target_xyz"], *rgb.size)
    draw_sketch(rgb, object_px, target_px).save(sketch_path)
    meta_path.write_text(json.dumps({"frame": 0, "side": steer["side"], "object_px": list(object_px),
                                     "target_px": list(target_px), "A_xyz": steer["A_xyz"],
                                     "target_xyz": steer["target_xyz"], "image_size": list(rgb.size)}, indent=2))


def make_episode(index, raw, instructions_dir, bundle, tts, robotwin=DEFAULT_ROBOTWIN, task_name=None,
                 language_num=100):
    """Create the files of one episode; returns (side name, list of actions)."""
    actions = []
    target = episode_dir(bundle, index)
    target.mkdir(parents=True, exist_ok=True)
    scene_info = json.loads((raw / "scene_info.json").read_text())
    episode_info = scene_info[f"episode_{index}"]
    steer = episode_info["steer"]
    text_path = target / "text.json"
    if not text_path.exists():
        if instructions_dir is not None:
            shutil.copyfile(instructions_dir / f"episode{index}.json", text_path)
        else:
            texts = generate_texts(robotwin, task_name or raw.parent.name, episode_info["info"], language_num, index)
            text_path.write_text(json.dumps(texts, indent=2))
        actions.append("text")
    texts = json.loads(text_path.read_text())
    if len(texts.get("seen", [])) < TEXT_VARIANTS:
        print(f"episode {index}: only {len(texts.get('seen', []))} seen texts (< {TEXT_VARIANTS}); embedding will cycle")
    sketch_path = target / "sketch.png"
    if not sketch_path.exists():
        make_sketch(raw / "data" / f"episode{index}.hdf5", steer, sketch_path, target / "sketch_meta.json")
        actions.append("sketch")
    if tts is not None:
        seen = texts["seen"]
        if not seen:
            raise ValueError(f"episode {index}: no seen texts to synthesise")
        for k in range(AUDIO_VARIANTS):
            wav_path = target / f"audio_{k}.wav"
            if wav_path.exists():
                continue
            save_wav(wav_path, tts.synthesize(seen[k % len(seen)]))
            actions.append(f"audio_{k}")
    return steer["side"], actions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="Raw collection, e.g. data/raw/place_a2b_side/leap_clean")
    parser.add_argument("--zarr", type=Path, required=True, help="Converted dataset; the bundle goes next to it")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--episodes", type=int, default=None, help="Default: the Zarr's episode count")
    parser.add_argument("--instructions", type=Path, default=None, help="Override the instructions directory")
    parser.add_argument("--robotwin", type=Path, default=DEFAULT_ROBOTWIN)
    parser.add_argument("--no-audio", action="store_true", help="Skip TTS (bundle without audio)")
    parser.add_argument("--bundle", type=Path, default=None, help="Override the bundle directory")
    parser.add_argument("--task", default=None, help="RoboTwin task name (default: parent directory of --raw)")
    parser.add_argument("--language-num", type=int, default=100,
                        help="Instructions per episode when generating them (RoboTwin's language_num)")
    args = parser.parse_args()

    raw = args.raw.resolve()
    bundle = args.bundle or bundle_path(args.zarr)
    episodes = args.episodes
    if args.zarr.exists():
        zarr_episodes = len(read_episode_ends(args.zarr))
        if episodes is None:
            episodes = zarr_episodes
        elif episodes != zarr_episodes:
            print(f"warning: --episodes {episodes} differs from the Zarr's {zarr_episodes} episodes")
    elif episodes is None:
        parser.error(f"{args.zarr} does not exist; pass --episodes")
    instructions_dir = find_instructions_dir(raw, args.robotwin, args.instructions)
    task_name = args.task or raw.parent.name
    if instructions_dir is None:
        print(f"no instructions/episode*.json found; generating texts for '{task_name}' with the official generator")
    tts = None if args.no_audio else PiperTTS(args.voice)
    bundle.mkdir(parents=True, exist_ok=True)

    sides, done = [], {"text": 0, "sketch": 0, "audio": 0}
    for index in range(episodes):
        side, actions = make_episode(index, raw, instructions_dir, bundle, tts, args.robotwin, task_name,
                                     args.language_num)
        sides.append(side)
        for action in actions:
            done[action.split("_")[0]] += 1
        print(f"episode {index}: side={side} " + (", ".join(actions) if actions else "up to date"))
    manifest = {"episodes": episodes, "side": [SIDE_INDEX[s] for s in sides], "side_name": sides,
                "text_variants": TEXT_VARIANTS, "audio_variants": 0 if args.no_audio else AUDIO_VARIANTS,
                "raw": str(raw), "zarr": str(args.zarr.resolve()), "task": task_name,
                "instructions_dir": None if instructions_dir is None else str(instructions_dir),
                "voice": None if args.no_audio else str(args.voice)}
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    counts = {s: sides.count(s) for s in SIDE_INDEX}
    print(f"bundle {bundle}: {episodes} episodes, sides {counts}, created {done}")


if __name__ == "__main__":
    main()

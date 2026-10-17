"""Steering instructions for LIBERO-Spatial: text, sketch and audio for one task + scene, and their embeddings.

Shared by ``scripts/vla/build_reversal_dataset.py`` (writes the instruction artifacts), ``scripts/vla/
embed_vla_instructions.py`` (embeds them) and the evaluator, so all describe the target bowl the same way:

- text:   the task's own language (``task_text``), e.g. "pick up the black bowl between the plate and
          the ramekin and place it on the plate";
- sketch: the policy-view agentview frame with a red circle on ``akita_black_bowl_1`` and a blue arrow
          to ``plate_1`` (``build_sketch`` -> ``leap.steer.sketch.render_sketch``), positions projected
          with ``leap.vla.libero_scene.agentview_camera``;
- audio:  Piper TTS of the text, float32 mono at 16 kHz (``build_audio`` -> ``leap.steer.tts.PiperTTS``).

``embed_instruction`` turns one of them into a 768-d vector with ``leap.steer.embedder.InstructionEmbedder``
(unit L2 norm); ``InstructionCache`` builds each (modality, key) once per episode and reuses it at every
policy call.

Dataset-side artifacts (one record per task in the reversal dataset directory, see ``write_record``):

- ``task_{i}_instructions.json``: ``{"task_id", "task_language", "texts": [variants], "demo_ids": [...],
  "sketches": ["task_{i}_episode{d}_sketch.png", ...] (one per demo, ascending demo id),
  "audio": ["task_{i}_audio{v}.wav", ...] (one per text variant, empty until synthesized), "sample_rate"}``
- the sketch PNGs and 16 kHz mono WAVs next to it.

Only numpy/PIL/json are imported at module level; sketch/tts/embedder are imported lazily.
"""
import json
import os
from pathlib import Path

import numpy as np

MODALITIES = ("text", "sketch", "audio")
INSTR_DIM = 768
SAMPLE_RATE = 16000


def task_text(task_language):
    """The instruction text: the task's own language, stripped (no paraphrasing)."""
    text = " ".join(str(task_language).split())
    if not text:
        raise ValueError("task language is empty")
    return text


def build_sketch(agentview_rgb, cam, layout):
    """PIL RGB sketch: circle the target bowl and draw an arrow to the plate on the policy-view frame.

    ``cam`` is ``(intrinsic_cv, extrinsic_cv)`` from ``agentview_camera``; ``layout`` is ``scene_layout``'s
    dict (needs ``akita_black_bowl_1`` and ``plate_1``).
    """
    from leap.steer.sketch import render_sketch
    from leap.vla.libero_scene import PLATE, TARGET_BOWL
    intrinsic, extrinsic = cam
    return render_sketch(np.asarray(agentview_rgb, dtype=np.uint8), intrinsic, extrinsic,
                         np.asarray(layout[TARGET_BOWL], dtype=np.float64),
                         np.asarray(layout[PLATE], dtype=np.float64))


def build_audio(text, tts):
    """Synthesize ``text`` with a ``PiperTTS``-like object (``synthesize(text) -> float32 mono 16 kHz``)."""
    wav = np.asarray(tts.synthesize(task_text(text)), dtype=np.float32).reshape(-1)
    if wav.size == 0:
        raise ValueError(f"TTS produced no audio for {text!r}")
    return wav


def _single(vectors, modality):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 2 and vectors.shape[0] == 1:
        vectors = vectors[0]
    if vectors.shape != (INSTR_DIM,):
        raise ValueError(f"{modality} embedding must be [{INSTR_DIM}], got {vectors.shape}")
    return vectors


def embed_instruction(embedder, modality, *, text=None, sketch=None, audio=None):
    """Embed one instruction: ``modality`` picks which of ``text`` (str), ``sketch`` (PIL) or ``audio`` (wave) is used.

    Returns a float32 [768] vector from ``embedder.embed_text/embed_images/embed_audio``.
    """
    if modality == "text":
        if text is None:
            raise ValueError("text modality needs text=")
        return _single(embedder.embed_text([task_text(text)]), modality)
    if modality == "sketch":
        if sketch is None:
            raise ValueError("sketch modality needs sketch=")
        return _single(embedder.embed_images([sketch]), modality)
    if modality == "audio":
        if audio is None:
            raise ValueError("audio modality needs audio=")
        return _single(embedder.embed_audio([np.asarray(audio, dtype=np.float32)]), modality)
    raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}")


def embed_all(embedder, text, sketch=None, audio=None):
    """``{"text": [768], "sketch": [768], "audio": [768]}`` for whichever inputs are given (text always)."""
    out = {"text": embed_instruction(embedder, "text", text=text)}
    if sketch is not None:
        out["sketch"] = embed_instruction(embedder, "sketch", sketch=sketch)
    if audio is not None:
        out["audio"] = embed_instruction(embedder, "audio", audio=audio)
    return out


class InstructionCache:
    """Per-episode cache: build and embed each (modality, key) once, return the same vector afterwards.

    ``get(modality, key, text=..., sketch=...)`` embeds text/sketch directly; for ``audio`` it synthesizes
    ``text`` through ``tts`` first.  ``key`` is whatever identifies the episode (e.g. ``(task_id, episode)``).
    """

    def __init__(self, embedder, tts=None):
        self.embedder = embedder
        self.tts = tts
        self._vectors = {}
        self._audio = {}

    def clear(self):
        self._vectors.clear()
        self._audio.clear()

    def audio(self, key, text):
        """The synthesized waveform for ``key`` (built once)."""
        if key not in self._audio:
            if self.tts is None:
                raise ValueError("InstructionCache needs a tts to build audio instructions")
            self._audio[key] = build_audio(text, self.tts)
        return self._audio[key]

    def get(self, modality, key, *, text=None, sketch=None):
        cache_key = (modality, key)
        if cache_key not in self._vectors:
            if modality == "audio":
                vector = embed_instruction(self.embedder, "audio", audio=self.audio(key, text))
            else:
                vector = embed_instruction(self.embedder, modality, text=text, sketch=sketch)
            self._vectors[cache_key] = vector
        return self._vectors[cache_key]


# --- dataset-side instruction records (shared by the builder and the embed script) --------------------------

def record_path(data_dir, task_id):
    return Path(data_dir) / f"task_{task_id}_instructions.json"


def sketch_file(task_id, demo_id):
    return f"task_{task_id}_episode{demo_id}_sketch.png"


def audio_file(task_id, variant):
    return f"task_{task_id}_audio{variant}.wav"


def new_record(task_id, task_language, demo_ids, texts=None):
    """A fresh instruction record for a task: one text variant (the language) and one sketch per demo."""
    language = task_text(task_language)
    demo_ids = [int(d) for d in demo_ids]
    return {"task_id": int(task_id), "task_language": language, "texts": list(texts or [language]),
            "demo_ids": demo_ids, "sketches": [sketch_file(task_id, d) for d in demo_ids],
            "audio": [], "sample_rate": SAMPLE_RATE}


def write_record(data_dir, record):
    path = record_path(data_dir, record["task_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    with open(temp, "w") as handle:
        json.dump(record, handle, indent=2)
    os.replace(temp, path)
    return path


def read_record(data_dir, task_id):
    path = record_path(data_dir, task_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/vla/build_reversal_dataset.py first")
    with open(path) as handle:
        return json.load(handle)


def ensure_audio(data_dir, record, tts):
    """Synthesize the WAV of every text variant that has none yet (``record["audio"]``), save and update the record.

    Returns the record; a no-op when every variant already has a file.  ``tts`` may be None only when nothing
    is missing.
    """
    from leap.steer.tts import save_wav
    data_dir = Path(data_dir)
    files = list(record.get("audio", []))
    if len(files) >= len(record["texts"]) and all((data_dir / f).exists() for f in files):
        return record
    if tts is None:
        raise ValueError(f"task {record['task_id']}: audio files missing and no TTS voice given")
    files = []
    for variant, text in enumerate(record["texts"]):
        name = audio_file(record["task_id"], variant)
        save_wav(data_dir / name, build_audio(text, tts), SAMPLE_RATE)
        files.append(name)
    record["audio"] = files
    write_record(data_dir, record)
    return record


# --- reversal dataset files -------------------------------------------------------------------------------------

def task_npz_path(data_dir, task_id):
    return Path(data_dir) / f"task_{task_id}.npz"


def load_task_npz(path):
    """Read a ``task_{i}.npz`` as a dict of arrays (string / int scalars unwrapped)."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    for key in ("task_id", "num_steps"):
        if key in out:
            out[key] = int(out[key])
    for key in ("task_language", "prompt"):
        if key in out:
            out[key] = str(out[key])
    return out


def save_task_npz(path, arrays):
    """Write the npz atomically (temp file + rename) so an interrupted run never leaves a partial task file."""
    path = Path(path)
    temp = path.with_suffix(".npz.tmp")
    with open(temp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, path)
    return path


def present_modalities(arrays):
    """Modalities whose ``instr_<m>`` array exists and is not all zeros."""
    return [m for m in MODALITIES if f"instr_{m}" in arrays and np.abs(arrays[f"instr_{m}"]).sum() > 0]


def update_manifest(data_dir, entries, defaults=None):
    """Merge per-task entries into ``<data_dir>/manifest.json`` (``tasks[str(task_id)]``), keeping other fields."""
    path = Path(data_dir) / "manifest.json"
    manifest = dict(defaults or {})
    manifest.setdefault("tasks", {})
    if path.exists():
        with open(path) as handle:
            existing = json.load(handle)
        existing_tasks = existing.pop("tasks", {})
        manifest.update(existing)
        manifest["tasks"].update(existing_tasks)
    for entry in entries:
        if entry.get("skipped"):
            continue
        current = manifest["tasks"].get(str(entry["task_id"]), {})
        current.update(entry)
        manifest["tasks"][str(entry["task_id"])] = current
    temp = path.with_suffix(".json.tmp")
    with open(temp, "w") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temp, path)
    return path

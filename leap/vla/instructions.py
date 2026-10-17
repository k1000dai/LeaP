"""Steering instructions for LIBERO-Spatial: text, sketch and audio for one task + scene, and their embeddings.

Shared by ``scripts/vla/build_reversal_dataset.py`` and the evaluator so both describe the target bowl
the same way:

- text:   the task's own language (``task_text``), e.g. "pick up the black bowl between the plate and
          the ramekin and place it on the plate";
- sketch: the policy-view agentview frame with a red circle on ``akita_black_bowl_1`` and a blue arrow
          to ``plate_1`` (``build_sketch`` -> ``leap.steer.sketch.render_sketch``), positions projected
          with ``leap.vla.libero_scene.agentview_camera``;
- audio:  Piper TTS of the text, float32 mono at 16 kHz (``build_audio`` -> ``leap.steer.tts.PiperTTS``).

``embed_instruction`` turns one of them into a 768-d vector with ``leap.steer.embedder.InstructionEmbedder``
(unit L2 norm); ``InstructionCache`` builds each (modality, key) once per episode and reuses it at every
policy call.  Only numpy/PIL are imported at module level; sketch/tts/embedder are imported lazily.
"""
import numpy as np

MODALITIES = ("text", "sketch", "audio")
INSTR_DIM = 768


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

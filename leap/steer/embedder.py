"""One embedding space for text, sketch images and speech: google/embeddinggemma-2 via sentence-transformers.

Input formats (sentence-transformers >= 6.1, transformers >= 5.18):

- text:  ``model.encode(texts, prompt_name="SearchQuery")`` (prefix ``task: search result | query: ``)
- image: ``model.encode({"text": "<|image|>", "image": [path]})``
- audio: ``model.encode({"text": "<|audio|>", "audio": [path]})`` with 16 kHz mono WAV files

Every method returns float32 [N, INSTR_DIM] with unit L2 norm.  The heavy imports (torch,
sentence_transformers) happen on first use so that importing this module stays cheap.
"""
from pathlib import Path
import tempfile

import numpy as np

from .bundle import INSTR_DIM
from .tts import SAMPLE_RATE, save_wav

DEFAULT_MODEL = "google/embeddinggemma-2"
TEXT_PROMPT = "SearchQuery"
IMAGE_TEMPLATE = "<|image|>"
AUDIO_TEMPLATE = "<|audio|>"


class InstructionEmbedder:
    def __init__(self, model=DEFAULT_MODEL, device="cuda", dtype="bfloat16", batch_size=16):
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch_dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
            self._model = SentenceTransformer(self.model_name, device=self.device,
                                              model_kwargs={"torch_dtype": torch_dtype})
        return self._model

    def _encode(self, inputs, **kwargs):
        model = self._load()
        if not inputs:
            return np.zeros((0, INSTR_DIM), dtype=np.float32)
        embeddings = model.encode(inputs, batch_size=self.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False, **kwargs)
        embeddings = np.asarray(embeddings, dtype=np.float32).reshape(len(inputs), -1)
        if embeddings.shape[1] != INSTR_DIM:
            raise ValueError(f"Expected {INSTR_DIM}-d embeddings, got {embeddings.shape[1]}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_text(self, texts):
        return self._encode([str(t) for t in texts], prompt_name=TEXT_PROMPT)

    def embed_images(self, images):
        """PIL images or paths; PIL images are written to temporary PNG files for the processor."""
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="leap-instr-img-") as temp:
            paths = []
            for index, image in enumerate(images):
                if isinstance(image, Image.Image):
                    path = Path(temp) / f"image{index}.png"
                    image.convert("RGB").save(path)
                    paths.append(str(path))
                else:
                    paths.append(str(image))
            return self._encode([{"text": IMAGE_TEMPLATE, "image": [p]} for p in paths])

    def embed_audio(self, clips):
        """WAV paths or float waveforms at 16 kHz mono; arrays are written to temporary WAV files."""
        with tempfile.TemporaryDirectory(prefix="leap-instr-audio-") as temp:
            paths = []
            for index, clip in enumerate(clips):
                if isinstance(clip, (str, Path)):
                    paths.append(str(clip))
                else:
                    path = Path(temp) / f"clip{index}.wav"
                    save_wav(path, np.asarray(clip, dtype=np.float32), SAMPLE_RATE)
                    paths.append(str(path))
            return self._encode([{"text": AUDIO_TEMPLATE, "audio": [p]} for p in paths])

"""Official RoboTwin policy interface: get_model, eval and reset_model.

Instruction-conditioned checkpoints (``policy.instr_mode != "none"``) receive one
768-d instruction embedding per episode, built from the live environment by
``InstructionBuilder`` at the first ``eval`` call after ``reset_model``.
"""
from collections import deque
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch

from leap.robotwin.checkpoint import load_policy

INSTRUCTION_MODALITIES = ("none", "text", "sketch", "audio")
INSTR_DIM = 768
DEFAULT_EMBEDDER = "google/embeddinggemma-2"


def encode_obs(observation):
    return {"agent_pos": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "point_cloud": np.asarray(observation["pointcloud"], dtype=np.float32)}


# ---- lazy leap.steer entry points (imported on first use; patched by tests) ----
def _make_embedder(model, device):
    from leap.steer.embedder import InstructionEmbedder
    return InstructionEmbedder(model=model, device=device)


def _make_tts(voice_path):
    from leap.steer.tts import PiperTTS
    return PiperTTS(voice_path)


def _save_wav(path, wav):
    from leap.steer.tts import save_wav
    save_wav(path, wav, sr=16000)


def _render_sketch(rgb, intrinsic_cv, extrinsic_cv, object_xyz, target_xyz):
    from leap.steer.sketch import render_sketch
    return render_sketch(rgb, intrinsic_cv, extrinsic_cv, object_xyz, target_xyz)


class InstructionBuilder:
    """Build and embed one instruction per episode from the live task environment.

    The embedder and TTS are created on first use. ``build`` returns a float32
    vector of ``INSTR_DIM`` and, with ``dump_dir``, writes
    ``episode_{n}/{instruction.txt, meta.json}`` plus ``sketch.png`` or
    ``audio.wav`` for the sketch/audio modalities (``n`` counts builds).
    """

    def __init__(self, modality, embedder=DEFAULT_EMBEDDER, tts_voice=None, device="cpu", dump_dir=None):
        if modality not in INSTRUCTION_MODALITIES[1:]:
            raise ValueError(f"instruction modality must be one of {INSTRUCTION_MODALITIES[1:]}, got {modality!r}")
        if modality == "audio":
            if not tts_voice:
                raise ValueError("tts_voice (Piper voice .onnx) is required for audio instructions")
            if not Path(tts_voice).is_file():
                raise FileNotFoundError(tts_voice)
        self.modality = modality
        self.embedder_name = embedder
        self.tts_voice = tts_voice
        self.device = device
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.episode = 0
        self._embedder = None
        self._tts = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _make_embedder(self.embedder_name, self.device)
        return self._embedder

    @property
    def tts(self):
        if self._tts is None:
            self._tts = _make_tts(self.tts_voice)
        return self._tts

    def build(self, task_env, observation):
        """Embed the current episode's instruction (text, sketch or speech)."""
        text = str(task_env.get_instruction())
        layout = task_env.steer_layout()
        with tempfile.TemporaryDirectory(prefix="leap-instr-") as temporary:
            directory = Path(temporary)
            if self.dump_dir is not None:
                directory = self.dump_dir / f"episode_{self.episode}"
                directory.mkdir(parents=True, exist_ok=True)
            embedding = self._embed(text, layout, observation, directory)
            if self.dump_dir is not None:
                self._write_meta(directory, text, layout, embedding)
        self.episode += 1
        return embedding

    def _embed(self, text, layout, observation, directory):
        if self.modality == "text":
            vectors = self.embedder.embed_text([text])
        elif self.modality == "sketch":
            camera = observation["observation"]["head_camera"]
            if "rgb" not in camera:
                raise KeyError("head_camera rgb missing from the observation; enable data_type.rgb in the task config")
            sketch = _render_sketch(np.asarray(camera["rgb"]), np.asarray(camera["intrinsic_cv"]),
                                    np.asarray(camera["extrinsic_cv"]), layout["A_xyz"], layout["target_xyz"])
            if self.dump_dir is not None:
                sketch.save(directory / "sketch.png")
            vectors = self.embedder.embed_images([sketch])
        else:
            # Embed from a 16 kHz wav file so evaluation matches the offline bundle.
            path = directory / "audio.wav"
            _save_wav(path, self.tts.synthesize(text))
            vectors = self.embedder.embed_audio([str(path)])
        embedding = np.asarray(vectors, dtype=np.float32).reshape(-1)
        if embedding.shape != (INSTR_DIM,):
            raise ValueError(f"expected one {INSTR_DIM}-d embedding, got shape {np.shape(vectors)}")
        return embedding

    def _write_meta(self, directory, text, layout, embedding):
        (directory / "instruction.txt").write_text(text + "\n")
        meta = {"episode": self.episode, "modality": self.modality, "instruction": text,
                "layout": layout, "embedding_norm": float(np.linalg.norm(embedding))}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable))


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


def make_instruction_builder(usr_args, instr_mode):
    """InstructionBuilder for ``usr_args`` or None, checked against the checkpoint's instr_mode."""
    modality = usr_args.get("instruction_modality", "none") or "none"
    if modality not in INSTRUCTION_MODALITIES:
        raise ValueError(f"instruction_modality must be one of {INSTRUCTION_MODALITIES}, got {modality!r}")
    if instr_mode != "none" and modality == "none":
        raise ValueError(f"checkpoint is instruction-conditioned (instr_mode={instr_mode!r}); "
                         "set instruction_modality to text, sketch or audio")
    if modality == "none":
        return None
    if instr_mode == "none":
        warnings.warn(f"instruction_modality={modality!r} ignored: the checkpoint has instr_mode 'none'")
        return None
    return InstructionBuilder(modality, embedder=usr_args.get("embedder") or DEFAULT_EMBEDDER,
                              tts_voice=usr_args.get("tts_voice"), device=usr_args.get("device", "cuda:0"),
                              dump_dir=usr_args.get("instr_dump_dir"))


class Model:
    def __init__(self, policy, instructions=None):
        self.policy = policy
        self.instructions = instructions  # InstructionBuilder or None
        self.obs = deque(maxlen=policy.n_obs_steps)
        self.instr = None  # the episode's embedding, np.float32 [INSTR_DIM]

    def get_action(self):
        history = list(self.obs)
        history = [history[0]] * (self.policy.n_obs_steps - len(history)) + history
        tensors = {key: torch.as_tensor(np.stack([obs[key] for obs in history]),
                                        device=self.policy.device).unsqueeze(0)
                   for key in history[0]}
        pc = tensors["point_cloud"]
        channels = self.policy.normalizer["point_cloud"].params_dict["scale"].numel()
        if pc.shape[-1] == 3 and channels == 6 and not self.policy.use_pc_color:
            tensors["point_cloud"] = torch.cat((pc, torch.zeros_like(pc)), dim=-1)
        if self.instructions is not None:
            tensors["instr"] = torch.as_tensor(self.instr, device=self.policy.device)[None]
        with torch.no_grad():
            return self.policy.predict_action(tensors)["action"][0].cpu().numpy()


def get_model(usr_args):
    policy = load_policy(usr_args["ckpt_path"], device=usr_args.get("device", "cuda:0"))
    # Optional inference-time prior intervention (see LeaPPolicy.set_prior_intervention).
    policy.set_prior_intervention(usr_args.get("intervention"))
    instructions = make_instruction_builder(usr_args, getattr(policy, "instr_mode", "none"))
    return Model(policy, instructions)


def eval(TASK_ENV, model, observation):
    if not model.obs:
        if model.instructions is not None:  # first call of the episode: build from the first frame
            model.instr = model.instructions.build(TASK_ENV, observation)
        model.obs.append(encode_obs(observation))
    for action in model.get_action():
        TASK_ENV.take_action(action, action_type="qpos")
        model.obs.append(encode_obs(TASK_ENV.get_obs()))


def reset_model(model):
    model.obs.clear()
    model.instr = None
    model.policy.reset()

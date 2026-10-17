"""CPU checks for the per-episode instruction builder in the RoboTwin deploy adapter."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from leap.robotwin import deploy
from leap.robotwin.deploy import INSTR_DIM, InstructionBuilder, Model

INSTRUCTION = "Place the bottle to the left of the cup."
MODALITY_INDEX = {"text": 0, "sketch": 1, "audio": 2}


def modality_vector(modality):
    vector = np.zeros(INSTR_DIM, dtype=np.float32)
    vector[MODALITY_INDEX[modality]] = 1.0
    return vector


class FakeEmbedder:
    """Deterministic unit vectors keyed by modality; records every call."""

    def __init__(self, model, device):
        self.model, self.device, self.calls = model, device, []

    def _embed(self, modality, inputs):
        self.calls.append((modality, list(inputs)))
        return np.stack([modality_vector(modality)] * len(inputs))

    def embed_text(self, texts):
        return self._embed("text", texts)

    def embed_images(self, images):
        return self._embed("sketch", images)

    def embed_audio(self, audios):
        self.audio_existed = [Path(audio).is_file() for audio in audios]
        return self._embed("audio", audios)


class FakeTTS:
    def __init__(self, voice_path):
        self.voice_path, self.texts = voice_path, []

    def synthesize(self, text):
        self.texts.append(text)
        return np.zeros(1600, dtype=np.float32)


def fake_save_wav(path, wav):
    Path(path).write_bytes(np.asarray(wav, dtype=np.float32).tobytes())


def fake_render_sketch(rgb, intrinsic_cv, extrinsic_cv, object_xyz, target_xyz):
    fake_render_sketch.calls.append((rgb, intrinsic_cv, extrinsic_cv, object_xyz, target_xyz))
    return Image.new("RGB", (8, 6), (255, 0, 0))


fake_render_sketch.calls = []


def make_observation(step=0):
    camera = {"rgb": np.full((6, 8, 3), step, dtype=np.uint8),
              "intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(3, 4)}
    return {"observation": {"head_camera": camera},
            "pointcloud": np.zeros((16, 3), dtype=np.float32),
            "joint_action": {"vector": np.full(4, step, dtype=np.float32)}}


class FakeTaskEnv:
    def __init__(self, instruction=INSTRUCTION, side="left"):
        self.instruction, self.steer_side, self.actions = instruction, side, []

    def get_instruction(self):
        return self.instruction

    def steer_layout(self):
        sign = -1.0 if self.steer_side == "left" else 1.0
        return {"side": self.steer_side, "arm": "right", "A_xyz": [0.1, -0.1, 0.75],
                "B_xyz": [0.0, -0.1, 0.75], "target_xyz": [sign * 0.13, -0.1, 0.75]}

    def get_obs(self):
        return make_observation(len(self.actions))

    def take_action(self, action, action_type="qpos"):
        self.actions.append(np.asarray(action))


class FakePolicy:
    def __init__(self, instr_mode="none", n_obs_steps=2, n_action_steps=3):
        if instr_mode is not None:
            self.instr_mode = instr_mode
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.device, self.use_pc_color = "cpu", False
        self.normalizer = {"point_cloud": SimpleNamespace(params_dict={"scale": torch.ones(3)})}
        self.received, self.resets, self.intervention = [], 0, None

    def predict_action(self, obs):
        self.received.append(obs)
        return {"action": torch.zeros(obs["agent_pos"].shape[0], self.n_action_steps, 4)}

    def reset(self):
        self.resets += 1

    def set_prior_intervention(self, intervention):
        self.intervention = intervention


def run_episode(model, env, calls=2):
    deploy.reset_model(model)
    for _ in range(calls):
        deploy.eval(env, model, env.get_obs())


class DeployInstructionTest(unittest.TestCase):
    def setUp(self):
        fake_render_sketch.calls.clear()
        patches = [patch.object(deploy, "_make_embedder", FakeEmbedder),
                   patch.object(deploy, "_make_tts", FakeTTS),
                   patch.object(deploy, "_save_wav", fake_save_wav),
                   patch.object(deploy, "_render_sketch", fake_render_sketch)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.voice = Path(self.temp.name) / "voice.onnx"
        self.voice.touch()

    def model(self, modality, instr_mode="cond", **kwargs):
        builder = InstructionBuilder(modality, embedder="org/model", tts_voice=str(self.voice),
                                     device="cpu", **kwargs)
        return Model(FakePolicy(instr_mode), builder)

    def test_text_built_once_per_episode_and_passed_to_policy(self):
        model, env = self.model("text"), FakeTaskEnv()
        run_episode(model, env, calls=3)
        embedder = model.instructions.embedder
        self.assertEqual(embedder.calls, [("text", [INSTRUCTION])])
        self.assertEqual((embedder.model, embedder.device), ("org/model", "cpu"))
        self.assertEqual(len(model.policy.received), 3)
        for obs in model.policy.received:
            self.assertEqual(obs["instr"].shape, (1, INSTR_DIM))
            self.assertEqual(obs["instr"].dtype, torch.float32)
            self.assertTrue(torch.equal(obs["instr"][0], torch.from_numpy(modality_vector("text"))))
            self.assertEqual(obs["agent_pos"].shape, (1, 2, 4))  # history keeps its time axis
        self.assertEqual(len(env.actions), 9)

    def test_sketch_uses_first_frame_and_live_layout(self):
        model, env = self.model("sketch"), FakeTaskEnv(side="right")
        run_episode(model, env, calls=2)
        self.assertEqual(len(fake_render_sketch.calls), 1)
        rgb, intrinsic, extrinsic, object_xyz, target_xyz = fake_render_sketch.calls[0]
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb == 0))  # frame 0, before any action
        self.assertEqual((intrinsic.shape, extrinsic.shape), ((3, 3), (3, 4)))
        self.assertEqual(object_xyz, [0.1, -0.1, 0.75])
        self.assertEqual(target_xyz, [0.13, -0.1, 0.75])
        calls = model.instructions.embedder.calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "sketch")
        self.assertIsInstance(calls[0][1][0], Image.Image)
        self.assertTrue(torch.equal(model.policy.received[-1]["instr"][0],
                                    torch.from_numpy(modality_vector("sketch"))))

    def test_audio_synthesises_the_text_and_embeds_a_wav_file(self):
        model, env = self.model("audio"), FakeTaskEnv()
        run_episode(model, env)
        self.assertEqual(model.instructions.tts.texts, [INSTRUCTION])
        self.assertEqual(model.instructions.tts.voice_path, str(self.voice))
        embedder = model.instructions.embedder
        self.assertEqual(embedder.calls[0][0], "audio")
        self.assertTrue(embedder.calls[0][1][0].endswith("audio.wav"))
        self.assertEqual(embedder.audio_existed, [True])
        self.assertTrue(torch.equal(model.policy.received[-1]["instr"][0],
                                    torch.from_numpy(modality_vector("audio"))))

    def test_reset_clears_cache_and_next_episode_rebuilds(self):
        model, env = self.model("text"), FakeTaskEnv("first")
        run_episode(model, env)
        self.assertIsNotNone(model.instr)
        deploy.reset_model(model)
        self.assertIsNone(model.instr)
        self.assertEqual(len(model.obs), 0)
        self.assertEqual(model.policy.resets, 2)
        env.instruction = "second"
        deploy.eval(env, model, env.get_obs())
        self.assertEqual(model.instructions.embedder.calls, [("text", ["first"]), ("text", ["second"])])
        self.assertEqual(model.instructions.episode, 2)

    def test_embedder_and_tts_are_created_lazily_once(self):
        model = self.model("audio")
        self.assertIsNone(model.instructions._embedder)
        self.assertIsNone(model.instructions._tts)
        with patch.object(deploy, "_make_embedder", wraps=FakeEmbedder) as make_embedder:
            run_episode(model, FakeTaskEnv())
            run_episode(model, FakeTaskEnv())
            make_embedder.assert_called_once_with("org/model", "cpu")
        self.assertEqual(len(model.instructions.tts.texts), 2)

    def test_dump_dir_gets_instruction_files_per_episode(self):
        dump = Path(self.temp.name) / "dump"
        for modality, extra in (("text", []), ("sketch", ["sketch.png"]), ("audio", ["audio.wav"])):
            model = self.model(modality, dump_dir=dump / modality)
            run_episode(model, FakeTaskEnv(side="right"))
            run_episode(model, FakeTaskEnv("again", side="left"))
            for episode, text in ((0, INSTRUCTION), (1, "again")):
                directory = dump / modality / f"episode_{episode}"
                self.assertEqual(sorted(path.name for path in directory.iterdir()),
                                 sorted(["instruction.txt", "meta.json"] + extra))
                self.assertEqual((directory / "instruction.txt").read_text().strip(), text)
                meta = json.loads((directory / "meta.json").read_text())
                self.assertEqual(meta["modality"], modality)
                self.assertEqual(meta["instruction"], text)
                self.assertEqual(meta["layout"]["side"], "right" if episode == 0 else "left")
                self.assertAlmostEqual(meta["embedding_norm"], 1.0)
            if modality == "sketch":
                with Image.open(dump / modality / "episode_0/sketch.png") as sketch:
                    self.assertEqual(sketch.size, (8, 6))

    def test_rejects_wrong_embedding_shape(self):
        model = self.model("text")
        with patch.object(FakeEmbedder, "embed_text", lambda self, texts: np.zeros((1, 512), np.float32)):
            with self.assertRaisesRegex(ValueError, "768"):
                run_episode(model, FakeTaskEnv())

    def test_get_model_consistency_checks(self):
        usr_args = {"ckpt_path": "x.ckpt", "device": "cpu"}
        with patch.object(deploy, "load_policy", return_value=FakePolicy("cond")):
            with self.assertRaisesRegex(ValueError, "instruction-conditioned"):
                deploy.get_model(usr_args)
            with self.assertRaisesRegex(ValueError, "instruction_modality"):
                deploy.get_model(dict(usr_args, instruction_modality="speech"))
            with self.assertRaisesRegex(ValueError, "tts_voice"):
                deploy.get_model(dict(usr_args, instruction_modality="audio"))
            with self.assertRaises(FileNotFoundError):
                deploy.get_model(dict(usr_args, instruction_modality="audio", tts_voice="/nonexistent.onnx"))
            model = deploy.get_model(dict(usr_args, instruction_modality="text", embedder="org/model",
                                          intervention={"sigma_scale": 0}, instr_dump_dir="d"))
            self.assertEqual(model.instructions.modality, "text")
            self.assertEqual(model.instructions.embedder_name, "org/model")
            self.assertEqual(model.instructions.device, "cpu")
            self.assertEqual(model.instructions.dump_dir, Path("d"))
            self.assertIsNone(model.instructions._embedder)  # lazy
            self.assertEqual(model.policy.intervention, {"sigma_scale": 0})
            model = deploy.get_model(dict(usr_args, instruction_modality="audio", tts_voice=str(self.voice)))
            self.assertEqual(model.instructions.tts_voice, str(self.voice))
        with patch.object(deploy, "load_policy", return_value=FakePolicy("none")):
            with self.assertWarnsRegex(UserWarning, "ignored"):
                model = deploy.get_model(dict(usr_args, instruction_modality="sketch"))
            self.assertIsNone(model.instructions)
            self.assertEqual(model.instructions, deploy.get_model(usr_args).instructions)
        with patch.object(deploy, "load_policy", return_value=FakePolicy(instr_mode=None)):
            self.assertIsNone(deploy.get_model(usr_args).instructions)  # policies without instr_mode
            with self.assertRaisesRegex(ValueError, "instruction_modality"):
                deploy.get_model(dict(usr_args, instruction_modality=5))

    def test_default_adapter_passes_no_instr(self):
        with patch.object(deploy, "load_policy", return_value=FakePolicy("none")):
            model = deploy.get_model({"ckpt_path": "x.ckpt", "device": "cpu"})
        env = FakeTaskEnv()
        run_episode(model, env, calls=2)
        self.assertEqual(len(model.policy.received), 2)
        for obs in model.policy.received:
            self.assertEqual(set(obs), {"agent_pos", "point_cloud"})
            self.assertEqual(obs["point_cloud"].shape, (1, 2, 16, 3))
        self.assertIsNone(model.instr)
        self.assertEqual(len(env.actions), 6)


if __name__ == "__main__":
    unittest.main()

"""CPU checks for scripts/vla/build_reversal_dataset.py and scripts/vla/embed_vla_instructions.py.

The frozen VLA (leap.vla.pi05_flow), the LIBERO env, the embedder and the TTS are replaced by fakes; the
fake flow is an affine map so ``forward(reverse(a)) == a`` and the reconstruction error is (numerically) zero.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "vla" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = load_script("build_reversal_dataset")
embed = load_script("embed_vla_instructions")

from leap.steer.tts import load_wav  # noqa: E402
from leap.vla import instructions as I  # noqa: E402
from leap.vla import libero_scene as S  # noqa: E402

MEAN = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
STD = np.linspace(0.5, 1.5, 7, dtype=np.float32)


def fake_flow_module():
    """A stand-in for leap.vla.pi05_flow with the same names; the flow is x_noise = 2 x_action + 1."""
    api = types.SimpleNamespace()
    api.Observation = lambda images, state, prompt: types.SimpleNamespace(images=images, state=state, prompt=prompt)
    api.calls = []

    def load_pi05(checkpoint, device, **kwargs):
        return {"checkpoint": checkpoint, "device": device}

    def prepare_batches(bundle, observations, flip_images=True):
        observations = list(observations)
        api.calls.append(("prepare", len(observations), observations[0].prompt))
        for obs in observations:
            assert obs.images["image"].dtype == np.uint8 and obs.images["image"].shape == (8, 8, 3)
            assert obs.state.shape == (8,)
        return {"state": np.stack([o.state for o in observations]).astype(np.float32),
                "image_mean": np.array([o.images["image"].astype(np.float64).mean() for o in observations])}

    def prefix_context(bundle, batch):
        return {"n": batch["state"].shape[0]}

    def prefix_features(bundle, batch, ctx=None):
        n = batch["state"].shape[0]
        feat = np.zeros((n, build.FEAT_DIM), dtype=np.float32)
        feat[:, :8] = batch["state"]
        feat[:, 8] = batch["image_mean"]
        return feat

    def normalize_actions(bundle, actions7):
        actions7 = np.asarray(actions7, dtype=np.float32)
        out = np.zeros(actions7.shape[:2] + (32,), dtype=np.float32)
        out[:, :, :7] = (actions7 - MEAN) / STD
        return out

    def reverse_actions(bundle, ctx, actions_norm_padded, num_steps):
        api.calls.append(("reverse", num_steps))
        return 2.0 * np.asarray(actions_norm_padded) + 1.0

    def forward_from_noise(bundle, ctx, noise, num_steps):
        api.calls.append(("forward", num_steps))
        return (np.asarray(noise) - 1.0) / 2.0

    api.load_pi05 = load_pi05
    api.prepare_batches = prepare_batches
    api.prefix_context = prefix_context
    api.prefix_features = prefix_features
    api.normalize_actions = normalize_actions
    api.reverse_actions = reverse_actions
    api.forward_from_noise = forward_from_noise
    return api


class FakeEnv:
    """Replays a flat state as a deterministic raw observation (image = state[0] everywhere)."""

    def __init__(self):
        self.camera_name = ["agentview_image", "robot0_eye_in_hand_image"]
        self.camera_name_mapping = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        self.observation_height = self.observation_width = 8
        self.task_description = S.NEUTRAL_PROMPT
        self.closed = False

    def reset(self, seed=None):
        return {}, {}

    def close(self):
        self.closed = True

    def replay(self, flat_state):
        value = int(flat_state[0]) % 256
        return {"agentview_image": np.full((8, 8, 3), value, np.uint8),
                "robot0_eye_in_hand_image": np.full((8, 8, 3), 255 - value, np.uint8),
                "robot0_eef_pos": np.array([flat_state[0], 0.0, 1.0]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.array([0.04, -0.04])}


class FakeSuite:
    name = "libero_spatial"

    def get_task(self, i):
        return types.SimpleNamespace(language=f"pick up the black bowl number {i} and place it on the plate",
                                     name=f"task{i}")

    def get_task_demonstration(self, i):
        return f"libero_spatial/task{i}_demo.hdf5"


class FakeEmbedder:
    """Vectors encode the modality in dim 0 and the input index in dim 1."""

    def __init__(self):
        self.calls = []

    def _vectors(self, tag, count):
        out = np.zeros((count, 768), np.float32)
        out[:, 0] = tag
        out[:, 1] = np.arange(count)
        return out

    def embed_text(self, texts):
        self.calls.append(("text", list(texts)))
        return self._vectors(1.0, len(texts))

    def embed_images(self, images):
        self.calls.append(("image", [im.size for im in images]))
        return self._vectors(2.0, len(images))

    def embed_audio(self, clips):
        self.calls.append(("audio", [int(np.asarray(c).shape[0]) for c in clips]))
        return self._vectors(3.0, len(clips))


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return np.linspace(-0.5, 0.5, 1600, dtype=np.float32)


def write_demos(path, lengths):
    import h5py
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for k, length in enumerate(lengths):
            group = data.create_group(f"demo_{k}")
            states = np.zeros((length, 92))
            states[:, 0] = np.arange(length) + 100 * k
            group.create_dataset("states", data=states)
            actions = np.tile(np.arange(length, dtype=np.float64)[:, None], (1, 7)) / 100 + 0.01 * k
            group.create_dataset("actions", data=actions)


class ChunkTest(unittest.TestCase):
    def test_padding_repeats_last_action_and_marks_padded_steps(self):
        actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
        chunk, pad_mask = build.action_chunk(actions, 10)
        self.assertEqual(chunk.shape, (50, 7))
        self.assertEqual(pad_mask.tolist(), [False, False] + [True] * 48)
        np.testing.assert_array_equal(chunk[0], actions[10])
        np.testing.assert_array_equal(chunk[1], actions[11])
        np.testing.assert_array_equal(chunk[2:], np.repeat(actions[11:12], 48, axis=0))

    def test_full_chunk_has_no_padding(self):
        actions = np.random.default_rng(0).normal(size=(80, 7)).astype(np.float32)
        chunk, pad_mask = build.action_chunk(actions, 3)
        self.assertFalse(pad_mask.any())
        np.testing.assert_array_equal(chunk, actions[3:53])
        with self.assertRaises(IndexError):
            build.action_chunk(actions, 80)

    def test_frames_tasks_and_relative_l2(self):
        self.assertEqual(build.sample_frames(7, 2), [0, 2, 4, 6])
        self.assertEqual(build.parse_tasks("0-2,5"), [0, 1, 2, 5])
        target = np.ones((2, 50, 32))
        pred = target.copy()
        pred[0, :, :7] += 0.1
        pred[1, 25:, :7] += 5.0  # only on padded steps, which the error ignores
        pad_mask = np.zeros((2, 50), bool)
        pad_mask[1, 25:] = True
        err = build.relative_l2(pred, target, pad_mask)
        self.assertAlmostEqual(float(err[0]), 0.1, places=5)
        self.assertEqual(float(err[1]), 0.0)


class ReverserTest(unittest.TestCase):
    def test_run_measures_recon_only_where_asked(self):
        api = fake_flow_module()
        with mock.patch.object(build, "flow_api", lambda: api):
            reverser = build.Reverser(bundle={}, prompt=S.NEUTRAL_PROMPT, num_steps=10)
            images = [{"image": np.zeros((8, 8, 3), np.uint8), "image2": np.zeros((8, 8, 3), np.uint8)}] * 3
            states = [np.arange(8, dtype=np.float32)] * 3
            actions = np.random.default_rng(1).normal(size=(3, 50, 7)).astype(np.float32)
            pad_mask = np.zeros((3, 50), bool)
            feat, noise, recon = reverser.run(images, states, actions, pad_mask, np.array([True, False, True]))
        self.assertEqual(feat.shape, (3, build.FEAT_DIM))
        self.assertEqual(noise.shape, (3, 50, 7))
        np.testing.assert_allclose(noise, 2 * (actions - MEAN) / STD + 1, rtol=1e-5)
        self.assertAlmostEqual(float(recon[0]), 0.0, places=5)
        self.assertTrue(np.isnan(recon[1]))
        self.assertAlmostEqual(float(recon[2]), 0.0, places=5)
        self.assertEqual([c[0] for c in api.calls], ["prepare", "reverse", "forward"])
        self.assertEqual(api.calls[0][2], S.NEUTRAL_PROMPT)


class DatasetTest(unittest.TestCase):
    """build_task with fakes, then embed_vla_instructions on its output."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-vla-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "out"
        (self.root / "raw").mkdir()
        write_demos(self.root / "raw" / "task0_demo.hdf5", [7, 60])
        self.api = fake_flow_module()
        self.env = FakeEnv()
        self.suite = FakeSuite()
        patches = [mock.patch.object(build, "flow_api", lambda: self.api),
                   mock.patch.object(S, "make_env", lambda suite, task_id, prompt_override=None, **k: self.env),
                   mock.patch.object(S, "replay_state", lambda env, state: env.replay(state)),
                   mock.patch.object(S, "scene_layout", lambda env, names=S.SCENE_OBJECTS: {
                       S.TARGET_BOWL: np.array([0.0, 0.0, 2.0]), S.DISTRACTOR_BOWL: np.array([0.3, 0.0, 2.0]),
                       S.PLATE: np.array([0.0, 0.2, 2.0]), "initial_z": {}}),
                   mock.patch.object(S, "agentview_camera", lambda env: (
                       np.array([[-4.0, 0, 3.5], [0, 4.0, 4.0], [0, 0, 1]]), np.hstack([np.eye(3), np.zeros((3, 1))])))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def build_args(self, *extra):
        return build.parse_args(["--raw", str(self.root / "raw"), "--out", str(self.out), "--tasks", "0",
                                 "--stride", "3", "--batch-size", "4", "--recon-fraction", "1.0",
                                 "--num-steps", "4", *extra])

    def run_build(self, tts=None, *extra):
        args = self.build_args(*extra)
        entry = build.build_task(args, self.suite, 0, {}, np.random.default_rng(0), tts)
        build_manifest = I.update_manifest(args.out, [entry], build.manifest_defaults(args))
        return entry, args, build_manifest

    def test_npz_and_artifacts(self):
        entry, args, manifest_path = self.run_build()
        data = I.load_task_npz(self.out / "task_0.npz")
        frames = [0, 3, 6] + list(range(0, 60, 3))
        n = len(frames)
        self.assertEqual(data["feat"].shape, (n, build.FEAT_DIM))
        self.assertEqual(data["state"].shape, (n, 8))
        self.assertEqual(data["noise"].shape, (n, 50, 7))
        self.assertEqual(data["chunk_pad_mask"].shape, (n, 50))
        self.assertEqual(data["demo_id"].tolist(), [0] * 3 + [1] * 20)
        self.assertEqual(data["t"].tolist(), frames)
        self.assertEqual(data["task_id"], 0)
        self.assertEqual(data["prompt"], S.NEUTRAL_PROMPT)
        self.assertEqual(data["num_steps"], 4)
        self.assertIn("bowl number 0", data["task_language"])
        self.assertFalse(any(key.startswith("instr_") for key in data))
        # replayed states and the 8-d state come from the fake env: eef x = frame index (+100 per demo)
        np.testing.assert_allclose(data["state"][:3, 0], [0, 3, 6])
        np.testing.assert_allclose(data["state"][3:, 0], np.arange(0, 60, 3) + 100)
        np.testing.assert_allclose(data["feat"][:, :8], data["state"], rtol=1e-6)
        # True = padded: demo 0 has 7 steps, so frame 6 keeps 1 real step; demo 1 frame 57 keeps 3
        self.assertEqual(int(data["chunk_pad_mask"][2].sum()), 49)
        self.assertEqual(int(data["chunk_pad_mask"][-1].sum()), 47)
        self.assertFalse(data["chunk_pad_mask"][3].any())
        self.assertTrue(np.all(np.isfinite(data["recon_err"])))
        np.testing.assert_allclose(data["recon_err"], 0.0, atol=1e-5)
        self.assertEqual(entry["frames"], n)
        self.assertEqual(entry["demo_ids"], [0, 1])
        self.assertEqual(entry["audio_files"], [])
        self.assertTrue(self.env.closed)
        self.assertTrue(all(c[1] == 4 for c in self.api.calls if c[0] in ("reverse", "forward")))
        record = I.read_record(self.out, 0)
        self.assertEqual(record["demo_ids"], [0, 1])
        self.assertEqual(record["texts"], [data["task_language"]])
        self.assertEqual(record["sketches"], ["task_0_episode0_sketch.png", "task_0_episode1_sketch.png"])
        for name in record["sketches"]:
            with Image.open(self.out / name) as sketch:
                self.assertEqual(sketch.size, (8, 8))
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["chunk_pad_mask"], "true_means_padded")
        self.assertEqual(manifest["tasks"]["0"]["frames"], n)
        self.assertEqual(manifest["num_steps"], 4)

    def test_builder_writes_audio_when_tts_given(self):
        tts = FakeTTS()
        entry, args, _ = self.run_build(tts)
        self.assertEqual(entry["audio_files"], ["task_0_audio0.wav"])
        self.assertEqual(len(tts.calls), 1)
        wav, rate = load_wav(self.out / "task_0_audio0.wav")
        self.assertEqual(rate, 16000)
        self.assertEqual(wav.shape, (1600,))
        self.assertEqual(I.read_record(self.out, 0)["audio"], ["task_0_audio0.wav"])

    def test_idempotent_build(self):
        entry, args, _ = self.run_build()
        calls = len(self.api.calls)
        again = build.build_task(args, self.suite, 0, {}, np.random.default_rng(0))
        self.assertTrue(again["skipped"])
        self.assertEqual(len(self.api.calls), calls)
        manifest = json.loads(I.update_manifest(args.out, [again], build.manifest_defaults(args)).read_text())
        self.assertEqual(manifest["tasks"]["0"]["frames"], entry["frames"])

    def test_max_demos(self):
        entry, args, _ = self.run_build(None, "--max-demos", "1")
        data = I.load_task_npz(self.out / "task_0.npz")
        self.assertEqual(data["demo_id"].tolist(), [0, 0, 0])
        self.assertEqual(I.read_record(self.out, 0)["demo_ids"], [0])

    def test_embed_script_fills_arrays(self):
        entry, args, manifest_path = self.run_build()
        embedder, tts = FakeEmbedder(), FakeTTS()
        result = embed.embed_task(self.out, 0, embedder, ["text", "sketch", "audio"], tts)
        self.assertEqual(result["modalities"], ["text", "sketch", "audio"])
        self.assertEqual(result["instr_variants"], {"text": 1, "sketch": 1, "audio": 1})
        self.assertEqual(tts.calls, [entry["task_language"]])
        data = I.load_task_npz(self.out / "task_0.npz")
        self.assertEqual(data["instr_modalities"].tolist(), ["text", "sketch", "audio"])
        self.assertEqual(data["instr_text"].shape, (2, 1, 768))
        self.assertEqual(data["instr_sketch"].shape, (2, 1, 768))
        self.assertEqual(data["instr_audio"].shape, (2, 1, 768))
        np.testing.assert_array_equal(data["instr_text"][:, 0, 0], [1.0, 1.0])
        np.testing.assert_array_equal(data["instr_sketch"][:, 0, :2], [[2.0, 0.0], [2.0, 1.0]])  # one row per demo
        np.testing.assert_array_equal(data["instr_audio"][:, 0, 0], [3.0, 3.0])
        self.assertEqual(data["feat"].shape[0], entry["frames"])  # reversal arrays untouched
        self.assertEqual(data["chunk_pad_mask"].shape, (entry["frames"], 50))
        self.assertEqual([c[0] for c in embedder.calls], ["text", "image", "audio"])
        self.assertEqual(embedder.calls[2][1], [1600])
        self.assertTrue(embed.embed_task(self.out, 0, embedder, ["text", "sketch", "audio"], tts)["skipped"])
        manifest = json.loads(I.update_manifest(self.out, [result]).read_text())
        self.assertEqual(manifest["tasks"]["0"]["modalities"], ["text", "sketch", "audio"])
        self.assertEqual(manifest["tasks"]["0"]["frames"], entry["frames"])
        self.assertEqual(manifest["chunk_pad_mask"], "true_means_padded")

    def test_embed_script_partial_then_audio_needs_voice(self):
        self.run_build()
        embedder = FakeEmbedder()
        embed.embed_task(self.out, 0, embedder, ["text", "sketch"])
        data = I.load_task_npz(self.out / "task_0.npz")
        self.assertEqual(data["instr_modalities"].tolist(), ["text", "sketch"])
        self.assertNotIn("instr_audio", data)
        with self.assertRaises(ValueError):
            embed.embed_task(self.out, 0, embedder, ["audio"], tts=None)
        embed.embed_task(self.out, 0, embedder, ["audio"], tts=FakeTTS())
        data = I.load_task_npz(self.out / "task_0.npz")
        self.assertEqual(data["instr_modalities"].tolist(), ["text", "sketch", "audio"])
        self.assertTrue(embed.embed_task(self.out, 0, embedder, ["text"])["skipped"])
        self.assertFalse(embed.embed_task(self.out, 0, embedder, ["text"], force=True).get("skipped"))
        self.assertEqual(embed.find_tasks(self.out), [0])
        self.assertEqual(embed.parse_tasks("1-2"), [1, 2])

    def test_embed_script_uses_builder_audio(self):
        self.run_build(FakeTTS())
        embedder, tts = FakeEmbedder(), FakeTTS()
        embed.embed_task(self.out, 0, embedder, ["audio"], tts)
        self.assertEqual(tts.calls, [])  # the builder's WAV is reused
        with self.assertRaises(ValueError):
            record = I.read_record(self.out, 0)
            record["demo_ids"] = [5]
            embed.record_demo_ids(record, I.load_task_npz(self.out / "task_0.npz"))


if __name__ == "__main__":
    unittest.main()

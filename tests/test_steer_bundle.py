"""CPU checks for leap.steer: projection, sketch rendering, bundle checks and the offline scripts (mocked models)."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import h5py
import numpy as np
from PIL import Image
import zarr

from leap.steer.bundle import (AUDIO_VARIANTS, INSTR_DIM, TEXT_VARIANTS, bundle_path, check_bundle,
                               load_embeddings, standardize_stats, validate_embeddings, zarr_path_for_bundle)
from leap.steer.sketch import clip_pixel, project_points, render_sketch
from leap.steer.tts import SAMPLE_RATE, load_wav, save_wav

ROOT = Path(__file__).resolve().parents[1]

# Head camera of data/raw/beat_block_hammer/leap_clean episode 0 (RoboTwin D435 at 320x240) and the
# right-arm TCP at the grasp frame; the circle drawn at this pixel sits on the gripper (checked visually).
REAL_INTRINSIC = np.array([[358.64218, 0.0, 160.0], [0.0, 358.64218, 120.0], [0.0, 0.0, 1.0]])
REAL_EXTRINSIC = np.array([[1.0, 0.0, 0.0, 0.03200001], [0.0, -0.8, -0.6, 0.45], [0.0, 0.6, -0.8, 1.35]])
REAL_TCP = np.array([-0.00396226, -0.09100199, 0.91107464])
REAL_PIXEL = np.array([177.749, 104.906])


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"leap_script_{name}", ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_vector(seed_text):
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    vector = rng.normal(size=INSTR_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Deterministic stand-in for InstructionEmbedder (no model download)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_text(self, texts):
        return np.stack([unit_vector("text:" + t) for t in texts])

    def embed_images(self, images):
        return np.stack([unit_vector("image:" + Path(p).name) for p in images])

    def embed_audio(self, clips):
        return np.stack([unit_vector("audio:" + Path(p).name) for p in clips])


class FakeTTS:
    def __init__(self, voice_path):
        self.voice_path = voice_path

    def synthesize(self, text):
        t = np.arange(int(0.2 * SAMPLE_RATE)) / SAMPLE_RATE
        return (0.5 * np.sin(2 * np.pi * (200 + len(text)) * t)).astype(np.float32)


class ProjectionAndSketchTest(unittest.TestCase):
    def test_round_trip_with_synthetic_camera(self):
        rng = np.random.default_rng(0)
        angle = 0.4
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        translation = np.array([0.1, -0.2, 1.5])
        extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)
        intrinsic = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
        camera_points = np.column_stack([rng.uniform(-0.3, 0.3, 5), rng.uniform(-0.3, 0.3, 5), rng.uniform(0.5, 2, 5)])
        world_points = (camera_points - translation) @ rotation  # R^T (p - t)
        expected = np.column_stack([300.0 * camera_points[:, 0] / camera_points[:, 2] + 160.0,
                                    300.0 * camera_points[:, 1] / camera_points[:, 2] + 120.0])
        np.testing.assert_allclose(project_points(world_points, intrinsic, extrinsic), expected, atol=1e-9)
        # The optical axis maps to the principal point; a 4x4 extrinsic is accepted too.
        axis_point = (np.array([0, 0, 1.0]) - translation) @ rotation
        np.testing.assert_allclose(project_points(axis_point, intrinsic, np.vstack([extrinsic, [0, 0, 0, 1]])),
                                   [[160.0, 120.0]], atol=1e-9)
        behind = (np.array([0, 0, -1.0]) - translation) @ rotation
        self.assertTrue(np.isnan(project_points(behind, intrinsic, extrinsic)).all())
        with self.assertRaises(ValueError):
            clip_pixel((np.nan, 1.0), 320, 240)

    def test_real_robotwin_camera_matrices(self):
        pixel = project_points(REAL_TCP, REAL_INTRINSIC, REAL_EXTRINSIC)[0]
        np.testing.assert_allclose(pixel, REAL_PIXEL, atol=1e-2)
        self.assertTrue(0 <= pixel[0] < 320 and 0 <= pixel[1] < 240)

    def test_render_sketch_draws_circle_and_arrow(self):
        rgb = np.full((240, 320, 3), 200, dtype=np.uint8)
        object_xyz = np.array([-0.1, -0.1, 0.75])
        target_xyz = np.array([0.1, -0.1, 0.75])
        sketch = render_sketch(rgb, REAL_INTRINSIC, REAL_EXTRINSIC, object_xyz, target_xyz)
        self.assertIsInstance(sketch, Image.Image)
        self.assertEqual(sketch.size, (320, 240))
        pixels = np.asarray(sketch)
        red = (pixels[..., 0] == 255) & (pixels[..., 1] == 0) & (pixels[..., 2] == 0)
        blue = (pixels[..., 0] == 0) & (pixels[..., 2] == 255)
        self.assertGreater(red.sum(), 50)
        self.assertGreater(blue.sum(), 50)
        object_px, target_px = project_points(np.stack([object_xyz, target_xyz]), REAL_INTRINSIC, REAL_EXTRINSIC)
        red_center = np.argwhere(red).mean(axis=0)[::-1]
        np.testing.assert_allclose(red_center, object_px, atol=2.0)
        blue_center = np.argwhere(blue).mean(axis=0)[::-1]
        self.assertGreater(blue_center[0], object_px[0])
        self.assertLess(blue_center[0], target_px[0])
        # Points outside the frame are clamped, not an error.
        far = render_sketch(rgb, REAL_INTRINSIC, REAL_EXTRINSIC, np.array([-2.0, -0.1, 0.75]), target_xyz)
        self.assertEqual(far.size, (320, 240))

    def test_wav_round_trip(self):
        wav = (0.3 * np.sin(np.linspace(0, 100, 1600))).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            save_wav(Path(temp) / "a.wav", wav)
            loaded, rate = load_wav(Path(temp) / "a.wav")
        self.assertEqual(rate, SAMPLE_RATE)
        np.testing.assert_allclose(loaded, wav, atol=1e-4)


class BundleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-steer-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.zarr_path = self.root / "demo.zarr"
        group = zarr.open_group(str(self.zarr_path), mode="w")
        group.create_group("meta").create_dataset("episode_ends", data=np.array([11, 22, 30], dtype=np.int64))
        self.bundle = bundle_path(self.zarr_path)
        self.bundle.mkdir()

    def good_arrays(self):
        rng = np.random.default_rng(1)

        def normalised(shape):
            array = rng.normal(size=shape).astype(np.float32)
            return array / np.linalg.norm(array, axis=-1, keepdims=True)

        return {"text": normalised((3, TEXT_VARIANTS, INSTR_DIM)), "sketch": normalised((3, 1, INSTR_DIM)),
                "audio": normalised((3, AUDIO_VARIANTS, INSTR_DIM)), "side": np.array([0, 1, 0], dtype=np.int64),
                "episode_ends": np.array([11, 22, 30], dtype=np.int64)}

    def test_paths(self):
        self.assertEqual(bundle_path("data/x.zarr"), Path("data/x.instr"))
        self.assertEqual(zarr_path_for_bundle(Path("data/x.instr")), Path("data/x.zarr"))
        with self.assertRaises(ValueError):
            bundle_path("data/x.npz")

    def test_check_bundle_accepts_good_and_rejects_bad(self):
        arrays = self.good_arrays()
        np.savez(self.bundle / "embeddings.npz", **arrays)
        self.assertEqual(check_bundle(self.bundle, self.zarr_path), ["text", "sketch", "audio"])
        loaded = load_embeddings(self.bundle)
        self.assertEqual(loaded["text"].shape, (3, TEXT_VARIANTS, INSTR_DIM))
        self.assertEqual(loaded["side"].dtype, np.int64)
        bad_cases = {
            "ends": dict(arrays, episode_ends=np.array([11, 22, 31], dtype=np.int64)),
            "text shape": dict(arrays, text=arrays["text"][:, :-1]),
            "not normalised": dict(arrays, sketch=arrays["sketch"] * 2),
            "side dtype": dict(arrays, side=arrays["side"].astype(np.int32)),
            "side value": dict(arrays, side=np.array([0, 2, 0], dtype=np.int64)),
            "float64": dict(arrays, audio=arrays["audio"].astype(np.float64)),
            "non finite": dict(arrays, text=np.where(np.arange(INSTR_DIM) == 0, np.nan, arrays["text"]).astype(np.float32)),
        }
        for name, case in bad_cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_embeddings(case, np.array([11, 22, 30]))
        missing = {k: v for k, v in arrays.items() if k not in ("text", "audio")}
        self.assertEqual(validate_embeddings(missing, np.array([11, 22, 30])), ["sketch"])
        with self.assertRaises(ValueError):
            validate_embeddings({"side": arrays["side"], "episode_ends": arrays["episode_ends"]})

    def test_standardize_stats_pools_modalities_and_floors_std(self):
        arrays = self.good_arrays()
        arrays["text"][..., 5] = 0.25  # constant dimension
        mean, std = standardize_stats({k: arrays[k] for k in ("text", "sketch", "audio")})
        self.assertEqual((mean.shape, std.shape, mean.dtype, std.dtype), ((INSTR_DIM,), (INSTR_DIM,), np.float32, np.float32))
        pooled = np.concatenate([arrays[k].reshape(-1, INSTR_DIM) for k in ("text", "sketch", "audio")])
        np.testing.assert_allclose(mean, pooled.mean(axis=0), atol=1e-6)
        standardised = (pooled - mean) / std
        np.testing.assert_allclose(standardised.mean(axis=0)[:5], 0, atol=1e-4)
        np.testing.assert_allclose(standardised.std(axis=0)[:5], 1, atol=1e-3)
        only_text = standardize_stats({"text": np.full((2, 1, INSTR_DIM), 0.5, dtype=np.float32)})[1]
        np.testing.assert_allclose(only_text, 1e-6, rtol=1e-6)


class OfflineScriptsTest(unittest.TestCase):
    """make_instructions -> embed_instructions -> instr_probe on a synthetic raw collection with mocked models."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-steer-scripts-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw/place_a2b_side/leap_clean"
        (self.raw / "data").mkdir(parents=True)
        (self.raw / "instructions").mkdir()
        self.episodes = 4
        scene_info = {}
        for i in range(self.episodes):
            side = "left" if i % 2 == 0 else "right"
            a = [0.1 * (i - 1.5), -0.1, 0.75]
            b = [0.0, -0.1 + 0.12 * (1 if i % 2 else -1), 0.75]
            target = [b[0] + (-0.13 if side == "left" else 0.13), b[1], b[2]]
            scene_info[f"episode_{i}"] = {"info": {"{A}": "047_mouse/base0", "{B}": "048_stapler/base0",
                                                   "{a}": "right", "{side}": side},
                                          "steer": {"side": side, "arm": "right", "A_xyz": a, "B_xyz": b,
                                                    "target_xyz": target}}
            buffer = io.BytesIO()
            Image.fromarray(np.full((240, 320, 3), 90 + 20 * i, dtype=np.uint8)).save(buffer, format="JPEG")
            frames = 6 + i
            with h5py.File(self.raw / "data" / f"episode{i}.hdf5", "w") as episode:
                episode.create_dataset("observation/head_camera/rgb",
                                       data=np.array([buffer.getvalue()] * frames, dtype=f"S{len(buffer.getvalue())}"))
                episode.create_dataset("observation/head_camera/intrinsic_cv",
                                       data=np.repeat(REAL_INTRINSIC[None].astype(np.float32), frames, axis=0))
                episode.create_dataset("observation/head_camera/extrinsic_cv",
                                       data=np.repeat(REAL_EXTRINSIC[None].astype(np.float32), frames, axis=0))
                episode.create_dataset("joint_action/vector", data=np.zeros((frames, 14), dtype=np.float32))
                episode.create_dataset("pointcloud", data=np.zeros((frames, 8, 6), dtype=np.float32))
            seen = [f"Place the mouse on the {side} of the stapler ({k})." for k in range(3)]
            (self.raw / "instructions" / f"episode{i}.json").write_text(json.dumps({"seen": seen, "unseen": seen[:1]}))
        (self.raw / "scene_info.json").write_text(json.dumps(scene_info))
        self.zarr_path = self.root / "place_a2b_side-leap_clean-4.zarr"
        from leap.robotwin.data import convert_episodes
        convert_episodes(self.raw / "data", self.zarr_path, self.episodes)
        self.bundle = bundle_path(self.zarr_path)

    def test_make_embed_probe(self):
        argv = ["--raw", str(self.raw), "--zarr", str(self.zarr_path), "--voice", "unused.onnx"]
        with patch.object(sys, "argv", ["make_instructions.py", *argv]):
            make = load_script("make_instructions")
            with patch.object(make, "PiperTTS", FakeTTS):
                make.main()
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        self.assertEqual(manifest["episodes"], self.episodes)
        self.assertEqual(manifest["side"], [0, 1, 0, 1])
        for i in range(self.episodes):
            folder = self.bundle / f"episode{i}"
            self.assertTrue((folder / "text.json").exists())
            self.assertEqual(Image.open(folder / "sketch.png").size, (320, 240))
            meta = json.loads((folder / "sketch_meta.json").read_text())
            self.assertEqual(meta["side"], "left" if i % 2 == 0 else "right")
            self.assertTrue(all(0 <= meta["object_px"][k] < (320, 240)[k] for k in range(2)))
            for k in range(AUDIO_VARIANTS):
                wav, rate = load_wav(folder / f"audio_{k}.wav")
                self.assertEqual(rate, SAMPLE_RATE)
                self.assertGreater(len(wav), 1000)
        # Idempotent: a second run keeps the files (same mtimes) and the sketch stays byte-identical.
        before = {p: p.stat().st_mtime_ns for p in self.bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
        with patch.object(sys, "argv", ["make_instructions.py", *argv]):
            with patch.object(make, "PiperTTS", FakeTTS):
                make.main()
        after = {p: p.stat().st_mtime_ns for p in self.bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
        self.assertEqual(before, after)

        embed = load_script("embed_instructions")
        with patch.object(sys, "argv", ["embed_instructions.py", "--bundle", str(self.bundle), "--device", "cpu"]), \
             patch.object(embed, "InstructionEmbedder", FakeEmbedder):
            embed.main()
        self.assertEqual(check_bundle(self.bundle, self.zarr_path), ["text", "sketch", "audio"])
        arrays = load_embeddings(self.bundle)
        self.assertEqual(arrays["text"].shape, (self.episodes, TEXT_VARIANTS, INSTR_DIM))
        self.assertEqual(arrays["sketch"].shape, (self.episodes, 1, INSTR_DIM))
        self.assertEqual(arrays["audio"].shape, (self.episodes, AUDIO_VARIANTS, INSTR_DIM))
        np.testing.assert_array_equal(arrays["side"], [0, 1, 0, 1])
        np.testing.assert_array_equal(arrays["episode_ends"], zarr.open_group(str(self.zarr_path))["meta/episode_ends"][:])
        # Only 3 seen texts: variants cycle, so variant 3 equals variant 0.
        np.testing.assert_array_equal(arrays["text"][0, 3], arrays["text"][0, 0])
        self.assertFalse(np.array_equal(arrays["text"][0, 1], arrays["text"][0, 0]))

        probe = load_script("instr_probe")
        report = self.root / "probe.json"
        with patch.object(sys, "argv", ["instr_probe.py", "--bundle", str(self.bundle), "--json", str(report), "--folds", "2"]):
            probe.main()
        results = json.loads(report.read_text())
        self.assertEqual(sorted(results["accuracy"]), ["audio", "sketch", "text"])
        for train in results["accuracy"].values():
            for value in train.values():
                self.assertTrue(0.0 <= value <= 1.0)

    def test_make_instructions_without_audio_and_missing_instructions(self):
        make = load_script("make_instructions")
        argv = ["--raw", str(self.raw), "--zarr", str(self.zarr_path), "--no-audio"]
        with patch.object(sys, "argv", ["make_instructions.py", *argv]):
            make.main()
        self.assertFalse((self.bundle / "episode0/audio_0.wav").exists())
        self.assertEqual(json.loads((self.bundle / "manifest.json").read_text())["audio_variants"], 0)
        embed = load_script("embed_instructions")
        with patch.object(sys, "argv", ["embed_instructions.py", "--bundle", str(self.bundle), "--device", "cpu"]), \
             patch.object(embed, "InstructionEmbedder", FakeEmbedder):
            embed.main()
        self.assertEqual(check_bundle(self.bundle, self.zarr_path), ["text", "sketch"])
        self.assertIsNone(make.find_instructions_dir(self.root / "nowhere/task/setting", self.root / "no-robotwin"))
        with self.assertRaises(FileNotFoundError):
            make.find_instructions_dir(self.raw, self.root / "no-robotwin", explicit=self.root / "missing")

    def test_texts_are_generated_with_the_official_generator_when_files_are_missing(self):
        make = load_script("make_instructions")
        shutil.rmtree(self.raw / "instructions")
        calls = []

        def fake_generate(robotwin, task_name, info, language_num, seed):
            calls.append((task_name, dict(info), language_num, seed))
            return {"seen": [f"Put {info['{A}']} {info['{side}']} of {info['{B}']} ({k})" for k in range(language_num)],
                    "unseen": ["unseen one"]}

        argv = ["--raw", str(self.raw), "--zarr", str(self.zarr_path), "--no-audio", "--language-num", "12"]
        with patch.object(sys, "argv", ["make_instructions.py", *argv]), patch.object(make, "generate_texts", fake_generate):
            make.main()
        self.assertEqual([c[0] for c in calls], ["place_a2b_side"] * self.episodes)
        self.assertEqual([c[3] for c in calls], list(range(self.episodes)))
        self.assertEqual(calls[1][1]["{side}"], "right")
        texts = json.loads((self.bundle / "episode1/text.json").read_text())
        self.assertEqual(len(texts["seen"]), 12)
        self.assertIn("right", texts["seen"][0])
        self.assertIsNone(json.loads((self.bundle / "manifest.json").read_text())["instructions_dir"])


if __name__ == "__main__":
    unittest.main()

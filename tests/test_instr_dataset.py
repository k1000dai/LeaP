"""CPU checks for instruction sampling in RobotDataset and the instr-aware trainer."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml
import zarr

from leap.robotwin.checkpoint import load_policy
from leap.robotwin.data import RobotDataset

ROOT = Path(__file__).resolve().parents[1]
DIM = 768
VARIANTS = {"text": 10, "sketch": 1, "audio": 3}
MODALITY_ID = {"text": 0, "sketch": 1, "audio": 2}
EPISODE_ENDS = [11, 22]


def write_dataset(root):
    """Two 11-frame episodes plus a bundle whose vectors encode (episode, modality, variant)."""
    rng = np.random.default_rng(0)
    zarr_path = root / "demo.zarr"
    group = zarr.open_group(str(zarr_path), mode="w")
    data = group.create_group("data")
    data.create_dataset("state", data=rng.normal(size=(22, 14)).astype(np.float32))
    data.create_dataset("action", data=rng.normal(size=(22, 14)).astype(np.float32))
    data.create_dataset("point_cloud", data=rng.normal(size=(22, 32, 6)).astype(np.float32))
    group.create_group("meta").create_dataset("episode_ends", data=np.asarray(EPISODE_ENDS, dtype=np.int64))
    bundle = {}
    for name, count in VARIANTS.items():
        array = np.zeros((2, count, DIM), dtype=np.float32)
        for episode in range(2):
            for variant in range(count):
                array[episode, variant] = 100 * episode + 10 * MODALITY_ID[name] + variant
        bundle[name] = array
    instr_dir = root / "demo.instr"
    instr_dir.mkdir()
    np.savez(instr_dir / "embeddings.npz", side=np.asarray([0, 1], dtype=np.int64),
             episode_ends=np.asarray(EPISODE_ENDS, dtype=np.int64), **bundle)
    return zarr_path, instr_dir / "embeddings.npz"


def decode(vector):
    code = int(round(float(vector[0])))
    return code // 100, (code % 100) // 10, code % 10


class InstrDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="leap-instr-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.zarr_path, self.npz = write_dataset(self.root)

    def test_without_bundle_nothing_changes(self):
        dataset = RobotDataset(self.zarr_path)
        self.assertIsNone(dataset.instr)
        self.assertNotIn("instr", dataset[0]["obs"])
        self.assertNotIn("instr", dataset.shape_meta["obs"])
        with self.assertRaises(RuntimeError):
            dataset.instr_stats()

    def test_samples_come_from_the_window_episode(self):
        dataset = RobotDataset(self.zarr_path, instr_path=self.npz)
        self.assertEqual(dataset.shape_meta["obs"]["instr"], {"shape": [DIM]})
        np.testing.assert_array_equal(dataset.instr_side, [0, 1])
        self.assertEqual(len(dataset), 22)
        torch.manual_seed(0)
        seen = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            instr = sample["obs"]["instr"]
            self.assertEqual(instr.shape, (DIM,))
            self.assertEqual(instr.dtype, torch.float32)
            self.assertTrue(torch.equal(instr, instr[0].expand(DIM)))
            episode, modality, variant = decode(instr)
            self.assertEqual(episode, index // 11)
            self.assertLess(variant, list(VARIANTS.values())[modality])
            seen.add(modality)
            self.assertEqual(sample["obs"]["agent_pos"].shape, (8, 14))
        for _ in range(50):
            seen.add(decode(dataset.sample_instruction(1))[1])
        self.assertEqual(seen, {0, 1, 2})
        # Drawing uses torch's RNG, so seeding reproduces the draws.
        torch.manual_seed(4)
        first = [decode(dataset[i]["obs"]["instr"]) for i in range(22)]
        torch.manual_seed(4)
        second = [decode(dataset[i]["obs"]["instr"]) for i in range(22)]
        self.assertEqual(first, second)

    def test_modality_restriction_and_validation_split(self):
        dataset = RobotDataset(self.zarr_path, instr_path=self.npz, instr_modalities=("sketch",))
        for index in range(len(dataset)):
            self.assertEqual(decode(dataset[index]["obs"]["instr"])[1:], (1, 0))
        dataset = RobotDataset(self.zarr_path, instr_path=self.npz, instr_modalities=["audio", "text"],
                               val_ratio=0.5, seed=0)
        validation = dataset.get_validation_dataset()
        self.assertEqual(len(dataset) + len(validation), 22)
        self.assertIs(validation.instr, dataset.instr)
        train_episode = int(dataset.train_episodes[0])
        for index in range(len(validation)):
            episode, modality, _ = decode(validation[index]["obs"]["instr"])
            self.assertEqual(episode, 1 - train_episode)
            self.assertIn(modality, (0, 2))
        mean, std = dataset.instr_stats()
        self.assertEqual(mean.shape, (DIM,))
        self.assertEqual(std.shape, (DIM,))
        self.assertEqual(mean.dtype, torch.float32)
        codes = [100 * train_episode + 10 * MODALITY_ID[name] + v
                 for name in ("audio", "text") for v in range(VARIANTS[name])]
        self.assertAlmostEqual(float(mean[0]), np.mean(codes), places=4)
        self.assertAlmostEqual(float(std[0]), np.std(codes), places=4)
        self.assertTrue((std >= 1e-6).all())      # constant dims are floored, not zero

    def test_episode_subset_and_bad_bundles_raise(self):
        dataset = RobotDataset(self.zarr_path, instr_path=self.npz, episodes=1)
        self.assertEqual(len(dataset.instr["text"]), 1)
        np.testing.assert_array_equal(dataset.instr_side, [0])
        with self.assertRaises(ValueError):
            RobotDataset(self.zarr_path, instr_path=self.npz, instr_modalities=("text", "video"))
        with self.assertRaises(ValueError):
            RobotDataset(self.zarr_path, instr_path=self.npz, instr_modalities=())
        with np.load(self.npz) as bundle:
            arrays = dict(bundle)
        bad = self.root / "bad.npz"
        np.savez(bad, **dict(arrays, episode_ends=np.asarray([10, 22], dtype=np.int64)))
        with self.assertRaises(ValueError):
            RobotDataset(self.zarr_path, instr_path=bad)
        np.savez(bad, **dict(arrays, sketch=arrays["sketch"][:, :, :512]))
        with self.assertRaises(ValueError):
            RobotDataset(self.zarr_path, instr_path=bad)

    def test_train_smoke_with_steer_both_config(self):
        config = yaml.safe_load((ROOT / "configs/steer_both.yaml").read_text())
        self.assertEqual(config["policy"]["instr_mode"], "both")
        config["policy"].update(encoder_output_dim=16, encoder_hidden_dims=[16, 16],
                                 flow_net_hidden_dim=32, flow_net_num_layers=1,
                                 flow_net_time_embed_dim=16)
        config["training"]["warmup_steps"] = 0
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        command = [sys.executable, "-m", "leap.robotwin.train", "--config", str(config_path),
                   "--data", str(self.zarr_path), "--device", "cpu", "--batch-size", "2",
                   "--epochs", "1", "--max-steps", "2"]
        # The bundle next to the Zarr is found without --instr.
        subprocess.run(command + ["--output", str(self.root / "run"), "--instr-modalities", "text,audio"],
                       check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        run = json.loads((self.root / "run/run.json").read_text())
        self.assertEqual(run["instr"]["mode"], "both")
        self.assertEqual(run["instr"]["modalities"], ["text", "audio"])
        self.assertEqual(Path(run["instr"]["path"]), self.npz.resolve())
        self.assertEqual(run["policy_config"]["shape_meta"]["obs"]["instr"], {"shape": [DIM]})
        policy = load_policy(self.root / "run/checkpoints/latest.ckpt")
        self.assertEqual(policy.instr_mode, "both")
        dataset = RobotDataset(self.zarr_path, instr_path=self.npz, instr_modalities=("text", "audio"),
                               val_ratio=config["training"]["val_ratio"], seed=0)
        mean, std = dataset.instr_stats()
        torch.testing.assert_close(policy.instr_mean, mean)
        torch.testing.assert_close(policy.instr_std, std)
        obs = {"agent_pos": torch.zeros(1, 3, 14), "point_cloud": torch.zeros(1, 3, 32, 6),
               "instr": dataset.sample_instruction(0)[None]}
        self.assertEqual(policy.predict_action(obs)["action"].shape, (1, 6, 14))
        # --instr accepts the bundle directory as well as the npz file.
        subprocess.run(command + ["--output", str(self.root / "run2"), "--instr", str(self.npz.parent)],
                       check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        run = json.loads((self.root / "run2/run.json").read_text())
        self.assertEqual(Path(run["instr"]["path"]), self.npz.resolve())
        self.assertEqual(run["instr"]["modalities"], ["text", "sketch", "audio"])
        # Without a bundle the trainer refuses to start an instruction-conditioned run.
        failed = subprocess.run(command + ["--output", str(self.root / "run3"), "--instr",
                                           str(self.root / "missing.instr")],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("instruction bundle", failed.stdout)
        self.assertFalse((self.root / "run3").exists())


if __name__ == "__main__":
    unittest.main()

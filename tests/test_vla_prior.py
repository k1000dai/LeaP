"""CPU checks for leap.vla.prior (VLAPrior) and scripts/vla/train_vla_prior.py on a synthetic reversal dataset."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from leap.vla.prior import (LOG_VAR_MAX, LOG_VAR_MIN, STD_FLOOR, VLAPrior, gaussian_nll, masked_mse,
                            standard_normal_nll)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_vla_prior", ROOT / "scripts/vla/train_vla_prior.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

SMALL = dict(feat_dim=16, instr_dim=8, hidden=(32, 16), horizon=5, action_dim=3)


def make_prior(seed=0, **overrides):
    torch.manual_seed(seed)
    return VLAPrior(**dict(SMALL, **overrides))


def write_synthetic_dataset(folder, tasks=((12, 3, 2), (8, 2, 1)), feat_dim=24, horizon=6, instr_dim=16, seed=0,
                            valid_mask=True, modalities=("text", "sketch", "audio")):
    """task_i.npz files with (samples, demos, text variants) per task in the reversal-dataset layout.

    Like build_reversal_dataset.py, ``chunk_pad_mask`` marks real steps (True) and ``demo_id`` holds hdf5 demo
    numbers (task 1 uses the non-contiguous ids 3 and 7); ``valid_mask=False`` stores True = padded instead.
    ``modalities`` are the instr_* arrays embed_vla_instructions.py has added (empty = builder output only).
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (n, d, text_variants) in enumerate(tasks):
        ids = np.arange(d) if i == 0 else np.arange(d) * 4 + 3
        demo = np.repeat(ids, int(math.ceil(n / d)))[:n]
        valid = np.arange(horizon)[None, :] < rng.integers(1, horizon + 1, size=(n, 1))   # real steps, then padding
        arrays = dict(feat=rng.normal(size=(n, feat_dim)).astype(np.float32),
                      state=rng.normal(size=(n, 8)).astype(np.float32),
                      noise=rng.normal(size=(n, horizon, 7)).astype(np.float32),
                      chunk_pad_mask=valid if valid_mask else ~valid,
                      demo_id=demo, t=np.arange(n),
                      recon_err=np.where(rng.random(n) < 0.5, rng.random(n), np.nan).astype(np.float32),
                      task_id=i, task_language=f"pick up the bowl of task {i}")
        variants = {"text": text_variants, "sketch": 1, "audio": 1}
        for modality in modalities:
            arrays[f"instr_{modality}"] = rng.normal(size=(d, variants[modality], instr_dim)).astype(np.float32)
        if modalities:
            arrays["instr_modalities"] = np.asarray(list(modalities))
        np.savez(folder / f"task_{i}.npz", **arrays)
    return folder


class VLAPriorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.feat = torch.randn(4, 16)
        self.instr = torch.randn(4, 8)
        self.noise = torch.randn(4, 5, 3)
        self.pad = torch.zeros(4, 5, dtype=torch.bool)
        self.pad[:, 3:] = True          # the last two chunk steps are padding
        self.pad[0] = True              # one fully padded row

    def test_forward_shapes_and_clamp(self):
        prior = make_prior()
        mu, log_var = prior(self.feat, self.instr)
        self.assertEqual(mu.shape, (4, 5, 3))
        self.assertEqual(log_var.shape, (4, 5, 3))
        self.assertTrue((log_var >= LOG_VAR_MIN).all() and (log_var <= LOG_VAR_MAX).all())
        self.assertEqual(prior.config()["instr_mode"], "concat")
        with self.assertRaises(ValueError):
            prior(self.feat)                       # concat needs an instruction
        with self.assertRaises(ValueError):
            prior(self.feat, self.instr[:, :4])    # wrong instruction width
        plain = make_prior(instr_mode="none")
        mu, _ = plain(self.feat)
        self.assertEqual(mu.shape, (4, 5, 3))
        with self.assertRaises(ValueError):
            plain(self.feat, self.instr)           # none takes no instruction
        with self.assertRaises(ValueError):
            VLAPrior(instr_mode="both")

    def test_nll_masking(self):
        prior = make_prior()
        mu, log_var = prior(self.feat, self.instr)
        full = gaussian_nll(mu, log_var, self.noise)
        masked = gaussian_nll(mu, log_var, self.noise, self.pad)
        per_element = 0.5 * (log_var + (self.noise - mu) ** 2 / log_var.exp() + math.log(2 * math.pi))
        keep = ~self.pad.unsqueeze(-1).expand_as(mu)
        torch.testing.assert_close(full, per_element.mean())
        torch.testing.assert_close(masked, per_element[keep].mean())
        # Changing the labels inside the padding leaves the masked NLL unchanged.
        noisy = self.noise.clone()
        noisy[~keep] += 100.0
        torch.testing.assert_close(gaussian_nll(mu, log_var, noisy, self.pad), masked)
        torch.testing.assert_close(masked_mse(mu, noisy, self.pad), ((self.noise - mu) ** 2)[keep].mean())
        self.assertGreater(prior.nll(mu, log_var, self.noise, self.pad, mse_weight=0.5).item(), masked.item())
        # An N(0, I) source scores 0.5 * log(2 pi) on zero labels and the empirical value otherwise.
        self.assertAlmostEqual(float(standard_normal_nll(torch.zeros(2, 5, 3))), 0.5 * math.log(2 * math.pi), places=6)
        self.assertAlmostEqual(float(standard_normal_nll(self.noise, self.pad)),
                               float((0.5 * (self.noise ** 2 + math.log(2 * math.pi)))[keep].mean()), places=5)

    def test_sampling(self):
        prior = make_prior()
        mu, log_var = prior(self.feat, self.instr)
        torch.testing.assert_close(prior.sample(mu, log_var, sigma_scale=0.0), mu)
        sample = prior.sample(mu, log_var)
        self.assertEqual(sample.shape, mu.shape)
        self.assertFalse(torch.equal(sample, mu))
        generator = torch.Generator().manual_seed(5)
        first = prior.sample(mu, log_var, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(prior.sample(mu, log_var, generator=generator), first)
        # sigma_scale scales the deviation from mu.
        half = prior.sample(mu, log_var, sigma_scale=0.5, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(half - mu, 0.5 * (first - mu))

    def test_steered_gain(self):
        prior = make_prior().eval()   # dropout off: steered() must reproduce the plain forward
        prior.set_stats(np.zeros(SMALL["feat_dim"]), np.ones(SMALL["feat_dim"]),
                        instr_mean=np.full(SMALL["instr_dim"], 0.5), instr_std=np.full(SMALL["instr_dim"], 2.0))
        mu, log_var = prior(self.feat, self.instr)
        # gain 1 is the plain forward; gain 0 is the response to the mean instruction (a standardised zero).
        mu1, lv1 = prior.steered(self.feat, self.instr, gain=1.0)
        torch.testing.assert_close(mu1, mu)
        torch.testing.assert_close(lv1, log_var)
        anchor = torch.full_like(self.instr, 0.5)
        mu0, _ = prior.steered(self.feat, self.instr, gain=0.0)
        torch.testing.assert_close(mu0, prior(self.feat, anchor)[0])
        self.assertFalse(torch.allclose(mu0, mu))
        # gain 2 extrapolates linearly along the instruction's shift; log_var stays the instruction's own.
        mu2, lv2 = prior.steered(self.feat, self.instr, gain=2.0)
        torch.testing.assert_close(mu2, mu0 + 2.0 * (mu - mu0))
        torch.testing.assert_close(lv2, log_var)
        # instr_mode none ignores the gain.
        none = make_prior(instr_mode="none").eval()
        mu_n, lv_n = none(self.feat)
        mu_g, lv_g = none.steered(self.feat, None, gain=3.0)
        torch.testing.assert_close(mu_g, mu_n)
        torch.testing.assert_close(lv_g, lv_n)

    def test_stats_standardisation(self):
        prior = make_prior()
        feat_mean, feat_std = np.full(16, 2.0, dtype=np.float32), np.full(16, 4.0, dtype=np.float32)
        instr_mean, instr_std = np.full(8, -1.0, dtype=np.float32), np.full(8, 0.5, dtype=np.float32)
        prior.set_stats(feat_mean, feat_std, instr_mean, instr_std)
        raw = make_prior()                     # same weights, identity statistics
        raw.load_state_dict({k: v for k, v in prior.state_dict().items() if "mean" not in k and "std" not in k},
                            strict=False)
        prior.eval()
        raw.eval()
        expected = raw((self.feat - 2.0) / 4.0, (self.instr + 1.0) / 0.5)
        actual = prior(self.feat, self.instr)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        prior.set_stats(feat_mean, np.zeros(16), instr_mean, np.zeros(8))
        self.assertTrue(torch.equal(prior.feat_std, torch.full((16,), STD_FLOOR)))
        with self.assertRaises(ValueError):
            prior.set_stats(feat_mean, feat_std, instr_mean)        # instr stats come together
        with self.assertRaises(ValueError):
            prior.set_stats(np.zeros(3), np.ones(3))

    def test_save_load_round_trip(self):
        prior = make_prior(hidden=(12,), dropout=0.0)
        prior.set_stats(np.arange(16, dtype=np.float32), np.linspace(1, 2, 16, dtype=np.float32),
                        np.zeros(8, dtype=np.float32), np.full(8, 3.0, dtype=np.float32))
        prior.eval()
        mu, log_var = prior(self.feat, self.instr)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prior.pt"
            prior.save(path, {"step": 7})
            loaded, extra = VLAPrior.load(path)
        self.assertEqual(extra, {"step": 7})
        self.assertEqual(loaded.config(), prior.config())
        self.assertFalse(loaded.training)
        torch.testing.assert_close(loaded.feat_std, prior.feat_std)
        mu2, log_var2 = loaded(self.feat, self.instr)
        torch.testing.assert_close(mu2, mu)
        torch.testing.assert_close(log_var2, log_var)


class TrainScriptTest(unittest.TestCase):
    def run_training(self, folder, *extra):
        out = Path(folder) / "prior-run"
        train.main(["--data", str(Path(folder) / "data"), "--out", str(out), "--steps", "5", "--batch", "4",
                    "--log-every", "2", "--hidden", "8", "--device", "cpu", "--val-fraction", "0.34", *extra])
        return out

    def test_concat_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            write_synthetic_dataset(Path(temp) / "data")
            out = self.run_training(temp, "--instr-mode", "concat", "--modalities", "text,sketch")
            self.assertTrue((out / "prior.pt").exists() and (out / "best.pt").exists())
            rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [1, 2, 4, 5])
            for row in rows:
                self.assertEqual({"val_nll_text", "val_nll_sketch", "val_nll", "train_nll", "train_loss"} - set(row),
                                 set())
                self.assertTrue(all(math.isfinite(row[k]) for k in ("val_nll", "train_nll")))
            run = json.loads((out / "run.json").read_text())
            self.assertEqual(run["model"]["instr_mode"], "concat")
            self.assertEqual(run["model"]["feat_dim"], 24)
            self.assertEqual(run["model"]["instr_dim"], 16)
            self.assertEqual(run["n_train"] + run["n_val"], 20)
            for task in run["split"].values():          # every task keeps >= 1 demo in each split
                self.assertGreaterEqual(len(task["val_demos"]), 1)
                self.assertLess(len(task["val_demos"]), task["demos"])
            self.assertIn("gauss_val_nll", run["references"])
            self.assertIn("empirical_val_nll", run["references"])
            self.assertEqual(run["recon_err"]["measured"] > 0, True)
            self.assertEqual(run["best"]["step"], min(rows, key=lambda r: r["val_nll"])["step"])
            prior, extra = VLAPrior.load(out / "prior.pt")
            self.assertEqual(extra["step"], 5)
            self.assertEqual(extra["modalities"], ["text", "sketch"])
            self.assertFalse(torch.equal(prior.feat_mean, torch.zeros(24)))      # stats were stored
            self.assertFalse(torch.equal(prior.instr_std, torch.ones(16)))
            mu, log_var = prior(torch.randn(2, 24), torch.randn(2, 16))
            self.assertEqual(mu.shape, (2, 6, 7))
            with self.assertRaises(SystemExit):     # refuses to clobber a finished run
                self.run_training(temp, "--instr-mode", "concat")

    def test_none_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            write_synthetic_dataset(Path(temp) / "data")
            out = self.run_training(temp, "--instr-mode", "none")
            prior, _ = VLAPrior.load(out / "prior.pt")
            self.assertEqual(prior.instr_mode, "none")
            self.assertTrue(torch.equal(prior.instr_mean, torch.zeros(768)))
            rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertNotIn("val_nll_text", rows[-1])
            self.assertTrue(math.isfinite(rows[-1]["val_nll"]))

    def test_helpers(self):
        with tempfile.TemporaryDirectory() as temp:
            tasks = train.load_tasks(write_synthetic_dataset(Path(temp) / "data"), ["text", "audio"])
            self.assertEqual([t["task_id"] for t in tasks], [0, 1])
            self.assertEqual([t["mask_convention"] for t in tasks], ["valid", "valid"])
            self.assertFalse(tasks[0]["pad"][:, 0].any())                   # normalised to True = padded
            self.assertEqual(tasks[1]["demo_id"].tolist(), [3, 3, 3, 3, 7, 7, 7, 7])
            self.assertEqual(tasks[1]["demo_row"].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
            masks, split = train.split_demos(tasks, 0.34, seed=0)
            self.assertEqual(split["1"]["demos"], 2)
            data = train.stack_dataset(tasks, masks, ["text", "audio"])
            self.assertEqual(data["instr"]["text"].shape, (5, 2, 16))     # variants padded to the max (2)
            self.assertEqual(data["instr_count"]["text"].tolist(), [2, 2, 2, 1, 1])
            self.assertEqual(data["demo_row"].max(), 4)
            rng = np.random.default_rng(0)
            instr = train.gather_instr(data, np.arange(20), ["text", "audio"], rng=rng)
            self.assertEqual(instr.shape, (20, 16))
            fixed = train.gather_instr(data, np.array([0, 17]), ["text", "audio"], modality="audio")
            np.testing.assert_array_equal(fixed[0], tasks[0]["instr_audio"][0, 0])
            np.testing.assert_array_equal(fixed[1], tasks[1]["instr_audio"][1, 0])      # demo 7 -> row 1
            stats = train.compute_stats(data, ["text", "audio"])
            self.assertEqual(stats["feat_mean"].shape, (24,))
            self.assertTrue((stats["instr_std"] >= STD_FLOOR).all())
            with self.assertRaises(SystemExit):        # a modality the files do not carry
                train.load_tasks(Path(temp) / "data", ["video"])
            with self.assertRaises(FileNotFoundError):
                train.load_tasks(Path(temp) / "missing", [])
            # A file that already stores True = padded loads to the same masks.
            padded = train.load_tasks(write_synthetic_dataset(Path(temp) / "padded", valid_mask=False), [])
            self.assertEqual(padded[0]["mask_convention"], "padded")
            np.testing.assert_array_equal(padded[0]["pad"], tasks[0]["pad"])

    def test_builder_output_without_embeddings(self):
        """A file straight from build_reversal_dataset.py (no instr_* keys) trains with --instr-mode none only."""
        with tempfile.TemporaryDirectory() as temp:
            write_synthetic_dataset(Path(temp) / "data", modalities=())
            out = self.run_training(temp, "--instr-mode", "none")
            self.assertTrue((out / "prior.pt").exists())
            with self.assertRaises(SystemExit) as raised:
                train.load_tasks(Path(temp) / "data", ["text"])
            self.assertIn("embed_vla_instructions.py", str(raised.exception))
            # Partially embedded (text only): sketch is refused, text loads.
            write_synthetic_dataset(Path(temp) / "partial", modalities=("text",))
            with self.assertRaises(SystemExit):
                train.load_tasks(Path(temp) / "partial", ["text", "sketch"])
            self.assertEqual(len(train.load_tasks(Path(temp) / "partial", ["text"])), 2)

    def test_padded_mask_polarity(self):
        valid = np.array([[True, True, False], [True, False, False]])
        mask, convention = train.padded_mask(valid, "x")
        self.assertEqual(convention, "valid")
        np.testing.assert_array_equal(mask, ~valid)
        mask, convention = train.padded_mask(~valid, "x")
        self.assertEqual(convention, "padded")
        np.testing.assert_array_equal(mask, ~valid)
        with self.assertRaises(ValueError):
            train.padded_mask(np.array([[True, False], [False, True]]), "x")


if __name__ == "__main__":
    unittest.main()

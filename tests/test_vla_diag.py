"""CPU checks for scripts/vla/probe_cf_labels.py and scripts/vla/score_prior_forward.py on synthetic counterfactual data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import unittest.mock

import numpy as np
import torch

from leap.vla import counterfactual as C
from leap.vla.prior import VLAPrior

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"diag_{name}", ROOT / "scripts" / "vla" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_script("probe_cf_labels")
score = load_script("score_prior_forward")

FEAT, HORIZON, ACTION, INSTR = 12, 5, 3, 8


def write_cf_dataset(folder, tasks=(0, 1), bowl2_tasks=(0, 1), states=6, queries=4, shift=3.0, seed=0,
                     feat_dim=FEAT, horizon=HORIZON, action_dim=ACTION, instr_dim=INSTR, extra_bowl1_task=2):
    """task_{100 b + t}.npz files in the counterfactual layout with a planted bowl shift in the noise labels.

    Rows: every init state (demo_id = bowl * 1000 + state) at queries t = 0, 2, ...; the label of (state, q) is a
    state/query-specific base plus ``shift`` times a fixed direction for bowl 2 (minus it for bowl 1) plus small noise.
    The bowl-1 feat at q = 0 equals the bowl-2 feat (shared observation).  Instruction tables: one 768-like vector per
    demo, bowl-specific for text and audio, demo-specific for sketch.
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    direction = rng.normal(size=(horizon, action_dim))
    direction /= np.linalg.norm(direction)
    all_tasks = [(t, 1) for t in tasks] + [(t, 2) for t in bowl2_tasks]
    if extra_bowl1_task is not None:
        all_tasks.append((extra_bowl1_task, 1))
    written = {}
    for t, bowl in all_tasks:
        rng_task = np.random.default_rng(1000 + t)                            # same bases for both bowls of a task
        base_feat = rng_task.normal(size=(states, feat_dim))
        base_noise = rng_task.normal(size=(states, queries, horizon, action_dim))
        feat, noise, demo, tt = [], [], [], []
        for s in range(states):
            for q in range(queries):
                f = base_feat[s] + (0.0 if q == 0 else 0.3 * rng.normal(size=feat_dim)) + (0.0 if q == 0 else 0.5 * (bowl - 1.5))
                n = base_noise[s, q] + (1.0 if bowl == 2 else -1.0) * shift * direction + 0.1 * rng.normal(size=(horizon, action_dim))
                feat.append(f)
                noise.append(n)
                demo.append(bowl * 1000 + s)
                tt.append(2 * q)
        n_rows = len(feat)
        e_bowl = np.random.default_rng(10 * bowl + t).normal(size=instr_dim)
        arrays = dict(feat=np.asarray(feat, dtype=np.float32), noise=np.asarray(noise, dtype=np.float32),
                      state=np.zeros((n_rows, 8), np.float32), demo_id=np.asarray(demo, dtype=np.int64),
                      t=np.asarray(tt, dtype=np.int64), chunk_pad_mask=np.zeros((n_rows, horizon), bool),
                      recon_err=np.full(n_rows, np.nan, np.float32), task_id=np.int64(t),
                      task_language=np.str_(f"pick up the black bowl of task {t} and place it on the plate"),
                      bowl=np.full(n_rows, bowl, np.int64), episode=np.asarray(demo, np.int64) % 1000,
                      success=np.ones(n_rows, bool), dataset_index=np.int64(100 * bowl + t),
                      instr_text=np.repeat(e_bowl[None, None], states, axis=0).astype(np.float32),
                      instr_audio=np.repeat((e_bowl + 0.1)[None, None], states, axis=0).astype(np.float32),
                      instr_sketch=(e_bowl[None, None] + 0.05 * rng.normal(size=(states, 1, instr_dim))).astype(np.float32),
                      instr_modalities=np.asarray(["text", "sketch", "audio"]))
        np.savez(folder / f"task_{100 * bowl + t}.npz", **arrays)
        written[100 * bowl + t] = arrays
    return written


def save_prior(path, instr_mode, seed=0):
    torch.manual_seed(seed)
    prior = VLAPrior(feat_dim=FEAT, instr_dim=INSTR, hidden=(16,), horizon=HORIZON, action_dim=ACTION, instr_mode=instr_mode)
    prior.set_stats(np.zeros(FEAT), np.ones(FEAT), np.zeros(INSTR), np.ones(INSTR))
    prior.eval()
    prior.save(path, {"step": 1})
    return prior


class ProbeLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        self.arrays = write_cf_dataset(self.data)
        self.files = probe.load_cf_dataset(self.data)

    def test_loader_and_pairs(self):
        self.assertEqual(sorted(self.files), [100, 101, 102, 200, 201])
        self.assertEqual(probe.paired_tasks(self.files), [0, 1])
        f1, f2 = self.files[100], self.files[200]
        np.testing.assert_array_equal(f1["init_state"], f1["demo_id"] % 100)
        np.testing.assert_array_equal(f1["query"], np.tile(np.arange(4), 6))
        np.testing.assert_array_equal(probe.query_index(np.array([5, 5, 7, 5]), np.array([4, 0, 0, 2])), [2, 0, 0, 1])
        pairs = probe.matched_pairs(f1, f2, max_query=1)
        self.assertEqual(len(pairs), 12)                      # 6 init states x queries 0, 1
        self.assertEqual(pairs[0], (0, 0, 0, 0))
        self.assertEqual(pairs[1], (0, 1, 1, 1))
        for s, q, r1, r2 in pairs:
            self.assertEqual((f1["init_state"][r1], f1["query"][r1]), (s, q))
            self.assertEqual((f2["init_state"][r2], f2["query"][r2]), (s, q))
        self.assertEqual(probe.matched_pairs(f1, f2, max_query=0)[-1][0], 5)
        # Repeats (demo 1100 = repeat 1 of init state 0) pair with every bowl-2 rollout of that state.
        f1b = dict(f1, init_state=np.where(np.arange(len(f1["demo_id"])) < 4, 0, f1["init_state"]),
                   query=f1["query"], demo_id=f1["demo_id"])
        self.assertEqual(len(probe.matched_pairs(f1b, f2, max_query=0)), 6)

    def test_label_shift_detects_planted_direction(self):
        f1, f2 = self.files[100], self.files[200]
        pairs = probe.matched_pairs(f1, f2, max_query=3)
        stats = probe.label_shift(f1, f2, pairs, queries=(0, 3))
        for q in (0, 3):
            self.assertEqual(stats[q]["pairs"], 6)
            self.assertGreater(stats[q]["pairwise_cos"], 0.9)        # one shared direction
            self.assertGreater(stats[q]["shared_fraction"], 0.9)
            self.assertGreater(stats[q]["d_prime"], 1.0)
            self.assertEqual(len(stats[q]["per_step"]), HORIZON)
            self.assertEqual(len(stats[q]["per_dim"]), ACTION)
            self.assertAlmostEqual(np.linalg.norm(stats[q]["mean_delta"]), 6.0, delta=0.2)      # 2 x the planted shift
        self.assertEqual(list(probe.band_summary(stats[0]["per_step"], bands=((0, 2), (2, 5)))), ["0-1", "2-4"])
        # No planted shift: no consistent direction, small effect size.
        write_cf_dataset(Path(self.temp.name) / "flat", shift=0.0, seed=3)
        files = probe.load_cf_dataset(Path(self.temp.name) / "flat")
        flat = probe.label_shift(files[100], files[200], probe.matched_pairs(files[100], files[200], 0), queries=(0,))[0]
        self.assertLess(abs(flat["pairwise_cos"]), 0.5)
        self.assertLess(flat["d_prime"], stats[0]["d_prime"] / 3)
        cos = probe.direction_cosines({0: stats[0]["mean_delta"], 1: -stats[0]["mean_delta"]})
        self.assertAlmostEqual(cos["mean_off_diagonal"], -1.0, places=5)

    def test_probes(self):
        f1, f2 = self.files[100], self.files[200]
        pairs = probe.matched_pairs(f1, f2, max_query=3)
        X, y, groups = probe.probe_rows(f1, f2, pairs)
        self.assertEqual(X.shape, (48, HORIZON * ACTION))
        self.assertEqual(set(y.tolist()), {-1.0, 1.0})
        result = probe.loso_probe(X, y, groups)
        self.assertEqual((result["n"], result["groups"]), (48, 6))
        self.assertGreater(result["ridge"]["accuracy"], 0.9)
        self.assertGreater(result["mean_direction"]["accuracy"], 0.9)
        self.assertGreater(result["mean_direction"]["mean_margin"], 0.0)
        # Shuffled labels: near chance.
        rng = np.random.default_rng(0)
        shuffled = probe.loso_probe(X, rng.permutation(y), groups)
        self.assertLess(shuffled["mean_direction"]["accuracy"], 0.8)
        folds = probe.group_folds(np.array([0, 0, 1, 1, 2, 2, 3, 3]), k=2)
        self.assertEqual(len(folds), 2)
        self.assertEqual(sorted(np.concatenate([f[1] for f in folds]).tolist()), list(range(8)))
        self.assertEqual(len(probe.group_folds(np.array([0, 0, 1, 1]))), 2)
        # Primal and dual ridge agree.
        Xs, ys = X[:10, :4], y[:10]
        w_dual, b_dual = probe.ridge_fit(Xs[:3], ys[:3], 1.0)              # n <= d
        w_primal, b_primal = probe.ridge_fit(Xs, ys, 1.0)                 # n > d
        self.assertEqual(w_dual.shape, w_primal.shape)
        Xc = Xs - Xs.mean(axis=0)
        expected = np.linalg.solve(Xc.T @ Xc + np.eye(4), Xc.T @ (ys - ys.mean()))
        np.testing.assert_allclose(w_primal, expected)
        np.testing.assert_allclose(Xc.T @ np.linalg.solve(Xc @ Xc.T + np.eye(10), ys - ys.mean()), expected, atol=1e-8)

    def test_shift_stats_and_r2(self):
        delta = np.random.default_rng(0).normal(size=(4, HORIZON, ACTION))
        stats = probe.shift_stats(0.5 * delta, delta)
        self.assertAlmostEqual(stats["norm_ratio"], 0.5, places=6)
        self.assertAlmostEqual(stats["projection"], 0.5, places=6)
        self.assertAlmostEqual(stats["cos"], 1.0, places=6)
        orthogonal = probe.shift_stats(np.flip(delta, axis=1) * 0 + 1e-9, delta)
        self.assertAlmostEqual(orthogonal["norm_ratio"], 0.0, places=6)
        noise = np.random.default_rng(1).normal(size=(10, HORIZON, ACTION))
        self.assertAlmostEqual(probe.r_squared(noise, noise), 1.0, places=6)
        self.assertAlmostEqual(probe.r_squared(np.broadcast_to(noise.mean(axis=0), noise.shape), noise), 0.0, places=6)

    def test_prior_reproduction_and_main(self):
        concat = save_prior(Path(self.temp.name) / "concat.pt", "concat")
        none = save_prior(Path(self.temp.name) / "none.pt", "none")
        f1, f2 = self.files[100], self.files[200]
        pairs = probe.matched_pairs(f1, f2, max_query=3)
        val1, val2 = np.zeros(len(f1["demo_id"]), bool), np.zeros(len(f2["demo_id"]), bool)
        val1[:8] = True
        cell = probe.prior_reproduction(concat, f1, f2, pairs, "text", 2.0, val1, val2)
        for key in ("norm_ratio", "projection", "cos", "r2_train", "r2_val", "delta_mu_norm"):
            self.assertTrue(np.isfinite(cell[key]), key)
        self.assertEqual(cell["val_rows"], 8)
        self.assertEqual(cell["shift_train"]["pairs"] + cell["shift_val"]["pairs"], len(pairs))
        self.assertEqual(cell["shift_val"]["pairs"], 8)                  # the 8 held-out bowl-1 rows each pair once
        self.assertIn("instr_bowl2", cell["mu_says_bowl2"])
        self.assertLess(abs(cell["embedding"]["cos_standardised"]), 1.0)
        # The cached gain formula reproduces VLAPrior.steered.
        rows = np.array([p[2] for p in pairs[:5]])
        e1, _ = probe.embeddings_for(f1, f2, pairs[:5], "text")
        cache = probe.PriorCache(concat)
        expected, _ = concat.steered(torch.as_tensor(f1["feat"][rows]), torch.as_tensor(e1), 3.0)
        np.testing.assert_allclose(cache.steered(f1, rows, e1, ("text", 1), 3.0), expected.numpy(), rtol=1e-5, atol=1e-6)
        # gain 0 removes the instruction: no shift at all.
        zero = probe.prior_reproduction(concat, f1, f2, pairs, "text", 0.0, val1, val2)
        self.assertAlmostEqual(zero["delta_mu_norm"], 0.0, places=5)
        # sketch embeddings are per demo; audio works too; a missing modality raises.
        probe.prior_reproduction(concat, f1, f2, pairs, "sketch", 1.0, val1, val2)
        with self.assertRaises(KeyError):
            probe.embeddings_for(f1, {k: v for k, v in f2.items() if k != "instr_text"}, pairs, "text")
        ref = probe.prior_reproduction(none, f1, f2, [p for p in pairs if p[1] == 0], None, 1.0, val1, val2)
        self.assertAlmostEqual(ref["delta_mu_norm"], 0.0, places=5)     # same observation at q0 -> no shift
        self.assertNotIn("mu_says_bowl2", ref)
        # End to end: tables printed, JSON + markdown written, the trainer split holds out one demo per file.
        out = Path(self.temp.name) / "probe" / "report.json"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            report = probe.main(["--data", str(self.data), "--prior", str(Path(self.temp.name) / "concat.pt"),
                                 str(Path(self.temp.name) / "none.pt"), "--gains", "1,3", "--max-query", "2",
                                 "--modalities", "text,sketch", "--out", str(out)])
        text = captured.getvalue()
        self.assertIn("| pooled |", text)
        self.assertIn("text/g3", text)
        self.assertTrue(out.exists() and out.with_suffix(".md").exists())
        saved = json.loads(out.read_text())
        self.assertEqual(sorted(saved["tasks"]), ["0", "1"])
        self.assertEqual(saved["tasks"]["0"]["pairs"], 18)
        self.assertIn("concat", saved["tasks"]["0"]["priors"])
        self.assertIn("none", saved["tasks"]["0"]["priors"]["none"])
        self.assertGreater(saved["pooled"]["probe_q0"]["ridge"]["accuracy"], 0.9)
        self.assertEqual(report["tasks"]["1"]["priors"]["concat"]["text/g1"]["q0"]["val_rows"], 2)
        with contextlib.redirect_stdout(io.StringIO()):
            held = probe.main(["--data", str(self.data), "--val-init-states", "0,1", "--max-query", "0", "--tasks", "1"])
        self.assertEqual(sorted(held["tasks"]), ["1"])
        val = probe.val_masks(self.files, 0.1, 0, [0, 1])
        self.assertEqual(int(val[100].sum()), 8)


if __name__ == "__main__":
    unittest.main()


# ---- forward score -------------------------------------------------------------------------------------------------------
SIZE = 8
MARKER = 255
MEAN = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
STD = np.linspace(0.5, 1.5, 7, dtype=np.float32)
EEF = np.array([0.0, 0.0, 1.0])
BOWL1 = np.array([0.3, 0.0, 0.9])      # +x from the gripper
BOWL2 = np.array([0.0, 0.3, 0.9])      # +y from the gripper
PLATE = np.array([0.1, 0.1, 0.9])


def planted_chunk(bowl, seed):
    """[50, 7] un-normalised chunk whose xy actions head for the bowl."""
    rng = np.random.default_rng(seed)
    chunk = 0.05 * rng.normal(size=(50, 7)).astype(np.float32)
    chunk[:, 0 if bowl == 1 else 1] += 0.5
    return chunk


def write_episode(folder, task, bowl, episode, success=True, lifted=None, seed=0):
    """A tiny rollout file in the counterfactual layout (policy-view images carry a MARKER at the last pixel)."""
    rng = np.random.default_rng(seed)
    steps, queries = 6, 2
    image = np.full((steps, SIZE, SIZE, 3), 100, dtype=np.uint8)
    image[:, -1, -1] = MARKER                              # raw_images rotates it to pixel [0, 0]
    state = np.zeros((steps, 8), np.float32)
    state[:, :3] = EEF
    state[:, 6:] = [0.04, -0.04]
    chunk = np.stack([planted_chunk(bowl, seed + i) for i in range(queries)])
    arrays = {"agentview": image, "wrist": image.copy(), "state": state, "action": np.zeros((steps, 7), np.float32),
              "positions": np.repeat(np.stack([BOWL1, BOWL2, PLATE])[None], steps, axis=0), "on_plate": np.zeros(steps, bool),
              "pred_step": np.array([0, 2]), "chunk": chunk, "noise": rng.normal(size=(queries, 50, 32)).astype(np.float32),
              "executed": np.array([True, False]), "cam_intrinsic": np.eye(3), "cam_extrinsic": np.eye(3, 4)}
    scalars = {"task_id": task, "bowl": bowl, "episode": episode, "prompt": "p", "task_language": "l", "success": success,
               "lifted_bowl": bowl if lifted is None else lifted, "bowl_choice": "target", "steps": steps}
    C.save_episode(C.episode_path(folder, task, bowl, episode), arrays, scalars)
    return chunk


def fake_flow(record):
    """pi05_flow stand-in: the flow is x_action = 0.5 (x_noise - 1), i.e. noise = 2 a + 1 (as the reversal test's fake)."""
    from leap.vla import pi05_flow as real
    api = types.SimpleNamespace(Observation=real.Observation, pad_noise=real.pad_noise)
    bundle = types.SimpleNamespace(config=types.SimpleNamespace(chunk_size=50, max_action_dim=32), device="cpu")
    api.load_pi05 = lambda checkpoint, device, **kwargs: bundle

    def prepare_batches(bundle, observations, flip_images=True):
        observations = list(observations)
        record["prompts"].append(observations[0].prompt)
        record["batch_sizes"].append(len(observations))
        for obs in observations:
            assert obs.images["image"][0, 0, 0] == MARKER       # raw orientation restored before the flip
            assert obs.state.shape == (8,)
        return {"state": np.stack([o.state for o in observations])}

    def prefix_features(bundle, batch, ctx=None):
        feat = np.zeros((batch["state"].shape[0], 4104), np.float32)
        feat[:, :8] = batch["state"]
        return feat

    def forward_from_noise(bundle, ctx, noise, num_steps=10):
        assert noise.shape[0] == ctx["n"]
        record["num_steps"].append(num_steps)
        return 0.5 * (noise - 1.0)

    def normalize_actions(bundle, actions7):
        out = np.zeros(np.asarray(actions7).shape[:2] + (32,), np.float32)
        out[:, :, :7] = (np.asarray(actions7) - MEAN) / STD
        return torch.as_tensor(out)

    api.prepare_batches = prepare_batches
    api.prefix_context = lambda bundle, batch: {"n": batch["state"].shape[0]}
    api.prefix_features = prefix_features
    api.forward_from_noise = forward_from_noise
    api.normalize_actions = normalize_actions
    api.unnormalize_actions = lambda bundle, x: torch.as_tensor(np.asarray(x)[..., :7] * STD + MEAN)
    return api


class ScorePriorForwardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.rollouts, self.data = root / "rollouts", root / "dataset"
        self.chunks = {}
        for task in (1, 3):
            for s in range(3):
                for bowl in (1, 2):
                    self.chunks[(task, bowl, s)] = write_episode(self.rollouts, task, bowl, s, seed=10 * task + 2 * s + bowl)
        write_episode(self.rollouts, 1, 1, 3, seed=7)                                  # state 3: bowl 2 failed
        write_episode(self.rollouts, 1, 2, 3, success=False, seed=8)
        write_episode(self.rollouts, 1, 1, 4, seed=9)                                  # state 4: bowl 2 lifted bowl 1 first
        write_episode(self.rollouts, 1, 2, 4, lifted=1, seed=11)
        # Dataset files: stored labels = 2 norm(chunk) + 1 at t = 0 (the fake flow regenerates the chunk from them).
        self.data.mkdir()
        for task in (1, 3):
            for bowl in (1, 2):
                demos = [C.demo_id_for(bowl, s) for s in range(3)]
                labels = np.stack([2.0 * (self.chunks[(task, bowl, s)][0] - MEAN) / STD + 1.0 for s in range(3)])
                e_bowl = np.random.default_rng(bowl).normal(size=INSTR).astype(np.float32)
                np.savez(self.data / f"task_{100 * bowl + task}.npz", noise=np.concatenate([labels, labels]).astype(np.float32),
                         demo_id=np.array(demos + demos), t=np.array([0] * 3 + [2] * 3),
                         instr_text=np.repeat(e_bowl[None, None], 3, axis=0),
                         instr_sketch=np.repeat(e_bowl[None, None], 3, axis=0) + 0.01,
                         instr_audio=np.repeat(e_bowl[None, None], 3, axis=0), instr_modalities=np.asarray(["text", "sketch", "audio"]))
        torch.manual_seed(0)
        prior = VLAPrior(feat_dim=4104, instr_dim=INSTR, hidden=(8,), horizon=50, action_dim=7, instr_mode="concat")
        prior.set_stats(np.zeros(4104), np.ones(4104), np.zeros(INSTR), np.ones(INSTR))
        prior.save(root / "concat.pt")
        none = VLAPrior(feat_dim=4104, instr_dim=INSTR, hidden=(8,), horizon=50, action_dim=7, instr_mode="none")
        none.set_stats(np.zeros(4104), np.ones(4104))
        none.save(root / "none.pt")
        self.record = {"prompts": [], "batch_sizes": [], "num_steps": []}
        patcher = unittest.mock.patch.object(score, "flow_api", lambda: fake_flow(self.record))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_classifiers(self):
        c1, c2 = self.chunks[(1, 1, 0)][0], self.chunks[(1, 2, 0)][0]
        bowls, margin = score.nearest_chunk(np.stack([c1, c2, 0.5 * (c1 + c2) + 0.01 * (c2 - c1)]), c1, c2)
        np.testing.assert_array_equal(bowls, [1, 2, 2])
        self.assertLess(margin[0], 0)
        self.assertGreater(margin[1], 0)
        bowls, margin = score.geometric_bowl(np.stack([c1, c2]), EEF, BOWL1, BOWL2, horizon=20)
        np.testing.assert_array_equal(bowls, [1, 2])
        self.assertGreater(margin[1], 0.5)

    def test_state_selection(self):
        chosen, skipped = score.select_states(self.rollouts, 1, [0, 1, 2, 3, 4, 5])
        self.assertEqual([s for s, _, _ in chosen], [0, 1, 2])
        self.assertIn("bowl2 failed", skipped[3])
        self.assertIn("lifted bowl 1", skipped[4])
        self.assertTrue(skipped[5].startswith("missing"))
        chosen, _ = score.select_states(self.rollouts, 1, [4], require_lifted=False)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(score.episode_of(2, 7), 207)
        self.assertEqual(score.parse_spec("3,0-1"), [0, 1, 3])

    def test_dataset_helpers(self):
        files = score.load_dataset_files(self.data, [1, 3, 5])
        self.assertEqual(sorted(files), [101, 103, 201, 203])
        label = score.stored_label(files, 1, 2, 1)
        np.testing.assert_allclose(label, 2.0 * (self.chunks[(1, 2, 1)][0] - MEAN) / STD + 1.0, rtol=1e-5)
        self.assertIsNone(score.stored_label(files, 1, 2, 9))
        self.assertIsNone(score.stored_label(files, 5, 1, 0))
        self.assertEqual(score.instruction_embedding(files, 1, 2, 0, "text").shape, (INSTR,))
        self.assertEqual(score.instruction_embedding(files, 1, 1, 9, "text").shape, (INSTR,))     # unknown demo: row 0
        self.assertIsNone(score.instruction_embedding(files, 1, 1, 9, "sketch"))                 # sketch is per demo
        with self.assertRaises(KeyError):
            score.instruction_embedding(files, 5, 1, 0, "text")

    def test_main_end_to_end(self):
        root = Path(self.temp.name)
        out = root / "score" / "concat.json"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            report = score.main(["--rollouts", str(self.rollouts), "--data", str(self.data), "--prior", str(root / "concat.pt"),
                                 str(root / "none.pt"), "--modalities", "text,sketch", "--gains", "1,2", "--samples", "3",
                                 "--tasks", "1,3", "--init-states", "0-4", "--batch-size", "4", "--num-steps", "3",
                                 "--device", "cpu", "--out", str(out)])
        summary = report["summary"]
        conditions = ["stored-b1", "stored-b2", "gauss", "concat/text/g1/instr-b1", "concat/text/g1/instr-b2",
                      "concat/text/g2/instr-b1", "concat/text/g2/instr-b2", "concat/sketch/g1/instr-b1", "concat/sketch/g1/instr-b2",
                      "concat/sketch/g2/instr-b1", "concat/sketch/g2/instr-b2", "none/none"]
        self.assertEqual(list(summary), conditions)
        # The stored labels regenerate their own chunks: bowl 1 -> 0 %, bowl 2 -> 100 % with both classifiers.
        for key in ("p_bowl2_nearest", "p_bowl2_geometric"):
            self.assertEqual(summary["stored-b1"]["overall"][key], 0.0)
            self.assertEqual(summary["stored-b2"]["overall"][key], 1.0)
        self.assertEqual(summary["stored-b2"]["overall"]["states"], 6)            # 3 usable states x 2 tasks
        self.assertEqual(summary["gauss"]["overall"]["samples"], 18)               # 3 draws x 6 states
        self.assertEqual(summary["concat/text/g1/instr-b2"]["tasks"]["1"]["samples"], 9)
        self.assertEqual(summary["none/none"]["overall"]["samples"], 18)
        self.assertEqual(report["skipped"]["1"], {"3": "bowl2 failed", "4": "bowl2 lifted bowl 1 first"})
        self.assertEqual(set(self.record["prompts"]), {"pick up the black bowl and place it on the plate"})
        self.assertEqual(set(self.record["batch_sizes"]), {4})
        self.assertEqual(set(self.record["num_steps"]), {3})
        for info in report["info"].values():
            self.assertEqual(info["state_gap"], 0.0)
            self.assertEqual(info["recorded_geometric"], [1, 2])
            self.assertLess(max(info["recon_err"].values()), 1e-5)
        self.assertTrue(out.exists() and out.with_suffix(".md").exists())
        self.assertIn("| stored-b2 |", captured.getvalue())
        saved = json.loads(out.read_text())
        self.assertEqual(saved["args"]["samples"], 3)
        self.assertIn("1", saved["per_state"])
        # References can be skipped; the init-state list restricts the states.
        with contextlib.redirect_stdout(io.StringIO()):
            small = score.main(["--rollouts", str(self.rollouts), "--data", str(self.data), "--prior", str(root / "concat.pt"),
                                "--tasks", "1", "--init-states", "1,2", "--samples", "2", "--batch-size", "3", "--device", "cpu",
                                "--no-references"])
        self.assertEqual(list(small["summary"]), ["concat/text/g1/instr-b1", "concat/text/g1/instr-b2"])
        self.assertEqual(small["summary"]["concat/text/g1/instr-b2"]["overall"]["states"], 2)

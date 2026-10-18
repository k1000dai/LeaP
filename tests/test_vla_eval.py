"""CPU checks for scripts/vla/eval_vla_steer.py (rollout loop with fakes), collect_vla_results.py and the campaign driver."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from leap.vla.prior import VLAPrior

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/vla"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_script("eval_vla_steer")
collector = load_script("collect_vla_results")

FEAT_DIM, INSTR_DIM, HORIZON, ACTION_DIM, MAX_ACTION_DIM = 16, 8, 50, 7, 32
LANGUAGES = {0: "pick up the black bowl next to the ramekin and place it on the plate",
             1: "pick up the black bowl on the stove and place it on the plate"}


# ---- fakes ------------------------------------------------------------------------------------------------------
class FakeTask:
    def __init__(self, language):
        self.language = language


class FakeSuite:
    name = "libero_spatial"

    def get_task(self, task_id):
        return FakeTask(LANGUAGES[task_id])


class FakeEnv:
    """LiberoEnv-like: succeeds after ``succeed_at`` steps on task 0, never on task 1."""

    def __init__(self, task_id, succeed_at=13):
        self.task_id = task_id
        self.succeed_at = succeed_at if task_id == 0 else None
        self.init_state_id = 0
        self.steps = 0
        self.resets = []
        self.actions = []
        self.closed = False

    def _obs(self):
        value = (self.steps * 7 + self.init_state_id) % 255
        return {"pixels": {"image": np.full((8, 8, 3), value, dtype=np.uint8),
                           "image2": np.full((8, 8, 3), 255 - value, dtype=np.uint8)},
                "robot_state": {"eef": {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                                "gripper": {"qpos": np.array([0.04, -0.04])}}}

    def reset(self, seed=None):
        self.resets.append((self.init_state_id, seed))
        self.steps = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.steps += 1
        success = self.succeed_at is not None and self.steps >= self.succeed_at
        return self._obs(), float(success), success, False, {"is_success": success}

    def close(self):
        self.closed = True


class FakeTracker:
    """Task 0 lifts the target bowl at step 5, task 1 the distractor at step 8."""

    def __init__(self, layout):
        self.layout = layout
        self.choice = "none"

    def update(self, env):
        if self.choice == "none":
            if env.task_id == 0 and env.steps >= 5:
                self.choice = "target"
            elif env.task_id == 1 and env.steps >= 8:
                self.choice = "distractor"
        return self.choice


def make_fake_scene(record):
    def make_env(suite, task_id, prompt_override=None, **kwargs):
        record["prompt_override"].append(prompt_override)
        env = FakeEnv(task_id)
        record["envs"].append(env)
        return env

    return types.SimpleNamespace(
        NEUTRAL_PROMPT="pick up the black bowl and place it on the plate",
        load_suite=lambda name: FakeSuite(),
        make_env=make_env,
        scene_layout=lambda env: {"akita_black_bowl_1": np.zeros(3), "plate_1": np.ones(3), "initial_z": {}},
        agentview_camera=lambda env: (np.eye(3), np.eye(3, 4)),
        BowlChoiceTracker=FakeTracker,
        max_steps=lambda suite: 30,
        policy_view=lambda image: np.ascontiguousarray(np.asarray(image)[::-1, ::-1]),
        quat_to_axis_angle=lambda quat: np.zeros(3, dtype=np.float32),
        state_vector=lambda raw: np.zeros(8, dtype=np.float32))


def make_fake_flow(record):
    def prepare_batch(bundle, images, state, prompt):
        record["prompts"].append(prompt)
        record["states"].append(np.asarray(state))
        return {"image": images["image"], "state": state}

    def prefix_features(bundle, batch, ctx=None):
        record["ctx_reuse"].append(ctx is not None)
        return torch.full((1, FEAT_DIM), float(batch["image"][0, 0, 0]) / 255.0)

    def pad_noise(noise7, max_action_dim=MAX_ACTION_DIM, generator=None):
        noise7 = torch.as_tensor(noise7, dtype=torch.float32)
        return torch.cat([noise7, torch.randn(*noise7.shape[:-1], max_action_dim - noise7.shape[-1])], dim=-1)

    def sample_actions_with_noise(bundle, ctx, noise=None, num_steps=10):
        record["noises"].append(None if noise is None else noise.detach().clone())
        record["num_steps"].append(num_steps)
        return np.zeros((1, HORIZON, ACTION_DIM), dtype=np.float32)

    return types.SimpleNamespace(
        load_pi05=lambda checkpoint, device, n_action_steps=10: {"checkpoint": checkpoint, "device": device},
        prepare_batch=prepare_batch, prefix_context=lambda bundle, batch: {"ctx": True},
        prefix_features=prefix_features, pad_noise=pad_noise, sample_actions_with_noise=sample_actions_with_noise)


def make_fake_instructions(record):
    def embed_instruction(embedder, modality, *, text=None, sketch=None, audio=None):
        record["embedded"].append((modality, text, sketch is not None, audio is not None))
        return np.full(INSTR_DIM, {"text": 0.1, "sketch": 0.2, "audio": 0.3}[modality], dtype=np.float32)

    return types.SimpleNamespace(
        task_text=lambda language: " ".join(str(language).split()),
        build_sketch=lambda rgb, cam, layout: Image.fromarray(np.asarray(rgb, dtype=np.uint8)),
        build_audio=lambda text, tts: np.zeros(160, dtype=np.float32),
        embed_instruction=embed_instruction)


def save_prior(path, instr_mode):
    torch.manual_seed(0)
    prior = VLAPrior(feat_dim=FEAT_DIM, instr_dim=INSTR_DIM, hidden=(8,), horizon=HORIZON, action_dim=ACTION_DIM,
                     instr_mode=instr_mode, dropout=0.0)
    prior.save(path, {"step": 1})
    return path


class EvalLoopTest(unittest.TestCase):
    def setUp(self):
        self.record = {"prompt_override": [], "envs": [], "prompts": [], "states": [], "noises": [], "num_steps": [],
                       "ctx_reuse": [], "embedded": [], "embedders": [], "tts": []}
        self.patches = [
            patch.object(evaluator, "import_pi05_flow", lambda: make_fake_flow(self.record)),
            patch.object(evaluator, "import_libero_scene", lambda: make_fake_scene(self.record)),
            patch.object(evaluator, "import_instructions", lambda: make_fake_instructions(self.record)),
            patch.object(evaluator, "make_tts", lambda voice: self.record["tts"].append(voice) or "tts"),
            patch.object(evaluator, "make_embedder",
                         lambda model, device, python=None: self.record["embedders"].append((model, device, python))
                         or "embedder"),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in self.patches])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"

    def run_main(self, *extra, tasks="0", episodes=2):
        evaluator.main(["--tasks", tasks, "--episodes", str(episodes), "--out", str(self.out), "--device", "cpu",
                        "--n-action-steps", "4", "--num-steps", "3", *extra])
        lines = [json.loads(l) for l in (self.out / "episodes.jsonl").read_text().splitlines()]
        return lines, json.loads((self.out / "summary.json").read_text())

    def test_neutral_gauss(self):
        lines, summary = self.run_main("--condition", "neutral-gauss", tasks="0,1")
        self.assertEqual([(l["task_id"], l["episode"]) for l in lines], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertTrue(all(noise is None for noise in self.record["noises"]))
        self.assertEqual(set(self.record["num_steps"]), {3})
        neutral = "pick up the black bowl and place it on the plate"
        self.assertEqual(self.record["prompt_override"], [neutral, neutral])
        self.assertEqual(set(self.record["prompts"]), {neutral})
        self.assertEqual(self.record["embedders"], [])
        # Task 0 succeeds at step 13 (the policy is called every 4 steps -> stops at 13); task 1 runs to max_steps.
        self.assertEqual([l["success"] for l in lines], [True, True, False, False])
        self.assertEqual([l["steps"] for l in lines], [13, 13, 30, 30])
        self.assertEqual([l["bowl_choice"] for l in lines], ["target", "target", "distractor", "distractor"])
        self.assertEqual([l["task_language"] for l in lines][::2], [LANGUAGES[0], LANGUAGES[1]])
        self.assertEqual(lines[0]["seed"], 0)
        self.assertEqual(lines[3]["seed"], 1001)
        self.assertNotIn("prior_mu_abs", lines[0])
        # Episode e is run on init state e; the env is closed per task; the state comes from robot_state.
        self.assertEqual([r[0] for r in self.record["envs"][0].resets], [0, 1])
        self.assertTrue(all(env.closed for env in self.record["envs"]))
        np.testing.assert_allclose(self.record["states"][0], [0.1, 0.2, 0.3, 0, 0, 0, 0.04, -0.04], atol=1e-6)
        self.assertEqual(summary["overall"]["episodes"], 4)
        self.assertAlmostEqual(summary["overall"]["success_rate"], 0.5)
        self.assertAlmostEqual(summary["overall"]["bowl_choice_accuracy"], 0.5)
        self.assertEqual(summary["overall"]["bowl_choice_counts"], {"target": 2, "distractor": 2, "none": 0})
        self.assertEqual(summary["tasks"]["1"]["success_rate"], 0.0)
        self.assertEqual(summary["tasks"]["0"]["task_language"], LANGUAGES[0])
        self.assertEqual(summary["condition"], "neutral-gauss")
        self.assertEqual(summary["modality"], "none")

    def test_first_episode_offsets_init_states(self):
        lines, summary = self.run_main("--condition", "neutral-gauss", "--first-episode", "20", tasks="0,1")
        self.assertEqual([(l["task_id"], l["episode"]) for l in lines], [(0, 20), (0, 21), (1, 20), (1, 21)])
        self.assertEqual([r[0] for r in self.record["envs"][0].resets], [20, 21])
        self.assertEqual(summary["first_episode"], 20)
        self.assertEqual(summary["overall"]["episodes"], 4)
        # A rerun with the default window runs the states 0-1 and summarises only those.
        self.record["envs"].clear()
        lines, summary = self.run_main("--condition", "neutral-gauss", tasks="0,1")
        self.assertEqual([r[0] for r in self.record["envs"][0].resets], [0, 1])
        self.assertEqual(len(lines), 8)
        self.assertEqual(summary["overall"]["episodes"], 4)

    def test_full_prompt_uses_task_language(self):
        lines, _ = self.run_main("--condition", "full-prompt", episodes=1)
        self.assertEqual(self.record["prompt_override"], [None])
        self.assertEqual(set(self.record["prompts"]), {LANGUAGES[0]})
        self.assertEqual(lines[0]["prompt"], LANGUAGES[0])

    def test_neutral_prior_with_sketch_and_resume(self):
        prior = save_prior(Path(self.temp.name) / "concat.pt", "concat")
        lines, summary = self.run_main("--condition", "neutral-prior", "--instr-modality", "sketch", "--prior",
                                       str(prior), "--sigma-scale", "1.0", "--dump-instructions", "--voice", "v.onnx",
                                       "--embedder", "emb", "--embedder-python", "/some/python")
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(noise is not None and tuple(noise.shape) == (1, HORIZON, MAX_ACTION_DIM)
                            for noise in self.record["noises"]))
        self.assertTrue(all(self.record["ctx_reuse"]))
        self.assertEqual(self.record["embedders"], [("emb", "cpu", "/some/python")])
        self.assertEqual(self.record["tts"], [])                                   # sketch needs no TTS
        self.assertEqual(self.record["embedded"], [("sketch", LANGUAGES[0], True, False)] * 2)   # once per episode
        self.assertIn("prior_mu_abs", lines[0])
        self.assertIn("prior_sigma", lines[0])
        self.assertEqual(lines[0]["modality"], "sketch")
        self.assertEqual(lines[0]["sigma_scale"], 1.0)
        folder = self.out / "instructions"
        self.assertTrue((folder / "task00_ep000.png").exists() and (folder / "task00_ep001.txt").exists())
        self.assertEqual((folder / "task00_ep000.txt").read_text().strip(), LANGUAGES[0])
        self.assertEqual(summary["prior"], str(prior))
        # Restart: nothing new is run, the log is unchanged; a third episode appends exactly one line.
        calls = len(self.record["noises"])
        again, _ = self.run_main("--condition", "neutral-prior", "--instr-modality", "sketch", "--prior", str(prior))
        self.assertEqual(again, lines)
        self.assertEqual(len(self.record["noises"]), calls)
        more, summary = self.run_main("--condition", "neutral-prior", "--instr-modality", "sketch", "--prior",
                                      str(prior), episodes=3)
        self.assertEqual([l["episode"] for l in more], [0, 1, 2])
        self.assertEqual(summary["overall"]["episodes"], 3)

    def test_sigma_zero_is_deterministic(self):
        prior = save_prior(Path(self.temp.name) / "none.pt", "none")
        self.run_main("--condition", "neutral-prior", "--prior", str(prior), "--sigma-scale", "0", episodes=1)
        first = [n[0, :, :ACTION_DIM] for n in self.record["noises"]]
        self.assertGreater(len(first), 1)
        self.assertEqual(self.record["embedders"], [])
        # Features change with the frame, so mu differs between calls, but a second run reproduces it exactly.
        self.record["noises"].clear()
        (self.out / "episodes.jsonl").unlink()
        self.run_main("--condition", "neutral-prior", "--prior", str(prior), "--sigma-scale", "0", episodes=1)
        second = [n[0, :, :ACTION_DIM] for n in self.record["noises"]]
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_prior_modality_mismatch(self):
        prior = save_prior(Path(self.temp.name) / "none.pt", "none")
        with self.assertRaises(SystemExit):
            self.run_main("--condition", "neutral-prior", "--instr-modality", "text", "--prior", str(prior))
        prior = save_prior(Path(self.temp.name) / "concat.pt", "concat")
        with self.assertRaises(SystemExit):
            self.run_main("--condition", "neutral-prior", "--prior", str(prior))

    def test_argument_validation(self):
        self.assertEqual(evaluator.parse_tasks("0-3"), [0, 1, 2, 3])
        self.assertEqual(evaluator.parse_tasks("5,1, 3"), [1, 3, 5])
        self.assertEqual(evaluator.parse_tasks("7"), [7])
        with self.assertRaises(ValueError):
            evaluator.parse_tasks(" ")
        base = ["--out", "x", "--device", "cpu"]
        for bad in (["--condition", "neutral-gauss", "--instr-modality", "text"],   # gauss takes no instruction
                    ["--condition", "neutral-prior"],                                # needs --prior
                    []):                                                             # needs --condition
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                evaluator.parse_args(base + bad)
        args = evaluator.parse_args(base + ["--condition", "neutral-prior", "--prior", "p.pt", "--tasks", "2-3"])
        self.assertEqual(args.task_ids, [2, 3])
        self.assertEqual(args.embedder_device, "cpu")
        self.assertEqual(evaluator.episode_seed(2, 3, 4), 2_003_004)


class SubprocessEmbedderTest(unittest.TestCase):
    """The worker protocol end to end, with a stand-in interpreter that fakes leap.steer.embedder."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            wrapper = Path(temp) / "fake_python"
            wrapper.write_text(f"""#!{sys.executable}
import runpy, sys, types
import numpy as np
sys.path.insert(0, {str(ROOT)!r})
import leap.steer
fake = types.ModuleType("leap.steer.embedder")
class InstructionEmbedder:
    def __init__(self, model, device="cpu"):
        print("chatter on stdout while loading", model, device)
    def embed_text(self, texts):
        return np.full((len(texts), 4), 1.0, dtype=np.float32)
    def embed_images(self, paths):
        assert all(p.endswith(".png") for p in paths), paths
        return np.full((len(paths), 4), 2.0, dtype=np.float32)
    def embed_audio(self, paths):
        assert all(p.endswith(".wav") for p in paths), paths
        return np.full((len(paths), 4), 3.0, dtype=np.float32)
fake.InstructionEmbedder = InstructionEmbedder
sys.modules["leap.steer.embedder"] = fake
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
""")
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
            embedder = evaluator.SubprocessEmbedder(wrapper, "fake-model", "cpu")
            try:
                np.testing.assert_array_equal(embedder.embed_text(["a", "b"]), np.full((2, 4), 1.0))
                image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
                np.testing.assert_array_equal(embedder.embed_images([image]), np.full((1, 4), 2.0))
                np.testing.assert_array_equal(embedder.embed_audio([np.zeros(16, dtype=np.float32)]),
                                              np.full((1, 4), 3.0))
                with self.assertRaises(RuntimeError):
                    embedder._request("video", [])
            finally:
                embedder.close()
            self.assertEqual(embedder.process.returncode, 0)


class CollectResultsTest(unittest.TestCase):
    def write_summary(self, folder, name, condition, modality, sigma, success, choice, episodes=4):
        run = Path(folder) / name
        run.mkdir(parents=True)
        tasks = {"0": {"episodes": episodes // 2, "success_rate": success, "bowl_choice_accuracy": choice,
                       "task_language": LANGUAGES[0]},
                 "1": {"episodes": episodes // 2, "success_rate": success / 2, "bowl_choice_accuracy": choice,
                       "task_language": LANGUAGES[1]}}
        (run / "summary.json").write_text(json.dumps({
            "condition": condition, "modality": modality, "sigma_scale": sigma, "seed": 0,
            "overall": {"episodes": episodes, "success_rate": success, "bowl_choice_accuracy": choice,
                        "bowl_choice_counts": {"target": 3, "distractor": 1, "none": 0}, "mean_steps": 20.0},
            "tasks": tasks}))

    def test_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            self.write_summary(temp, "neutral-prior-sketch-s1", "neutral-prior", "sketch", 1.0, 0.75, 0.75)
            self.write_summary(temp, "full-prompt-none", "full-prompt", "none", 1.0, 1.0, 1.0)
            self.write_summary(temp, "neutral-prior-sketch-s0", "neutral-prior", "sketch", 0.0, 0.5, 0.5)
            self.write_summary(temp, "neutral-gauss-none", "neutral-gauss", "none", 1.0, 0.25, 0.25)
            prior = Path(temp) / "prior-concat"
            prior.mkdir()
            (prior / "run.json").write_text(json.dumps({
                "args": {"instr_mode": "concat", "modalities": ["text", "sketch"]}, "steps": 10, "n_train": 9,
                "n_val": 1, "best": {"step": 8, "val_nll": 1.2}, "final": {"val_nll": 1.3, "val_nll_text": 1.25,
                                                                           "val_nll_sketch": 1.35},
                "references": {"gauss_val_nll": 1.4, "empirical_val_nll": 1.5},
                "recon_err": {"measured": 3, "median": 0.05}}))
            collector.main(["--out-dir", temp])
            runs = collector.load_runs(temp)
            self.assertEqual([r["run"] for r in runs], ["full-prompt-none", "neutral-gauss-none",
                                                        "neutral-prior-sketch-s1", "neutral-prior-sketch-s0"])
            table = (Path(temp) / "results.md").read_text()
            self.assertIn("| full-prompt-none | full-prompt | none |  | 4 | 100.0 | 100.0 | 3/1/0 |", table)
            self.assertIn("| neutral-prior-sketch-s0 | neutral-prior | sketch | 0 | 4 | 50.0 | 50.0 |", table)
            self.assertIn("| task 0 | task 1 |", table)
            self.assertIn("| neutral-gauss-none | 25.0 | 12.5 |", table)
            self.assertIn("| prior-concat | concat | 10 | 9/1 | 1.200 | text 1.250, sketch 1.350 | 1.400 | 1.500 | 0.050 |",
                          table)
            results = json.loads((Path(temp) / "results.json").read_text())
            self.assertEqual(len(results["runs"]), 4)
            self.assertEqual(results["priors"][0]["best_val_nll"], 1.2)
            csv_lines = (Path(temp) / "results_per_task.csv").read_text().splitlines()
            self.assertEqual(len(csv_lines), 1 + 4 * 2)
            self.assertTrue(csv_lines[1].startswith("full-prompt-none,full-prompt,none,1.0,0,"))


class CounterfactualCampaignScriptTest(unittest.TestCase):
    SCRIPT = SCRIPTS / "run_vla_counterfactual.sh"

    def test_syntax(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True)
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_dry_run_cells_and_held_out_states(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, DRY_RUN="1", LOG_DIR=str(Path(temp) / "logs"), OUT_DIR=str(Path(temp) / "out"),
                       DATA=str(Path(temp) / "data"), VENV=str(Path(temp) / "no-venv"), TASKS="0-1", EPISODES="2",
                       MODALITIES="text sketch", EMBED_PYTHON="/opt/py",
                       CELLS="full-prompt-b2 neutral-prior-concat-sketch-s1-b2 neutral-prior-none-none-s1-b1")
            result = subprocess.run(["bash", str(self.SCRIPT)], env=env, capture_output=True, text=True, check=True)
            out = result.stdout
            data = str(Path(temp) / "data")
            self.assertIn(f"/opt/py scripts/vla/embed_vla_instructions.py --data {data} "
                          "--voice data/voices/en_US-lessac-medium.onnx --device cuda --tasks 100-109,200-209 "
                          "--modalities text sketch --embedder-model google/embeddinggemma-2", out)
            self.assertIn(f"train_vla_prior.py --data {data} --out {Path(temp) / 'out' / 'prior-cf-concat'} "
                          "--instr-mode concat --modalities text,sketch", out)
            self.assertIn("--instr-mode none", out)
            evals = [line for line in out.splitlines() if "eval_vla_steer.py" in line]
            self.assertEqual(len(evals), 3)
            self.assertTrue(all("--first-episode 20" in line for line in evals))
            self.assertIn("--condition full-prompt --instr-modality none --target-bowl 2", evals[0])
            self.assertIn("--condition neutral-prior --instr-modality sketch --target-bowl 2", evals[1])
            self.assertIn(f"--prior {Path(temp) / 'out' / 'prior-cf-concat' / 'best.pt'} --sigma-scale 1", evals[1])
            self.assertIn("--embedder-python /opt/py", evals[1])
            self.assertIn("--condition neutral-prior --instr-modality none --target-bowl 1", evals[2])
            self.assertIn("prior-cf-none/best.pt", evals[2])
            self.assertNotIn("neutral-gauss", out.split("=== eval")[1] if "=== eval" in out else "")
            self.assertIn("campaign finished", out)


class CampaignScriptTest(unittest.TestCase):
    SCRIPT = SCRIPTS / "run_vla_campaign.sh"

    def test_syntax(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True)
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_dry_run_lists_every_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, DRY_RUN="1", LOG_DIR=str(Path(temp) / "logs"), OUT_DIR=str(Path(temp) / "out"),
                       DATA=str(Path(temp) / "data"), VENV=str(Path(temp) / "no-venv"), TASKS="0-1", EPISODES="2",
                       MODALITIES="text sketch", SIGMAS="1 0", EMBED_PYTHON="/opt/py", STAGES="data prior eval results")
            result = subprocess.run(["bash", str(self.SCRIPT)], env=env, capture_output=True, text=True, check=True)
            out = result.stdout
            data = str(Path(temp) / "data")
            self.assertIn(f"python scripts/vla/build_reversal_dataset.py --raw data/vla/raw/libero_spatial --out {data} "
                          "--checkpoint lerobot/pi05_libero_finetuned_v044 --suite libero_spatial --tasks 0-1 --stride 2 "
                          "--num-steps 10 --batch-size 16 --device cuda --seed 0 "
                          "--voice data/voices/en_US-lessac-medium.onnx", out)
            self.assertIn(f"/opt/py scripts/vla/embed_vla_instructions.py --data {data} "
                          "--voice data/voices/en_US-lessac-medium.onnx --device cuda --tasks 0-1 "
                          "--modalities text sketch --embedder-model google/embeddinggemma-2", out)
            self.assertNotIn("--stage", out)
            self.assertIn("PYTHONPATH=" + str(ROOT), out)
            self.assertIn("MUJOCO_GL=egl PYOPENGL_PLATFORM=egl", out)
            self.assertIn("train_vla_prior.py", out)
            self.assertIn("--instr-mode none", out)
            self.assertIn("--instr-mode concat", out)
            self.assertIn("--modalities text,sketch", out)
            for name in ("full-prompt-none", "neutral-gauss-none", "neutral-prior-none-s1", "neutral-prior-none-s0",
                         "neutral-prior-text-s1", "neutral-prior-sketch-s0"):
                self.assertIn(f"=== eval {name}", out)
            self.assertNotIn("neutral-prior-audio", out)
            self.assertIn("--condition neutral-prior --instr-modality sketch", out)
            self.assertIn("--prior " + str(Path(temp) / "out/prior-concat/best.pt"), out)
            self.assertIn("--prior " + str(Path(temp) / "out/prior-none/best.pt"), out)
            self.assertIn("--embedder-python /opt/py", out)
            self.assertIn("--sigma-scale 0", out)
            self.assertIn("collect_vla_results.py", out)
            self.assertIn("campaign finished", out)
            self.assertFalse((Path(temp) / "out/prior-none").exists())        # dry run creates no outputs


if __name__ == "__main__":
    unittest.main()

"""CPU checks for leap.vla.counterfactual, scripts/vla/rollout_counterfactual.py, scripts/vla/reverse_rollouts.py and
the --target-bowl option of scripts/vla/eval_vla_steer.py (frozen VLA, LIBERO env, embedder and TTS replaced by fakes).

The BDDL init blocks and languages below are those of the 10 LIBERO-Spatial tasks (hf-libero, suite order 0-9).
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from leap.vla import counterfactual as C
from leap.vla import instructions as I
from leap.vla import libero_scene as S
from leap.vla.prior import VLAPrior

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "vla"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"cf_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout = load_script("rollout_counterfactual")
reverse = load_script("reverse_rollouts")
evaluator = load_script("eval_vla_steer")
trainer = load_script("train_vla_prior")
embed = load_script("embed_vla_instructions")

# task id -> (language, bowl-1 init predicate, bowl-1 region, bowl-2 region) as in the hf-libero BDDL files.
TASKS = {
    0: ("pick up the black bowl between the plate and the ramekin and place it on the plate", "On",
        "main_table_between_plate_ramekin_region", "main_table_next_to_ramekin_region"),
    1: ("pick up the black bowl next to the ramekin and place it on the plate", "On",
        "main_table_next_to_ramekin_region", "main_table_next_to_box_region"),
    2: ("pick up the black bowl from table center and place it on the plate", "On",
        "main_table_table_center", "main_table_next_to_plate_region"),
    3: ("pick up the black bowl on the cookie box and place it on the plate", "On", "cookies_1", "wooden_cabinet_1_top_side"),
    4: ("pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", "In",
        "wooden_cabinet_1_top_region", "wooden_cabinet_1_top_side"),
    5: ("pick up the black bowl on the ramekin and place it on the plate", "On", "glazed_rim_porcelain_ramekin_1", "cookies_1"),
    6: ("pick up the black bowl next to the cookie box and place it on the plate", "On",
        "main_table_next_to_box_region", "flat_stove_1_cook_region"),
    7: ("pick up the black bowl on the stove and place it on the plate", "On", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side"),
    8: ("pick up the black bowl next to the plate and place it on the plate", "On",
        "main_table_next_to_plate_region", "main_table_next_to_ramekin_region"),
    9: ("pick up the black bowl on the wooden cabinet and place it on the plate", "On",
        "wooden_cabinet_1_top_side", "flat_stove_1_cook_region"),
}
EXPECTED_BOWL2 = {0: "next to the ramekin", 1: "next to the cookie box", 2: "next to the plate", 3: "on the wooden cabinet",
                  4: "on the wooden cabinet", 5: "on the cookie box", 6: "on the stove", 7: "on the wooden cabinet",
                  8: "next to the ramekin", 9: "on the stove"}


def bddl_text(task_id):
    language, pred, region1, region2 = TASKS[task_id]
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the akita black bowl and place it on the plate)
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    plate_1 - plate
  )
  (:init
    ({pred} akita_black_bowl_1 {region1})
    (On akita_black_bowl_2 {region2})
    (Open wooden_cabinet_1_top_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""


# robosuite pinhole of the agentview camera at 256x256 (task 0) and resting object positions (tests/test_vla_scene.py).
PINHOLE_K = np.array([[309.01933598, 0.0, 128.0], [0.0, 309.01933598, 128.0], [0.0, 0.0, 1.0]])
PINHOLE_E = np.array([[0.0, 1.0, 0.0, 0.0], [0.62826645, 0.0, -0.77799824, 0.83906493],
                      [-0.77799824, 0.0, -0.62826645, 1.52412879]])
OBJECTS = {S.TARGET_BOWL: np.array([-0.06347437, 0.20202566, 0.89840387]),
           S.DISTRACTOR_BOWL: np.array([-0.18872204, 0.32034761, 0.89840387]),
           S.PLATE: np.array([0.05341727, 0.20518007, 0.90249958])}
PLATE_TOP = OBJECTS[S.PLATE] + np.array([0.0, 0.0, 0.01])
SIZE = 8
MARKER = 255


# ---- fakes ------------------------------------------------------------------------------------------------------------
class FakeSim:
    def __init__(self, positions):
        self.data = types.SimpleNamespace(body_xpos=np.stack([positions[n] for n in positions]).astype(np.float64))


class FakeRobosuiteEnv:
    def __init__(self, positions):
        self.sim = FakeSim(positions)
        self.obj_body_id = {name: index for index, name in enumerate(positions)}

    def set(self, name, xyz):
        self.sim.data.body_xpos[self.obj_body_id[name]] = np.asarray(xyz, dtype=np.float64)

    def move(self, name, delta):
        self.sim.data.body_xpos[self.obj_body_id[name]] += np.asarray(delta, dtype=np.float64)


class FakeTask:
    def __init__(self, language):
        self.language = language


class FakeSuite:
    """get_task / get_task_bddl_file_path over the BDDL fixtures written to a temp dir."""
    name = "libero_spatial"

    def __init__(self, folder):
        self.folder = Path(folder)
        for task_id in TASKS:
            (self.folder / f"task{task_id}.bddl").write_text(bddl_text(task_id))

    def get_task(self, task_id):
        return FakeTask(TASKS[task_id][0])

    def get_task_bddl_file_path(self, task_id):
        return str(self.folder / f"task{task_id}.bddl")


class FakeRolloutEnv:
    """LiberoEnv-like sim: the bowl the prompt names is lifted at step LIFT_AT and lands on the plate at PLACE_AT.

    Bowl 2 on init state 1 is never placed (a failed counterfactual episode).  ``info["is_success"]`` is LIBERO's
    bowl-1 flag.  Raw images carry a marker at pixel [0, 0] (the policy view moves it to the opposite corner).
    """
    LIFT_AT, PLACE_AT = 3, 9

    def __init__(self, task_id, prompt):
        self.task_id = task_id
        self.prompt = prompt
        self.bowl = 1 if prompt == TASKS[task_id][0] else 2
        self._env = types.SimpleNamespace(env=FakeRobosuiteEnv({k: v.copy() for k, v in OBJECTS.items()}))
        self.camera_name = ["agentview_image", "robot0_eye_in_hand_image"]
        self.camera_name_mapping = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        self.observation_height = self.observation_width = SIZE
        self.init_state_id = 0
        self.steps = 0
        self.resets = []
        self.closed = False

    def _ensure_env(self):
        pass

    def _obs(self):
        value = (self.steps * 7 + self.init_state_id + 1) % 250
        image = np.full((SIZE, SIZE, 3), value, dtype=np.uint8)
        image[0, 0] = MARKER
        wrist = np.full((SIZE, SIZE, 3), 250 - value, dtype=np.uint8)
        wrist[0, 0] = MARKER
        return {"pixels": {"image": image, "image2": wrist},
                "robot_state": {"eef": {"pos": np.array([0.1, 0.2, 0.3 + 0.01 * self.steps]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                                "gripper": {"qpos": np.array([0.04, -0.04])}}}

    def reset(self, seed=None):
        self.resets.append((self.init_state_id, seed))
        self.steps = 0
        for name, xyz in OBJECTS.items():
            self._env.env.set(name, xyz)
        return self._obs(), {"is_success": False}

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self.steps += 1
        name = C.bowl_name(self.bowl)
        fails = self.bowl == 2 and self.init_state_id == 1
        if self.steps == self.LIFT_AT:
            self._env.env.move(name, [0, 0, 0.1])
        if self.steps == self.PLACE_AT and not fails:
            self._env.env.set(name, PLATE_TOP)
        is_success = C.on_plate_geometric(self._env.env.sim.data.body_xpos[0], self._env.env.sim.data.body_xpos[2])
        return self._obs(), float(is_success), is_success, False, {"is_success": is_success}

    def close(self):
        self.closed = True


def make_fake_scene(record, folder):
    """The real libero_scene helpers (they work on the fake sim) with the suite/env/camera entry points replaced."""
    names = {k: getattr(S, k) for k in dir(S) if not k.startswith("_")}

    def make_env(suite, task_id, prompt_override=None, **kwargs):
        env = FakeRolloutEnv(task_id, prompt_override)
        record["envs"].append(env)
        return env

    names.update(load_suite=lambda name: FakeSuite(folder), make_env=make_env,
                 agentview_camera=lambda env: (S.policy_view_intrinsic(PINHOLE_K, 256), PINHOLE_E),
                 max_steps=lambda suite: 40)
    return types.SimpleNamespace(**names)


def make_fake_rollout_flow(record):
    bundle = types.SimpleNamespace(device="cpu", config=types.SimpleNamespace(chunk_size=50, max_action_dim=32))

    def prepare_batch(bundle, images, state, prompt):
        assert images["image"].dtype == np.uint8 and images["image"][0, 0, 0] == MARKER   # raw orientation
        record["prompts"].append(prompt)
        return {"state": np.asarray(state)}

    def sample_actions_with_noise(bundle, ctx, noise=None, num_steps=10):
        record["num_steps"].append(num_steps)
        return (noise[:, :, :7] * 0.01).numpy().astype(np.float32)

    return types.SimpleNamespace(
        CHUNK_SIZE=50, MAX_ACTION_DIM=32,
        load_pi05=lambda checkpoint, device, n_action_steps=10: bundle,
        prepare_batch=prepare_batch, prefix_context=lambda bundle, batch: {"ctx": True},
        sample_noise=lambda shape, device: torch.randn(*shape),
        sample_actions_with_noise=sample_actions_with_noise)


MEAN = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
STD = np.linspace(0.5, 1.5, 7, dtype=np.float32)


def make_fake_reverse_flow(record):
    """A stand-in for leap.vla.pi05_flow whose flow is x_noise = 2 x_action + 1 (so the reconstruction error is 0)."""
    api = types.SimpleNamespace()
    api.Observation = lambda images, state, prompt: types.SimpleNamespace(images=images, state=state, prompt=prompt)

    def prepare_batches(bundle, observations, flip_images=True):
        observations = list(observations)
        record["prepare"].append((len(observations), observations[0].prompt))
        for obs in observations:
            image = obs.images["image"]
            assert image.dtype == np.uint8 and image.shape == (SIZE, SIZE, 3)
            assert image[0, 0, 0] == MARKER and image[-1, -1, 0] != MARKER   # raw orientation restored
            assert obs.state.shape == (8,)
        return {"state": np.stack([o.state for o in observations]).astype(np.float32),
                "image_mean": np.array([o.images["image"].astype(np.float64).mean() for o in observations])}

    def prefix_features(bundle, batch, ctx=None):
        n = batch["state"].shape[0]
        feat = np.zeros((n, reverse.FEAT_DIM), dtype=np.float32)
        feat[:, :8] = batch["state"]
        feat[:, 8] = batch["image_mean"]
        return feat

    def normalize_actions(bundle, actions7):
        actions7 = np.asarray(actions7, dtype=np.float32)
        out = np.zeros(actions7.shape[:2] + (32,), dtype=np.float32)
        out[:, :, :7] = (actions7 - MEAN) / STD
        return out

    api.load_pi05 = lambda checkpoint, device, **kwargs: {"checkpoint": checkpoint}
    api.prepare_batches = prepare_batches
    api.prefix_context = lambda bundle, batch: {"n": batch["state"].shape[0]}
    api.prefix_features = prefix_features
    api.normalize_actions = normalize_actions
    api.reverse_actions = lambda bundle, ctx, x, num_steps: 2.0 * np.asarray(x) + 1.0
    api.forward_from_noise = lambda bundle, ctx, noise, num_steps: (np.asarray(noise) - 1.0) / 2.0
    return api


# ---- phrase table --------------------------------------------------------------------------------------------------------
class PhraseTableTest(unittest.TestCase):
    def test_bowl_one_phrase_reproduces_every_language(self):
        for task_id, (language, _, region1, region2) in TASKS.items():
            phrases = C.bowl_phrases_from_bddl(bddl_text(task_id), language)
            self.assertEqual(C.prompt_for_bowl(language, phrases[1]), language, task_id)
            self.assertEqual(phrases[2], EXPECTED_BOWL2[task_id], task_id)
            self.assertEqual(C.bowl_regions(bddl_text(task_id)), {1: region1, 2: region2})
            self.assertEqual(C.target_prompt(language, phrases, 1), language)
            self.assertEqual(C.target_prompt(language, phrases, 2),
                             f"pick up the black bowl {EXPECTED_BOWL2[task_id]} and place it on the plate")
        # every bowl-2 phrase is the language phrase of some other task (in-distribution prompts)
        own = {C.split_language(t[0])[1] for t in TASKS.values()}
        self.assertTrue(set(EXPECTED_BOWL2.values()) <= own)

    def test_bowl_phrases_from_suite_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = FakeSuite(temp)
            for task_id in TASKS:
                self.assertEqual(C.bowl_phrases(suite, task_id)[2], EXPECTED_BOWL2[task_id])
        with self.assertRaises(ValueError):
            C.bowl_phrases_from_bddl(bddl_text(0), TASKS[1][0])      # wrong language for this BDDL

    def test_parsing_errors(self):
        self.assertEqual(C.parse_init(bddl_text(4))[0], ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region"))
        self.assertNotIn(("On", "akita_black_bowl_1", "plate_1"), C.parse_init(bddl_text(4)))   # goal is not init
        with self.assertRaises(ValueError):
            C.bowl_regions("(:init (On akita_black_bowl_1 x)) (:goal)")
        with self.assertRaises(KeyError):
            C.region_phrase("main_table_table_front")
        with self.assertRaises(ValueError):
            C.split_language("put the bowl on the plate")
        with self.assertRaises(ValueError):
            C.prompt_for_bowl(TASKS[0][0], "  ")
        with self.assertRaises(ValueError):
            C.bowl_name(3)
        self.assertEqual(C.bowl_name(2), S.DISTRACTOR_BOWL)
        self.assertEqual(C.other_bowl(2), 1)


# ---- success and choice ---------------------------------------------------------------------------------------------------
class OnPlateTest(unittest.TestCase):
    def test_geometric_tolerances(self):
        plate = np.array([0.05, 0.2, 0.9])
        self.assertTrue(C.on_plate_geometric(plate + [0.02, 0.0, 0.01], plate))
        self.assertFalse(C.on_plate_geometric(plate + [0.03, 0.0, 0.01], plate))      # 3 cm radius is exclusive
        self.assertFalse(C.on_plate_geometric(plate + [0.0, 0.0, -0.004], plate))     # resting on the table: below the plate
        self.assertFalse(C.on_plate_geometric(plate + [0.0, 0.0, 0.06], plate))       # held above the plate
        self.assertTrue(C.on_plate_geometric(plate + [0.0, 0.0, 0.049], plate))

    def test_bowl_on_plate_uses_libero_predicate_when_present(self):
        env = FakeRolloutEnv(0, TASKS[0][0])
        calls = []

        class State:
            def __init__(self, name):
                self.name = name

            def check_ontop(self, other):
                calls.append((self.name, other.name))
                return other.name == S.DISTRACTOR_BOWL

        env._env.env.object_states_dict = {n: State(n) for n in (S.TARGET_BOWL, S.DISTRACTOR_BOWL, S.PLATE)}
        self.assertFalse(C.bowl_on_plate(env, 1))
        self.assertTrue(C.bowl_on_plate(env, 2))
        self.assertEqual(calls, [(S.PLATE, S.TARGET_BOWL), (S.PLATE, S.DISTRACTOR_BOWL)])   # On(bowl, plate) = plate.check_ontop(bowl)

    def test_bowl_on_plate_geometric_fallback(self):
        env = FakeRolloutEnv(0, TASKS[0][0])
        self.assertFalse(C.bowl_on_plate(env, 2))
        env._env.env.set(S.DISTRACTOR_BOWL, PLATE_TOP)
        self.assertTrue(C.bowl_on_plate(env, 2))
        self.assertFalse(C.bowl_on_plate(env, 1))
        np.testing.assert_allclose(C.object_positions(env)[1], PLATE_TOP)

    def test_choice_relabelling(self):
        self.assertEqual(C.intended_choice("target", 1), "target")
        self.assertEqual(C.intended_choice("target", 2), "distractor")
        self.assertEqual(C.intended_choice("distractor", 2), "target")
        self.assertEqual(C.intended_choice("none", 2), "none")
        self.assertEqual([C.lifted_bowl(c) for c in ("target", "distractor", "none")], [1, 2, 0])

    def test_bowl_sketch_circles_the_chosen_bowl(self):
        frame = np.full((256, 256, 3), 200, np.uint8)
        cam = (S.policy_view_intrinsic(PINHOLE_K, 256), PINHOLE_E)
        layout = C.layout_from_positions(np.stack([OBJECTS[n] for n in C.POSITION_OBJECTS]))
        for bowl, expected_u in ((1, 65.1), (2, 37.5)):
            pixels = np.asarray(C.build_bowl_sketch(frame, cam, layout, bowl))
            red = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 60)
            self.assertLess(abs(np.nonzero(red)[1].mean() - expected_u), 3, bowl)


# ---- episode records ------------------------------------------------------------------------------------------------------
class EpisodeRecordTest(unittest.TestCase):
    def test_recorder_round_trip(self):
        recorder = C.EpisodeRecorder()
        for t in range(3):
            image = np.full((SIZE, SIZE, 3), t, np.uint8)
            recorder.add_step(image, image, np.arange(8, dtype=np.float32), np.ones((3, 3)) * t)
            if t % 2 == 0:
                recorder.add_prediction(t, np.ones((50, 7)) * t, np.zeros((50, 32)), executed=True)
            recorder.add_result(np.full(7, 0.5), on_plate=(t == 2))
        arrays = recorder.arrays()
        self.assertEqual(arrays["agentview"].shape, (3, SIZE, SIZE, 3))
        self.assertEqual(arrays["chunk"].shape, (2, 50, 7))
        self.assertEqual(arrays["noise"].shape, (2, 50, 32))
        np.testing.assert_array_equal(arrays["pred_step"], [0, 2])
        np.testing.assert_array_equal(arrays["on_plate"], [False, False, True])
        with tempfile.TemporaryDirectory() as temp:
            path = C.save_episode(Path(temp) / C.episode_file(3, 2, 7), arrays, {"task_id": 3, "bowl": 2, "episode": 7,
                                                                                 "success": True, "prompt": "p", "steps": 3})
            self.assertEqual(C.parse_episode_name(path.name), (3, 2, 7))
            scalars = C.episode_scalars(path)
            self.assertEqual(scalars, {"task_id": 3, "bowl": 2, "episode": 7, "success": True, "prompt": "p", "steps": 3})
            data = C.load_episode(path)
            self.assertEqual((data["prompt"], data["bowl"]), ("p", 2))
            np.testing.assert_array_equal(data["chunk"], arrays["chunk"])
            self.assertEqual(C.find_episodes(temp), [(3, 2, 7, path)])
            self.assertEqual(C.find_episodes(temp, bowl=1), [])
        with self.assertRaises(ValueError):
            C.EpisodeRecorder().arrays()
        self.assertEqual(C.dataset_index(4, 2), 204)
        self.assertEqual(C.demo_id_for(1, 12), 1012)

    def test_raw_images_is_the_inverse_rotation(self):
        raw = np.arange(SIZE * SIZE * 3, dtype=np.uint8).reshape(SIZE, SIZE, 3)
        policy = S.policy_view(raw)
        restored = C.raw_images(policy, policy)
        np.testing.assert_array_equal(restored["image"], raw)
        np.testing.assert_array_equal(restored["image2"], raw)

    def test_summary_table(self):
        records = [{"task_id": 0, "bowl": 2, "success": True, "success_geometric": True, "libero_success": False,
                    "bowl_choice": "target", "steps": 10, "elapsed_s": 2.0},
                   {"task_id": 0, "bowl": 2, "success": False, "success_geometric": False, "libero_success": True,
                    "bowl_choice": "distractor", "steps": 30, "elapsed_s": 4.0}]
        table = C.summarize_episodes(records)
        self.assertEqual(table["0:2"]["success_rate"], 0.5)
        self.assertEqual(table["0:2"]["bowl_choice_counts"], {"target": 1, "distractor": 1, "none": 0})
        self.assertIn("| 0 | 2 | 2 | 1 (50 %) | 1 | 1 | 1/1/0 | 20.0 | 3.0 |", C.format_table(table))


# ---- rollout script ---------------------------------------------------------------------------------------------------------
class RolloutScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bddl = Path(self.temp.name) / "bddl"
        self.bddl.mkdir()
        self.out = Path(self.temp.name) / "rollouts"
        self.record = {"envs": [], "prompts": [], "num_steps": []}
        patches = [patch.object(rollout, "import_pi05_flow", lambda: make_fake_rollout_flow(self.record)),
                   patch.object(rollout, "import_libero_scene", lambda: make_fake_scene(self.record, self.bddl))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_main(self, *extra, tasks="0,1", bowls="1,2", episodes=2, out=None):
        with contextlib.redirect_stdout(io.StringIO()):
            rollout.main(["--tasks", tasks, "--bowls", bowls, "--episodes", str(episodes), "--out", str(out or self.out),
                          "--device", "cpu", "--n-action-steps", "4", "--num-steps", "3", "--max-steps", "40", *extra])
        folder = Path(out or self.out)
        lines = [json.loads(l) for l in (folder / "rollouts.jsonl").read_text().splitlines()]
        return lines, json.loads((folder / "manifest.json").read_text())

    def test_rollouts_prompts_success_and_records(self):
        lines, manifest = self.run_main("--predict-every", "3")
        self.assertEqual([(l["task_id"], l["bowl"], l["episode"]) for l in lines],
                         [(0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
        by_key = {(l["task_id"], l["bowl"], l["episode"]): l for l in lines}
        self.assertEqual(by_key[(0, 1, 0)]["prompt"], TASKS[0][0])
        self.assertEqual(by_key[(0, 2, 0)]["prompt"], "pick up the black bowl next to the ramekin and place it on the plate")
        self.assertEqual(by_key[(1, 2, 0)]["prompt"], "pick up the black bowl next to the cookie box and place it on the plate")
        self.assertEqual(manifest["prompts"]["1:2"], by_key[(1, 2, 0)]["prompt"])
        # The fake env places the named bowl after 9 steps (bowl 2 never on init state 1).
        self.assertEqual([by_key[k]["success"] for k in sorted(by_key)], [True, True, True, False, True, True, True, False])
        ok = by_key[(0, 2, 0)]
        self.assertEqual((ok["steps"], ok["success_step"], ok["geometric_step"]), (9, 8, 8))
        self.assertTrue(ok["success_geometric"])
        self.assertFalse(ok["libero_success"])                       # bowl 1 never moved
        self.assertEqual((ok["bowl_choice"], ok["lifted_bowl"]), ("target", 2))
        bowl1 = by_key[(0, 1, 0)]
        self.assertTrue(bowl1["libero_success"] and bowl1["success"])
        self.assertEqual((bowl1["bowl_choice"], bowl1["lifted_bowl"]), ("target", 1))
        failed = by_key[(0, 2, 1)]
        self.assertEqual((failed["steps"], failed["success_step"], failed["bowl_choice"]), (40, -1, "target"))
        self.assertEqual(by_key[(1, 2, 1)]["seed"], 12_001)
        # Episode e reset on init state e; one env per (task, bowl), closed afterwards.
        self.assertEqual([r[0] for r in self.record["envs"][0].resets], [0, 1])
        self.assertEqual(len(self.record["envs"]), 4)
        self.assertTrue(all(env.closed for env in self.record["envs"]))
        self.assertEqual(set(self.record["num_steps"]), {3})
        # Stored arrays: policy view (marker moved to the far corner), predictions at executed and extra steps.
        data = C.load_episode(C.episode_path(self.out, 0, 2, 0))
        self.assertEqual(data["agentview"].shape, (9, SIZE, SIZE, 3))
        self.assertEqual(int(data["agentview"][0, -1, -1, 0]), MARKER)
        self.assertNotEqual(int(data["agentview"][0, 0, 0, 0]), MARKER)
        np.testing.assert_array_equal(data["pred_step"], [0, 3, 4, 6, 8])
        np.testing.assert_array_equal(data["executed"], [True, False, True, False, True])
        self.assertEqual(data["chunk"].shape, (5, 50, 7))
        self.assertEqual(data["noise"].shape, (5, 50, 32))
        np.testing.assert_allclose(data["chunk"], data["noise"][:, :, :7] * 0.01, rtol=1e-5)
        self.assertEqual(data["action"].shape, (9, 7))
        np.testing.assert_allclose(data["action"][5], data["chunk"][2][1])      # step 5 runs action 1 of the chunk from step 4
        self.assertEqual(data["positions"].shape, (9, 3, 3))
        np.testing.assert_array_equal(data["on_plate"], [False] * 8 + [True])
        self.assertEqual(data["cam_intrinsic"].shape, (3, 3))
        self.assertEqual((data["image_view"], data["predict_every"], data["n_action_steps"]), ("policy", 3, 4))
        self.assertEqual(data["prompt"], ok["prompt"])
        self.assertEqual(data["state"].shape, (9, 8))
        self.assertAlmostEqual(float(data["state"][3, 2]), 0.33, places=5)
        # Table
        self.assertEqual(manifest["table"]["0:2"]["success"], 1)
        self.assertEqual(manifest["table"]["0:1"]["libero_success"], 2)
        self.assertEqual(manifest["episodes"], 8)
        # Idempotent: a second run touches no env and keeps the log; a third episode appends one line per pair.
        before = len(self.record["prompts"])
        again, manifest2 = self.run_main("--predict-every", "3")
        self.assertEqual(again, lines)
        self.assertEqual(len(self.record["prompts"]), before)
        self.assertEqual(manifest2["table"], manifest["table"])
        more, _ = self.run_main("--predict-every", "3", episodes=3)
        self.assertEqual(len(more), 12)

    def test_phrase_override_and_executed_only(self):
        out = Path(self.temp.name) / "alt"
        lines, _ = self.run_main("--phrase-override", "2=on the stove", tasks="0", bowls="2", episodes=1, out=out)
        self.assertEqual(lines[0]["prompt"], "pick up the black bowl on the stove and place it on the plate")
        self.assertEqual(lines[0]["bowl_choice"], "target")
        data = C.load_episode(C.episode_path(out, 0, 2, 0))
        np.testing.assert_array_equal(data["pred_step"], [0, 4, 8])          # predict-every 0: executed calls only
        self.assertTrue(data["executed"].all())

    def test_argument_validation(self):
        self.assertEqual(rollout.parse_tasks("1,0"), [0, 1])
        self.assertEqual(rollout.parse_phrase_overrides(["2= next   to the plate "]), {2: "next to the plate"})
        self.assertEqual(rollout.episode_seed(1, 2, 1, 3), 1_021_003)
        for bad in (["--bowls", "3"], ["--episodes", "0"], ["--phrase-override", "x"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                rollout.parse_args(["--out", "x", *bad])


# ---- reverse script ---------------------------------------------------------------------------------------------------------
class ReverseScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bddl = Path(self.temp.name) / "bddl"
        self.bddl.mkdir()
        self.rollouts = Path(self.temp.name) / "rollouts"
        self.out = Path(self.temp.name) / "dataset"
        record = {"envs": [], "prompts": [], "num_steps": []}
        with patch.object(rollout, "import_pi05_flow", lambda: make_fake_rollout_flow(record)), \
                patch.object(rollout, "import_libero_scene", lambda: make_fake_scene(record, self.bddl)), \
                contextlib.redirect_stdout(io.StringIO()):
            rollout.main(["--tasks", "0,1", "--bowls", "1,2", "--episodes", "2", "--out", str(self.rollouts), "--device", "cpu",
                          "--n-action-steps", "4", "--num-steps", "3", "--max-steps", "40", "--predict-every", "3"])
        self.record = {"prepare": []}
        patcher = patch.object(reverse, "flow_api", lambda: make_fake_reverse_flow(self.record))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, *extra, out=None):
        with contextlib.redirect_stdout(io.StringIO()):
            reverse.main(["--rollouts", str(self.rollouts), "--out", str(out or self.out), "--tasks", "0,1", "--bowls", "1,2",
                          "--device", "cpu", "--num-steps", "3", "--batch-size", "4", *extra])

    def test_dataset_format(self):
        self.run_main("--successful-only", "--recon-fraction", "1.0")
        self.assertEqual(sorted(p.name for p in self.out.glob("task_*.npz")), ["task_100.npz", "task_101.npz", "task_200.npz", "task_201.npz"])
        arrays = I.load_task_npz(self.out / "task_200.npz")
        n = arrays["feat"].shape[0]
        self.assertEqual(n, 5)                                                   # one successful bowl-2 episode x 5 queries
        self.assertEqual(arrays["feat"].shape, (n, 4104))
        self.assertEqual(arrays["state"].shape, (n, 8))
        self.assertEqual(arrays["noise"].shape, (n, 50, 7))
        self.assertEqual(arrays["chunk_pad_mask"].shape, (n, 50))
        self.assertFalse(arrays["chunk_pad_mask"].any())
        np.testing.assert_array_equal(arrays["demo_id"], [2000] * 5)
        np.testing.assert_array_equal(arrays["t"], [0, 3, 4, 6, 8])
        np.testing.assert_array_equal(arrays["bowl"], [2] * 5)
        np.testing.assert_array_equal(arrays["episode"], [0] * 5)
        np.testing.assert_array_equal(arrays["executed"], [True, False, True, False, True])
        self.assertTrue(arrays["success"].all())
        self.assertEqual((arrays["task_id"], arrays["num_steps"]), (0, 3))
        self.assertEqual(arrays["task_language"], TASKS[0][0])
        self.assertEqual(arrays["prompt"], S.NEUTRAL_PROMPT)
        self.assertEqual(str(arrays["target_prompt"]), "pick up the black bowl next to the ramekin and place it on the plate")
        self.assertEqual((int(arrays["target_bowl"]), int(arrays["dataset_index"])), (2, 200))
        self.assertTrue(np.isfinite(arrays["recon_err"]).all())
        np.testing.assert_allclose(arrays["recon_err"], 0.0, atol=1e-5)
        # The reversal ran under the neutral prompt and the noise is the fake flow of the normalised stored chunk.
        self.assertEqual({p[1] for p in self.record["prepare"]}, {S.NEUTRAL_PROMPT})
        episode = C.load_episode(C.episode_path(self.rollouts, 0, 2, 0))
        expected = 2.0 * (episode["chunk"] - MEAN) / STD + 1.0
        np.testing.assert_allclose(arrays["noise"], expected, rtol=1e-5)
        np.testing.assert_allclose(arrays["state"], episode["state"][[0, 3, 4, 6, 8]])
        # Bowl-1 file: both episodes succeed -> 10 rows with demo ids 1000 and 1001.
        bowl1 = I.load_task_npz(self.out / "task_100.npz")
        np.testing.assert_array_equal(np.unique(bowl1["demo_id"]), [1000, 1001])
        self.assertEqual(bowl1["feat"].shape[0], 10)
        # Instruction record: the bowl's prompt as the single text, one sketch per episode, audio left to the embed script.
        record = I.read_record(self.out, 200)
        self.assertEqual(record["texts"], [str(arrays["target_prompt"])])
        self.assertEqual(record["demo_ids"], [2000])
        self.assertEqual((record["suite_task_id"], record["bowl"], record["audio"]), (0, 2, []))
        self.assertEqual(record["sketches"], ["task_200_episode2000_sketch.png"])
        with Image.open(self.out / record["sketches"][0]) as sketch:
            self.assertEqual(sketch.size, (SIZE, SIZE))
        self.assertEqual(embed.record_demo_ids(record, arrays), [2000])
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["tasks"]["200"]["frames"], 5)
        self.assertEqual(manifest["tasks"]["201"]["suite_task_id"], 1)
        self.assertEqual(manifest["chunk_pad_mask"], "true_means_padded")
        # The trainer reads the files unchanged: all-False masks are recognised as the "padded" convention.
        tasks = trainer.load_tasks(self.out, [])
        self.assertEqual([t["task_id"] for t in tasks], [0, 1, 0, 1])
        self.assertEqual({t["mask_convention"] for t in tasks}, {"padded"})
        self.assertFalse(any(t["pad"].any() for t in tasks))
        self.assertEqual([t["demos"] for t in tasks], [2, 2, 1, 1])
        # Idempotent: a second run reverses nothing.
        calls = len(self.record["prepare"])
        self.run_main("--successful-only")
        self.assertEqual(len(self.record["prepare"]), calls)

    def test_all_episodes_stride_and_executed_only(self):
        out = Path(self.temp.name) / "all"
        self.run_main("--stride", "2", out=out)
        arrays = I.load_task_npz(out / "task_200.npz")
        np.testing.assert_array_equal(np.unique(arrays["demo_id"]), [2000, 2001])       # the failed episode is included
        self.assertEqual(sorted(set(arrays["success"].tolist())), [False, True])
        np.testing.assert_array_equal(arrays["t"][arrays["demo_id"] == 2000], [0, 4, 8])
        out2 = Path(self.temp.name) / "executed"
        self.run_main("--executed-only", "--max-episodes", "1", out=out2)
        arrays = I.load_task_npz(out2 / "task_100.npz")
        np.testing.assert_array_equal(arrays["t"], [0, 4, 8])
        self.assertTrue(arrays["executed"].all())
        np.testing.assert_array_equal(np.unique(arrays["demo_id"]), [1000])

    def test_argument_validation(self):
        for bad in (["--bowls", "0"], ["--stride", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                reverse.parse_args(["--rollouts", "r", "--out", "o", *bad])


# ---- eval --target-bowl -----------------------------------------------------------------------------------------------------
class EvalFakeEnv:
    """LiberoEnv-like: bowl 1 succeeds after 13 steps on task 0 (LIBERO flag); never on task 1."""

    def __init__(self, task_id):
        self.task_id = task_id
        self.init_state_id = 0
        self.steps = 0
        self.closed = False

    def _obs(self):
        value = (self.steps * 7 + self.init_state_id) % 255
        return {"pixels": {"image": np.full((8, 8, 3), value, dtype=np.uint8), "image2": np.full((8, 8, 3), 255 - value, dtype=np.uint8)},
                "robot_state": {"eef": {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                                "gripper": {"qpos": np.array([0.04, -0.04])}}}

    def reset(self, seed=None):
        self.steps = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self.steps += 1
        success = self.task_id == 0 and self.steps >= 13
        return self._obs(), float(success), success, False, {"is_success": success}

    def close(self):
        self.closed = True


class EvalFakeTracker:
    """Task 0 lifts bowl 1 at step 5, task 1 lifts bowl 2 at step 8 (bowl-1 view, as BowlChoiceTracker reports it)."""

    def __init__(self, layout):
        self.choice = "none"

    def update(self, env):
        if self.choice == "none":
            if env.task_id == 0 and env.steps >= 5:
                self.choice = "target"
            elif env.task_id == 1 and env.steps >= 8:
                self.choice = "distractor"
        return self.choice


class EvalTargetBowlTest(unittest.TestCase):
    def setUp(self):
        self.record = {"prompts": [], "sketch_bowls": [], "embedded": [], "on_plate_calls": 0, "imported": 0}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"
        record = self.record

        def make_fake_scene():
            return types.SimpleNamespace(
                NEUTRAL_PROMPT=S.NEUTRAL_PROMPT,
                load_suite=lambda name: FakeSuite(Path(self.temp.name)),
                make_env=lambda suite, task_id, prompt_override=None, **kw: EvalFakeEnv(task_id),
                scene_layout=lambda env: {S.TARGET_BOWL: np.zeros(3), S.DISTRACTOR_BOWL: np.ones(3), S.PLATE: np.ones(3), "initial_z": {}},
                agentview_camera=lambda env: (np.eye(3), np.eye(3, 4)), BowlChoiceTracker=EvalFakeTracker,
                max_steps=lambda suite: 30, policy_view=S.policy_view, quat_to_axis_angle=S.quat_to_axis_angle,
                state_vector=lambda raw: np.zeros(8, dtype=np.float32))

        def make_fake_flow():
            def prepare_batch(bundle, images, state, prompt):
                record["prompts"].append(prompt)
                return {"image": images["image"]}

            def pad_noise(noise7, max_action_dim=32, generator=None):
                noise7 = torch.as_tensor(noise7, dtype=torch.float32)
                return torch.cat([noise7, torch.randn(*noise7.shape[:-1], max_action_dim - noise7.shape[-1])], dim=-1)

            return types.SimpleNamespace(
                load_pi05=lambda checkpoint, device, n_action_steps=10: {"device": device},
                prepare_batch=prepare_batch, prefix_context=lambda bundle, batch: {"ctx": True},
                prefix_features=lambda bundle, batch, ctx=None: torch.full((1, 16), float(batch["image"][0, 0, 0]) / 255.0),
                pad_noise=pad_noise,
                sample_actions_with_noise=lambda bundle, ctx, noise=None, num_steps=10: np.zeros((1, 50, 7), dtype=np.float32))

        def make_fake_instructions():
            def embed_instruction(embedder, modality, *, text=None, sketch=None, audio=None):
                record["embedded"].append((modality, text, sketch is not None))
                return np.full(8, 0.1, dtype=np.float32)

            return types.SimpleNamespace(task_text=I.task_text, embed_instruction=embed_instruction,
                                         build_sketch=lambda rgb, cam, layout: Image.fromarray(np.asarray(rgb, dtype=np.uint8)),
                                         build_audio=lambda text, tts: np.zeros(160, dtype=np.float32))

        def make_fake_counterfactual():
            record["imported"] += 1

            def bowl_on_plate(env, bowl):
                record["on_plate_calls"] += 1
                assert bowl == 2
                return env.task_id == 1 and env.steps >= 9

            def build_bowl_sketch(rgb, cam, layout, bowl):
                record["sketch_bowls"].append(bowl)
                return Image.fromarray(np.asarray(rgb, dtype=np.uint8))

            return types.SimpleNamespace(bowl_phrases=C.bowl_phrases, target_prompt=C.target_prompt,
                                         intended_choice=C.intended_choice, bowl_on_plate=bowl_on_plate,
                                         build_bowl_sketch=build_bowl_sketch)

        patches = [patch.object(evaluator, "import_pi05_flow", make_fake_flow),
                   patch.object(evaluator, "import_libero_scene", make_fake_scene),
                   patch.object(evaluator, "import_instructions", make_fake_instructions),
                   patch.object(evaluator, "import_counterfactual", make_fake_counterfactual),
                   patch.object(evaluator, "make_tts", lambda voice: "tts"),
                   patch.object(evaluator, "make_embedder", lambda model, device, python=None: "embedder")]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_main(self, *extra, tasks="0,1", episodes=1):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.main(["--tasks", tasks, "--episodes", str(episodes), "--out", str(self.out), "--device", "cpu",
                            "--n-action-steps", "4", "--num-steps", "3", *extra])
        lines = [json.loads(l) for l in (self.out / "episodes.jsonl").read_text().splitlines()]
        return lines, json.loads((self.out / "summary.json").read_text())

    def test_full_prompt_names_bowl_two_and_judges_bowl_two(self):
        lines, summary = self.run_main("--condition", "full-prompt", "--target-bowl", "2")
        bowl2 = {0: "pick up the black bowl next to the ramekin and place it on the plate",
                 1: "pick up the black bowl next to the cookie box and place it on the plate"}
        self.assertEqual(set(self.record["prompts"]), set(bowl2.values()))
        self.assertEqual([l["prompt"] for l in lines], [bowl2[0], bowl2[1]])
        self.assertEqual([l["target_language"] for l in lines], [bowl2[0], bowl2[1]])
        self.assertEqual([l["task_language"] for l in lines], [TASKS[0][0], TASKS[1][0]])
        # Task 0: LIBERO's bowl-1 success at 13 ends the episode as a bowl-2 failure; task 1: bowl 2 on the plate at step 9.
        self.assertEqual([(l["success"], l["libero_success"], l["steps"]) for l in lines], [(False, True, 13), (True, False, 9)])
        self.assertEqual([l["bowl_choice"] for l in lines], ["distractor", "target"])
        self.assertEqual([l["target_bowl"] for l in lines], [2, 2])
        self.assertEqual(summary["target_bowl"], 2)
        self.assertAlmostEqual(summary["overall"]["success_rate"], 0.5)
        self.assertAlmostEqual(summary["overall"]["bowl_choice_accuracy"], 0.5)
        self.assertGreater(self.record["on_plate_calls"], 0)

    def test_neutral_prior_sketch_circles_bowl_two(self):
        torch.manual_seed(0)
        prior = VLAPrior(feat_dim=16, instr_dim=8, hidden=(8,), horizon=50, action_dim=7, instr_mode="concat", dropout=0.0)
        path = Path(self.temp.name) / "concat.pt"
        prior.save(path, {"step": 1})
        lines, _ = self.run_main("--condition", "neutral-prior", "--instr-modality", "sketch", "--prior", str(path),
                                 "--target-bowl", "2", "--dump-instructions", tasks="1")
        self.assertEqual(set(self.record["prompts"]), {S.NEUTRAL_PROMPT})
        self.assertEqual(self.record["sketch_bowls"], [2])
        self.assertEqual(self.record["embedded"], [("sketch", "pick up the black bowl next to the cookie box and place it on the plate", True)])
        self.assertEqual((self.out / "instructions" / "task01_ep000.txt").read_text().strip(),
                         "pick up the black bowl next to the cookie box and place it on the plate")
        self.assertTrue(lines[0]["success"])

    def test_default_bowl_one_is_unchanged(self):
        lines, summary = self.run_main("--condition", "neutral-gauss")
        self.assertEqual(self.record["imported"], 0)                           # counterfactual module untouched
        self.assertEqual([(l["success"], l["libero_success"], l["bowl_choice"], l["steps"]) for l in lines],
                         [(True, True, "target", 13), (False, False, "distractor", 30)])
        self.assertEqual([l["target_language"] for l in lines], [TASKS[0][0], TASKS[1][0]])
        self.assertEqual(summary["target_bowl"], 1)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            evaluator.parse_args(["--condition", "neutral-gauss", "--out", "x", "--target-bowl", "3"])


if __name__ == "__main__":
    unittest.main()

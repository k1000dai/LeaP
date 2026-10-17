"""CPU checks for the place_a2b_side task file, its instruction templates and the installer (no simulator)."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "leap/robotwin/tasks"
TASK_SOURCE = TASK_DIR / "place_a2b_side.py"
TEMPLATES = json.loads((TASK_DIR / "place_a2b_side.json").read_text())
GENERATOR = ROOT / "third_party/RoboTwin/description/utils/generate_episode_instructions.py"
PARAMS = {"{A}": "047_mouse/base0", "{B}": "048_stapler/base0", "{a}": "right", "{side}": "left"}

sys.path.insert(0, str(ROOT / "scripts"))
import install_task


def load_generator():
    spec = importlib.util.spec_from_file_location("robotwin_generate_episode_instructions", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemplateTest(unittest.TestCase):
    def test_every_template_uses_the_side_placeholder_once(self):
        self.assertGreaterEqual(len(TEMPLATES["seen"]), 20)
        self.assertGreaterEqual(len(TEMPLATES["unseen"]), 10)
        self.assertEqual(len(set(TEMPLATES["seen"])), len(TEMPLATES["seen"]))
        self.assertEqual(len(set(TEMPLATES["unseen"])), len(TEMPLATES["unseen"]))
        self.assertFalse(set(TEMPLATES["seen"]) & set(TEMPLATES["unseen"]))
        for template in TEMPLATES["seen"] + TEMPLATES["unseen"]:
            with self.subTest(template):
                self.assertEqual(template.count("{side}"), 1)
                self.assertIn("{A}", template)
                self.assertIn("{B}", template)
                self.assertNotRegex(template.lower(), r"\b(left|right)\b")

    @unittest.skipUnless(GENERATOR.exists(), "pinned RoboTwin checkout not present")
    def test_official_generator_accepts_and_renders_templates(self):
        generator = load_generator()
        for key, minimum in (("seen", 20), ("unseen", 10)):
            kept = generator.filter_instructions(list(TEMPLATES[key]), PARAMS)
            self.assertEqual(sorted(kept), sorted(TEMPLATES[key]), f"{key} templates rejected by filter_instructions")
            self.assertGreaterEqual(len(kept), minimum)
        with_arm = next(t for t in TEMPLATES["seen"] if "{a}" in t)
        without_arm = next(t for t in TEMPLATES["seen"] if "{a}" not in t)
        for side in ("left", "right"):
            params = dict(PARAMS, **{"{side}": side})
            rendered = generator.replace_placeholders(with_arm, params)
            self.assertNotIn("{", rendered)
            self.assertIn(side, rendered)
            self.assertIn("the right arm", rendered)
            self.assertIn("mouse", rendered)
            self.assertIn("stapler", rendered)
            unseen = generator.replace_placeholders_unseen(without_arm, params)
            self.assertNotIn("{", unseen)
            self.assertIn(side, unseen)
            self.assertNotIn("arm", unseen)
            self.assertIn("mouse", unseen)


class StubbedTaskTest(unittest.TestCase):
    """Load place_a2b_side.py as envs.place_a2b_side on top of stub RoboTwin modules and exercise its logic."""

    class Pose:
        def __init__(self, p, q=None):
            self.p = np.asarray(p, dtype=np.float64)
            self.q = q

    class Actor:
        def __init__(self, pose):
            self.pose = pose
            self.mass = None

        def get_pose(self):
            return self.pose

        def set_mass(self, mass):
            self.mass = mass

    class ArmTag:
        def __init__(self, value):
            self.arm = value.arm if isinstance(value, StubbedTaskTest.ArmTag) else value

        def __str__(self):
            return self.arm

    class Robot:
        def is_left_gripper_open(self):
            return True

        def is_right_gripper_open(self):
            return True

    def setUp(self):
        test = self
        self.saved_modules = {name: sys.modules.get(name) for name in
                              ("envs", "envs._base_task", "envs.utils", "envs._GLOBAL_CONFIGS",
                               "envs.place_a2b_left", "envs.place_a2b_side", "sapien")}

        def rand_pose(xlim, ylim, zlim=(0.741,), ylim_prop=False, rotate_rand=False, rotate_lim=(0, 0, 0), qpos=(1, 0, 0, 0)):
            x = np.random.uniform(xlim[0], xlim[1])
            y = np.random.uniform(ylim[0], ylim[1])
            z = np.random.uniform(zlim[0], zlim[-1])
            if rotate_rand:
                for limit in rotate_lim:
                    np.random.uniform(-limit, limit)
            return test.Pose([x, y, z], list(qpos))

        def create_actor(scene, pose, modelname, convex=False, model_id=0, **kwargs):
            return test.Actor(pose)

        envs = types.ModuleType("envs")
        envs.__path__ = []
        base = types.ModuleType("envs._base_task")

        class Base_Task:
            def __init__(self):
                pass

        base.Base_Task = Base_Task
        utils = types.ModuleType("envs.utils")
        utils.rand_pose, utils.create_actor, utils.ArmTag = rand_pose, create_actor, self.ArmTag
        utils.__all__ = ["rand_pose", "create_actor", "ArmTag"]
        configs = types.ModuleType("envs._GLOBAL_CONFIGS")
        configs.__all__ = []
        left = types.ModuleType("envs.place_a2b_left")

        class place_a2b_left(Base_Task):
            def add_prohibit_area(self, actor, padding):
                self.prohibited.append((actor, padding))

        left.place_a2b_left = place_a2b_left
        sys.modules.update({"envs": envs, "envs._base_task": base, "envs.utils": utils,
                            "envs._GLOBAL_CONFIGS": configs, "envs.place_a2b_left": left,
                            "sapien": types.ModuleType("sapien")})
        spec = importlib.util.spec_from_file_location("envs.place_a2b_side", TASK_SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules["envs.place_a2b_side"] = self.module
        spec.loader.exec_module(self.module)
        self.glob_patch = patch("glob.glob", return_value=["assets/objects/x/model_data0.json", "assets/objects/x/model_data1.json"])
        self.glob_patch.start()

    def tearDown(self):
        self.glob_patch.stop()
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def make_task(self):
        task = self.module.place_a2b_side()
        task.prohibited = []
        task.info = {"info": {}}
        task.robot = self.Robot()
        task.moves = []
        task.move = lambda *actions: task.moves.append(actions)
        task.grasp_actor = lambda actor, arm_tag, pre_grasp_dis: ("grasp", str(arm_tag))
        task.move_by_displacement = lambda arm_tag, z, move_axis: ("lift", z)
        task.place_actor = lambda actor, arm_tag, target_pose: ("place", list(target_pose))
        return task

    def test_side_is_first_draw_and_layout_is_symmetric(self):
        sides = []
        for seed in range(200):
            np.random.seed(seed)
            expected_side = "left" if np.random.rand() < 0.5 else "right"
            np.random.seed(seed)
            task = self.make_task()
            task.load_actors()
            self.assertEqual(task.steer_side, expected_side)
            a, b = task.object.get_pose().p, task.target_object.get_pose().p
            self.assertTrue(-0.22 <= a[0] <= 0.22 and -0.2 <= a[1] <= 0)
            self.assertTrue(-0.10 <= b[0] <= 0.10 and -0.2 <= b[1] <= 0)
            self.assertGreaterEqual(np.hypot(*(a[:2] - b[:2])), 0.1)
            self.assertGreaterEqual(abs(a[1] - b[1]), 0.1)
            self.assertEqual(task.object.mass, 0.05)
            self.assertEqual(len(task.prohibited), 2)
            self.assertNotEqual(task.selected_modelname_A, task.selected_modelname_B)
            sides.append(task.steer_side)
        self.assertGreater(sides.count("left"), 60)
        self.assertGreater(sides.count("right"), 60)
        # Same seed -> same layout (reproducible in collection and evaluation).
        np.random.seed(7)
        first = self.make_task()
        first.load_actors()
        np.random.seed(7)
        second = self.make_task()
        second.load_actors()
        self.assertEqual(first.steer_layout(), second.steer_layout())

    def test_play_once_records_layout_and_placeholders(self):
        for seed, side in ((0, None), (1, None)):
            np.random.seed(seed)
            task = self.make_task()
            task.load_actors()
            layout = task.steer_layout()
            info = task.play_once()
            self.assertEqual(info["steer"], layout)
            self.assertEqual(set(layout), {"side", "arm", "A_xyz", "B_xyz", "target_xyz"})
            self.assertTrue(all(isinstance(v, float) for key in ("A_xyz", "B_xyz", "target_xyz") for v in layout[key]))
            json.dumps(info)
            offset = -0.13 if layout["side"] == "left" else 0.13
            np.testing.assert_allclose(layout["target_xyz"], np.add(layout["B_xyz"], [offset, 0, 0]))
            self.assertEqual(layout["arm"], "right" if layout["A_xyz"][0] > 0 else "left")
            self.assertEqual(set(info["info"]), {"{A}", "{B}", "{a}", "{side}"})
            self.assertEqual(info["info"]["{side}"], layout["side"])
            self.assertEqual(info["info"]["{a}"], layout["arm"])
            self.assertEqual([m[0][0] for m in task.moves], ["grasp", "lift", "place"])
            self.assertEqual(task.moves[2][0][1], layout["target_xyz"])

    def test_check_success_sign_and_steer_log(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "steer.jsonl"

            def logged_lines():
                return log.read_text().splitlines() if log.exists() else []

            for index, (side, correct_x) in enumerate((("left", -0.13), ("right", 0.13))):
                task = self.make_task()
                task.steer_side = side
                task.target_object = self.Actor(self.Pose([0.0, -0.1, 0.75]))
                task.object = self.Actor(self.Pose([correct_x, -0.1, 0.75]))
                task.ep_num, task.step_lim, task.take_action_cnt = 3, 400, 0
                with patch.dict(os.environ, {"LEAP_STEER_LOG": str(log)}):
                    self.assertTrue(task.check_success())            # expert check: not logged
                    self.assertEqual(len(logged_lines()), index)
                    task.take_action_cnt = 120
                    self.assertTrue(task.check_success())
                    self.assertTrue(task.check_success())            # logged once only
                    task.object = self.Actor(self.Pose([-correct_x, -0.1, 0.75]))
                    self.assertFalse(task.check_success())           # mirrored side fails
                    task.object = self.Actor(self.Pose([correct_x, -0.03, 0.75]))
                    self.assertFalse(task.check_success())           # y tolerance
                    task.object = self.Actor(self.Pose([correct_x * 2, -0.1, 0.75]))
                    self.assertFalse(task.check_success())           # too far
            lines = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual([line["side"] for line in lines], ["left", "right"])
            self.assertTrue(all(line["success"] and line["side_correct"] and line["ep_num"] == 3 and line["steps"] == 120
                                for line in lines))
            # A failed rollout is logged when the step limit is reached, with side_correct separate from success.
            task = self.make_task()
            task.steer_side = "left"
            task.target_object = self.Actor(self.Pose([0.0, -0.1, 0.75]))
            task.object = self.Actor(self.Pose([-0.05, -0.1, 0.75]))  # correct side, too close
            task.ep_num, task.step_lim, task.take_action_cnt = 4, 400, 399
            with patch.dict(os.environ, {"LEAP_STEER_LOG": str(log)}):
                self.assertFalse(task.check_success())
                task.take_action_cnt = 400
                self.assertFalse(task.check_success())
                self.assertFalse(task.check_success())
            lines = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertEqual((lines[-1]["success"], lines[-1]["side_correct"], lines[-1]["ep_num"]), (False, True, 4))
            # Without the variable nothing is written.
            task = self.make_task()
            task.steer_side = "right"
            task.target_object = self.Actor(self.Pose([0.0, -0.1, 0.75]))
            task.object = self.Actor(self.Pose([0.13, -0.1, 0.75]))
            task.take_action_cnt, task.step_lim = 5, 400
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(task.check_success())
            self.assertEqual(len(log.read_text().splitlines()), 3)


class InstallTaskTest(unittest.TestCase):
    def test_source_structure(self):
        tree = ast.parse(TASK_SOURCE.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        self.assertEqual([c.name for c in classes], ["place_a2b_side"])
        self.assertEqual([b.id for b in classes[0].bases], ["place_a2b_left"])
        methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"load_actors", "play_once", "check_success", "steer_layout"} <= methods)

    def test_install_check_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temp:
            robotwin = Path(temp)
            (robotwin / "envs").mkdir()
            (robotwin / "description/task_instruction").mkdir(parents=True)
            (robotwin / "task_config").mkdir()
            limits = robotwin / "task_config/_eval_step_limit.yml"
            limits.write_text("beat_block_hammer: 400\nturn_switch: 400")  # no trailing newline, like upstream
            self.assertEqual(len(install_task.missing_items(robotwin)), 3)
            actions = install_task.install(robotwin)
            self.assertEqual(len(actions), 3)
            self.assertEqual(install_task.missing_items(robotwin), [])
            self.assertEqual(install_task.install(robotwin), [])
            self.assertEqual(limits.read_text(), "beat_block_hammer: 400\nturn_switch: 400\nplace_a2b_side: 400\n")
            installed = robotwin / "envs/place_a2b_side.py"
            self.assertEqual(installed.read_bytes(), TASK_SOURCE.read_bytes())
            self.assertEqual((robotwin / "description/task_instruction/place_a2b_side.json").read_bytes(),
                             (TASK_DIR / "place_a2b_side.json").read_bytes())
            installed.write_text(installed.read_text() + "\n# stale\n")
            self.assertEqual(len(install_task.missing_items(robotwin)), 1)
            with patch.object(sys, "argv", ["install_task.py", "--robotwin", temp, "--check"]), \
                 self.assertRaises(SystemExit) as exit_code:
                install_task.main()
            self.assertEqual(exit_code.exception.code, 1)
            with patch.object(sys, "argv", ["install_task.py", "--robotwin", temp]):
                install_task.main()
            with patch.object(sys, "argv", ["install_task.py", "--robotwin", temp, "--check"]), \
                 self.assertRaises(SystemExit) as exit_code:
                install_task.main()
            self.assertEqual(exit_code.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

"""place_a2b_side: place object A to the left or right of object B, side drawn per episode.

Variant of RoboTwin's place_a2b_left / place_a2b_right with a symmetric layout so that the side is
not inferable from the observation (design: docs/superpowers/specs/2026-10-17-leap-multimodal-steering-design.md):

- ``self.steer_side`` in {"left", "right"} with p = 0.5 is the FIRST random draw of ``load_actors``
  (``np.random.seed(seed)`` runs before it in ``_init_task_env_``, so the side is reproducible from
  the seed in collection and evaluation alike).
- A: x in [-0.22, 0.22], y in [-0.2, 0]; arm = right if A.x > 0 else left.
- B: x in [-0.10, 0.10], y in [-0.2, 0]; resampled until xy-distance(A, B) >= 0.10 and |A.y - B.y| >= 0.10.
- Expert: grasp A, lift 0.1 m, place at B + (-0.13, 0, 0) for left and B + (+0.13, 0, 0) for right.
- ``self.info["info"]`` carries the template placeholders {A}, {B}, {a}, {side} for the official
  instruction generator; ``self.info["steer"]`` carries the initial layout (saved to scene_info.json).
- ``steer_layout()`` returns that layout from the live poses; call it before the robot moves
  (the evaluator does so at the first policy step) to get the same values as scene_info.json.
- ``check_success`` is the original test with the x-order chosen by the side.  With the environment
  variable ``LEAP_STEER_LOG`` set it appends one JSON line per policy-driven episode to that file:
  at the first successful check or when ``take_action_cnt`` reaches ``step_lim``.
"""
import glob
import json
import os

from ._base_task import Base_Task
from .place_a2b_left import place_a2b_left
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np

SIDE_OFFSET = 0.13
STEER_LOG_ENV = "LEAP_STEER_LOG"
OBJECT_LIST = [
    "047_mouse",
    "048_stapler",
    "050_bell",
    "057_toycar",
    "073_rubikscube",
    "075_bread",
    "077_phone",
    "081_playingcards",
    "086_woodenblock",
    "112_tea-box",
    "113_coffee-box",
    "107_soap",
]


class place_a2b_side(place_a2b_left):

    def load_actors(self):

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        # First random draw of the episode: the side (reproducible from the seed).
        self.steer_side = "left" if np.random.rand() < 0.5 else "right"
        self._steer_logged = False

        object_list = list(OBJECT_LIST)

        def sample_pose(xlim):
            return rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

        def layout_ok(pose_a, pose_b):
            distance = np.sqrt((pose_b.p[0] - pose_a.p[0])**2 + (pose_b.p[1] - pose_a.p[1])**2)
            return distance >= 0.1 and np.abs(pose_b.p[1] - pose_a.p[1]) >= 0.1

        try_lim = 100
        rand_pos = target_rand_pose = None
        for _ in range(try_lim):
            rand_pos = sample_pose([-0.22, 0.22])
            for _ in range(try_lim):
                target_rand_pose = sample_pose([-0.10, 0.10])
                if layout_ok(rand_pos, target_rand_pose):
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("place_a2b_side: no valid layout after %d tries" % try_lim)

        self.selected_modelname_A = np.random.choice(object_list)

        available_model_ids = get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")

        self.selected_model_id_A = np.random.choice(available_model_ids)
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        self.selected_modelname_B = np.random.choice(object_list)
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = np.random.choice(object_list)

        available_model_ids = get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")

        self.selected_model_id_B = np.random.choice(available_model_ids)

        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )
        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

    def steer_arm(self):
        """Arm used for A, from its live pose (same rule as play_once)."""
        return "right" if self.object.get_pose().p[0] > 0 else "left"

    def steer_target_xyz(self):
        """Place position: B shifted by -0.13 m (left) or +0.13 m (right) along world x."""
        target = self.target_object.get_pose().p.tolist()
        target[0] += -SIDE_OFFSET if self.steer_side == "left" else SIDE_OFFSET
        return target

    def steer_layout(self):
        """JSON-serialisable layout from the live poses (A, B, target, side, arm)."""
        return {
            "side": self.steer_side,
            "arm": self.steer_arm(),
            "A_xyz": [float(v) for v in self.object.get_pose().p],
            "B_xyz": [float(v) for v in self.target_object.get_pose().p],
            "target_xyz": [float(v) for v in self.steer_target_xyz()],
        }

    def play_once(self):
        # Layout before anything moves: this is what sketches are drawn from.
        self.info["steer"] = self.steer_layout()
        arm_tag = ArmTag(self.steer_arm())

        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.steer_target_xyz()
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))

        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
            "{side}": self.steer_side,
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        target_pos = self.target_object.get_pose().p
        distance = np.sqrt(np.sum((object_pose[:2] - target_pos[:2])**2))
        if self.steer_side == "left":
            side_correct = bool(object_pose[0] < target_pos[0])
        else:
            side_correct = bool(object_pose[0] > target_pos[0])
        success = bool(distance < 0.2 and distance > 0.08 and side_correct
                       and abs(object_pose[1] - target_pos[1]) < 0.05 and self.robot.is_left_gripper_open()
                       and self.robot.is_right_gripper_open())
        self._log_steer_result(object_pose, target_pos, side_correct, success)
        return success

    def _log_steer_result(self, object_pose, target_pos, side_correct, success):
        """One JSON line per policy episode under $LEAP_STEER_LOG (success, or the step limit reached)."""
        log_path = os.environ.get(STEER_LOG_ENV)
        if not log_path or getattr(self, "_steer_logged", False):
            return
        try:
            steps = int(getattr(self, "take_action_cnt", 0) or 0)
            step_lim = getattr(self, "step_lim", None)
        except Exception:
            steps, step_lim = 0, None
        if steps <= 0:  # expert demonstration / collection, not a policy rollout
            return
        if not (success or (step_lim is not None and steps >= step_lim)):
            return
        self._steer_logged = True
        record = {
            "side": self.steer_side,
            "object_xy": [float(object_pose[0]), float(object_pose[1])],
            "target_xy": [float(target_pos[0]), float(target_pos[1])],
            "side_correct": bool(side_correct),
            "success": bool(success),
            "ep_num": int(getattr(self, "ep_num", -1)),
            "steps": steps,
        }
        try:
            with open(log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as error:
            print(f"place_a2b_side: cannot write {STEER_LOG_ENV}={log_path}: {error}")

"""CPU checks for leap.vla.libero_scene and leap.vla.instructions with fakes (no LIBERO / LeRobot needed).

The camera and state conventions were verified once against the real sim (LIBERO-Spatial task 0,
LeRobot 0.6.1 clone): ``state_vector`` matched ``LiberoProcessorStep`` to 0.0 and the projected
bowl/plate positions landed on the objects in the policy-view frame.  The numbers below are those
sim values, so the tests pin the convention without the simulator.
"""
import unittest

import numpy as np
from PIL import Image

from leap.steer.sketch import project_points
from leap.vla import instructions as I
from leap.vla import libero_scene as S

# robosuite pinhole of the agentview camera at 256x256 (task 0, after reset), world->camera [R | t].
PINHOLE_K = np.array([[309.01933598, 0.0, 128.0], [0.0, 309.01933598, 128.0], [0.0, 0.0, 1.0]])
PINHOLE_E = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.62826645, 0.0, -0.77799824, 0.83906493],
                      [-0.77799824, 0.0, -0.62826645, 1.52412879]])
OBJECTS = {
    S.TARGET_BOWL: np.array([-0.06347437, 0.20202566, 0.89840387]),
    S.DISTRACTOR_BOWL: np.array([-0.18872204, 0.32034761, 0.89840387]),
    S.PLATE: np.array([0.05341727, 0.20518007, 0.90249958]),
}
# Pixels of those objects in the policy-view (180-degree rotated) agentview frame, read off the render.
POLICY_VIEW_PIXELS = {S.TARGET_BOWL: (65.1, 158.7), S.DISTRACTOR_BOWL: (37.5, 134.0), S.PLATE: (57.7, 185.5)}


class FakeSim:
    def __init__(self, positions):
        self.data = type("Data", (), {})()
        self.data.body_xpos = np.stack(list(positions.values()))
        self.names = list(positions)


class FakeRobosuiteEnv:
    def __init__(self, positions):
        self.sim = FakeSim(positions)
        self.obj_body_id = {name: index for index, name in enumerate(positions)}

    def move(self, name, delta):
        self.sim.data.body_xpos[self.obj_body_id[name]] += np.asarray(delta)


class FakeLiberoEnv:
    """Mimics the parts of LeRobot's LiberoEnv that libero_scene touches."""

    def __init__(self, positions):
        self._env = type("Raw", (), {})()
        self._env.env = FakeRobosuiteEnv(positions)
        self.camera_name = ["agentview_image", "robot0_eye_in_hand_image"]
        self.camera_name_mapping = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        self.observation_height = self.observation_width = 256

    def _ensure_env(self):
        pass


class CameraConventionTest(unittest.TestCase):
    def test_policy_view_intrinsic_mirrors_u(self):
        mirrored = S.policy_view_intrinsic(PINHOLE_K, 256)
        pinhole = project_points(OBJECTS[S.PLATE], PINHOLE_K, PINHOLE_E)[0]
        policy = project_points(OBJECTS[S.PLATE], mirrored, PINHOLE_E)[0]
        self.assertAlmostEqual(policy[0], 255 - pinhole[0], places=6)
        self.assertAlmostEqual(policy[1], pinhole[1], places=6)

    def test_projection_matches_rendered_positions(self):
        intrinsic = S.policy_view_intrinsic(PINHOLE_K, 256)
        for name, expected in POLICY_VIEW_PIXELS.items():
            pixel = project_points(OBJECTS[name], intrinsic, PINHOLE_E)[0]
            np.testing.assert_allclose(pixel, expected, atol=0.2, err_msg=name)

    def test_synthetic_camera_round_trip(self):
        """A point straight ahead of the camera lands on the principal point; +x_world goes left in the policy view."""
        intrinsic = S.policy_view_intrinsic(np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1]]), 100)
        extrinsic = np.hstack([np.eye(3), np.zeros((3, 1))])
        centre = project_points(np.array([[0.0, 0.0, 2.0]]), intrinsic, extrinsic)[0]
        np.testing.assert_allclose(centre, [99 - 50, 50])
        right = project_points(np.array([[0.5, 0.0, 2.0]]), intrinsic, extrinsic)[0]
        self.assertLess(right[0], centre[0])
        down = project_points(np.array([[0.0, 0.5, 2.0]]), intrinsic, extrinsic)[0]
        self.assertGreater(down[1], centre[1])

    def test_policy_view_rotates_180(self):
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        rotated = S.policy_view(image)
        np.testing.assert_array_equal(rotated[0, 0], image[-1, -1])
        np.testing.assert_array_equal(rotated[-1, -1], image[0, 0])
        self.assertTrue(rotated.flags["C_CONTIGUOUS"])


class StateVectorTest(unittest.TestCase):
    def test_layout_and_axis_angle(self):
        raw = {"robot0_eef_pos": np.array([-0.21124163, -0.0110906, 1.17424953]),
               "robot0_eef_quat": np.array([9.99609615e-01, -9.06336483e-04, -2.79235845e-02, -2.64635271e-04]),
               "robot0_gripper_qpos": np.array([0.03872337, -0.0387222]),
               "robot0_joint_pos": np.zeros(7), "agentview_image": np.zeros((2, 2, 3), np.uint8)}
        state = S.state_vector(raw)
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(state.shape, (8,))
        np.testing.assert_allclose(state[:3], raw["robot0_eef_pos"], rtol=1e-6)
        np.testing.assert_allclose(state[6:], raw["robot0_gripper_qpos"], rtol=1e-6)
        # value LiberoProcessorStep produced for this quaternion in the real sim
        np.testing.assert_allclose(state[3:6], [3.1408956, -0.00284782, -0.08773931], atol=1e-5)
        self.assertIs(S.libero_state_vector, S.state_vector)

    def test_axis_angle_identity_is_zero(self):
        np.testing.assert_array_equal(S.quat_to_axis_angle([0, 0, 0, 1]), np.zeros(3))
        half_turn_x = S.quat_to_axis_angle([1, 0, 0, 0])
        np.testing.assert_allclose(half_turn_x, [np.pi, 0, 0], atol=1e-6)

    def test_images_from_raw_uses_env_mapping(self):
        env = FakeLiberoEnv(OBJECTS)
        raw = {"agentview_image": np.ones((4, 4, 3), np.uint8), "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
               "robot0_eef_pos": np.zeros(3)}
        images = S.images_from_raw(raw, env)
        self.assertEqual(set(images), {"image", "image2"})
        self.assertEqual(int(images["image"].sum()), 4 * 4 * 3)


class BowlChoiceTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeLiberoEnv({k: v.copy() for k, v in OBJECTS.items()})
        self.layout = S.scene_layout(self.env)

    def test_layout_contents(self):
        for name in S.SCENE_OBJECTS:
            np.testing.assert_allclose(self.layout[name], OBJECTS[name])
            self.assertAlmostEqual(self.layout["initial_z"][name], OBJECTS[name][2])
        with self.assertRaises(KeyError):
            S.object_position(self.env, "no_such_object")

    def test_tracker_latches_first_lifted_bowl(self):
        tracker = S.BowlChoiceTracker(self.layout)
        self.assertEqual(tracker.update(self.env), "none")
        self.env._env.env.move(S.DISTRACTOR_BOWL, [0, 0, 0.03])
        self.assertEqual(tracker.update(self.env), "none")
        self.env._env.env.move(S.DISTRACTOR_BOWL, [0, 0, 0.03])
        self.assertEqual(tracker.update(self.env), "distractor")
        self.env._env.env.move(S.TARGET_BOWL, [0, 0, 0.2])
        self.assertEqual(tracker.update(self.env), "distractor")
        self.assertEqual(tracker.choice, "distractor")

    def test_stateless_choice_prefers_higher_bowl(self):
        self.env._env.env.move(S.TARGET_BOWL, [0, 0, 0.06])
        self.assertEqual(S.bowl_choice(self.env, self.layout), "target")
        self.env._env.env.move(S.DISTRACTOR_BOWL, [0, 0, 0.1])
        self.assertEqual(S.bowl_choice(self.env, self.layout), "distractor")
        self.assertEqual(S.bowl_choice(self.env, self.layout, lift=0.5), "none")


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def _vec(self, tag):
        vector = np.zeros(I.INSTR_DIM, dtype=np.float32)
        vector[0] = {"text": 1, "image": 2, "audio": 3}[tag]
        return vector[None]

    def embed_text(self, texts):
        self.calls.append(("text", list(texts)))
        return np.concatenate([self._vec("text") for _ in texts])

    def embed_images(self, images):
        self.calls.append(("image", len(images)))
        return np.concatenate([self._vec("image") for _ in images])

    def embed_audio(self, clips):
        self.calls.append(("audio", [np.asarray(c).shape for c in clips]))
        return np.concatenate([self._vec("audio") for _ in clips])


class FakeTTS:
    def __init__(self):
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        return np.linspace(-0.5, 0.5, 1600, dtype=np.float32)


class InstructionsTest(unittest.TestCase):
    def test_task_text_normalises_whitespace(self):
        self.assertEqual(I.task_text("  pick up  the bowl\n"), "pick up the bowl")
        with self.assertRaises(ValueError):
            I.task_text("   ")

    def test_build_sketch_marks_target_and_plate(self):
        frame = np.full((256, 256, 3), 200, np.uint8)
        cam = (S.policy_view_intrinsic(PINHOLE_K, 256), PINHOLE_E)
        layout = S.scene_layout(FakeLiberoEnv(OBJECTS))
        sketch = I.build_sketch(frame, cam, layout)
        self.assertIsInstance(sketch, Image.Image)
        pixels = np.asarray(sketch)
        red = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 60)
        blue = (pixels[:, :, 2] > 200) & (pixels[:, :, 0] < 60)
        tu, tv = POLICY_VIEW_PIXELS[S.TARGET_BOWL]
        pu, pv = POLICY_VIEW_PIXELS[S.PLATE]
        red_v, red_u = np.nonzero(red)
        self.assertTrue(red.any() and blue.any())
        self.assertLess(abs(red_u.mean() - tu), 3)
        self.assertLess(abs(red_v.mean() - tv), 3)
        self.assertTrue(blue[int(pv) - 3:int(pv) + 4, int(pu) - 3:int(pu) + 4].any(), "arrow tip is on the plate")

    def test_embed_instruction_dispatch(self):
        embedder = FakeEmbedder()
        self.assertEqual(I.embed_instruction(embedder, "text", text="go")[0], 1)
        self.assertEqual(I.embed_instruction(embedder, "sketch", sketch=Image.new("RGB", (4, 4)))[0], 2)
        self.assertEqual(I.embed_instruction(embedder, "audio", audio=np.zeros(16))[0], 3)
        self.assertEqual(I.embed_instruction(embedder, "text", text="go").shape, (I.INSTR_DIM,))
        with self.assertRaises(ValueError):
            I.embed_instruction(embedder, "text")
        with self.assertRaises(ValueError):
            I.embed_instruction(embedder, "video", text="x")

    def test_build_audio_and_cache_reuse(self):
        embedder, tts = FakeEmbedder(), FakeTTS()
        wav = I.build_audio("pick up the bowl", tts)
        self.assertEqual(wav.dtype, np.float32)
        cache = I.InstructionCache(embedder, tts)
        key = (0, 3)
        first = cache.get("audio", key, text="pick up the bowl")
        second = cache.get("audio", key, text="pick up the bowl")
        self.assertIs(first, second)
        self.assertEqual(tts.calls, 2)  # one direct build_audio call + one through the cache
        cache.get("text", key, text="pick up the bowl")
        cache.get("text", key, text="pick up the bowl")
        self.assertEqual(sum(1 for call in embedder.calls if call[0] == "text"), 1)
        with self.assertRaises(ValueError):
            I.InstructionCache(embedder).get("audio", key, text="x")
        out = I.embed_all(embedder, "pick", sketch=Image.new("RGB", (4, 4)))
        self.assertEqual(set(out), {"text", "sketch"})


if __name__ == "__main__":
    unittest.main()

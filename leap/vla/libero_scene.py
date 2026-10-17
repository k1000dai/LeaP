"""LIBERO-Spatial scene access on top of LeRobot's ``LiberoEnv`` (envs/libero.py, LeRobot 0.6.1, 9459727e7).

Conventions, all verified in the LIBERO sim (see tests/test_vla_scene.py and the lead report):

Images
    ``LiberoEnv`` returns ``pixels = {"image": agentview, "image2": wrist}`` as uint8 HxWx3 arrays
    straight from robosuite; MuJoCo renders them upside down.  ``lerobot-eval`` then runs
    ``LiberoProcessorStep`` which rotates every image by 180 degrees (flip H and W) before the policy
    sees it.  ``policy_view(image)`` applies that rotation; everything the VLA (and a human) looks at
    is the policy view.

Camera
    ``agentview_camera(env)`` returns ``(intrinsic_cv, extrinsic_cv)`` for ``leap.steer.sketch.project_points``
    (pixel = K [R | t] X, u along the width, v along the height, origin top-left) *for the policy
    view*.  robosuite's ``get_camera_intrinsic_matrix`` / ``get_camera_extrinsic_matrix`` describe a
    pinhole whose image is the raw render flipped vertically; after the 180-degree rotation the
    policy view is that pinhole image mirrored horizontally, which is folded into the intrinsic as
    ``fx -> -fx`` and ``cx -> W - 1 - cx``.  The extrinsic is the plain OpenCV world->camera [R | t].

State
    ``state_vector(raw_obs)`` is the 8-d ``observation.state`` that ``LiberoProcessorStep`` builds from
    ``LiberoEnv._format_raw_obs``: ``[eef_pos (3), axis_angle(eef_quat xyzw) (3), gripper_qpos (2)]``,
    float32.  The axis-angle conversion is a numpy port of ``LiberoProcessorStep._quat2axisangle``.

Objects
    LIBERO-Spatial BDDL names: ``akita_black_bowl_1`` (the bowl the goal asks for), ``akita_black_bowl_2``
    (the distractor) and ``plate_1``; body positions are read from ``sim.data.body_xpos`` through the
    env's ``obj_body_id`` map.

Replay
    ``replay_state(env, flat_state)`` restores a stored demo state (``sim.get_state().flatten()``) with
    LIBERO's ``regenerate_obs_from_state`` (``set_state_from_flattened`` + ``forward`` + observables) and
    returns the raw observation dict, re-rendered at the env's resolution.

Heavy imports (lerobot, libero, robosuite) happen inside the functions so the module imports on CPU.
"""
import numpy as np

NEUTRAL_PROMPT = "pick up the black bowl and place it on the plate"
TARGET_BOWL = "akita_black_bowl_1"
DISTRACTOR_BOWL = "akita_black_bowl_2"
PLATE = "plate_1"
SCENE_OBJECTS = (TARGET_BOWL, DISTRACTOR_BOWL, PLATE)
AGENTVIEW_CAMERA = "agentview"
DEFAULT_RESOLUTION = 256
STATE_DIM = 8
LIFT_THRESHOLD = 0.05


# --- suite / env construction -------------------------------------------------------------------

def load_suite(name="libero_spatial"):
    """Instantiate a LIBERO benchmark suite (default task order) by name."""
    from libero.libero import benchmark
    suites = benchmark.get_benchmark_dict()
    if name not in suites:
        raise ValueError(f"Unknown LIBERO suite {name!r}; available: {sorted(suites)}")
    return suites[name]()


def suite_name(suite):
    """The suite's registered name (``suite.name``), e.g. ``libero_spatial``."""
    return str(suite.name)


def make_env(suite, task_id, prompt_override=None, resolution=DEFAULT_RESOLUTION, **kwargs):
    """Build LeRobot's ``LiberoEnv`` for ``suite`` / ``task_id`` with agentview + wrist at ``resolution``.

    ``prompt_override`` replaces ``env.task_description`` (the string ``lerobot-eval`` and our rollout loop
    feed to the policy as ``observation["task"]``); ``task_description`` is a plain attribute set in
    ``LiberoEnv.__init__`` from ``task.language``, so it is overwritten after construction.  The original
    language stays available as ``env.task_language``.  Extra ``kwargs`` go to ``LiberoEnv`` (for example
    ``episode_index``, ``init_states``, ``num_steps_wait``).  The underlying sim is created lazily on the
    first ``reset()``.
    """
    from lerobot.envs.libero import LiberoEnv
    options = {"obs_type": "pixels_agent_pos", "observation_height": resolution,
               "observation_width": resolution}
    options.update(kwargs)
    env = LiberoEnv(task_suite=suite, task_id=int(task_id), task_suite_name=suite_name(suite), **options)
    env.task_language = suite.get_task(int(task_id)).language
    if prompt_override is not None:
        env.task_description = str(prompt_override)
    return env


def raw_env(env):
    """The underlying LIBERO ``OffScreenRenderEnv`` (created on first use, as ``LiberoEnv`` does)."""
    env._ensure_env()
    return env._env


def robosuite_env(env):
    """The robosuite task env inside the LIBERO wrapper (has ``obj_body_id``, ``sim``, ``_get_observations``)."""
    return raw_env(env).env


def max_steps(suite_name_):
    """Episode length LeRobot uses for a suite (``TASK_SUITE_MAX_STEPS``)."""
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
    if suite_name_ not in TASK_SUITE_MAX_STEPS:
        raise ValueError(f"No max steps for suite {suite_name_!r}; known: {sorted(TASK_SUITE_MAX_STEPS)}")
    return int(TASK_SUITE_MAX_STEPS[suite_name_])


# --- images ------------------------------------------------------------------------------------

def policy_view(image):
    """Rotate a raw robosuite render by 180 degrees (what ``LiberoProcessorStep`` feeds the policy)."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def images_from_raw(raw_obs, env):
    """``{"image": agentview, "image2": wrist}`` uint8 HxWx3 exactly as ``LiberoEnv._format_raw_obs`` builds ``pixels``."""
    return {env.camera_name_mapping[name]: np.asarray(raw_obs[name]) for name in env.camera_name}


# --- state ---------------------------------------------------------------------------------------

def quat_to_axis_angle(quat_xyzw):
    """numpy port of ``LiberoProcessorStep._quat2axisangle`` for one (x, y, z, w) quaternion."""
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.float32(w))
    return (quat[:3] / np.float32(den) * np.float32(angle)).astype(np.float32)


def state_vector(raw_obs, env=None):
    """8-d ``observation.state``: ``[robot0_eef_pos, axis_angle(robot0_eef_quat), robot0_gripper_qpos]`` float32.

    ``env`` is accepted for symmetry with the other helpers and unused: the layout does not depend on it.
    """
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    axis_angle = quat_to_axis_angle(raw_obs["robot0_eef_quat"])
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
    state = np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must be {STATE_DIM}-d, got {state.shape}")
    return state


libero_state_vector = state_vector


# --- scene objects --------------------------------------------------------------------------------

def object_position(env, name):
    """World xyz of a BDDL object's main body (``obj_body_id[name]`` -> ``sim.data.body_xpos``)."""
    rs_env = robosuite_env(env)
    if name not in rs_env.obj_body_id:
        raise KeyError(f"{name!r} is not an object of this task; have {sorted(rs_env.obj_body_id)}")
    return np.array(rs_env.sim.data.body_xpos[rs_env.obj_body_id[name]], dtype=np.float64)


def scene_layout(env, names=SCENE_OBJECTS):
    """World xyz of the target bowl, distractor bowl and plate plus their current (initial) heights.

    Returns ``{name: xyz (3,) float64 for name in names, "initial_z": {name: z}}``; call it right after
    ``reset`` so ``initial_z`` is the resting height ``BowlChoiceTracker`` compares against.
    """
    layout = {name: object_position(env, name) for name in names}
    layout["initial_z"] = {name: float(layout[name][2]) for name in names}
    return layout


def lifted_bowls(env, layout, lift=LIFT_THRESHOLD):
    """Bowls currently at least ``lift`` metres above their ``initial_z`` -> {name: rise}."""
    rises = {}
    for name in (TARGET_BOWL, DISTRACTOR_BOWL):
        rise = float(object_position(env, name)[2]) - layout["initial_z"][name]
        if rise >= lift:
            rises[name] = rise
    return rises


def bowl_choice(env, layout, lift=LIFT_THRESHOLD):
    """Stateless check of the current frame: ``"target"``, ``"distractor"`` (the higher one if both) or ``"none"``."""
    rises = lifted_bowls(env, layout, lift)
    if not rises:
        return "none"
    name = max(rises, key=rises.get)
    return "target" if name == TARGET_BOWL else "distractor"


class BowlChoiceTracker:
    """Remember which bowl first rose ``lift`` metres above its initial height across the steps of an episode.

    ``update(env)`` reads the sim and returns the latched choice (``"target"`` / ``"distractor"`` / ``"none"``);
    once a bowl has been lifted the choice no longer changes.
    """

    def __init__(self, layout, lift=LIFT_THRESHOLD):
        self.layout = layout
        self.lift = lift
        self.choice = "none"

    def update(self, env):
        if self.choice == "none":
            self.choice = bowl_choice(env, self.layout, self.lift)
        return self.choice


# --- camera -----------------------------------------------------------------------------------------

def pinhole_camera(sim, camera_name, height, width):
    """robosuite's pinhole (K 3x3, world->camera [R | t] 3x4) for a sim camera at ``height`` x ``width``."""
    from robosuite.utils import camera_utils
    from robosuite.utils import transform_utils
    intrinsic = camera_utils.get_camera_intrinsic_matrix(sim, camera_name, height, width)
    camera_pose = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
    extrinsic = transform_utils.pose_inv(camera_pose)[:3]
    return np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)


def policy_view_intrinsic(intrinsic, width):
    """Mirror a pinhole intrinsic horizontally: the policy-view pixel is ``(W - 1 - u, v)`` of the pinhole pixel."""
    mirrored = np.array(intrinsic, dtype=np.float64)
    mirrored[0, 0] = -mirrored[0, 0]
    mirrored[0, 2] = width - 1 - mirrored[0, 2]
    return mirrored


def agentview_camera(env, camera_name=AGENTVIEW_CAMERA):
    """``(intrinsic_cv, extrinsic_cv)`` projecting world points onto the policy-view agentview frame.

    See the module docstring: robosuite's pinhole image is the raw render flipped vertically; the policy
    view (raw rotated 180 degrees) is the pinhole image mirrored horizontally, folded into the intrinsic.
    """
    sim = raw_env(env).sim
    height, width = env.observation_height, env.observation_width
    intrinsic, extrinsic = pinhole_camera(sim, camera_name, height, width)
    return policy_view_intrinsic(intrinsic, width), extrinsic


# --- replay -----------------------------------------------------------------------------------------

def replay_state(env, flat_state):
    """Restore a flattened MuJoCo state in the env and return the raw observation dict (images re-rendered).

    Uses LIBERO's ``regenerate_obs_from_state`` = ``sim.set_state_from_flattened`` + ``sim.forward`` +
    ``_update_observables(force=True)``; the env must have been reset once (``raw_env`` does that).
    """
    flat_state = np.asarray(flat_state, dtype=np.float64).reshape(-1)
    return raw_env(env).regenerate_obs_from_state(flat_state)


def sim_state(env):
    """The current flattened MuJoCo state (same layout as the stored demo ``states``)."""
    return np.asarray(raw_env(env).get_sim_state(), dtype=np.float64)

"""Counterfactual bowl-2 rollouts for LIBERO-Spatial (phase 3b): per-bowl phrases and prompts, success for either
bowl, sketches of either bowl, and the episode record shared by ``scripts/vla/rollout_counterfactual.py`` (writes
it), ``scripts/vla/reverse_rollouts.py`` (reads it) and ``scripts/vla/eval_vla_steer.py --target-bowl 2``.

Why
    Every LIBERO-Spatial demo moves ``akita_black_bowl_1``, so within a layout the instruction is constant and
    redundant with the observation.  Prompting the frozen pi0.5 with the spatial phrase of ``akita_black_bowl_2``
    (read from the task's BDDL ``:init`` block) gives rollouts in which the same layout leads to the other bowl,
    so the reversed noises depend on the instruction and not only on the scene.

Phrases
    The suite languages are the BDDL file names: "pick up the black bowl <phrase> and place it on the plate".
    ``REGION_PHRASES`` maps every ``:init`` region of the two bowls to the wording of those file names; every
    bowl-2 region is the bowl-1 region of another task, so each counterfactual prompt is a sentence the checkpoint
    was fine-tuned on, with the bowls swapped (suite order, LIBERO default task order):

    task  bowl-1 phrase (= task language)              bowl-2 region -> phrase
    0     between the plate and the ramekin             next_to_ramekin_region -> next to the ramekin
    1     next to the ramekin                           next_to_box_region -> next to the cookie box
    2     from table center                             next_to_plate_region -> next to the plate
    3     on the cookie box                             wooden_cabinet_1_top_side -> on the wooden cabinet
    4     in the top drawer of the wooden cabinet       wooden_cabinet_1_top_side -> on the wooden cabinet
    5     on the ramekin                                cookies_1 -> on the cookie box
    6     next to the cookie box                        flat_stove_1_cook_region -> on the stove
    7     on the stove                                  wooden_cabinet_1_top_side -> on the wooden cabinet
    8     next to the plate                             next_to_ramekin_region -> next to the ramekin
    9     on the wooden cabinet                         flat_stove_1_cook_region -> on the stove

    ``bowl_phrases`` checks, per task, that the bowl-1 phrase reproduces the task's own language, so a LIBERO
    version with other region names fails loudly instead of prompting nonsense.

Success
    LIBERO's goal ``(On akita_black_bowl_1 plate_1)`` is evaluated as ``plate.check_ontop(bowl)``
    (libero/envs/object_states/base_object_states.py): ``plate_z <= bowl_z``, the two bodies are in contact, and
    ``|plate_xy - bowl_xy| < 0.03`` (body origins).  ``bowl_on_plate(env, bowl)`` evaluates exactly that predicate
    for either bowl through the env's ``object_states_dict``.  ``on_plate_geometric`` is the contact-free
    approximation (same 3 cm radius, ``plate_z <= bowl_z <= plate_z + Z_MAX`` with ``Z_MAX`` = 5 cm as the contact
    proxy) used where no sim is available; the rollout script logs it next to the predicate.  Neither test asks
    the gripper to have released the bowl: LIBERO does not either.

Episode files
    ``task_{i}_bowl{b}_episode{e}.npz`` (``np.savez_compressed``), written by ``EpisodeRecorder.arrays``:

    agentview, wrist [T, H, W, 3] u8   policy view (raw render rotated 180 degrees = what the VLA sees);
                                       ``raw_images`` rotates them back for ``pi05_flow.prepare_batches``
    state [T, 8] f32                   LeRobot ``observation.state`` before action t
    action [T, 7] f32                  the action executed at step t
    positions [T, 3, 3] f64            xyz of (bowl 1, bowl 2, plate) before action t (``POSITION_OBJECTS``)
    on_plate [T] bool                  LIBERO predicate for the intended bowl after action t
    pred_step [C] i64                  env steps at which the VLA was queried (observation of that step)
    chunk [C, 50, 7] f32               its un-normalised 50-step prediction (``sample_actions_with_noise``)
    noise [C, 50, 32] f32              the Gaussian source noise of that prediction
    executed [C] bool                  True for the predictions whose first n_action_steps actions were run
    cam_intrinsic [3, 3], cam_extrinsic [3, 4]   ``libero_scene.agentview_camera`` (policy view)
    scalars: task_id, bowl, episode, prompt, task_language, success, success_geometric, libero_success,
             success_step, lifted_bowl, bowl_choice, steps, seed, checkpoint, num_steps, n_action_steps,
             image_view = "policy"

Only numpy (and ``leap.vla.libero_scene``, which imports nothing heavy) at module level.
"""
import json
import os
from pathlib import Path
import re

import numpy as np

from leap.vla import libero_scene as S

BOWLS = (1, 2)
BOWL_NAMES = {1: S.TARGET_BOWL, 2: S.DISTRACTOR_BOWL}
POSITION_OBJECTS = (S.TARGET_BOWL, S.DISTRACTOR_BOWL, S.PLATE)
XY_TOLERANCE = 0.03     # LIBERO check_ontop: |plate_xy - bowl_xy| < 0.03
Z_MAX = 0.05            # geometric contact proxy: bowl origin at most 5 cm above the plate origin
IMAGE_VIEW = "policy"

# BDDL ``:init`` region of a bowl -> the wording of the LIBERO-Spatial task languages (BDDL file names).
REGION_PHRASES = {
    "main_table_between_plate_ramekin_region": "between the plate and the ramekin",
    "main_table_next_to_ramekin_region": "next to the ramekin",
    "main_table_table_center": "from table center",
    "cookies_1": "on the cookie box",
    "wooden_cabinet_1_top_region": "in the top drawer of the wooden cabinet",
    "glazed_rim_porcelain_ramekin_1": "on the ramekin",
    "main_table_next_to_box_region": "next to the cookie box",
    "flat_stove_1_cook_region": "on the stove",
    "main_table_next_to_plate_region": "next to the plate",
    "wooden_cabinet_1_top_side": "on the wooden cabinet",
}
LANGUAGE_RE = re.compile(r"^(pick up the black bowl )(.+?)( and place it on the plate)$")
INIT_PREDICATE_RE = re.compile(r"\(\s*(On|In)\s+(\S+)\s+(\S+)\s*\)")
EPISODE_RE = re.compile(r"^task_(\d+)_bowl(\d)_episode(\d+)\.npz$")


# --- phrases and prompts -------------------------------------------------------------------------------------

def bowl_name(bowl):
    """``1`` -> ``akita_black_bowl_1`` (LIBERO's target), ``2`` -> ``akita_black_bowl_2`` (the distractor)."""
    bowl = int(bowl)
    if bowl not in BOWL_NAMES:
        raise ValueError(f"bowl must be one of {BOWLS}, got {bowl}")
    return BOWL_NAMES[bowl]


def other_bowl(bowl):
    return 2 if int(bowl) == 1 else 1


def parse_init(bddl_text):
    """``[(predicate, object, region), ...]`` of the ``(:init ...)`` block of a BDDL problem (``On`` / ``In`` only)."""
    text = str(bddl_text)
    start = text.find("(:init")
    end = text.find("(:goal", start)
    if start < 0 or end < 0:
        raise ValueError("BDDL text has no (:init ...) block followed by (:goal ...)")
    return [(pred, obj, region) for pred, obj, region in INIT_PREDICATE_RE.findall(text[start:end])]


def bowl_regions(bddl_text):
    """``{1: region, 2: region}``: where the two bowls start according to the BDDL ``:init`` block."""
    regions = {}
    for _, obj, region in parse_init(bddl_text):
        for bowl, name in BOWL_NAMES.items():
            if obj == name:
                regions[bowl] = region
    missing = [bowl for bowl in BOWLS if bowl not in regions]
    if missing:
        raise ValueError(f"BDDL init block does not place bowl(s) {missing}")
    return regions


def region_phrase(region):
    """The task-language wording of a BDDL init region (``REGION_PHRASES``)."""
    if region not in REGION_PHRASES:
        raise KeyError(f"no phrase for BDDL region {region!r}; known: {sorted(REGION_PHRASES)}")
    return REGION_PHRASES[region]


def split_language(task_language):
    """``(prefix, phrase, suffix)`` of "pick up the black bowl <phrase> and place it on the plate"."""
    text = " ".join(str(task_language).split())
    match = LANGUAGE_RE.match(text)
    if match is None:
        raise ValueError(f"task language does not follow the LIBERO-Spatial pattern: {text!r}")
    return match.group(1), match.group(2), match.group(3)


def prompt_for_bowl(task_language, phrase):
    """The task language with its spatial phrase replaced: "pick up the black bowl <phrase> and place it on the plate".

    ``prompt_for_bowl(language, bowl_phrases(...)[1]) == language`` for every LIBERO-Spatial task.
    """
    prefix, _, suffix = split_language(task_language)
    phrase = " ".join(str(phrase).split())
    if not phrase:
        raise ValueError("phrase is empty")
    return prefix + phrase + suffix


def bowl_phrases_from_bddl(bddl_text, task_language=None):
    """``{1: phrase, 2: phrase}`` from a BDDL problem text; with ``task_language`` the bowl-1 phrase is checked against it."""
    phrases = {bowl: region_phrase(region) for bowl, region in bowl_regions(bddl_text).items()}
    if task_language is not None:
        _, own, _ = split_language(task_language)
        if phrases[1] != own:
            raise ValueError(f"bowl-1 phrase {phrases[1]!r} from the BDDL does not reproduce the task language phrase {own!r}")
    return phrases


def task_bddl_text(suite, task_id):
    """The BDDL problem text of ``suite`` / ``task_id`` (``suite.get_task_bddl_file_path``)."""
    path = suite.get_task_bddl_file_path(int(task_id))
    with open(path) as handle:
        return handle.read()


def bowl_phrases(suite, task_id):
    """``{1: phrase, 2: phrase}`` for a suite task, checked against the task's own language."""
    language = suite.get_task(int(task_id)).language
    return bowl_phrases_from_bddl(task_bddl_text(suite, task_id), language)


def target_prompt(task_language, phrases, bowl):
    """The full prompt naming ``bowl``: the language itself for bowl 1, ``prompt_for_bowl`` with the bowl-2 phrase otherwise."""
    bowl = int(bowl)
    if bowl == 1:
        return " ".join(str(task_language).split())
    return prompt_for_bowl(task_language, phrases[bowl])


# --- success and bowl choice for either bowl ----------------------------------------------------------------

def on_plate_geometric(bowl_xyz, plate_xyz, xy_tolerance=XY_TOLERANCE, z_max=Z_MAX):
    """Contact-free mirror of LIBERO's ``check_ontop``: ``plate_z <= bowl_z <= plate_z + z_max`` and ``|dxy| < xy_tolerance``."""
    bowl_xyz = np.asarray(bowl_xyz, dtype=np.float64).reshape(3)
    plate_xyz = np.asarray(plate_xyz, dtype=np.float64).reshape(3)
    dxy = float(np.linalg.norm(bowl_xyz[:2] - plate_xyz[:2]))
    dz = float(bowl_xyz[2] - plate_xyz[2])
    return bool(dxy < xy_tolerance and 0.0 <= dz <= z_max)


def bowl_on_plate(env, bowl):
    """LIBERO's ``On(bowl, plate)`` predicate for either bowl: ``plate.check_ontop(bowl)`` on the env's object states.

    Falls back to ``on_plate_geometric`` on the body positions when the robosuite env has no
    ``object_states_dict`` (fakes in tests).
    """
    name = bowl_name(bowl)
    rs_env = S.robosuite_env(env)
    states = getattr(rs_env, "object_states_dict", None)
    if states and S.PLATE in states and name in states:
        return bool(states[S.PLATE].check_ontop(states[name]))
    return on_plate_geometric(S.object_position(env, name), S.object_position(env, S.PLATE))


def intended_choice(choice, bowl):
    """Re-label ``BowlChoiceTracker``'s bowl-1 view ("target" = bowl 1) so that "target" means the intended ``bowl``."""
    if int(bowl) == 1 or choice == "none":
        return choice
    return {"target": "distractor", "distractor": "target"}[choice]


def lifted_bowl(choice):
    """``BowlChoiceTracker`` choice (bowl-1 view) -> 1, 2 or 0 (nothing lifted)."""
    return {"target": 1, "distractor": 2, "none": 0}[choice]


def object_positions(env, names=POSITION_OBJECTS):
    """``[len(names), 3]`` float64 world positions of the listed bodies."""
    return np.stack([S.object_position(env, name) for name in names]).astype(np.float64)


# --- sketches --------------------------------------------------------------------------------------------------

def build_bowl_sketch(agentview_rgb, cam, layout, bowl):
    """``instructions.build_sketch`` for an arbitrary bowl: red circle on the bowl, blue arrow to the plate, white canvas."""
    from leap.steer.sketch import render_sketch
    intrinsic, extrinsic = cam
    return render_sketch(np.asarray(agentview_rgb, dtype=np.uint8), intrinsic, extrinsic,
                         np.asarray(layout[bowl_name(bowl)], dtype=np.float64), np.asarray(layout[S.PLATE], dtype=np.float64))


def layout_from_positions(positions, names=POSITION_OBJECTS):
    """``scene_layout``-like dict from a ``[len(names), 3]`` position row (for sketches without a sim)."""
    positions = np.asarray(positions, dtype=np.float64).reshape(len(names), 3)
    layout = {name: positions[index] for index, name in enumerate(names)}
    layout["initial_z"] = {name: float(layout[name][2]) for name in names}
    return layout


# --- episode records ---------------------------------------------------------------------------------------------

def episode_file(task_id, bowl, episode):
    return f"task_{int(task_id)}_bowl{int(bowl)}_episode{int(episode)}.npz"


def episode_path(rollouts_dir, task_id, bowl, episode):
    return Path(rollouts_dir) / episode_file(task_id, bowl, episode)


def parse_episode_name(name):
    """``task_{i}_bowl{b}_episode{e}.npz`` -> ``(i, b, e)`` or None."""
    match = EPISODE_RE.match(Path(name).name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def find_episodes(rollouts_dir, task_id=None, bowl=None):
    """Sorted ``[(task_id, bowl, episode, path), ...]`` of the episode files under ``rollouts_dir`` (optionally filtered)."""
    found = []
    for path in Path(rollouts_dir).iterdir():
        key = parse_episode_name(path.name)
        if key is None:
            continue
        if task_id is not None and key[0] != int(task_id):
            continue
        if bowl is not None and key[1] != int(bowl):
            continue
        found.append(key + (path,))
    return sorted(found)


def demo_id_for(bowl, episode):
    """Row identity of a rollout in the reversal npz: ``bowl * 1000 + episode``."""
    return int(bowl) * 1000 + int(episode)


def dataset_index(task_id, bowl):
    """``task_{n}.npz`` index of the (task, bowl) file in the counterfactual reversal dataset: ``100 * bowl + task_id``.

    One file per (task, bowl) keeps ``scripts/vla/embed_vla_instructions.py`` unchanged: it embeds one text per
    file and repeats it over the file's demos, so the text can only vary between files.
    """
    return 100 * int(bowl) + int(task_id)


class EpisodeRecorder:
    """Accumulate one rollout step by step; ``arrays()`` packs everything for ``np.savez_compressed``.

    Call ``add_step`` with the observation of env step ``t`` (before the action), ``add_prediction`` for every
    VLA query made from that observation, then ``add_result`` with the executed action and the post-step flags.
    """

    def __init__(self):
        self.agentview, self.wrist, self.state, self.positions = [], [], [], []
        self.action, self.on_plate = [], []
        self.pred_step, self.chunk, self.noise, self.executed = [], [], [], []

    @property
    def steps(self):
        return len(self.action)

    def add_step(self, agentview_policy, wrist_policy, state, positions):
        agentview_policy = np.asarray(agentview_policy)
        wrist_policy = np.asarray(wrist_policy)
        for image in (agentview_policy, wrist_policy):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"images must be uint8 HxWx3, got {image.dtype} {image.shape}")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (S.STATE_DIM,):
            raise ValueError(f"state must be {S.STATE_DIM}-d, got {state.shape}")
        self.agentview.append(agentview_policy)
        self.wrist.append(wrist_policy)
        self.state.append(state)
        self.positions.append(np.asarray(positions, dtype=np.float64).reshape(len(POSITION_OBJECTS), 3))

    def add_prediction(self, step, chunk, noise, executed):
        chunk = np.asarray(chunk, dtype=np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        if chunk.ndim != 2 or noise.ndim != 2 or chunk.shape[0] != noise.shape[0]:
            raise ValueError(f"chunk {chunk.shape} and noise {noise.shape} must be [H, A] and [H, A_max]")
        self.pred_step.append(int(step))
        self.chunk.append(chunk)
        self.noise.append(noise)
        self.executed.append(bool(executed))

    def add_result(self, action, on_plate):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.action.append(action)
        self.on_plate.append(bool(on_plate))

    def arrays(self):
        if not self.action or len(self.action) != len(self.state):
            raise ValueError(f"{len(self.state)} observations but {len(self.action)} actions")
        chunk_shape = self.chunk[0].shape if self.chunk else (0, 0)
        noise_shape = self.noise[0].shape if self.noise else (0, 0)
        return {
            "agentview": np.stack(self.agentview), "wrist": np.stack(self.wrist),
            "state": np.stack(self.state).astype(np.float32), "action": np.stack(self.action).astype(np.float32),
            "positions": np.stack(self.positions), "on_plate": np.asarray(self.on_plate, dtype=bool),
            "pred_step": np.asarray(self.pred_step, dtype=np.int64),
            "chunk": (np.stack(self.chunk) if self.chunk else np.zeros((0,) + chunk_shape, np.float32)),
            "noise": (np.stack(self.noise) if self.noise else np.zeros((0,) + noise_shape, np.float32)),
            "executed": np.asarray(self.executed, dtype=bool),
        }


def save_episode(path, arrays, scalars):
    """Write an episode npz atomically (compressed); ``scalars`` are stored as 0-d arrays (str / int / bool / float)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    for key, value in scalars.items():
        payload[key] = np.asarray(value)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temp, path)
    return path


def episode_scalars(path, keys=("task_id", "bowl", "episode", "success", "success_geometric", "libero_success",
                                "success_step", "lifted_bowl", "bowl_choice", "steps", "prompt", "seed", "elapsed_s")):
    """Read the small scalar entries of an episode npz without touching the image arrays."""
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            if key in data.files:
                out[key] = _unwrap(data[key])
    return out


def load_episode(path):
    """All arrays of an episode npz as a dict (scalars unwrapped to Python values)."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            out[key] = _unwrap(value)
    return out


def _unwrap(value):
    value = np.asarray(value)
    if value.ndim != 0:
        return value
    item = value.item()
    return str(item) if isinstance(item, (bytes, np.str_)) else item


def raw_images(agentview_policy, wrist_policy):
    """Policy-view renders back to the raw robosuite orientation ``{"image", "image2"}`` for ``prepare_batches``.

    The policy view is the raw render rotated by 180 degrees (``libero_scene.policy_view``); the rotation is an
    involution, so applying it again restores what ``LiberoEnv`` returned and ``prepare_batches`` (which rotates
    once more, as ``LiberoProcessorStep`` does) feeds the VLA exactly the stored picture.
    """
    return {"image": S.policy_view(agentview_policy), "image2": S.policy_view(wrist_policy)}


# --- reversal under the neutral prompt (shared by reverse_rollouts.py) ----------------------------------------------

def relative_l2(pred, target, action_dim=7):
    """Relative L2 error over the chunk and the first ``action_dim`` dims, per sample ``[B]``."""
    pred = np.asarray(pred, dtype=np.float64)[:, :, :action_dim]
    target = np.asarray(target, dtype=np.float64)[:, :, :action_dim]
    num = np.sqrt(((pred - target) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((target ** 2).sum(axis=(1, 2)))
    return (num / np.maximum(den, 1e-8)).astype(np.float32)


def reverse_chunks(flow, bundle, prompt, images, states, chunks, num_steps, measure_recon):
    """Prefix features and reversed noises of recorded VLA chunks under ``prompt``.

    ``images``: list of raw-orientation ``{"image", "image2"}`` dicts (see ``raw_images``), ``states``: 8-d vectors,
    ``chunks`` ``[B, 50, 7]`` un-normalised, ``measure_recon`` ``[B]`` bool.  Returns
    ``(feat [B, 4104], noise [B, 50, 7], recon_err [B] with NaN where not measured)`` as float32 numpy.
    """
    observations = [flow.Observation(images=pixels, state=np.asarray(state, dtype=np.float32), prompt=prompt)
                    for pixels, state in zip(images, states)]
    batch = flow.prepare_batches(bundle, observations)
    ctx = flow.prefix_context(bundle, batch)
    feat = _numpy(flow.prefix_features(bundle, batch, ctx))
    target = flow.normalize_actions(bundle, np.asarray(chunks, dtype=np.float32))
    noise = flow.reverse_actions(bundle, ctx, target, num_steps)
    measure_recon = np.asarray(measure_recon, dtype=bool)
    recon = np.full(feat.shape[0], np.nan, dtype=np.float32)
    if measure_recon.any():
        recon_pred = _numpy(flow.forward_from_noise(bundle, ctx, noise, num_steps))
        errors = relative_l2(recon_pred, _numpy(target))
        recon[measure_recon] = errors[measure_recon]
    return feat.astype(np.float32), _numpy(noise)[:, :, :np.asarray(chunks).shape[-1]].astype(np.float32), recon


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


# --- manifests ------------------------------------------------------------------------------------------------------

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp, path)
    return path


def summarize_episodes(records):
    """Per (task, bowl) success table of episode records (dicts with task_id, bowl, success, ... as in ``episode_scalars``)."""
    groups = {}
    for record in records:
        groups.setdefault((int(record["task_id"]), int(record["bowl"])), []).append(record)
    table = {}
    for (task_id, bowl), rows in sorted(groups.items()):
        n = len(rows)
        counts = {choice: sum(r.get("bowl_choice") == choice for r in rows) for choice in ("target", "distractor", "none")}
        table[f"{task_id}:{bowl}"] = {
            "task_id": task_id, "bowl": bowl, "episodes": n,
            "success": sum(bool(r["success"]) for r in rows),
            "success_rate": sum(bool(r["success"]) for r in rows) / n,
            "success_geometric": sum(bool(r.get("success_geometric", False)) for r in rows),
            "libero_success": sum(bool(r.get("libero_success", False)) for r in rows),
            "bowl_choice_counts": counts,
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_seconds": float(np.mean([r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]))
            if any(r.get("elapsed_s") is not None for r in rows) else None,
        }
    return table


def format_table(table):
    """Markdown rows of ``summarize_episodes``."""
    lines = ["| task | bowl | episodes | success | geometric | libero (bowl 1) | choice target/distractor/none | mean steps | s/episode |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in table.values():
        counts = row["bowl_choice_counts"]
        seconds = "" if row["mean_seconds"] is None else f"{row['mean_seconds']:.1f}"
        lines.append(f"| {row['task_id']} | {row['bowl']} | {row['episodes']} | {row['success']} ({100 * row['success_rate']:.0f} %) "
                     f"| {row['success_geometric']} | {row['libero_success']} | {counts['target']}/{counts['distractor']}/{counts['none']} "
                     f"| {row['mean_steps']:.1f} | {seconds} |")
    return "\n".join(lines)

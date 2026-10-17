"""RoboTwin joint-position demos and boundary-padded action windows."""
from pathlib import Path
import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from leap.normalizer import LinearNormalizer


def convert_episodes(input_dir, output, episodes):
    """Align observation[t] with joint_position[t+1], as RoboTwin DP3 does."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    arrays = {key: [] for key in ("point_cloud", "state", "action")}
    ends = []
    for i in range(episodes):
        path = Path(input_dir) / f"episode{i}.hdf5"
        with h5py.File(path, "r") as episode:
            state = np.asarray(episode["joint_action/vector"], dtype=np.float32)
            pc = np.asarray(episode["pointcloud"], dtype=np.float32)
        if state.ndim != 2 or pc.ndim != 3 or len(state) != len(pc) or len(state) < 2:
            raise ValueError(f"Invalid episode shapes in {path}: {state.shape}, {pc.shape}")
        if not np.isfinite(state).all() or not np.isfinite(pc).all():
            raise ValueError(f"Non-finite data in {path}")
        arrays["state"].append(state[:-1])
        arrays["action"].append(state[1:])
        arrays["point_cloud"].append(pc[:-1])
        ends.append((ends[-1] if ends else 0) + len(state) - 1)
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    root = zarr.open_group(str(output), mode="w-")
    data = root.create_group("data")
    for key, value in arrays.items():
        data.create_dataset(key, data=value, chunks=(min(100, len(value)), *value.shape[1:]),
                            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1))
    root.create_group("meta").create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    return {"episodes": episodes, "frames": ends[-1], "output": str(output)}


def load_instructions(instr_path, episode_ends, modalities):
    """Read an instruction bundle's embeddings.npz for the episodes of a Zarr.

    Returns ({modality: [E, variants, D] float32}, side [E] int64). The file's
    episode_ends must equal the Zarr's; `modalities` selects the arrays used.
    """
    modalities = tuple(modalities)
    if not modalities:
        raise ValueError("instr_modalities must name at least one modality")
    with np.load(instr_path) as bundle:
        if not np.array_equal(np.asarray(bundle["episode_ends"]), np.asarray(episode_ends)):
            raise ValueError(f"episode_ends in {instr_path} do not match the dataset")
        missing = [name for name in modalities if name not in bundle.files]
        if missing:
            raise ValueError(f"Modalities {missing} are not in {instr_path} (has {bundle.files})")
        embeddings = {name: np.asarray(bundle[name], dtype=np.float32) for name in modalities}
        side = np.asarray(bundle["side"], dtype=np.int64)
    dims = {value.shape[-1] for value in embeddings.values()}
    if any(value.ndim != 3 or len(value) != len(episode_ends) or value.shape[1] < 1
           for value in embeddings.values()) or len(dims) != 1 or len(side) != len(episode_ends):
        raise ValueError(f"Instruction arrays in {instr_path} must be [episodes, variants, dim] with one dim")
    if not all(np.isfinite(value).all() for value in embeddings.values()):
        raise ValueError(f"Non-finite embeddings in {instr_path}")
    return embeddings, side


class RobotDataset(Dataset):
    """Matches the research SequenceSampler's edge padding without its framework.

    With `instr_path` (an instruction bundle's embeddings.npz) every window also
    carries obs["instr"], one [D] embedding of its episode: a modality drawn
    uniformly from `instr_modalities`, then a variant drawn uniformly. The
    draws use torch's RNG, which the DataLoader seeds per worker.
    """
    def __init__(self, path, horizon=8, n_obs_steps=3, n_action_steps=6, episodes=None,
                 val_ratio=0.0, seed=0, instr_path=None, instr_modalities=("text", "sketch", "audio")):
        if not (horizon >= 1 and 1 <= n_obs_steps <= horizon
                and 1 <= n_action_steps <= horizon - n_obs_steps + 1):
            raise ValueError("Invalid horizon, observation steps or action steps")
        root = zarr.open_group(str(path), mode="r")
        ends = np.asarray(root["meta/episode_ends"])
        self.instr = self.instr_side = None
        self.instr_modalities = tuple(instr_modalities)
        if instr_path is not None:
            self.instr, self.instr_side = load_instructions(instr_path, ends, self.instr_modalities)
        if episodes is not None:
            if not 1 <= episodes <= len(ends):
                raise ValueError("episodes exceeds available episodes or is non-positive")
            ends = ends[:episodes]
            if self.instr is not None:
                self.instr = {key: value[:episodes] for key, value in self.instr.items()}
                self.instr_side = self.instr_side[:episodes]
        if not len(ends) or ends[0] <= 0 or np.any(np.diff(ends) <= 0):
            raise ValueError("episode_ends must be positive and strictly increasing")
        self.arrays = {key: np.asarray(root[f"data/{key}"][:int(ends[-1])], dtype=np.float32)
                       for key in ("state", "action", "point_cloud")}
        if any(len(value) != ends[-1] for value in self.arrays.values()):
            raise ValueError("Dataset lengths do not match episode_ends")
        self.horizon, self.ends = horizon, ends
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        val_mask = np.zeros(len(ends), dtype=bool)
        if val_ratio > 0:
            count = min(max(1, round(len(ends) * val_ratio)), len(ends) - 1)
            val_mask[np.random.default_rng(seed).choice(len(ends), size=count, replace=False)] = True
        self.train_episodes = np.flatnonzero(~val_mask)
        self.indices = []
        self.val_indices = []
        begin = 0
        for episode, end in enumerate(ends):
            for offset in range(-(n_obs_steps - 1), int(end) - begin - horizon + n_action_steps):
                target = self.val_indices if val_mask[episode] else self.indices
                target.append((begin, int(end), offset, episode))
            begin = int(end)
        if not self.indices:
            raise ValueError("No training windows; episodes are too short")

    def __len__(self):
        return len(self.indices)

    def get_validation_dataset(self):
        dataset = copy.copy(self)
        dataset.indices = self.val_indices
        return dataset

    def __getitem__(self, index):
        begin, end, offset, episode = self.indices[index]
        rows = np.clip(begin + offset + np.arange(self.horizon), begin, end - 1)
        sample = {key: torch.from_numpy(value[rows]) for key, value in self.arrays.items()}
        obs = {"agent_pos": sample["state"], "point_cloud": sample["point_cloud"]}
        if self.instr is not None:
            obs["instr"] = self.sample_instruction(episode)
        return {"obs": obs, "action": sample["action"]}

    def sample_instruction(self, episode):
        """One [D] float32 embedding of `episode`: uniform modality, then uniform variant."""
        modality = self.instr_modalities[int(torch.randint(len(self.instr_modalities), ()))]
        variants = self.instr[modality][episode]
        return torch.from_numpy(variants[int(torch.randint(len(variants), ()))])

    def instr_stats(self):
        """Per-dimension (mean, std) of all variants of all `instr_modalities` over the
        training episodes, as float32 tensors [D]; std is floored at 1e-6."""
        if self.instr is None:
            raise RuntimeError("Dataset was built without instr_path")
        vectors = np.concatenate([self.instr[name][self.train_episodes].reshape(-1, self.instr_dim)
                                  for name in self.instr_modalities]).astype(np.float64)
        mean = vectors.mean(axis=0)
        std = np.maximum(vectors.std(axis=0), 1e-6)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))

    @property
    def instr_dim(self):
        return None if self.instr is None else next(iter(self.instr.values())).shape[-1]

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer.fit({"agent_pos": self.arrays["state"], "action": self.arrays["action"],
                        "point_cloud": self.arrays["point_cloud"]}, last_n_dims=1, mode="limits")
        return normalizer

    @property
    def shape_meta(self):
        obs = {"agent_pos": {"shape": list(self.arrays["state"].shape[1:])},
               "point_cloud": {"shape": list(self.arrays["point_cloud"].shape[1:])}}
        if self.instr is not None:
            obs["instr"] = {"shape": [self.instr_dim]}
        return {"obs": obs, "action": {"shape": list(self.arrays["action"].shape[1:])}}
